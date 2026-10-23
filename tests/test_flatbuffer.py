import pytest
import flatbuffers

from imx_faceid.flatbuffer import (
    BODY_TYPES,
    FlatBufferError,
    make_debug_response,
    make_id_result,
    make_user_info_response,
    parse_envelope,
    read_byte_vector,
    read_keypoints,
    read_float_vector,
)


def test_id_result_includes_distance_and_confidence():
    payload = make_id_result(
        "track-1",
        [
            {
                "user_id": "user-1",
                "name": "Person",
                "confidence": 0.9,
                "raw_distance": 0.2,
            }
        ],
        True,
        time_unix_ms=1234,
    )

    envelope = parse_envelope(payload)
    assert envelope.body_type == BODY_TYPES["IdResultMsg"]
    assert envelope.body.string(0) == "track-1"
    assert envelope.body.scalar(2, "<q") == 1234
    assert envelope.body.scalar(3, "<?") is True
    candidate = envelope.body.table_vector(1)[0]
    assert candidate.string(0) == "user-1"
    assert candidate.string(1) == "Person"
    assert candidate.scalar(2, "<f") == pytest.approx(0.9)
    assert candidate.scalar(3, "<f") == pytest.approx(0.2)


def test_user_info_returns_each_embedding_and_timestamp():
    payload = make_user_info_response(
        "user-1",
        [
            {"embedding": [0.6, 0.8], "time_unix_ms": 77},
            {"embedding": [1.0, 0.0], "time_unix_ms": 88},
        ],
        True,
    )
    body = parse_envelope(payload).body
    records = body.table_vector(2)

    assert body.string(0) == "user-1"
    assert body.scalar(1, "<I") == 2
    assert [read_float_vector(record, 0) for record in records] == [
        pytest.approx([0.6, 0.8]),
        pytest.approx([1.0, 0.0]),
    ]
    assert [record.scalar(1, "<q") for record in records] == [77, 88]


def test_debug_response_uses_schema_field_indexes():
    payload = make_debug_response(
        database_type="qdrant",
        database_version="1.15",
        inference_backend="onnx",
        model_name="face.onnx",
        embedding_dim=192,
        nats_url="nats://localhost:4222",
        success=True,
        reason="",
        time_unix_ms=12,
    )
    body = parse_envelope(payload).body

    assert body.string(0) == "qdrant"
    assert body.string(1) == "1.15"
    assert body.string(2) == "onnx"
    assert body.string(3) == "face.onnx"
    assert body.scalar(4, "<I") == 192
    assert body.string(5) == "nats://localhost:4222"
    assert body.scalar(6, "<?") is True
    assert body.string(7) == ""
    assert body.scalar(8, "<q") == 12


def test_parse_rejects_non_face_flatbuffer():
    with pytest.raises(FlatBufferError, match="FACE"):
        parse_envelope(b"not a flatbuffer")


def test_parse_enrol_request_field_indexes_and_keypoints():
    builder = flatbuffers.Builder(256)
    user_id = builder.CreateString("user-1")
    name = builder.CreateString("Name")
    image = builder.CreateByteVector(b"bmp data")
    reply_to = builder.CreateString("faceid.response")
    custom = builder.CreateString("source")
    builder.StartVector(8, 2, 4)
    for x, y in reversed(((2.0, 3.0), (4.0, 5.0))):
        builder.Prep(4, 8)
        builder.PrependFloat32(y)
        builder.PrependFloat32(x)
    keypoints = builder.EndVector()
    builder.StartObject(11)
    builder.PrependUOffsetTRelativeSlot(0, user_id, 0)
    builder.PrependUOffsetTRelativeSlot(1, name, 0)
    builder.PrependUOffsetTRelativeSlot(2, image, 0)
    builder.PrependUint32Slot(3, 112, 0)
    builder.PrependUint32Slot(4, 112, 0)
    builder.PrependBoolSlot(5, True, False)
    builder.PrependUint8Slot(6, 3, 0)
    builder.PrependUOffsetTRelativeSlot(7, keypoints, 0)
    builder.PrependUOffsetTRelativeSlot(8, reply_to, 0)
    builder.PrependInt64Slot(9, 123, 0)
    builder.PrependUOffsetTRelativeSlot(10, custom, 0)
    enrolment = builder.EndObject()
    builder.StartObject(2)
    builder.PrependUint8Slot(0, BODY_TYPES["EnrolMsg"], 0)
    builder.PrependUOffsetTRelativeSlot(1, enrolment, 0)
    envelope = builder.EndObject()
    builder.Finish(envelope, file_identifier=b"FACE")

    body = parse_envelope(bytes(builder.Output())).body
    assert body.string(0) == "user-1"
    assert body.string(1) == "Name"
    assert read_byte_vector(body, 2, required=True) == b"bmp data"
    assert body.scalar(3, "<I") == 112  # height
    assert body.scalar(4, "<I") == 112  # width
    assert body.scalar(5, "<?") is True
    assert body.scalar(6, "<B") == 3
    assert read_keypoints(body, 7) == [(2.0, 3.0), (4.0, 5.0)]
    assert body.string(8) == "faceid.response"
    assert body.scalar(9, "<q") == 123
    assert body.string(10) == "source"
