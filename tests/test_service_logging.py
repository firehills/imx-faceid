from pathlib import Path
from types import SimpleNamespace

from imx_faceid.flatbuffer import (
    make_debug_response,
    make_delete_response,
    make_enrol_result,
    make_id_result,
    make_user_info_response,
)
from imx_faceid.service import FaceIdService


class DummyStore:
    def close(self):
        pass


def _service(debug: str) -> FaceIdService:
    config = SimpleNamespace(
        debug=debug,
        nats=SimpleNamespace(url="nats://localhost:4222"),
        model=SimpleNamespace(runtime="onnx", path=Path("model.onnx")),
    )
    model = SimpleNamespace(embedding_dim=192)
    return FaceIdService(config=config, model=model, store=DummyStore())


def test_verbose_request_logging_enabled(caplog):
    service = _service("verbose")

    with caplog.at_level("DEBUG"):
        service._log_verbose_request("query", track_id="t-1", image_len=123)

    assert "query request: track_id='t-1', image_len=123" in caplog.text


def test_verbose_request_logging_disabled_for_info(caplog):
    service = _service("info")

    with caplog.at_level("DEBUG"):
        service._log_verbose_request("query", track_id="t-1", image_len=123)

    assert caplog.text == ""


def test_verbose_response_logging_decodes_outgoing_messages(caplog):
    service = _service("verbose")
    payloads = [
        make_enrol_result("user-1", True, time_unix_ms=1),
        make_id_result(
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
            time_unix_ms=2,
        ),
        make_user_info_response(
            "user-1",
            [{"embedding": [0.6, 0.8], "time_unix_ms": 3}],
            True,
            time_unix_ms=4,
        ),
        make_delete_response("user-1", True, time_unix_ms=5),
        make_debug_response(
            database_type="qdrant",
            database_version="1.15",
            inference_backend="onnx",
            model_name="face.onnx",
            embedding_dim=192,
            nats_url="nats://localhost:4222",
            success=True,
            time_unix_ms=6,
        ),
    ]

    with caplog.at_level("DEBUG"):
        for payload in payloads:
            service._log_verbose_response("faceid.reply", payload)

    assert "enrol_result response: reply_to='faceid.reply', user_id='user-1'" in caplog.text
    assert (
        "query_result response: reply_to='faceid.reply', track_id='track-1', "
        "success=True" in caplog.text
    )
    assert "candidate_count=1" in caplog.text
    assert "user_info_result response: reply_to='faceid.reply', user_id='user-1'" in caplog.text
    assert "'embedding_dim': 2" in caplog.text
    assert "delete_user_result response: reply_to='faceid.reply', user_id='user-1'" in caplog.text
    assert (
        "debug_info_result response: reply_to='faceid.reply', database_type='qdrant'"
        in caplog.text
    )


def test_verbose_response_logging_disabled_for_info(caplog):
    service = _service("info")

    with caplog.at_level("DEBUG"):
        service._log_verbose_response("faceid.reply", make_enrol_result("user-1", True))

    assert caplog.text == ""
