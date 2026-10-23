"""FlatBuffer readers and writers for the message types in schemas/faceid.fbs."""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from typing import Any

import flatbuffers
import numpy as np


class FlatBufferError(ValueError):
    """Raised for malformed or unsupported FlatBuffer messages."""


BODY_TYPES = {
    "IdRequestMsg": 1,
    "IdResultMsg": 2,
    "EnrolMsg": 3,
    "EnrolResultMsg": 4,
    "UserInfoMsg": 5,
    "UserInfoResponseMsg": 6,
    "DeleteUserMsg": 7,
    "DeleteUserResponseMsg": 8,
    "DebugInfoMsg": 9,
    "DebugInfoResponseMsg": 10,
}


class Table:
    """Bounds-checked view of a FlatBuffers table."""

    def __init__(self, buffer: bytes, position: int) -> None:
        self.buffer = buffer
        self.position = position
        self._check(position, 4)
        vtable_delta = struct.unpack_from("<i", buffer, position)[0]
        self.vtable = position - vtable_delta
        self._check(self.vtable, 4)
        self.vtable_length = struct.unpack_from("<H", buffer, self.vtable)[0]

    def _check(self, position: int, size: int) -> None:
        if position < 0 or size < 0 or position + size > len(self.buffer):
            raise FlatBufferError("FlatBuffer field is outside the message")

    def field(self, index: int, size: int) -> int | None:
        entry = self.vtable + 4 + index * 2
        if entry + 2 > self.vtable + self.vtable_length:
            return None
        offset = struct.unpack_from("<H", self.buffer, entry)[0]
        if offset == 0:
            return None
        position = self.position + offset
        self._check(position, size)
        return position

    def scalar(self, index: int, fmt: str, default: Any = 0) -> Any:
        size = struct.calcsize(fmt)
        position = self.field(index, size)
        return default if position is None else struct.unpack_from(fmt, self.buffer, position)[0]

    def indirect(self, index: int) -> int | None:
        position = self.field(index, 4)
        if position is None:
            return None
        target = position + struct.unpack_from("<I", self.buffer, position)[0]
        self._check(target, 4)
        return target

    def string(self, index: int, *, required: bool = False) -> str | None:
        target = self.indirect(index)
        if target is None:
            if required:
                raise FlatBufferError("Required string field is missing")
            return None
        length = struct.unpack_from("<I", self.buffer, target)[0]
        start = target + 4
        self._check(start, length + 1)
        if self.buffer[start + length] != 0:
            raise FlatBufferError("FlatBuffer string is not null-terminated")
        try:
            return self.buffer[start : start + length].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FlatBufferError("FlatBuffer string is not valid UTF-8") from exc

    def vector(self, index: int, item_size: int) -> tuple[int, int] | None:
        target = self.indirect(index)
        if target is None:
            return None
        length = struct.unpack_from("<I", self.buffer, target)[0]
        start = target + 4
        self._check(start, length * item_size)
        return start, length

    def table_vector(self, index: int) -> list[Table]:
        vector = self.vector(index, 4)
        if vector is None:
            return []
        start, length = vector
        result = []
        for item_index in range(length):
            item = start + item_index * 4
            target = item + struct.unpack_from("<I", self.buffer, item)[0]
            result.append(Table(self.buffer, target))
        return result


@dataclass(frozen=True)
class EnvelopeMessage:
    body_type: int
    body: Table


def parse_envelope(buffer: bytes) -> EnvelopeMessage:
    """Parse the root envelope and its union table."""
    if len(buffer) < 8 or buffer[4:8] != b"FACE":
        raise FlatBufferError("Message does not have the FACE FlatBuffer identifier")
    root = struct.unpack_from("<I", buffer, 0)[0]
    envelope = Table(buffer, root)
    body_type = int(envelope.scalar(0, "<B"))
    body_position = envelope.indirect(1)
    if body_position is None:
        raise FlatBufferError("Envelope does not contain a body")
    if body_type not in BODY_TYPES.values():
        raise FlatBufferError(f"Unsupported Envelope body type: {body_type}")
    return EnvelopeMessage(body_type=body_type, body=Table(buffer, body_position))


def read_byte_vector(table: Table, field: int, *, required: bool = False) -> bytes:
    vector = table.vector(field, 1)
    if vector is None:
        if required:
            raise FlatBufferError("Required image vector is missing")
        return b""
    start, length = vector
    return table.buffer[start : start + length]


def read_float_vector(table: Table, field: int) -> list[float]:
    vector = table.vector(field, 4)
    if vector is None:
        return []
    start, length = vector
    return list(struct.unpack_from(f"<{length}f", table.buffer, start))


def read_keypoints(table: Table, field: int) -> list[tuple[float, float]]:
    vector = table.vector(field, 8)
    if vector is None:
        return []
    start, length = vector
    return [
        struct.unpack_from("<ff", table.buffer, start + index * 8)
        for index in range(length)
    ]


def _create_string(builder: flatbuffers.Builder, value: str | None) -> int:
    return 0 if value is None else builder.CreateString(value)


def _float_vector(builder: flatbuffers.Builder, values: list[float] | np.ndarray) -> int:
    builder.StartVector(4, len(values), 4)
    for value in reversed(values):
        builder.PrependFloat32(float(value))
    return builder.EndVector()


def _table_vector(builder: flatbuffers.Builder, offsets: list[int]) -> int:
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return builder.EndVector()


def _finish_message(builder: flatbuffers.Builder, body_type: int, body_offset: int) -> bytes:
    builder.StartObject(2)
    builder.PrependUint8Slot(0, body_type, 0)
    builder.PrependUOffsetTRelativeSlot(1, body_offset, 0)
    envelope_offset = builder.EndObject()
    builder.Finish(envelope_offset, file_identifier=b"FACE")
    return bytes(builder.Output())


def _finish_table(
    builder: flatbuffers.Builder,
    body_type: int,
    field_count: int,
    add_fields: Any,
) -> bytes:
    builder.StartObject(field_count)
    add_fields(builder)
    return _finish_message(builder, body_type, builder.EndObject())


def make_enrol_result(
    user_id: str, success: bool, reason: str = "", time_unix_ms: int | None = None
) -> bytes:
    builder = flatbuffers.Builder(256)
    user_offset = _create_string(builder, user_id)
    reason_offset = _create_string(builder, reason)

    def add_fields(b: flatbuffers.Builder) -> None:
        b.PrependUOffsetTRelativeSlot(0, user_offset, 0)
        b.PrependBoolSlot(1, success, False)
        b.PrependUOffsetTRelativeSlot(2, reason_offset, 0)
        b.PrependInt64Slot(3, time_unix_ms or int(time.time() * 1000), 0)

    return _finish_table(builder, BODY_TYPES["EnrolResultMsg"], 4, add_fields)


def make_id_result(
    track_id: str,
    candidates: list[dict[str, Any]],
    success: bool,
    reason: str = "",
    time_unix_ms: int | None = None,
) -> bytes:
    builder = flatbuffers.Builder(512)
    track_offset = _create_string(builder, track_id)
    candidate_offsets = []
    for candidate in candidates:
        user_offset = _create_string(builder, candidate["user_id"])
        name_offset = _create_string(builder, candidate["name"])
        builder.StartObject(4)
        builder.PrependUOffsetTRelativeSlot(0, user_offset, 0)
        builder.PrependUOffsetTRelativeSlot(1, name_offset, 0)
        builder.PrependFloat32Slot(2, candidate["confidence"], 0.0)
        builder.PrependFloat32Slot(3, candidate["raw_distance"], 0.0)
        candidate_offsets.append(builder.EndObject())
    candidates_offset = _table_vector(builder, candidate_offsets)
    reason_offset = _create_string(builder, reason)

    def add_fields(b: flatbuffers.Builder) -> None:
        b.PrependUOffsetTRelativeSlot(0, track_offset, 0)
        b.PrependUOffsetTRelativeSlot(1, candidates_offset, 0)
        b.PrependInt64Slot(2, time_unix_ms or int(time.time() * 1000), 0)
        b.PrependBoolSlot(3, success, False)
        b.PrependUOffsetTRelativeSlot(4, reason_offset, 0)

    return _finish_table(builder, BODY_TYPES["IdResultMsg"], 5, add_fields)


def make_user_info_response(
    user_id: str,
    enrolments: list[dict[str, Any]],
    success: bool,
    reason: str = "",
    time_unix_ms: int | None = None,
) -> bytes:
    builder = flatbuffers.Builder(512)
    user_offset = _create_string(builder, user_id)
    enrolment_offsets = []
    for enrolment in enrolments:
        embedding_offset = _float_vector(builder, enrolment["embedding"])
        builder.StartObject(2)
        builder.PrependUOffsetTRelativeSlot(0, embedding_offset, 0)
        builder.PrependInt64Slot(1, enrolment["time_unix_ms"], 0)
        enrolment_offsets.append(builder.EndObject())
    enrolments_offset = _table_vector(builder, enrolment_offsets)
    reason_offset = _create_string(builder, reason)

    def add_fields(b: flatbuffers.Builder) -> None:
        b.PrependUOffsetTRelativeSlot(0, user_offset, 0)
        b.PrependUint32Slot(1, len(enrolments), 0)
        b.PrependUOffsetTRelativeSlot(2, enrolments_offset, 0)
        b.PrependBoolSlot(3, success, False)
        b.PrependUOffsetTRelativeSlot(4, reason_offset, 0)
        b.PrependInt64Slot(5, time_unix_ms or int(time.time() * 1000), 0)

    return _finish_table(builder, BODY_TYPES["UserInfoResponseMsg"], 6, add_fields)


def make_delete_response(
    user_id: str, success: bool, reason: str = "", time_unix_ms: int | None = None
) -> bytes:
    builder = flatbuffers.Builder(256)
    user_offset = _create_string(builder, user_id)
    reason_offset = _create_string(builder, reason)

    def add_fields(b: flatbuffers.Builder) -> None:
        b.PrependUOffsetTRelativeSlot(0, user_offset, 0)
        b.PrependBoolSlot(1, success, False)
        b.PrependUOffsetTRelativeSlot(2, reason_offset, 0)
        b.PrependInt64Slot(3, time_unix_ms or int(time.time() * 1000), 0)

    return _finish_table(builder, BODY_TYPES["DeleteUserResponseMsg"], 4, add_fields)


def make_debug_response(
    *,
    database_type: str,
    database_version: str,
    inference_backend: str,
    model_name: str,
    embedding_dim: int,
    nats_url: str,
    success: bool,
    reason: str = "",
    time_unix_ms: int | None = None,
) -> bytes:
    builder = flatbuffers.Builder(512)
    strings = [
        _create_string(builder, value)
        for value in (
            database_type,
            database_version,
            inference_backend,
            model_name,
            nats_url,
            reason,
        )
    ]

    def add_fields(b: flatbuffers.Builder) -> None:
        for index, offset in enumerate(strings[:4]):
            b.PrependUOffsetTRelativeSlot(index, offset, 0)
        b.PrependUint32Slot(4, embedding_dim, 0)
        b.PrependUOffsetTRelativeSlot(5, strings[4], 0)
        b.PrependBoolSlot(6, success, False)
        b.PrependUOffsetTRelativeSlot(7, strings[5], 0)
        b.PrependInt64Slot(8, time_unix_ms or int(time.time() * 1000), 0)

    return _finish_table(builder, BODY_TYPES["DebugInfoResponseMsg"], 9, add_fields)
