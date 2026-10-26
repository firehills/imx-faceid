"""NATS request handlers for enrolment, identification, and user management."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import nats
from nats.aio.client import Client as NatsClient
from nats.aio.msg import Msg

from imx_faceid.config import AppConfig
from imx_faceid.flatbuffer import (
    BODY_TYPES,
    FlatBufferError,
    EnvelopeMessage,
    Table,
    make_debug_response,
    make_delete_response,
    make_enrol_result,
    make_id_result,
    make_user_info_response,
    parse_envelope,
    read_byte_vector,
    read_float_vector,
    read_keypoints,
)
from imx_faceid.inference import EmbeddingModel
from imx_faceid.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _required_string(body: Table, field: int, name: str) -> str:
    value = body.string(field, required=True)
    if not value:
        raise FlatBufferError(f"Required field '{name}' must not be empty")
    return value


def _reply_to(body: Table, field: int) -> str | None:
    value = body.string(field)
    return value or None


class FaceIdService:
    """Subscribe to configured NATS subjects and answer FlatBuffer requests."""

    def __init__(
        self,
        config: AppConfig,
        model: EmbeddingModel,
        store: VectorStore,
    ) -> None:
        self.config = config
        self.model = model
        self.store = store
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._nc: NatsClient | None = None

    async def run(self, stop: asyncio.Event) -> None:
        nats_config = self.config.nats
        connect_options: dict[str, Any] = {
            "servers": [nats_config.url],
            "name": "imx-faceid",
        }
        if nats_config.credentials_file:
            connect_options["user_credentials"] = nats_config.credentials_file
        elif nats_config.token:
            connect_options["token"] = nats_config.token
        elif nats_config.username:
            connect_options["user"] = nats_config.username
            connect_options["password"] = nats_config.password

        try:
            self._nc = await nats.connect(**connect_options)
            subscriptions = (
                (nats_config.enrol_subject, self.handle_enrol),
                (nats_config.query_subject, self.handle_query),
                (nats_config.user_info_subject, self.handle_user_info),
                (nats_config.delete_subject, self.handle_delete_user),
                (nats_config.debug_subject, self.handle_debug_info),
            )
            for subject, handler in subscriptions:
                await self._nc.subscribe(subject, cb=handler)
                logger.info(
                    "Listening for %s requests on %s", handler.__name__, subject
                )
            await stop.wait()
        finally:
            try:
                if self._nc is not None:
                    await self._nc.drain()
            finally:
                try:
                    if self._nc is not None:
                        await self._nc.close()
                finally:
                    self._nc = None
                    self.store.close()

    async def _publish(self, reply_to: str | None, payload: bytes) -> None:
        if not reply_to:
            return
        if self._nc is None:
            raise RuntimeError("NATS connection is not available")
        self._log_verbose_response(reply_to, payload)
        await self._nc.publish(reply_to, payload)

    @staticmethod
    def _parse(data: bytes, expected_body: str) -> EnvelopeMessage:
        message = parse_envelope(data)
        if message.body_type != BODY_TYPES[expected_body]:
            raise FlatBufferError(
                f"Expected {expected_body}, received body type {message.body_type}"
            )
        return message

    def _user_lock(self, user_id: str) -> asyncio.Lock:
        lock = self._user_locks.get(user_id)
        if lock is None:
            lock = asyncio.Lock()
            self._user_locks[user_id] = lock
        return lock

    def _log_verbose_request(self, operation: str, **fields: Any) -> None:
        if self.config.debug != "verbose":
            return
        details = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        logger.debug("%s request: %s", operation, details)

    def _log_verbose_response(self, reply_to: str, payload: bytes) -> None:
        if self.config.debug != "verbose":
            return
        envelope = parse_envelope(payload)
        body_type = envelope.body_type
        body = envelope.body
        if body_type == BODY_TYPES["EnrolResultMsg"]:
            fields = {
                "reply_to": reply_to,
                "user_id": body.string(0),
                "success": bool(body.scalar(1, "<?")),
                "reason": body.string(2) or "",
                "time_unix_ms": int(body.scalar(3, "<q")),
            }
            operation = "enrol_result"
        elif body_type == BODY_TYPES["IdResultMsg"]:
            candidates = body.table_vector(1)
            fields = {
                "reply_to": reply_to,
                "track_id": body.string(0),
                "success": bool(body.scalar(3, "<?")),
                "reason": body.string(4) or "",
                "time_unix_ms": int(body.scalar(2, "<q")),
                "candidate_count": len(candidates),
                "candidates": [
                    {
                        "user_id": candidate.string(0),
                        "name": candidate.string(1) or "",
                        "confidence": float(candidate.scalar(2, "<f")),
                        "raw_distance": float(candidate.scalar(3, "<f")),
                    }
                    for candidate in candidates
                ],
            }
            operation = "query_result"
        elif body_type == BODY_TYPES["UserInfoResponseMsg"]:
            enrolments = body.table_vector(2)
            fields = {
                "reply_to": reply_to,
                "user_id": body.string(0),
                "num_enrolments": int(body.scalar(1, "<I")),
                "success": bool(body.scalar(3, "<?")),
                "reason": body.string(4) or "",
                "time_unix_ms": int(body.scalar(5, "<q")),
                "enrolments": [
                    {
                        "embedding_dim": len(read_float_vector(record, 0)),
                        "time_unix_ms": int(record.scalar(1, "<q")),
                    }
                    for record in enrolments
                ],
            }
            operation = "user_info_result"
        elif body_type == BODY_TYPES["DeleteUserResponseMsg"]:
            fields = {
                "reply_to": reply_to,
                "user_id": body.string(0),
                "success": bool(body.scalar(1, "<?")),
                "reason": body.string(2) or "",
                "time_unix_ms": int(body.scalar(3, "<q")),
            }
            operation = "delete_user_result"
        elif body_type == BODY_TYPES["DebugInfoResponseMsg"]:
            fields = {
                "reply_to": reply_to,
                "database_type": body.string(0) or "",
                "database_version": body.string(1) or "",
                "inference_backend": body.string(2) or "",
                "model_name": body.string(3) or "",
                "embedding_dim": int(body.scalar(4, "<I")),
                "nats_url": body.string(5) or "",
                "success": bool(body.scalar(6, "<?")),
                "reason": body.string(7) or "",
                "time_unix_ms": int(body.scalar(8, "<q")),
            }
            operation = "debug_info_result"
        else:
            fields = {"reply_to": reply_to, "body_type": body_type}
            operation = "unknown_result"
        details = ", ".join(f"{key}={value!r}" for key, value in fields.items())
        logger.debug("%s response: %s", operation, details)

    async def handle_enrol(self, msg: Msg) -> None:
        body: Table | None = None
        user_id = ""
        reply_to = None
        try:
            envelope = self._parse(msg.data, "EnrolMsg")
            body = envelope.body
            user_id = _required_string(body, 0, "user_id")
            name = _required_string(body, 1, "name")
            reply_to = _reply_to(body, 8)
            image = read_byte_vector(body, 2, required=True)
            height = int(body.scalar(3, "<I"))
            width = int(body.scalar(4, "<I"))
            is_preprocessed = bool(body.scalar(5, "<?"))
            encoding = int(body.scalar(6, "<B"))
            keypoints = read_keypoints(body, 7)
            request_time_unix_ms = int(body.scalar(9, "<q"))
            custom = body.string(10) or ""
            self._log_verbose_request(
                "enrol",
                subject=msg.subject,
                user_id=user_id,
                name=name,
                reply_to=reply_to,
                encoding=encoding,
                width=width,
                height=height,
                is_preprocessed=is_preprocessed,
                keypoints=keypoints,
                keypoints_count=len(keypoints),
                image_len=len(image),
                time_unix_ms=request_time_unix_ms,
                custom=custom,
            )
            embedding = await asyncio.to_thread(
                self.model.embed,
                image,
                encoding=encoding,
                width=width,
                height=height,
                is_preprocessed=is_preprocessed,
                keypoints=keypoints,
            )
            time_unix_ms = request_time_unix_ms or int(time.time() * 1000)
            async with self._user_lock(user_id):
                await asyncio.to_thread(
                    self.store.enroll,
                    user_id=user_id,
                    name=name,
                    custom=custom,
                    embedding=embedding,
                    image_bmp=image,
                    time_unix_ms=time_unix_ms,
                )
            await self._publish(reply_to, make_enrol_result(user_id, True))
        except Exception as exc:
            logger.exception("Enrol request failed for user %s", user_id or "<unknown>")
            await self._publish(reply_to, make_enrol_result(user_id, False, str(exc)))

    async def handle_query(self, msg: Msg) -> None:
        track_id = ""
        reply_to = None
        try:
            envelope = self._parse(msg.data, "IdRequestMsg")
            body = envelope.body
            track_id = _required_string(body, 0, "track_id")
            reply_to = _required_string(body, 7, "reply_to")
            image = read_byte_vector(body, 1, required=True)
            height = int(body.scalar(2, "<I"))
            width = int(body.scalar(3, "<I"))
            is_preprocessed = bool(body.scalar(4, "<?"))
            encoding = int(body.scalar(5, "<B"))
            keypoints = read_keypoints(body, 6)
            min_confidence = float(body.scalar(8, "<f", 0.5))
            max_results = int(body.scalar(9, "<I", 1))
            request_pts = int(body.scalar(10, "<q"))
            request_time_unix_ms = int(body.scalar(11, "<q"))
            custom = body.string(12) or ""
            self._log_verbose_request(
                "query",
                subject=msg.subject,
                track_id=track_id,
                reply_to=reply_to,
                encoding=encoding,
                width=width,
                height=height,
                is_preprocessed=is_preprocessed,
                keypoints=keypoints,
                keypoints_count=len(keypoints),
                image_len=len(image),
                min_confidence=min_confidence,
                max_results=max_results,
                pts=request_pts,
                time_unix_ms=request_time_unix_ms,
                custom=custom,
            )
            embedding = await asyncio.to_thread(
                self.model.embed,
                image,
                encoding=encoding,
                width=width,
                height=height,
                is_preprocessed=is_preprocessed,
                keypoints=keypoints,
            )
            if max_results < 1:
                raise ValueError("max_results must be at least one")
            if not 0.0 <= min_confidence <= 1.0:
                raise ValueError("min_confidence must be in the range 0..1")
            matches = await asyncio.to_thread(
                self.store.query,
                embedding,
                max_results=max_results,
                min_confidence=min_confidence,
            )
            candidates = [
                {
                    "user_id": match.user_id,
                    "name": match.name,
                    "confidence": match.confidence,
                    "raw_distance": match.raw_distance,
                }
                for match in matches
            ]
            await self._publish(reply_to, make_id_result(track_id, candidates, True))
        except Exception as exc:
            logger.exception("Identity query failed for track %s", track_id or "<unknown>")
            await self._publish(
                reply_to, make_id_result(track_id, [], False, str(exc))
            )

    async def handle_user_info(self, msg: Msg) -> None:
        user_id = ""
        reply_to = None
        try:
            envelope = self._parse(msg.data, "UserInfoMsg")
            body = envelope.body
            user_id = _required_string(body, 0, "user_id")
            reply_to = _reply_to(body, 1)
            request_time_unix_ms = int(body.scalar(2, "<q"))
            self._log_verbose_request(
                "user_info",
                subject=msg.subject,
                user_id=user_id,
                reply_to=reply_to,
                time_unix_ms=request_time_unix_ms,
            )
            enrolments = await asyncio.to_thread(self.store.user_info, user_id)
            await self._publish(
                reply_to, make_user_info_response(user_id, enrolments, True)
            )
        except Exception as exc:
            logger.exception("User info request failed for user %s", user_id or "<unknown>")
            await self._publish(
                reply_to, make_user_info_response(user_id, [], False, str(exc))
            )

    async def handle_delete_user(self, msg: Msg) -> None:
        user_id = ""
        reply_to = None
        try:
            envelope = self._parse(msg.data, "DeleteUserMsg")
            body = envelope.body
            user_id = _required_string(body, 0, "user_id")
            reply_to = _reply_to(body, 1)
            request_time_unix_ms = int(body.scalar(2, "<q"))
            self._log_verbose_request(
                "delete_user",
                subject=msg.subject,
                user_id=user_id,
                reply_to=reply_to,
                time_unix_ms=request_time_unix_ms,
            )
            async with self._user_lock(user_id):
                await asyncio.to_thread(self.store.delete_user, user_id)
            await self._publish(reply_to, make_delete_response(user_id, True))
        except Exception as exc:
            logger.exception("User deletion failed for user %s", user_id or "<unknown>")
            await self._publish(
                reply_to, make_delete_response(user_id, False, str(exc))
            )

    async def handle_debug_info(self, msg: Msg) -> None:
        reply_to = None
        try:
            envelope = self._parse(msg.data, "DebugInfoMsg")
            reply_to = _reply_to(envelope.body, 0)
            request_time_unix_ms = int(envelope.body.scalar(1, "<q"))
            self._log_verbose_request(
                "debug_info",
                subject=msg.subject,
                reply_to=reply_to,
                time_unix_ms=request_time_unix_ms,
            )
            database_type, database_version = await asyncio.to_thread(
                self.store.debug_info
            )
            response = make_debug_response(
                database_type=database_type,
                database_version=database_version,
                inference_backend=self.config.model.runtime,
                model_name=self.config.model.path.name,
                embedding_dim=self.model.embedding_dim,
                nats_url=self.config.nats.url,
                success=True,
            )
            await self._publish(reply_to, response)
        except Exception as exc:
            logger.exception("Debug info request failed")
            response = make_debug_response(
                database_type="qdrant",
                database_version="",
                inference_backend=self.config.model.runtime,
                model_name=self.config.model.path.name,
                embedding_dim=self.model.embedding_dim,
                nats_url=self.config.nats.url,
                success=False,
                reason=str(exc),
            )
            await self._publish(reply_to, response)
