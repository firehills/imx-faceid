"""Qdrant gRPC operations for raw and per-user enrolment vectors."""

from __future__ import annotations

import base64
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import grpc
import numpy as np
from qdrant_client import QdrantClient, models

from imx_faceid.config import QdrantConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    user_id: str
    name: str
    raw_distance: float
    confidence: float


def score_to_distance_confidence(score: float, metric: str) -> tuple[float, float]:
    """Map Qdrant's metric score to raw distance and bounded similarity."""
    if metric == "cosine":
        raw_distance = 1.0 - score
        confidence = (1.0 + score) / 2.0
    elif metric == "euclidean":
        raw_distance = score
        confidence = 1.0 - (score * score) / 4.0
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
    return raw_distance, min(1.0, max(0.0, confidence))


class VectorStore:
    """Keep individual enrolments and a normalized mean per user in Qdrant."""

    def __init__(self, config: QdrantConfig, embedding_dim: int) -> None:
        self.config = config
        self.embedding_dim = embedding_dim
        self._distance = (
            models.Distance.COSINE
            if config.metric == "cosine"
            else models.Distance.EUCLID
        )
        self._client = self._new_client(prefer_grpc=True)
        try:
            self._ensure_collection(config.raw_collection)
            self._ensure_collection(config.normalized_collection)
        except grpc.RpcError as exc:
            self._client.close()
            LOGGER.warning(
                "Qdrant gRPC endpoint %s:%s is unavailable; retrying via REST on %s:%s",
                config.host,
                config.grpc_port,
                config.host,
                config.port,
            )
            self._client = self._new_client(prefer_grpc=False)
            try:
                self._ensure_collection(config.raw_collection)
                self._ensure_collection(config.normalized_collection)
            except Exception:
                self._client.close()
                raise RuntimeError(
                    "Unable to connect to Qdrant via gRPC or REST. "
                    f"Checked gRPC at {config.host}:{config.grpc_port} and REST at "
                    f"{config.host}:{config.port}. Ensure Qdrant is running and the "
                    "configured host/ports are reachable."
                ) from exc

    def _new_client(self, *, prefer_grpc: bool) -> QdrantClient:
        return QdrantClient(
            host=self.config.host,
            port=self.config.port,
            grpc_port=self.config.grpc_port,
            prefer_grpc=prefer_grpc,
            check_compatibility=False,
            api_key=self.config.api_key,
            https=self.config.https,
        )

    def close(self) -> None:
        self._client.close()

    def _ensure_collection(self, name: str) -> None:
        if not self._client.collection_exists(name):
            self._client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=self.embedding_dim, distance=self._distance
                ),
            )
            return

        info = self._client.get_collection(name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            raise ValueError(
                f"Collection '{name}' uses named vectors; this service requires a single vector"
            )
        if vectors.size != self.embedding_dim:
            raise ValueError(
                f"Collection '{name}' has vector size {vectors.size}; "
                f"the model returns {self.embedding_dim}"
            )
        if vectors.distance != self._distance:
            raise ValueError(
                f"Collection '{name}' uses {vectors.distance}; configured metric is "
                f"{self.config.metric}"
            )

    def _user_filter(self, user_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        )

    def _scroll_user(self, user_id: str, *, with_vectors: bool) -> list[Any]:
        points: list[Any] = []
        offset = None
        while True:
            page, offset = self._client.scroll(
                collection_name=self.config.raw_collection,
                scroll_filter=self._user_filter(user_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            points.extend(page)
            if offset is None:
                return points

    def enroll(
        self,
        *,
        user_id: str,
        name: str,
        custom: str,
        embedding: np.ndarray,
        image_bmp: bytes,
        time_unix_ms: int,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.embedding_dim:
            raise ValueError(
                f"Embedding size {vector.size} does not match collection size "
                f"{self.embedding_dim}"
            )
        raw_point_id = str(uuid.uuid4())
        self._client.upsert(
            collection_name=self.config.raw_collection,
            points=[
                models.PointStruct(
                    id=raw_point_id,
                    vector=vector.tolist(),
                    payload={
                        "user_id": user_id,
                        "name": name,
                        "custom": custom,
                        "time_unix_ms": time_unix_ms,
                        "image": base64.b64encode(image_bmp).decode("ascii"),
                    },
                )
            ],
            wait=True,
        )

        raw_points = self._scroll_user(user_id, with_vectors=True)
        if not raw_points:
            raise RuntimeError(f"Qdrant returned no enrolments for user '{user_id}'")
        if any(point.payload.get("name") != name for point in raw_points):
            self._client.set_payload(
                collection_name=self.config.raw_collection,
                payload={"name": name},
                points=[point.id for point in raw_points],
                wait=True,
            )

        vectors = []
        for point in raw_points:
            if not isinstance(point.vector, list):
                raise ValueError("Raw enrolment point does not contain a single vector")
            values = np.asarray(point.vector, dtype=np.float32)
            if values.size != self.embedding_dim:
                raise ValueError("Stored enrolment has an unexpected vector dimension")
            vectors.append(values)
        mean = np.mean(np.stack(vectors), axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(mean))
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("The normalized mean of this user's enrolments is zero")
        mean /= norm
        self._client.delete(
            collection_name=self.config.normalized_collection,
            points_selector=models.FilterSelector(filter=self._user_filter(user_id)),
            wait=True,
        )
        self._client.upsert(
            collection_name=self.config.normalized_collection,
            points=[
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"imx-faceid:{user_id}")),
                    vector=mean.tolist(),
                    payload={
                        "user_id": user_id,
                        "name": name,
                        "num_enrolments": len(raw_points),
                        "time_unix_ms": time_unix_ms,
                    },
                )
            ],
            wait=True,
        )

    def query(
        self, embedding: np.ndarray, *, max_results: int, min_confidence: float
    ) -> list[Match]:
        if max_results <= 0:
            return []
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.size != self.embedding_dim:
            raise ValueError(
                f"Query embedding size {vector.size} does not match collection size "
                f"{self.embedding_dim}"
            )
        result = self._client.query_points(
            collection_name=self.config.normalized_collection,
            query=vector.tolist(),
            limit=max_results,
            with_payload=True,
        )
        matches = []
        for point in result.points:
            if not point.payload:
                raise ValueError("Normalized enrolment is missing its payload")
            raw_distance, confidence = score_to_distance_confidence(
                float(point.score), self.config.metric
            )
            if confidence < min_confidence:
                continue
            matches.append(
                Match(
                    user_id=str(point.payload["user_id"]),
                    name=str(point.payload.get("name", "")),
                    raw_distance=raw_distance,
                    confidence=confidence,
                )
            )
        return matches

    def user_info(self, user_id: str) -> list[dict[str, Any]]:
        points = self._scroll_user(user_id, with_vectors=True)
        enrolments = []
        for point in points:
            if not isinstance(point.vector, list):
                raise ValueError("Raw enrolment point does not contain a single vector")
            payload = point.payload or {}
            enrolments.append(
                {
                    "embedding": point.vector,
                    "time_unix_ms": int(payload.get("time_unix_ms", 0)),
                }
            )
        return enrolments

    def delete_user(self, user_id: str) -> None:
        user_filter = self._user_filter(user_id)
        self._client.delete(
            collection_name=self.config.raw_collection,
            points_selector=models.FilterSelector(filter=user_filter),
            wait=True,
        )
        self._client.delete(
            collection_name=self.config.normalized_collection,
            points_selector=models.FilterSelector(filter=user_filter),
            wait=True,
        )

    def debug_info(self) -> tuple[str, str]:
        return "qdrant", str(self._client.info().version)
