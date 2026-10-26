import base64
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
from qdrant_client import models

import imx_faceid.vector_store as vector_store
from imx_faceid.config import QdrantConfig


class MemoryQdrant:
    instances = []

    def __init__(self, **kwargs):
        self.collections = {}
        self.instances.append(self)

    def collection_exists(self, name):
        return name in self.collections

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = {
            "vectors_config": vectors_config,
            "points": {},
        }

    def get_collection(self, name):
        vectors = self.collections[name]["vectors_config"]
        return SimpleNamespace(config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)))

    def upsert(self, collection_name, points, wait):
        for point in points:
            self.collections[collection_name]["points"][str(point.id)] = SimpleNamespace(
                id=str(point.id), vector=point.vector, payload=point.payload
            )

    def scroll(
        self,
        collection_name,
        scroll_filter,
        limit,
        offset,
        with_payload,
        with_vectors,
    ):
        user_id = scroll_filter.must[0].match.value
        points = [
            point
            for point in self.collections[collection_name]["points"].values()
            if point.payload.get("user_id") == user_id
        ]
        return points[:limit], None

    def set_payload(self, collection_name, payload, points, wait):
        for point_id in points:
            self.collections[collection_name]["points"][str(point_id)].payload.update(
                payload
            )

    def delete(self, collection_name, points_selector, wait):
        user_id = points_selector.filter.must[0].match.value
        points = self.collections[collection_name]["points"]
        for point_id in [
            point_id
            for point_id, point in points.items()
            if point.payload.get("user_id") == user_id
        ]:
            del points[point_id]

    def query_points(self, collection_name, query, limit, with_payload):
        points = []
        for stored in self.collections[collection_name]["points"].values():
            vector = np.asarray(stored.vector, dtype=np.float32)
            query_vector = np.asarray(query, dtype=np.float32)
            if self.collections[collection_name]["vectors_config"].distance == models.Distance.COSINE:
                score = float(np.dot(query_vector, vector))
            else:
                score = float(np.linalg.norm(query_vector - vector))
            points.append(
                SimpleNamespace(
                    payload=stored.payload,
                    score=score,
                )
            )
        if self.collections[collection_name]["vectors_config"].distance == models.Distance.COSINE:
            points.sort(key=lambda point: point.score, reverse=True)
        else:
            points.sort(key=lambda point: point.score)
        return SimpleNamespace(points=points[:limit])

    def info(self):
        return SimpleNamespace(version="test-qdrant")

    def close(self):
        pass


class FakeRpcError(grpc.RpcError):
    pass


class GrpcUnavailableThenRestQdrant(MemoryQdrant):
    instances = []

    def __init__(self, **kwargs):
        self.prefer_grpc = bool(kwargs.get("prefer_grpc", False))
        self.closed = False
        super().__init__(**kwargs)

    def collection_exists(self, name):
        if self.prefer_grpc:
            raise FakeRpcError("gRPC unavailable")
        return super().collection_exists(name)

    def close(self):
        self.closed = True


class AlwaysUnavailableQdrant(GrpcUnavailableThenRestQdrant):
    def collection_exists(self, name):
        raise FakeRpcError("Qdrant unavailable")


@pytest.fixture
def store(monkeypatch):
    MemoryQdrant.instances.clear()
    monkeypatch.setattr(vector_store, "QdrantClient", MemoryQdrant)
    config = QdrantConfig(
        host="localhost",
        port=6333,
        grpc_port=6334,
        api_key=None,
        https=False,
        metric="cosine",
        raw_collection="raw_enrol",
        normalized_collection="normalized_enrol",
    )
    result = vector_store.VectorStore(config, embedding_dim=2)
    return result, MemoryQdrant.instances[0]


def test_uses_rest_when_grpc_is_unavailable(monkeypatch):
    GrpcUnavailableThenRestQdrant.instances.clear()
    monkeypatch.setattr(vector_store, "QdrantClient", GrpcUnavailableThenRestQdrant)
    config = QdrantConfig(
        host="localhost",
        port=6333,
        grpc_port=6334,
        api_key=None,
        https=False,
        metric="cosine",
        raw_collection="raw_enrol",
        normalized_collection="normalized_enrol",
    )

    result = vector_store.VectorStore(config, embedding_dim=2)

    assert len(GrpcUnavailableThenRestQdrant.instances) == 2
    grpc_client, rest_client = GrpcUnavailableThenRestQdrant.instances
    assert grpc_client.prefer_grpc is True
    assert grpc_client.closed is True
    assert rest_client.prefer_grpc is False
    result.close()


def test_raises_clear_error_when_grpc_and_rest_are_unavailable(monkeypatch):
    AlwaysUnavailableQdrant.instances.clear()
    monkeypatch.setattr(vector_store, "QdrantClient", AlwaysUnavailableQdrant)
    config = QdrantConfig(
        host="localhost",
        port=6333,
        grpc_port=6334,
        api_key=None,
        https=False,
        metric="cosine",
        raw_collection="raw_enrol",
        normalized_collection="normalized_enrol",
    )

    with pytest.raises(RuntimeError, match="Unable to connect to Qdrant via gRPC or REST"):
        vector_store.VectorStore(config, embedding_dim=2)


def test_enrollment_recalculates_mean_updates_names_and_stores_payload(store):
    vector_store_instance, client = store
    vector_store_instance.enroll(
        user_id="user-1",
        name="Old name",
        custom="first",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        image_bmp=b"bmp bytes",
        time_unix_ms=100,
    )
    vector_store_instance.enroll(
        user_id="user-1",
        name="New name",
        custom="second",
        embedding=np.array([0.0, 1.0], dtype=np.float32),
        image_bmp=b"second image",
        time_unix_ms=200,
    )

    raw_points = client.collections["raw_enrol"]["points"].values()
    assert len(raw_points) == 2
    assert {point.payload["name"] for point in raw_points} == {"New name"}
    assert {point.payload["custom"] for point in raw_points} == {"first", "second"}
    assert {
        base64.b64decode(point.payload["image"])
        for point in raw_points
    } == {b"bmp bytes", b"second image"}

    normalized = list(client.collections["normalized_enrol"]["points"].values())
    assert len(normalized) == 1
    assert normalized[0].payload["num_enrolments"] == 2
    assert normalized[0].payload["name"] == "New name"
    assert normalized[0].vector == pytest.approx([2**-0.5, 2**-0.5])

    enrolments = vector_store_instance.user_info("user-1")
    assert len(enrolments) == 2
    assert {item["time_unix_ms"] for item in enrolments} == {100, 200}

    matches = vector_store_instance.query(
        np.array([1.0, 0.0], dtype=np.float32),
        max_results=1,
        min_confidence=0.8,
    )
    assert len(matches) == 1
    assert matches[0].name == "New name"
    assert matches[0].confidence == pytest.approx((1.0 + 2**-0.5) / 2.0)

    vector_store_instance.delete_user("user-1")
    assert client.collections["raw_enrol"]["points"] == {}
    assert client.collections["normalized_enrol"]["points"] == {}
