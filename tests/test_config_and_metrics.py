import yaml
import pytest

from imx_faceid.config import ConfigurationError, load_config
from imx_faceid.vector_store import score_to_distance_confidence


def valid_config(model_path):
    return {
        "nats": {"url": "nats://localhost:4222"},
        "qdrant": {
            "host": "localhost",
            "port": 6333,
            "grpc_port": 6334,
            "metric": "cosine",
        },
        "model": {
            "path": str(model_path),
            "runtime": "onnx",
            "input_layout": "NCHW",
            "input_size": 112,
            "color_order": "RGB",
            "pixel_scale": 1 / 127.5,
            "mean": [1.0, 1.0, 1.0],
            "std": [1.0, 1.0, 1.0],
            "alignment": "none",
        },
    }


def test_config_resolves_defaults_and_paths_relative_to_yaml(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    config_path = tmp_path / "config.yaml"
    raw = valid_config("model.onnx")
    raw["qdrant"]["metric"] = "euclidean"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    config = load_config(config_path)

    assert config.model.path == model
    assert config.debug == "info"
    assert config.nats.enrol_subject == "faceid.enrol"
    assert config.model.honor_preprocessed_flag is True
    assert config.qdrant.metric == "euclidean"


def test_config_requires_alignment_reference_points(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    raw = valid_config(model)
    raw["model"]["alignment"] = "five_point"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="reference_keypoints"):
        load_config(config_path)


def test_config_rejects_invalid_debug_level(tmp_path):
    model = tmp_path / "model.onnx"
    model.touch()
    raw = valid_config(model)
    raw["debug"] = "trace"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ConfigurationError, match="must be one of: none, error, info, verbose"):
        load_config(config_path)


@pytest.mark.parametrize(
    ("metric", "score", "expected_distance", "expected_confidence"),
    [
        ("cosine", 1.0, 0.0, 1.0),
        ("cosine", 0.0, 1.0, 0.5),
        ("cosine", -1.0, 2.0, 0.0),
        ("euclidean", 0.0, 0.0, 1.0),
        ("euclidean", 2**0.5, 2**0.5, 0.5),
        ("euclidean", 2.0, 2.0, 0.0),
        ("euclidean", 3.0, 3.0, 0.0),
    ],
)
def test_metric_score_mapping(metric, score, expected_distance, expected_confidence):
    distance, confidence = score_to_distance_confidence(score, metric)
    assert distance == pytest.approx(expected_distance)
    assert confidence == pytest.approx(expected_confidence)
