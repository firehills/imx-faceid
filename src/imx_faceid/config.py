"""Configuration loading and validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the YAML configuration is missing or invalid."""


@dataclass(frozen=True)
class NatsConfig:
    url: str
    enrol_subject: str
    query_subject: str
    user_info_subject: str
    delete_subject: str
    debug_subject: str
    token: str | None
    username: str | None
    password: str | None
    credentials_file: str | None


@dataclass(frozen=True)
class QdrantConfig:
    host: str
    port: int
    grpc_port: int
    api_key: str | None
    https: bool
    metric: str
    raw_collection: str
    normalized_collection: str


@dataclass(frozen=True)
class ModelConfig:
    path: Path
    runtime: str
    input_layout: str
    input_size: int
    input_name: str | None
    output_name: str | None
    color_order: str
    pixel_scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    alignment: str
    reference_keypoints: tuple[tuple[float, float], ...] | None
    honor_preprocessed_flag: bool
    delegate_library: str | None
    delegate_options: dict[str, Any]


@dataclass(frozen=True)
class AppConfig:
    nats: NatsConfig
    qdrant: QdrantConfig
    model: ModelConfig


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping or mapping[key] is None:
        raise ConfigurationError(f"Missing required configuration: {section}.{key}")
    return mapping[key]


def _section(root: dict[str, Any], name: str) -> dict[str, Any]:
    value = root.get(name)
    if not isinstance(value, dict):
        raise ConfigurationError(f"Configuration section '{name}' must be a mapping")
    return value


def _triple(value: Any, field: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigurationError(f"'{field}' must contain exactly three numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{field}' must contain exactly three numbers") from exc
    if not all(math.isfinite(item) for item in result):
        raise ConfigurationError(f"'{field}' values must be finite")
    return result


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{field}' must be a number") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"'{field}' must be finite")
    return result


def load_config(path: str | Path) -> AppConfig:
    """Load and validate the service configuration from YAML."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    except OSError as exc:
        raise ConfigurationError(f"Cannot read configuration '{config_path}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in '{config_path}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    nats_raw = _section(raw, "nats")
    subjects = nats_raw.get("subjects", {})
    if not isinstance(subjects, dict):
        raise ConfigurationError("'nats.subjects' must be a mapping")
    nats = NatsConfig(
        url=str(_required(nats_raw, "url", "nats")),
        enrol_subject=str(subjects.get("enrol", "faceid.enrol")),
        query_subject=str(subjects.get("query", "faceid.query")),
        user_info_subject=str(subjects.get("user_info", "faceid.user_info")),
        delete_subject=str(subjects.get("delete", "faceid.delete")),
        debug_subject=str(subjects.get("debug", "faceid.debug")),
        token=nats_raw.get("token"),
        username=nats_raw.get("username"),
        password=nats_raw.get("password"),
        credentials_file=nats_raw.get("credentials_file"),
    )
    subjects = (
        nats.enrol_subject,
        nats.query_subject,
        nats.user_info_subject,
        nats.delete_subject,
        nats.debug_subject,
    )
    if any(not subject for subject in subjects) or len(set(subjects)) != len(subjects):
        raise ConfigurationError("NATS subjects must be non-empty and unique")

    qdrant_raw = _section(raw, "qdrant")
    metric = str(_required(qdrant_raw, "metric", "qdrant")).lower()
    if metric not in {"cosine", "euclidean"}:
        raise ConfigurationError("'qdrant.metric' must be 'cosine' or 'euclidean'")
    try:
        qdrant_port = int(_required(qdrant_raw, "port", "qdrant"))
        qdrant_grpc_port = int(_required(qdrant_raw, "grpc_port", "qdrant"))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Qdrant ports must be integers") from exc
    qdrant = QdrantConfig(
        host=str(_required(qdrant_raw, "host", "qdrant")),
        port=qdrant_port,
        grpc_port=qdrant_grpc_port,
        api_key=qdrant_raw.get("api_key"),
        https=bool(qdrant_raw.get("https", False)),
        metric=metric,
        raw_collection=str(qdrant_raw.get("raw_collection", "raw_enrol")),
        normalized_collection=str(
            qdrant_raw.get("normalized_collection", "normalized_enrol")
        ),
    )
    if qdrant.raw_collection == qdrant.normalized_collection:
        raise ConfigurationError("Qdrant collection names must be different")
    if qdrant.port < 1 or qdrant.grpc_port < 1:
        raise ConfigurationError("Qdrant ports must be positive")

    model_raw = _section(raw, "model")
    model_path = Path(str(_required(model_raw, "path", "model")))
    if not model_path.is_absolute():
        model_path = config_path.parent / model_path
    runtime = str(model_raw.get("runtime", "auto")).lower()
    if runtime == "auto":
        suffix = model_path.suffix.lower()
        runtime = {".onnx": "onnx", ".tflite": "litert"}.get(suffix, "")
    if runtime not in {"onnx", "litert"}:
        raise ConfigurationError("Model runtime must be onnx or litert, or inferable from its file extension")
    input_layout = str(_required(model_raw, "input_layout", "model")).upper()
    if input_layout not in {"NCHW", "NHWC"}:
        raise ConfigurationError("'model.input_layout' must be NCHW or NHWC")
    try:
        input_size = int(_required(model_raw, "input_size", "model"))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("'model.input_size' must be an integer") from exc
    if input_size < 1:
        raise ConfigurationError("'model.input_size' must be positive")
    color_order = str(_required(model_raw, "color_order", "model")).upper()
    if color_order not in {"RGB", "BGR"}:
        raise ConfigurationError("'model.color_order' must be RGB or BGR")
    pixel_scale = _number(
        _required(model_raw, "pixel_scale", "model"), "model.pixel_scale"
    )
    mean = _triple(_required(model_raw, "mean", "model"), "model.mean")
    std = _triple(_required(model_raw, "std", "model"), "model.std")
    if any(item == 0 for item in std):
        raise ConfigurationError("Values in 'model.std' must be non-zero")
    alignment = str(_required(model_raw, "alignment", "model")).lower()
    if alignment not in {"none", "five_point"}:
        raise ConfigurationError("'model.alignment' must be none or five_point")
    reference_points_raw = model_raw.get("reference_keypoints")
    reference_points: tuple[tuple[float, float], ...] | None = None
    if reference_points_raw is not None:
        if (
            not isinstance(reference_points_raw, list)
            or len(reference_points_raw) != 5
            or any(not isinstance(point, list) or len(point) != 2 for point in reference_points_raw)
        ):
            raise ConfigurationError("'model.reference_keypoints' must be five [x, y] pairs")
        try:
            reference_points = tuple(
                (float(point[0]), float(point[1])) for point in reference_points_raw
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                "'model.reference_keypoints' values must be numeric"
            ) from exc
        if not all(
            math.isfinite(value) for point in reference_points for value in point
        ):
            raise ConfigurationError(
                "'model.reference_keypoints' values must be finite"
            )
    if alignment == "five_point" and reference_points is None:
        raise ConfigurationError(
            "'model.reference_keypoints' is required when alignment is five_point"
        )
    if runtime == "onnx" and model_raw.get("delegate_library"):
        raise ConfigurationError("LiteRT delegate settings cannot be used with an ONNX model")
    auth_modes = sum(
        (
            bool(nats.token),
            bool(nats.username or nats.password),
            bool(nats.credentials_file),
        )
    )
    if auth_modes > 1:
        raise ConfigurationError(
            "Configure only one NATS authentication method: token, username/password, "
            "or credentials_file"
        )
    if bool(nats.username) != bool(nats.password):
        raise ConfigurationError("NATS username and password must be configured together")
    delegate_options = model_raw.get("delegate_options", {})
    if not isinstance(delegate_options, dict):
        raise ConfigurationError("'model.delegate_options' must be a mapping")
    honor_preprocessed = model_raw.get("honor_preprocessed_flag", True)
    if not isinstance(honor_preprocessed, bool):
        raise ConfigurationError("'model.honor_preprocessed_flag' must be true or false")

    model = ModelConfig(
        path=model_path,
        runtime=runtime,
        input_layout=input_layout,
        input_size=input_size,
        input_name=model_raw.get("input_name"),
        output_name=model_raw.get("output_name"),
        color_order=color_order,
        pixel_scale=pixel_scale,
        mean=mean,
        std=std,
        alignment=alignment,
        reference_keypoints=reference_points,
        honor_preprocessed_flag=honor_preprocessed,
        delegate_library=model_raw.get("delegate_library"),
        delegate_options=delegate_options,
    )
    if not model.path.is_file():
        raise ConfigurationError(f"Model file does not exist: {model.path}")

    return AppConfig(nats=nats, qdrant=qdrant, model=model)
