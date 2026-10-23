"""Image decoding, model execution, and embedding normalization."""

from __future__ import annotations

import importlib
import io
import math
from typing import Any

import numpy as np
from PIL import Image

from imx_faceid.config import ModelConfig


class InferenceError(RuntimeError):
    """Raised when an image or model cannot produce a valid embedding."""


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Return a finite, unit-length float32 embedding."""
    embedding = np.asarray(vector, dtype=np.float32).reshape(-1)
    if embedding.size == 0 or not np.all(np.isfinite(embedding)):
        raise InferenceError("The model returned an empty or non-finite embedding")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm == 0:
        raise InferenceError("The model returned a zero-length embedding")
    return embedding / norm


def _align_five_points(
    image: Image.Image,
    source_points: list[tuple[float, float]],
    target_points: tuple[tuple[float, float], ...],
    size: int,
) -> Image.Image:
    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise InferenceError("The supplied keypoints must be finite")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    denominator = float(np.sum(source_zero * source_zero))
    if denominator == 0:
        raise InferenceError("The supplied keypoints do not define a usable alignment")

    covariance = target_zero.T @ source_zero
    u, singular_values, vh = np.linalg.svd(covariance)
    correction = np.eye(2)
    if np.linalg.det(u @ vh) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vh
    scale = float(np.sum(singular_values * np.diag(correction)) / denominator)
    if not math.isfinite(scale) or scale <= 0:
        raise InferenceError("The supplied keypoints produce an invalid alignment")

    forward = np.eye(3, dtype=np.float64)
    forward[:2, :2] = scale * rotation
    forward[:2, 2] = target_center - scale * rotation @ source_center
    inverse = np.linalg.inv(forward)
    coefficients = tuple(float(value) for value in inverse[:2].reshape(-1))
    return image.transform(
        (size, size),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
    )


def prepare_image(
    image_bytes: bytes,
    *,
    encoding: int,
    width: int,
    height: int,
    is_preprocessed: bool,
    keypoints: list[tuple[float, float]],
    config: ModelConfig,
) -> np.ndarray:
    """Decode a BMP and prepare the model's configured NCHW or NHWC tensor."""
    if encoding not in {0, 3}:  # Unknown or BMP in the schema.
        raise InferenceError("Only BMP image data is supported")
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            if source.format != "BMP":
                raise InferenceError("The image payload is not BMP data")
            image = source.convert("RGB")
    except (OSError, ValueError) as exc:
        raise InferenceError(f"Unable to decode BMP image: {exc}") from exc

    if image.width != image.height:
        raise InferenceError("Face images must be square")
    if width and width != image.width:
        raise InferenceError(f"Message width {width} does not match BMP width {image.width}")
    if height and height != image.height:
        raise InferenceError(f"Message height {height} does not match BMP height {image.height}")

    skip_spatial_preprocessing = (
        is_preprocessed and config.honor_preprocessed_flag
    )
    if skip_spatial_preprocessing:
        if image.size != (config.input_size, config.input_size):
            raise InferenceError(
                "An image marked preprocessed must match model.input_size when spatial "
                "preprocessing is skipped"
            )
    elif config.alignment == "five_point":
        if len(keypoints) != 5 or config.reference_keypoints is None:
            raise InferenceError(
                "Five-point alignment requires exactly five request keypoints and "
                "configured reference_keypoints"
            )
        image = _align_five_points(
            image, keypoints, config.reference_keypoints, config.input_size
        )
    else:
        image = image.resize(
            (config.input_size, config.input_size), Image.Resampling.BILINEAR
        )

    pixels = np.asarray(image, dtype=np.float32)
    if config.color_order == "BGR":
        pixels = pixels[..., ::-1]
    pixels = (pixels * config.pixel_scale - np.asarray(config.mean, dtype=np.float32)) / np.asarray(
        config.std, dtype=np.float32
    )
    if config.input_layout == "NCHW":
        pixels = np.transpose(pixels, (2, 0, 1))
    return np.expand_dims(pixels.astype(np.float32), axis=0)


class EmbeddingModel:
    """Small ONNX Runtime or LiteRT CPU inference adapter."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        if not config.path.is_file():
            raise InferenceError(f"Model file does not exist: {config.path}")
        if config.runtime == "onnx":
            self._load_onnx()
        else:
            self._load_litert()
        self.embedding_dim = int(self._infer_tensor(self._dummy_tensor()).size)
        if self.embedding_dim == 0:
            raise InferenceError("Model output has no embedding values")

    def _load_onnx(self) -> None:
        try:
            onnxruntime = importlib.import_module("onnxruntime")
        except ImportError as exc:
            raise InferenceError(
                "ONNX Runtime is missing; install the 'onnx' extra"
            ) from exc
        self._session = onnxruntime.InferenceSession(
            str(self.config.path), providers=["CPUExecutionProvider"]
        )
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        self._input_name = self._select_tensor_name(
            inputs, self.config.input_name, "input"
        )
        self._output_name = self._select_tensor_name(
            outputs, self.config.output_name, "output"
        )

    def _load_litert(self) -> None:
        try:
            module = importlib.import_module("ai_edge_litert.interpreter")
        except ImportError as exc:
            raise InferenceError(
                "LiteRT is missing; install the 'litert' extra"
            ) from exc
        delegates = []
        if self.config.delegate_library:
            load_delegate = getattr(module, "load_delegate", None)
            if load_delegate is None:
                raise InferenceError(
                    "Installed LiteRT runtime does not expose load_delegate"
                )
            delegates.append(
                load_delegate(
                    self.config.delegate_library,
                    options=self.config.delegate_options or None,
                )
            )
        try:
            self._interpreter = module.Interpreter(
                model_path=str(self.config.path),
                experimental_delegates=delegates,
            )
            self._interpreter.allocate_tensors()
        except (ValueError, RuntimeError, OSError) as exc:
            raise InferenceError(f"Unable to initialize LiteRT model: {exc}") from exc
        self._input_detail = self._select_detail(
            self._interpreter.get_input_details(),
            self.config.input_name,
            "input",
        )
        self._output_detail = self._select_detail(
            self._interpreter.get_output_details(),
            self.config.output_name,
            "output",
        )

    @staticmethod
    def _select_tensor_name(details: list[Any], name: str | None, kind: str) -> str:
        if name is not None:
            if not any(detail.name == name for detail in details):
                raise InferenceError(f"Configured model {kind} tensor '{name}' was not found")
            return name
        if len(details) != 1:
            raise InferenceError(
                f"Model has {len(details)} {kind} tensors; configure model.{kind}_name"
            )
        return details[0].name

    @staticmethod
    def _select_detail(
        details: list[dict[str, Any]], name: str | None, kind: str
    ) -> dict[str, Any]:
        if name is not None:
            matching = [detail for detail in details if detail["name"] == name]
            if not matching:
                raise InferenceError(f"Configured model {kind} tensor '{name}' was not found")
            return matching[0]
        if len(details) != 1:
            raise InferenceError(
                f"Model has {len(details)} {kind} tensors; configure model.{kind}_name"
            )
        return details[0]

    def _dummy_tensor(self) -> np.ndarray:
        shape = (1, 3, self.config.input_size, self.config.input_size)
        if self.config.input_layout == "NHWC":
            shape = (1, self.config.input_size, self.config.input_size, 3)
        return np.zeros(shape, dtype=np.float32)

    def _infer_tensor(self, tensor: np.ndarray) -> np.ndarray:
        if self.config.runtime == "onnx":
            outputs = self._session.run(
                [self._output_name], {self._input_name: tensor}
            )
            result = outputs[0]
        else:
            input_detail = self._input_detail
            input_dtype = input_detail["dtype"]
            if np.issubdtype(input_dtype, np.integer):
                scale, zero_point = self._quantization(input_detail)
                integer_limits = np.iinfo(input_dtype)
                tensor = np.clip(
                    np.rint(tensor / scale + zero_point),
                    integer_limits.min,
                    integer_limits.max,
                ).astype(input_dtype)
            elif np.issubdtype(input_dtype, np.floating):
                tensor = tensor.astype(input_dtype, copy=False)
            else:
                raise InferenceError(
                    f"Unsupported LiteRT input tensor type: {input_dtype}"
                )
            self._interpreter.set_tensor(input_detail["index"], tensor)
            self._interpreter.invoke()
            result = self._interpreter.get_tensor(self._output_detail["index"])
            output_dtype = self._output_detail["dtype"]
            if np.issubdtype(output_dtype, np.integer):
                scale, zero_point = self._quantization(self._output_detail)
                result = (result.astype(np.float32) - zero_point) * scale
        return np.asarray(result).reshape(-1)

    @staticmethod
    def _quantization(detail: dict[str, Any]) -> tuple[float, int]:
        parameters = detail.get("quantization_parameters", {})
        scales = np.asarray(parameters.get("scales", []), dtype=np.float32)
        zero_points = np.asarray(parameters.get("zero_points", []), dtype=np.int64)
        if scales.size != 1 or zero_points.size != 1 or scales[0] <= 0:
            raise InferenceError(
                "LiteRT quantized tensors must use per-tensor quantization with "
                "a positive scale"
            )
        return float(scales[0]), int(zero_points[0])

    def embed(
        self,
        image_bytes: bytes,
        *,
        encoding: int,
        width: int,
        height: int,
        is_preprocessed: bool,
        keypoints: list[tuple[float, float]],
    ) -> np.ndarray:
        tensor = prepare_image(
            image_bytes,
            encoding=encoding,
            width=width,
            height=height,
            is_preprocessed=is_preprocessed,
            keypoints=keypoints,
            config=self.config,
        )
        vector = normalize_embedding(self._infer_tensor(tensor))
        if vector.size != self.embedding_dim:
            raise InferenceError(
                f"Model output dimension changed from {self.embedding_dim} to {vector.size}"
            )
        return vector
