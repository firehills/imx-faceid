import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from imx_faceid.config import ModelConfig
from imx_faceid.inference import (
    EmbeddingModel,
    InferenceError,
    normalize_embedding,
    prepare_image,
)


def model_config(**overrides):
    config = ModelConfig(
        path=None,
        runtime="onnx",
        input_layout="NCHW",
        input_size=16,
        input_name=None,
        output_name=None,
        color_order="RGB",
        pixel_scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        alignment="none",
        reference_keypoints=None,
        honor_preprocessed_flag=True,
        delegate_library=None,
        delegate_options={},
    )
    return replace(config, **overrides)


def bitmap(size=(32, 32)):
    image = Image.new("RGB", size, (64, 128, 192))
    data = io.BytesIO()
    image.save(data, format="BMP")
    return data.getvalue()


def test_prepare_image_resizes_and_uses_requested_layout():
    tensor = prepare_image(
        bitmap(),
        encoding=3,
        width=32,
        height=32,
        is_preprocessed=False,
        keypoints=[],
        config=model_config(input_layout="NHWC"),
    )

    assert tensor.shape == (1, 16, 16, 3)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0] == pytest.approx([64, 128, 192] / np.float32(255))


def test_preprocessed_flag_can_skip_or_run_spatial_preprocessing():
    with pytest.raises(InferenceError, match="must match model.input_size"):
        prepare_image(
            bitmap(),
            encoding=3,
            width=32,
            height=32,
            is_preprocessed=True,
            keypoints=[],
            config=model_config(),
        )

    tensor = prepare_image(
        bitmap(),
        encoding=3,
        width=32,
        height=32,
        is_preprocessed=True,
        keypoints=[],
        config=model_config(honor_preprocessed_flag=False),
    )
    assert tensor.shape == (1, 3, 16, 16)


def test_five_point_alignment_requires_points_and_produces_model_shape():
    config = model_config(
        alignment="five_point",
        reference_keypoints=(
            (4.0, 5.0),
            (12.0, 5.0),
            (8.0, 8.0),
            (5.0, 12.0),
            (11.0, 12.0),
        ),
    )
    points = [
        (8.0, 10.0),
        (24.0, 10.0),
        (16.0, 16.0),
        (10.0, 24.0),
        (22.0, 24.0),
    ]
    tensor = prepare_image(
        bitmap(),
        encoding=3,
        width=32,
        height=32,
        is_preprocessed=False,
        keypoints=points,
        config=config,
    )
    assert tensor.shape == (1, 3, 16, 16)


def test_rejects_non_bmp_and_non_square_images():
    with pytest.raises(InferenceError, match="Only BMP"):
        prepare_image(
            bitmap(),
            encoding=1,
            width=32,
            height=32,
            is_preprocessed=False,
            keypoints=[],
            config=model_config(),
        )
    with pytest.raises(InferenceError, match="square"):
        prepare_image(
            bitmap((32, 24)),
            encoding=3,
            width=32,
            height=24,
            is_preprocessed=False,
            keypoints=[],
            config=model_config(),
        )


def test_normalize_embedding_rejects_zero_vector():
    with pytest.raises(InferenceError, match="zero-length"):
        normalize_embedding(np.zeros(3, dtype=np.float32))


def test_litert_quantization_requires_single_positive_scale():
    assert EmbeddingModel._quantization(
        {"quantization_parameters": {"scales": [0.25], "zero_points": [7]}}
    ) == (0.25, 7)
    with pytest.raises(InferenceError, match="per-tensor"):
        EmbeddingModel._quantization(
            {
                "quantization_parameters": {
                    "scales": [0.25, 0.5],
                    "zero_points": [7, 8],
                }
            }
        )
