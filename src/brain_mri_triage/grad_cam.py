from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image

from .config import GRADCAM_DIR, IMAGE_SIZE
from .predict import prepare_image_array


@dataclass(frozen=True)
class GradCamResult:
    image_path: Path
    image_url: str
    layer_name: str


def _call_inference(layer, value):
    try:
        return layer(value, training=False)
    except TypeError:
        return layer(value)


def _target_conv_layer(base_model, preferred_layer_name: str = "top_conv"):
    try:
        return base_model.get_layer(preferred_layer_name)
    except ValueError:
        pass

    import tensorflow as tf

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
    raise ValueError("Nao encontrei uma camada convolucional para gerar Grad-CAM.")


def _classifier_from_top_conv(model, base_model, conv_outputs):
    x = _call_inference(base_model.get_layer("top_bn"), conv_outputs)
    x = _call_inference(base_model.get_layer("top_activation"), x)

    base_index = model.layers.index(base_model)
    for layer in model.layers[base_index + 1 :]:
        x = _call_inference(layer, x)
    return x


def make_gradcam_heatmap(
    image,
    model,
    pred_index: int | None = None,
    base_model_name: str = "efficientnetb0",
    conv_layer_name: str = "top_conv",
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[np.ndarray, str]:
    import tensorflow as tf

    image_batch = prepare_image_array(image, image_size=image_size)
    base_model = model.get_layer(base_model_name)
    conv_layer = _target_conv_layer(base_model, conv_layer_name)
    conv_extractor = tf.keras.Model(base_model.inputs, conv_layer.output)

    with tf.GradientTape() as tape:
        x = _call_inference(model.get_layer("data_augmentation"), image_batch)
        conv_outputs = conv_extractor(x, training=False)
        tape.watch(conv_outputs)
        predictions = _classifier_from_top_conv(model, base_model, conv_outputs)

        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]).numpy())

        class_score = predictions[:, pred_index]

    grads = tape.gradient(class_score, conv_outputs)
    if grads is None:
        raise ValueError("Nao foi possivel calcular os gradientes do Grad-CAM.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) <= 0:
        return np.zeros(tuple(heatmap.shape), dtype=np.float32), conv_layer.name

    return (heatmap / max_value).numpy(), conv_layer.name


def overlay_heatmap(
    image,
    heatmap: np.ndarray,
    image_size: tuple[int, int] = IMAGE_SIZE,
    alpha: float = 0.42,
) -> Image.Image:
    import cv2

    if isinstance(image, Image.Image):
        original = np.asarray(image.convert("RGB"))
    else:
        original = np.asarray(Image.open(image).convert("RGB"))

    width, height = image_size[1], image_size[0]
    original = cv2.resize(original, (width, height))
    heatmap = cv2.resize(heatmap, (width, height))
    heatmap = np.uint8(255 * np.clip(heatmap, 0, 1))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 1 - alpha, heatmap, alpha, 0)
    return Image.fromarray(overlay)


def save_gradcam_overlay(
    image,
    model,
    pred_index: int | None = None,
    output_dir: str | Path = GRADCAM_DIR,
) -> GradCamResult:
    heatmap, layer_name = make_gradcam_heatmap(image, model, pred_index=pred_index)
    overlay = overlay_heatmap(image, heatmap)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.png"
    image_path = output_dir / filename
    overlay.save(image_path)
    return GradCamResult(
        image_path=image_path,
        image_url=f"/gradcam/{filename}",
        layer_name=layer_name,
    )
