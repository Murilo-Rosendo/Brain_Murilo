from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import json

import numpy as np

from .config import CLASS_NAMES_PATH, FINAL_MODEL_PATH, IMAGE_SIZE
from .rules import (
    LabAssessment,
    PriorityAssessment,
    SymptomAssessment,
    compose_response,
    define_final_priority,
    evaluate_labs,
    evaluate_symptoms,
)


@dataclass(frozen=True)
class PredictionResult:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


@dataclass(frozen=True)
class AnalysisResult:
    prediction: PredictionResult
    symptoms: SymptomAssessment
    labs: LabAssessment
    priority: PriorityAssessment
    text: str


def load_class_names(path: str | Path = CLASS_NAMES_PATH) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de classes nao encontrado: {path}. Rode o treino antes da predicao."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_trained_model(model_path: str | Path = FINAL_MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {model_path}. Rode `brain-train` antes de abrir o app."
        )
    from tensorflow.keras.models import load_model

    return load_model(model_path)


def load_runtime(
    model_path: str | Path = FINAL_MODEL_PATH,
    class_names_path: str | Path = CLASS_NAMES_PATH,
):
    return load_trained_model(model_path), load_class_names(class_names_path)


def prepare_image_array(image, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    import cv2
    from PIL import Image
    from tensorflow.keras.applications.efficientnet import preprocess_input

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"))
    else:
        array = np.asarray(image)

    if array.ndim == 2:
        array = np.stack([array, array, array], axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]

    resized = cv2.resize(array, image_size)
    resized = resized.astype(np.float32)
    resized = preprocess_input(resized)
    return np.expand_dims(resized, axis=0)


def predict_image(
    image,
    model,
    class_names: list[str],
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> PredictionResult:
    batch = prepare_image_array(image, image_size=image_size)
    raw_prediction = model.predict(batch, verbose=0)[0]
    predicted_index = int(np.argmax(raw_prediction))
    probabilities = {
        class_name: float(raw_prediction[index])
        for index, class_name in enumerate(class_names)
    }
    return PredictionResult(
        class_name=class_names[predicted_index],
        confidence=float(raw_prediction[predicted_index]),
        probabilities=probabilities,
    )


def analyze_case(
    image,
    symptoms_input: Mapping[str, bool] | None,
    labs_input: Mapping[str, float | None] | None,
    model,
    class_names: list[str],
) -> AnalysisResult:
    prediction = predict_image(image, model, class_names)
    symptoms = evaluate_symptoms(symptoms_input)
    labs = evaluate_labs(labs_input, predicted_class=prediction.class_name)
    priority = define_final_priority(
        predicted_class=prediction.class_name,
        confidence=prediction.confidence,
        symptoms=symptoms,
        labs=labs,
    )
    text = compose_response(
        predicted_class=prediction.class_name,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        symptoms=symptoms,
        labs=labs,
        priority=priority,
    )
    return AnalysisResult(
        prediction=prediction,
        symptoms=symptoms,
        labs=labs,
        priority=priority,
        text=text,
    )
