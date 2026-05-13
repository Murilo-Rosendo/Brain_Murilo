from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping
from uuid import uuid4
import csv
import json

from .config import HISTORY_PATH, VALIDATION_HISTORY_PATH, compact_class_name
from .predict import AnalysisResult

HISTORY_FIELDS = [
    "analysis_id",
    "data_hora",
    "nome_imagem",
    "imagem_salva",
    "classe_prevista",
    "confianca",
    "glioma_prob",
    "meningioma_prob",
    "pituitary_prob",
    "no_tumor_prob",
    "sintomas_informados",
    "prioridade_sintomas",
    "exames_informados",
    "avaliacao_exames",
    "prioridade_final",
    "gradcam_image",
    "observacoes",
]

VALIDATION_FIELDS = [
    "data_hora",
    "analysis_id",
    "nome_imagem",
    "imagem_salva",
    "classe_prevista",
    "classe_correta",
    "resultado",
    "confianca",
    "glioma_prob",
    "meningioma_prob",
    "pituitary_prob",
    "no_tumor_prob",
    "prioridade_final",
    "gradcam_image",
    "observacao",
]


def _probability_for(probabilities: Mapping[str, float], aliases: set[str]) -> float:
    for class_name, probability in probabilities.items():
        if compact_class_name(class_name) in aliases:
            return float(probability)
    return 0.0


def append_analysis_history(
    analysis: AnalysisResult,
    symptoms_input: Mapping[str, bool] | None,
    labs_input: Mapping[str, float | None] | None,
    image_name: str = "imagem_enviada",
    history_path: str | Path = HISTORY_PATH,
    gradcam_image: str | None = None,
    analysis_id: str | None = None,
    saved_image_path: str | None = None,
) -> dict[str, str | float]:
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_history_header(history_path)
    analysis_id = analysis_id or uuid4().hex

    row = {
        "analysis_id": analysis_id,
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "nome_imagem": image_name,
        "imagem_salva": saved_image_path or "",
        "classe_prevista": analysis.prediction.class_name,
        "confianca": round(analysis.prediction.confidence, 6),
        "glioma_prob": round(_probability_for(analysis.prediction.probabilities, {"glioma"}), 6),
        "meningioma_prob": round(_probability_for(analysis.prediction.probabilities, {"meningioma"}), 6),
        "pituitary_prob": round(
            _probability_for(analysis.prediction.probabilities, {"pituitary", "pituitario", "tumorpituitario"}),
            6,
        ),
        "no_tumor_prob": round(
            _probability_for(analysis.prediction.probabilities, {"notumor", "notumour", "semtumor"}),
            6,
        ),
        "sintomas_informados": json.dumps(symptoms_input or {}, ensure_ascii=False),
        "prioridade_sintomas": analysis.symptoms.priority,
        "exames_informados": json.dumps(labs_input or {}, ensure_ascii=False),
        "avaliacao_exames": analysis.labs.summary,
        "prioridade_final": analysis.priority.priority,
        "gradcam_image": gradcam_image or "",
        "observacoes": analysis.priority.warning,
    }

    file_exists = history_path.exists()
    with history_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_developer_validations(validation_path: str | Path = VALIDATION_HISTORY_PATH) -> list[dict[str, str]]:
    validation_path = Path(validation_path)
    if not validation_path.exists() or validation_path.stat().st_size == 0:
        return []

    with validation_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [{field: row.get(field, "") for field in VALIDATION_FIELDS} for row in reader]


def upsert_developer_validation(
    analysis_row: Mapping[str, object],
    true_class: str,
    note: str = "",
    validation_path: str | Path = VALIDATION_HISTORY_PATH,
) -> dict[str, str]:
    validation_path = Path(validation_path)
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_id = str(analysis_row.get("analysis_id", "") or "")
    predicted_class = str(analysis_row.get("classe_prevista", "") or "")
    result = "acerto" if compact_class_name(predicted_class) == compact_class_name(true_class) else "erro"

    row = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "analysis_id": analysis_id,
        "nome_imagem": str(analysis_row.get("nome_imagem", "") or ""),
        "imagem_salva": str(analysis_row.get("imagem_salva", "") or ""),
        "classe_prevista": predicted_class,
        "classe_correta": true_class,
        "resultado": result,
        "confianca": str(analysis_row.get("confianca", "") or ""),
        "glioma_prob": str(analysis_row.get("glioma_prob", "") or ""),
        "meningioma_prob": str(analysis_row.get("meningioma_prob", "") or ""),
        "pituitary_prob": str(analysis_row.get("pituitary_prob", "") or ""),
        "no_tumor_prob": str(analysis_row.get("no_tumor_prob", "") or ""),
        "prioridade_final": str(analysis_row.get("prioridade_final", "") or ""),
        "gradcam_image": str(analysis_row.get("gradcam_image", "") or ""),
        "observacao": note.strip(),
    }

    rows = [item for item in read_developer_validations(validation_path) if item.get("analysis_id") != analysis_id]
    rows.append(row)
    with validation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=VALIDATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return row


def _ensure_history_header(history_path: Path) -> None:
    if not history_path.exists() or history_path.stat().st_size == 0:
        return

    with history_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames == HISTORY_FIELDS:
            return
        rows = list(reader)

    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in HISTORY_FIELDS})
