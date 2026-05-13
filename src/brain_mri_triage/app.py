from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .config import (
    CLASSIFICATION_REPORT_PATH,
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    CASE_REPORTS_DIR,
    CASE_IMAGES_DIR,
    FINAL_MODEL_PATH,
    GRADCAM_DIR,
    HISTORY_PATH,
    TRAINING_HISTORY_PATH,
    VALIDATION_HISTORY_PATH,
    compact_class_name,
    display_class_name,
    is_no_tumor_class,
)
from .dashboard import class_distribution, confidence_distribution, read_history
from .grad_cam import save_gradcam_overlay
from .history import append_analysis_history, read_developer_validations, upsert_developer_validation
from .predict import analyze_case, load_runtime
from .report import create_case_report
from .rules import GENERAL_LAB_RANGES, HORMONE_LAB_RANGES, SYMPTOM_OPTIONS


LAB_LABELS = {
    "leucocitos": "Leucócitos",
    "neutrofilos": "Neutrófilos",
    "linfocitos": "Linfócitos",
    "monocitos": "Monócitos",
    "plaquetas": "Plaquetas",
    "hemoglobina": "Hemoglobina",
    "albumina": "Albumina",
    "pcr": "PCR",
    "fibrinogenio": "Fibrinogênio",
    "prolactina": "Prolactina",
    "gh": "GH",
    "igf1": "IGF-1",
    "acth": "ACTH",
    "cortisol": "Cortisol",
    "tsh": "TSH",
    "t3": "T3",
    "t4": "T4",
    "lh": "LH",
    "fsh": "FSH",
    "testosterona": "Testosterona",
    "estrogenio": "Estrogênio",
    "progesterona": "Progesterona",
}

GENERAL_DEFAULTS = {
    "leucocitos": 8000,
    "neutrofilos": 5000,
    "linfocitos": 2000,
    "monocitos": 500,
    "plaquetas": 250000,
    "hemoglobina": 13.5,
    "albumina": 4.0,
    "pcr": 2.0,
    "fibrinogenio": 300,
}


def _try_load_runtime():
    try:
        return (*load_runtime(FINAL_MODEL_PATH, CLASS_NAMES_PATH), None)
    except Exception as exc:
        return None, [], str(exc)


MODEL, CLASS_NAMES, MODEL_ERROR = _try_load_runtime()


def _label_for_lab(key: str) -> str:
    return LAB_LABELS.get(key, key.replace("_", " ").title())


def _range_text(normal_range: tuple[float, float]) -> str:
    lower, upper = normal_range
    return f"{lower:g} - {upper:g}"


def _symptom_dict(selected_keys: list[str] | None) -> dict[str, bool]:
    selected = set(selected_keys or [])
    return {item["key"]: item["key"] in selected for item in SYMPTOM_OPTIONS}


def _labs_dict(raw_labs: dict[str, Any] | None) -> dict[str, float | None]:
    raw_labs = raw_labs or {}
    lab_keys = list(GENERAL_LAB_RANGES.keys()) + list(HORMONE_LAB_RANGES.keys())
    return {
        key: raw_labs.get(key) if key in raw_labs and raw_labs.get(key) not in ("", None) else None
        for key in lab_keys
    }


def _frontend_config() -> dict[str, Any]:
    return {
        "classes": [{"key": class_name, "label": display_class_name(class_name)} for class_name in CLASS_NAMES],
        "symptoms": SYMPTOM_OPTIONS,
        "generalLabs": [
            {
                "key": key,
                "label": _label_for_lab(key),
                "range": _range_text(normal_range),
                "default": GENERAL_DEFAULTS.get(key),
            }
            for key, normal_range in GENERAL_LAB_RANGES.items()
        ],
        "hormoneLabs": [
            {
                "key": key,
                "label": _label_for_lab(key),
                "range": _range_text(normal_range),
                "default": None,
            }
            for key, normal_range in HORMONE_LAB_RANGES.items()
        ],
    }


def _safe_records(frame) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _value_counts(frame, column: str, label_column: str = "label") -> list[dict[str, Any]]:
    if frame.empty or column not in frame:
        return []
    counts = frame[column].fillna("-").astype(str).value_counts().reset_index()
    counts.columns = [label_column, "total"]
    return _safe_records(counts)


def _display_portuguese_text(value: Any) -> str:
    text = str(value or "-")
    replacements = (
        ("avaliacao", "avaliação"),
        ("revisao", "revisão"),
        ("diagnostico", "diagnóstico"),
        ("discordancia", "discordância"),
        ("saida", "saída"),
        ("clinica", "clínica"),
        ("nao", "não"),
        ("metastase", "metástase"),
        ("inflamatorios", "inflamatórios"),
        ("condicoes", "condições"),
        ("prioritario", "prioritário"),
        ("confianca", "confiança"),
        ("classificacao", "classificação"),
        ("analise", "análise"),
        ("atencao", "atenção"),
    )
    for source, target in replacements:
        text = text.replace(source, target)
    return text


def _save_case_image(image: Image.Image, original_name: str, analysis_id: str) -> str:
    CASE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(original_name or "imagem_web").stem[:60] or "imagem_web"
    output_path = CASE_IMAGES_DIR / f"{analysis_id}_{safe_stem}.png"
    image.save(output_path, format="PNG")
    return str(output_path)


def _training_payload() -> dict[str, Any]:
    import pandas as pd

    if not TRAINING_HISTORY_PATH.exists():
        return {"available": False, "history": [], "summary": {}}

    history = pd.read_csv(TRAINING_HISTORY_PATH)
    if history.empty:
        return {"available": False, "history": [], "summary": {}}

    numeric_columns = ["epoch", "accuracy", "loss", "val_accuracy", "val_loss", "learning_rate"]
    for column in numeric_columns:
        if column in history:
            history[column] = pd.to_numeric(history[column], errors="coerce")

    best_val_loss = history.loc[history["val_loss"].idxmin()].to_dict() if "val_loss" in history else {}
    best_val_accuracy = history.loc[history["val_accuracy"].idxmax()].to_dict() if "val_accuracy" in history else {}
    phase_counts = (
        history["phase"].fillna("-").astype(str).value_counts().rename_axis("phase").reset_index(name="total")
        if "phase" in history
        else pd.DataFrame(columns=["phase", "total"])
    )

    return {
        "available": True,
        "history": _safe_records(history),
        "summary": {
            "totalEpochs": int(len(history)),
            "lastEpoch": int(history["epoch"].max()) if "epoch" in history else int(len(history)),
            "bestValLossEpoch": int(best_val_loss.get("epoch", 0) or 0),
            "bestValLoss": round(float(best_val_loss.get("val_loss", 0) or 0), 4),
            "bestValAccuracyEpoch": int(best_val_accuracy.get("epoch", 0) or 0),
            "bestValAccuracy": round(float(best_val_accuracy.get("val_accuracy", 0) or 0) * 100, 1),
            "phaseCounts": _safe_records(phase_counts),
        },
    }


def _confusion_matrix_payload() -> dict[str, Any]:
    import pandas as pd

    if not CONFUSION_MATRIX_PATH.exists():
        return {"available": False, "labels": [], "matrix": []}

    matrix_frame = pd.read_csv(CONFUSION_MATRIX_PATH, index_col=0)
    labels = [display_class_name(str(label)) for label in matrix_frame.index.tolist()]
    matrix = matrix_frame.fillna(0).astype(int).values.tolist()
    per_class = []
    for index, label in enumerate(labels):
        row_total = sum(matrix[index])
        correct = matrix[index][index] if index < len(matrix[index]) else 0
        per_class.append(
            {
                "label": label,
                "correct": int(correct),
                "total": int(row_total),
                "recall": round((correct / row_total) * 100, 1) if row_total else 0,
            }
        )
    return {
        "available": True,
        "labels": labels,
        "matrix": matrix,
        "maxValue": max([max(row) for row in matrix], default=0),
        "perClass": per_class,
    }


def _classification_report_payload() -> dict[str, Any]:
    if not CLASSIFICATION_REPORT_PATH.exists():
        return {"available": False, "classes": [], "accuracy": None}

    class_names = {name: display_class_name(name) for name in CLASS_NAMES}
    rows = []
    accuracy = None
    for line in CLASSIFICATION_REPORT_PATH.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] in class_names and len(parts) >= 5:
            rows.append(
                {
                    "className": parts[0],
                    "label": class_names[parts[0]],
                    "precision": round(float(parts[1]) * 100, 1),
                    "recall": round(float(parts[2]) * 100, 1),
                    "f1": round(float(parts[3]) * 100, 1),
                    "support": int(float(parts[4])),
                }
            )
        elif parts[0] == "accuracy" and len(parts) >= 3:
            accuracy = round(float(parts[-2]) * 100, 1)

    return {"available": bool(rows), "classes": rows, "accuracy": accuracy}


def _usage_payload(history) -> dict[str, Any]:
    import pandas as pd

    if history.empty:
        return {
            "priorityDistribution": [],
            "confidenceTrend": [],
            "analysesByDay": [],
        }

    trend = history.tail(30).copy()
    if "confianca" in trend:
        trend["confianca_percent"] = pd.to_numeric(trend["confianca"], errors="coerce").fillna(0) * 100
    if "classe_prevista" in trend:
        trend["classe_prevista"] = trend["classe_prevista"].map(display_class_name)

    by_day = pd.DataFrame(columns=["date", "total"])
    if "data_hora" in history:
        dates = pd.to_datetime(history["data_hora"], errors="coerce").dt.date.astype(str)
        by_day = dates.value_counts().sort_index().rename_axis("date").reset_index(name="total")

    priority_history = history.copy()
    if "prioridade_final" in priority_history:
        priority_history["prioridade_final"] = priority_history["prioridade_final"].map(_display_portuguese_text)

    return {
        "priorityDistribution": _value_counts(priority_history, "prioridade_final"),
        "confidenceTrend": _safe_records(trend[["data_hora", "classe_prevista", "confianca_percent"]])
        if {"data_hora", "classe_prevista", "confianca_percent"}.issubset(trend.columns)
        else [],
        "analysesByDay": _safe_records(by_day.tail(14)),
    }


def _model_payload() -> dict[str, Any]:
    if MODEL is None:
        return {"available": False}

    layers = []
    nested_layers = []
    for layer in MODEL.layers:
        layers.append({"name": layer.name, "trainable": bool(layer.trainable)})
        if hasattr(layer, "layers"):
            nested_layers.extend(
                {"name": nested.name, "trainable": bool(nested.trainable)}
                for nested in layer.layers
            )

    all_layers = layers + nested_layers
    trainable_layers = sum(1 for layer in all_layers if layer["trainable"])
    frozen_layers = len(all_layers) - trainable_layers

    optimizer_name = "-"
    try:
        optimizer_name = MODEL.optimizer.__class__.__name__
    except Exception:
        pass

    return {
        "available": True,
        "name": MODEL.name,
        "optimizer": optimizer_name,
        "totalParams": int(MODEL.count_params()),
        "topLevelLayers": len(MODEL.layers),
        "totalLayers": len(all_layers),
        "trainableLayers": trainable_layers,
        "frozenLayers": frozen_layers,
    }


def _validation_display_row(row: dict[str, Any]) -> dict[str, Any]:
    confidence = 0.0
    try:
        confidence = float(row.get("confianca", 0) or 0) * 100
    except (TypeError, ValueError):
        pass
    result = str(row.get("resultado", "") or "")
    return {
        "analysisId": row.get("analysis_id", ""),
        "date": row.get("data_hora", ""),
        "imageName": row.get("nome_imagem", ""),
        "savedImage": row.get("imagem_salva", ""),
        "predictedClass": row.get("classe_prevista", ""),
        "predictedLabel": display_class_name(str(row.get("classe_prevista", ""))),
        "trueClass": row.get("classe_correta", ""),
        "trueLabel": display_class_name(str(row.get("classe_correta", ""))),
        "result": result,
        "resultLabel": "Acerto" if result == "acerto" else "Erro",
        "confidencePercent": round(confidence, 1),
        "note": row.get("observacao", ""),
    }


def _developer_confusion_matrix_payload(validations: list[dict[str, Any]]) -> dict[str, Any]:
    class_keys = list(CLASS_NAMES)
    if not class_keys:
        class_keys = sorted(
            {
                str(row.get("classe_prevista", "") or "")
                for row in validations
            }
            | {
                str(row.get("classe_correta", "") or "")
                for row in validations
            }
        )
    class_keys = [key for key in class_keys if key]
    if not validations or not class_keys:
        return {"available": False, "labels": [], "matrix": [], "maxValue": 0}

    index_by_class = {compact_class_name(class_name): index for index, class_name in enumerate(class_keys)}
    matrix = [[0 for _ in class_keys] for _ in class_keys]
    for row in validations:
        true_index = index_by_class.get(compact_class_name(str(row.get("classe_correta", ""))))
        predicted_index = index_by_class.get(compact_class_name(str(row.get("classe_prevista", ""))))
        if true_index is not None and predicted_index is not None:
            matrix[true_index][predicted_index] += 1

    return {
        "available": True,
        "labels": [display_class_name(class_name) for class_name in class_keys],
        "matrix": matrix,
        "maxValue": max([max(row) for row in matrix], default=0),
    }


def _developer_validation_payload() -> dict[str, Any]:
    validations = read_developer_validations(VALIDATION_HISTORY_PATH)
    total = len(validations)
    correct = sum(1 for row in validations if row.get("resultado") == "acerto")
    incorrect = sum(1 for row in validations if row.get("resultado") == "erro")
    accuracy = round((correct / total) * 100, 1) if total else None
    status_distribution = (
        [
            {"label": "Acertos", "total": correct},
            {"label": "Erros", "total": incorrect},
        ]
        if total
        else []
    )
    return {
        "summary": {
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
        },
        "statusDistribution": status_distribution,
        "confusionMatrix": _developer_confusion_matrix_payload(validations),
        "history": [_validation_display_row(row) for row in validations[-20:]],
    }


def _case_interpretation(probabilities: list[dict[str, Any]]) -> dict[str, Any]:
    sorted_probabilities = sorted(probabilities, key=lambda item: float(item["value"]), reverse=True)
    top = sorted_probabilities[0] if sorted_probabilities else None
    runner_up = sorted_probabilities[1] if len(sorted_probabilities) > 1 else None
    margin = (float(top["value"]) - float(runner_up["value"])) if top and runner_up else 0.0
    tumor_probability = sum(
        float(item["value"])
        for item in probabilities
        if not is_no_tumor_class(str(item["key"]))
    )
    no_tumor_probability = sum(
        float(item["value"])
        for item in probabilities
        if is_no_tumor_class(str(item["key"]))
    )

    if top and runner_up:
        margin_description = (
            f"{top['label']} ficou {margin * 100:.1f} pontos percentuais acima de "
            f"{runner_up['label']}. Quanto maior a margem, mais separada foi a decisao do modelo."
        )
    else:
        margin_description = "Não foi possível calcular a margem entre classes."

    return {
        "tumorVsNoTumor": {
            "tumorProbability": tumor_probability,
            "tumorPercent": round(tumor_probability * 100, 1),
            "noTumorProbability": no_tumor_probability,
            "noTumorPercent": round(no_tumor_probability * 100, 1),
            "description": "Soma das classes tumorais comparada com a classe sem tumor aparente.",
        },
        "classMargin": {
            "topClass": top,
            "runnerUpClass": runner_up,
            "margin": margin,
            "marginPercent": round(margin * 100, 1),
            "description": margin_description,
        },
    }


def _analysis_response(
    analysis,
    gradcam: dict[str, Any] | None = None,
    report_url: str | None = None,
    analysis_id: str | None = None,
) -> dict[str, Any]:
    has_discordance = (
        is_no_tumor_class(analysis.prediction.class_name)
        and (
            analysis.symptoms.priority == "alta"
            or analysis.labs.general_risk == "alto"
            or analysis.labs.hormone_alert
        )
    )
    display_name = display_class_name(analysis.prediction.class_name)
    if is_no_tumor_class(analysis.prediction.class_name):
        display_name = "Sem tumor aparente (saída do modelo)"

    probabilities = [
        {
            "key": class_name,
            "label": display_class_name(class_name),
            "value": float(probability),
            "percent": round(float(probability) * 100, 1),
        }
        for class_name, probability in analysis.prediction.probabilities.items()
    ]
    interpretation = _case_interpretation(probabilities)
    return {
        "analysisId": analysis_id,
        "prediction": {
            "className": analysis.prediction.class_name,
            "displayName": display_name,
            "confidence": float(analysis.prediction.confidence),
            "confidencePercent": round(float(analysis.prediction.confidence) * 100, 1),
            "discordance": has_discordance,
        },
        "probabilities": probabilities,
        "interpretation": interpretation,
        "gradcam": gradcam,
        "reportUrl": report_url,
        "symptoms": {
            "priority": analysis.symptoms.priority,
            "selectedLabels": analysis.symptoms.selected_labels,
            "summary": analysis.symptoms.summary,
        },
        "labs": {
            "risk": analysis.labs.general_risk,
            "hormoneAlert": analysis.labs.hormone_alert,
            "abnormalities": analysis.labs.abnormalities,
            "derivedIndices": analysis.labs.derived_indices,
            "summary": analysis.labs.summary,
        },
        "priority": {
            "label": analysis.priority.priority,
            "warning": analysis.priority.warning,
            "reasons": analysis.priority.reasons,
        },
        "text": analysis.text,
    }


def _dashboard_payload() -> dict[str, Any]:
    history = read_history(HISTORY_PATH)
    if history.empty:
        return {
            "total": 0,
            "meanConfidence": None,
            "latestClass": "-",
            "latestPriority": "-",
            "classDistribution": [],
            "confidenceDistribution": [],
            "history": [],
            "latestGradcam": None,
            "usage": _usage_payload(history),
            "developerValidation": _developer_validation_payload(),
            "training": _training_payload(),
            "evaluation": {
                "confusionMatrix": _confusion_matrix_payload(),
                "classificationReport": _classification_report_payload(),
            },
            "model": _model_payload(),
        }

    tail = history.tail(20).copy()
    if "classe_prevista" in tail:
        tail["classe_prevista"] = tail["classe_prevista"].map(display_class_name)
    if "prioridade_final" in tail:
        tail["prioridade_final"] = tail["prioridade_final"].map(_display_portuguese_text)

    mean_confidence = float(history["confianca"].astype(float).mean()) * 100
    latest = history.iloc[-1]
    latest_gradcam = str(latest.get("gradcam_image", "") or "")
    return {
        "total": int(len(history)),
        "meanConfidence": round(mean_confidence, 1),
        "latestClass": display_class_name(str(latest.get("classe_prevista", ""))),
        "latestPriority": _display_portuguese_text(latest.get("prioridade_final", "-")),
        "classDistribution": _safe_records(class_distribution(history)),
        "confidenceDistribution": _safe_records(confidence_distribution(history)),
        "history": _safe_records(tail),
        "latestGradcam": latest_gradcam or None,
        "usage": _usage_payload(history),
        "developerValidation": _developer_validation_payload(),
        "training": _training_payload(),
        "evaluation": {
            "confusionMatrix": _confusion_matrix_payload(),
            "classificationReport": _classification_report_payload(),
        },
        "model": _model_payload(),
    }


INDEX_HTML = r"""
<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Análise MRI cerebral</title>
  <style>
    :root {
      --blue: #1769d1;
      --blue-2: #0e56af;
      --blue-3: #e8f3ff;
      --blue-4: #f5faff;
      --ink: #152c45;
      --muted: #60758d;
      --line: #d8e7f7;
      --panel: #ffffff;
      --page: #f6faff;
      --green: #1f9d6a;
      --amber: #c47a16;
      --red: #c24141;
      --shadow: 0 18px 42px rgba(27, 86, 142, .08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-width: 320px;
      background: var(--page);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }

    button, input { font: inherit; }

    .shell {
      width: min(100% - 32px, 1320px);
      margin: 0 auto;
      padding: 18px 0 28px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .mark {
      width: 42px;
      height: 42px;
      border-radius: 12px;
      background: var(--blue);
      display: grid;
      place-items: center;
      color: #fff;
      font-weight: 800;
      box-shadow: 0 10px 24px rgba(23, 105, 209, .22);
      flex: 0 0 auto;
    }

    h1, h2, h3, p { margin: 0; }

    h1 {
      font-size: clamp(21px, 3vw, 32px);
      line-height: 1.05;
      color: var(--ink);
    }

    .subtitle {
      margin-top: 3px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .status-pill {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--blue-2);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 700;
      white-space: nowrap;
    }

    .tabs {
      display: inline-flex;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 12px;
      padding: 4px;
      gap: 4px;
      margin-bottom: 14px;
    }

    .tab {
      border: 0;
      border-radius: 9px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font-weight: 800;
      padding: 9px 14px;
    }

    .tab.active {
      background: var(--blue);
      color: #fff;
    }

    .view { display: none; }
    .view.active { display: block; }

    .workbench {
      display: grid;
      grid-template-columns: minmax(260px, .86fr) minmax(330px, 1.12fr) minmax(330px, 1fr);
      gap: 14px;
      align-items: start;
    }

    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px 10px;
      border-bottom: 1px solid #edf4fb;
    }

    .panel-title {
      font-size: 15px;
      font-weight: 900;
    }

    .panel-note {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }

    .panel-body { padding: 14px 16px 16px; }

    .upload-box {
      display: grid;
      place-items: center;
      min-height: 250px;
      border: 1px dashed #9dc6f0;
      border-radius: 12px;
      background: var(--blue-4);
      cursor: pointer;
      position: relative;
      overflow: hidden;
    }

    .upload-box:hover {
      border-color: var(--blue);
      background: #eef7ff;
    }

    .upload-copy {
      text-align: center;
      color: var(--muted);
      padding: 18px;
    }

    .upload-copy strong {
      display: block;
      color: var(--ink);
      margin-bottom: 4px;
      font-size: 15px;
    }

    #imagePreview {
      display: none;
      width: 100%;
      height: 100%;
      max-height: 330px;
      object-fit: contain;
      background: #06121f;
    }

    #imageInput { display: none; }

    .file-name {
      margin-top: 9px;
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .primary-button {
      width: 100%;
      min-height: 44px;
      border: 0;
      border-radius: 12px;
      background: var(--blue);
      color: #fff;
      font-weight: 900;
      cursor: pointer;
      margin-top: 12px;
      box-shadow: 0 12px 24px rgba(23, 105, 209, .18);
    }

    .primary-button:hover { background: var(--blue-2); }
    .primary-button:disabled { opacity: .6; cursor: wait; }

    .notice {
      margin-top: 12px;
      border-left: 3px solid var(--blue);
      background: var(--blue-3);
      border-radius: 10px;
      padding: 10px 11px;
      color: #31536f;
      font-size: 12px;
      line-height: 1.45;
    }

    .option-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }

    .check-card {
      position: relative;
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 40px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      padding: 8px 9px;
      cursor: pointer;
      color: #28445f;
      font-size: 13px;
      font-weight: 700;
    }

    .check-card input {
      width: 16px;
      height: 16px;
      accent-color: var(--blue);
      flex: 0 0 auto;
    }

    .check-card:has(input:checked) {
      border-color: #86bbf4;
      background: var(--blue-3);
      color: var(--blue-2);
    }

    details {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      margin-top: 12px;
      overflow: hidden;
    }

    summary {
      cursor: pointer;
      list-style: none;
      padding: 12px 13px;
      color: var(--ink);
      font-size: 14px;
      font-weight: 900;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    summary::-webkit-details-marker { display: none; }
    summary::after {
      content: "+";
      width: 24px;
      height: 24px;
      border-radius: 8px;
      background: var(--blue-3);
      color: var(--blue);
      display: grid;
      place-items: center;
      font-weight: 900;
    }

    details[open] summary::after { content: "-"; }

    .lab-list {
      display: grid;
      gap: 8px;
      padding: 0 12px 12px;
    }

    .lab-row {
      display: grid;
      grid-template-columns: minmax(122px, 1fr) minmax(92px, .8fr);
      gap: 8px;
      align-items: center;
      border: 1px solid #edf4fb;
      border-radius: 10px;
      padding: 8px;
      background: #fbfdff;
    }

    .lab-label {
      display: flex;
      align-items: center;
      gap: 8px;
      min-width: 0;
      font-size: 13px;
      font-weight: 800;
      color: #28445f;
    }

    .lab-label input {
      width: 16px;
      height: 16px;
      accent-color: var(--blue);
      flex: 0 0 auto;
    }

    .lab-meta {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      margin-top: 2px;
    }

    .lab-row input[type="number"] {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 9px;
      padding: 0 9px;
      color: var(--ink);
      background: #fff;
    }

    .lab-row input[type="number"]:disabled {
      color: #9aaaBB;
      background: #f2f6fa;
    }

    .result-empty {
      min-height: 360px;
      display: grid;
      place-items: center;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 12px;
      background: #fbfdff;
      padding: 20px;
      font-weight: 700;
    }

    .result-stack {
      display: none;
      gap: 12px;
    }

    .gradcam-card {
      border: 1px solid var(--line);
      background: #fbfdff;
      border-radius: 12px;
      overflow: hidden;
    }

    .gradcam-card img {
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      object-fit: contain;
      background: #07131f;
    }

    .gradcam-copy {
      padding: 10px 12px 12px;
    }

    .gradcam-copy strong {
      display: block;
      color: var(--ink);
      font-size: 14px;
      margin-bottom: 4px;
    }

    .gradcam-copy span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
      font-weight: 700;
    }

    .dashboard-gradcam {
      margin-top: 14px;
      display: none;
      max-width: 360px;
    }

    .history-gradcam {
      width: 74px;
      height: 74px;
      border-radius: 10px;
      border: 1px solid var(--line);
      object-fit: cover;
      background: #07131f;
      display: block;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .metric {
      border: 1px solid var(--line);
      background: #fbfdff;
      border-radius: 12px;
      padding: 12px;
    }

    .metric span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .metric strong {
      display: block;
      margin-top: 4px;
      font-size: 20px;
      line-height: 1.1;
      color: var(--ink);
    }

    .metric small {
      display: block;
      margin-top: 7px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      font-weight: 700;
    }

    .case-insight-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }

    .report-link {
      display: none;
      align-items: center;
      justify-content: center;
      min-height: 42px;
      border: 1px solid #b9d7f7;
      border-radius: 12px;
      background: var(--blue-3);
      color: var(--blue-2);
      text-decoration: none;
      font-weight: 900;
      text-align: center;
      padding: 10px 12px;
    }

    .report-link:hover {
      background: #dcefff;
    }

    .priority-box {
      border: 1px solid #b9d7f7;
      background: var(--blue-3);
      border-radius: 12px;
      padding: 12px;
    }

    .priority-box strong {
      display: block;
      color: var(--blue-2);
      margin-bottom: 6px;
    }

    .priority-box p {
      color: #31536f;
      font-size: 13px;
      line-height: 1.45;
    }

    .discordance-alert {
      display: none;
      border: 1px solid #f3c36f;
      background: #fff8e8;
      color: #72430a;
      border-radius: 12px;
      padding: 12px;
      font-size: 13px;
      line-height: 1.45;
      font-weight: 700;
    }

    .prob-list {
      display: grid;
      gap: 10px;
      margin-top: 10px;
    }

    .prob-row {
      display: grid;
      grid-template-columns: minmax(118px, 1fr) 58px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      font-weight: 800;
    }

    .bar {
      grid-column: 1 / -1;
      height: 8px;
      border-radius: 999px;
      background: #e7f0fa;
      overflow: hidden;
    }

    .bar > i {
      display: block;
      height: 100%;
      width: 0%;
      background: var(--blue);
      border-radius: inherit;
      transition: width .35s ease;
    }

    .section-mini {
      border-top: 1px solid #edf4fb;
      padding-top: 12px;
      color: #36536d;
      font-size: 13px;
      line-height: 1.5;
    }

    .section-mini strong {
      display: block;
      color: var(--ink);
      margin-bottom: 3px;
    }

    .developer-panel {
      border: 1px solid #cfe2f5;
      background: #fbfdff;
      border-radius: 12px;
      overflow: hidden;
    }

    .developer-panel summary {
      padding: 12px;
    }

    .developer-body {
      display: grid;
      gap: 10px;
      padding: 0 12px 12px;
    }

    .developer-note {
      color: #31536f;
      font-size: 12px;
      line-height: 1.45;
      background: var(--blue-3);
      border-left: 3px solid var(--blue);
      border-radius: 10px;
      padding: 10px;
    }

    .field-label {
      display: grid;
      gap: 6px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 900;
    }

    .developer-select,
    .developer-textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 10px;
    }

    .developer-textarea {
      min-height: 78px;
      resize: vertical;
    }

    .secondary-button {
      min-height: 42px;
      border: 1px solid #b9d7f7;
      border-radius: 12px;
      background: #fff;
      color: var(--blue-2);
      font-weight: 900;
      cursor: pointer;
    }

    .secondary-button:hover {
      background: var(--blue-3);
    }

    .secondary-button:disabled {
      opacity: .65;
      cursor: wait;
    }

    .developer-status {
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .developer-dashboard {
      margin-top: 18px;
      border-top: 1px solid #d8e7f7;
      padding-top: 18px;
    }

    .developer-dashboard h2 {
      font-size: 18px;
      margin-bottom: 10px;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 900;
    }

    .status-badge.acerto {
      background: #e8f7ee;
      color: #146b3a;
    }

    .status-badge.erro {
      background: #fff0f0;
      color: #9a2424;
    }

    .dashboard-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }

    .analytics-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 14px;
    }

    .chart-card {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 12px;
      padding: 14px;
      min-height: 260px;
    }

    .chart-card.wide {
      grid-column: 1 / -1;
    }

    .chart-title {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }

    .chart-title strong {
      display: block;
      color: var(--ink);
      font-size: 15px;
    }

    .chart-title span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.35;
      font-weight: 700;
    }

    .chart {
      width: 100%;
      min-height: 220px;
    }

    .chart svg {
      display: block;
      width: 100%;
      height: auto;
      overflow: visible;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }

    .legend-dot {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: var(--blue);
    }

    .bar-list {
      display: grid;
      gap: 10px;
      margin-top: 8px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: minmax(120px, 1fr) 68px;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      font-weight: 800;
    }

    .bar-row .bar {
      grid-column: 1 / -1;
    }

    .matrix-wrap {
      overflow: auto;
    }

    .matrix-table {
      min-width: 520px;
      border-collapse: separate;
      border-spacing: 4px;
    }

    .matrix-table th,
    .matrix-table td {
      border: 0;
      padding: 8px;
      text-align: center;
      border-radius: 8px;
      font-size: 12px;
    }

    .matrix-table th {
      background: #f4f8fd;
      color: var(--muted);
    }

    .matrix-cell {
      color: #07131f;
      font-weight: 900;
    }

    .model-summary {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 8px;
    }

    .mini-stat {
      border: 1px solid #edf4fb;
      border-radius: 10px;
      padding: 10px;
      background: #fbfdff;
    }

    .mini-stat span {
      display: block;
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
    }

    .mini-stat strong {
      display: block;
      margin-top: 4px;
      color: var(--ink);
      font-size: 18px;
    }

    .table-wrap {
      margin-top: 14px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
      font-size: 13px;
    }

    th, td {
      text-align: left;
      border-bottom: 1px solid #edf4fb;
      padding: 10px 11px;
      vertical-align: top;
    }

    th {
      color: var(--muted);
      background: #fbfdff;
      font-size: 12px;
      text-transform: uppercase;
    }

    .inline-list {
      margin: 0;
      padding-left: 17px;
      color: #36536d;
      font-size: 13px;
      line-height: 1.45;
    }

    .error {
      display: none;
      margin-top: 10px;
      color: #8a2424;
      background: #fff3f3;
      border: 1px solid #ffd1d1;
      border-radius: 10px;
      padding: 10px;
      font-size: 13px;
      font-weight: 700;
    }

    @media (max-width: 1120px) {
      .workbench {
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      }

      .result-panel {
        grid-column: 1 / -1;
      }

      .dashboard-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .analytics-grid {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 760px) {
      .shell {
        width: min(100% - 20px, 1320px);
        padding-top: 12px;
      }

      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .subtitle {
        white-space: normal;
      }

      .tabs {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        width: 100%;
      }

      .workbench,
      .dashboard-grid,
      .analytics-grid,
      .model-summary,
      .case-insight-grid,
      .metric-grid,
      .option-grid {
        grid-template-columns: 1fr;
      }

      .panel-header {
        align-items: flex-start;
        flex-direction: column;
      }

      .lab-row {
        grid-template-columns: 1fr;
      }

      .upload-box {
        min-height: 210px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <div class="mark">MRI</div>
        <div>
          <h1>Análise MRI cerebral</h1>
          <p class="subtitle">Interface local para triagem complementar por imagem, sintomas e exames.</p>
        </div>
      </div>
      <div class="status-pill">Modelo local ativo</div>
    </header>

    <nav class="tabs" aria-label="Navegação">
      <button class="tab active" data-view="analysisView" type="button">Análise</button>
      <button class="tab" data-view="dashboardView" type="button">Dashboard</button>
      <button class="tab" data-view="trainingView" type="button">Treino do Modelo</button>
    </nav>

    <main id="analysisView" class="view active">
      <form id="analysisForm" class="workbench">
        <section class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Imagem MRI</h2>
              <p class="panel-note">Entrada principal do modelo</p>
            </div>
          </div>
          <div class="panel-body">
            <label class="upload-box" for="imageInput">
              <img id="imagePreview" alt="Prévia da imagem MRI" />
              <span id="uploadCopy" class="upload-copy">
                <strong>Selecionar imagem</strong>
                JPG, PNG ou JPEG
              </span>
            </label>
            <input id="imageInput" name="image" type="file" accept="image/*" />
            <div id="fileName" class="file-name">Nenhum arquivo selecionado</div>
            <button id="analyzeButton" class="primary-button" type="submit">Analisar caso</button>
            <div id="errorBox" class="error"></div>
            <div class="notice">Este sistema não realiza diagnóstico médico. Use o resultado como apoio para revisão profissional.</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Triagem complementar</h2>
              <p class="panel-note">Sintomas e exames podem ficar vazios</p>
            </div>
          </div>
          <div class="panel-body">
            <h3 class="panel-title">Sintomas</h3>
            <div id="symptomGrid" class="option-grid"></div>

            <details>
              <summary>Exames gerais</summary>
              <div id="generalLabList" class="lab-list"></div>
            </details>

            <details>
              <summary>Exames hormonais</summary>
              <div id="hormoneLabList" class="lab-list"></div>
            </details>
          </div>
        </section>

        <section class="panel result-panel">
          <div class="panel-header">
            <div>
              <h2 class="panel-title">Resultado</h2>
              <p class="panel-note">Predição e prioridade final</p>
            </div>
          </div>
          <div class="panel-body">
            <div id="resultEmpty" class="result-empty">Envie uma imagem para visualizar a análise.</div>
            <div id="resultStack" class="result-stack">
              <a id="downloadReportLink" class="report-link" href="#" download>Baixar relatório da análise</a>
              <div id="gradcamCard" class="gradcam-card">
                <img id="gradcamImage" alt="Mapa Grad-CAM sobreposto a MRI" />
                <div class="gradcam-copy">
                  <strong>Mapa de atenção Grad-CAM</strong>
                  <span id="gradcamDescription">As áreas destacadas indicam regiões que mais influenciaram a previsão, não uma segmentação clínica precisa.</span>
                </div>
              </div>
              <div class="metric-grid">
                <div class="metric">
                  <span>Classe prevista</span>
                  <strong id="predictedClass">-</strong>
                </div>
                <div class="metric">
                  <span>Confiança</span>
                  <strong id="confidenceValue">-</strong>
                </div>
              </div>
              <div class="case-insight-grid">
                <div class="metric">
                  <span>Tumor vs sem tumor</span>
                  <strong id="tumorVsNoTumorValue">-</strong>
                  <small id="tumorVsNoTumorDetail">-</small>
                </div>
                <div class="metric">
                  <span>Margem entre classes</span>
                  <strong id="classMarginValue">-</strong>
                  <small id="classMarginDetail">-</small>
                </div>
              </div>
              <div class="priority-box">
                <strong id="priorityLabel">-</strong>
                <p id="priorityWarning">-</p>
              </div>
              <div id="discordanceAlert" class="discordance-alert">
                Alerta de discordância: a saída do modelo ficou como sem tumor aparente, mas os dados complementares indicam alto risco. Trate como caso prioritário para revisão profissional.
              </div>
              <div>
                <h3 class="panel-title">Probabilidades</h3>
                <div id="probabilities" class="prob-list"></div>
              </div>
              <div class="section-mini">
                <strong>Sintomas</strong>
                <span id="symptomSummary">-</span>
              </div>
              <div class="section-mini">
                <strong>Exames</strong>
                <span id="labSummary">-</span>
              </div>
              <div class="section-mini">
                <strong>Motivos</strong>
                <ul id="reasonList" class="inline-list"></ul>
              </div>
              <details id="developerPanel" class="developer-panel">
                <summary>Desenvolvedor</summary>
                <div class="developer-body">
                  <div class="developer-note">
                    Use apenas quando houver certeza do diagnóstico. A validação salva a classe correta, compara com a previsão da IA e alimenta as métricas de acerto/erro do dashboard.
                  </div>
                  <label class="field-label">
                    Classe correta
                    <select id="developerCorrectClass" class="developer-select"></select>
                  </label>
                  <label class="field-label">
                    Observação opcional
                    <textarea id="developerNote" class="developer-textarea" placeholder="Ex.: confirmado por laudo, revisão médica ou base rotulada."></textarea>
                  </label>
                  <button id="saveDeveloperValidation" class="secondary-button" type="button">Salvar validação</button>
                  <div id="developerStatus" class="developer-status"></div>
                </div>
              </details>
            </div>
          </div>
        </section>
      </form>
    </main>

    <section id="dashboardView" class="view">
      <div class="dashboard-grid">
        <div class="metric"><span>Total analisado</span><strong id="dashTotal">0</strong></div>
        <div class="metric"><span>Confiança média</span><strong id="dashMean">-</strong></div>
        <div class="metric"><span>Última classe</span><strong id="dashClass">-</strong></div>
        <div class="metric"><span>Última prioridade</span><strong id="dashPriority">-</strong></div>
      </div>
      <div class="analytics-grid">
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Classes analisadas no site</strong>
              <span>Distribuição das predições registradas no histórico.</span>
            </div>
          </div>
          <div id="classDistributionChart" class="bar-list"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Faixas de confiança</strong>
              <span>Concentração de predições por confiança.</span>
            </div>
          </div>
          <div id="confidenceDistributionChart" class="bar-list"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Prioridades finais</strong>
              <span>Resumo das regras de triagem aplicadas.</span>
            </div>
          </div>
          <div id="priorityDistributionChart" class="bar-list"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Confiança recente</strong>
              <span>Últimas análises registradas no dashboard.</span>
            </div>
          </div>
          <div id="confidenceTrendChart" class="chart"></div>
        </div>
      </div>
      <div class="developer-dashboard">
        <h2>Validação do Desenvolvedor</h2>
        <div class="dashboard-grid">
          <div class="metric"><span>Casos validados</span><strong id="devTotal">0</strong></div>
          <div class="metric"><span>Acertos da IA</span><strong id="devCorrect">0</strong></div>
          <div class="metric"><span>Erros da IA</span><strong id="devIncorrect">0</strong></div>
          <div class="metric"><span>Acurácia validada</span><strong id="devAccuracy">-</strong></div>
        </div>
        <div class="analytics-grid">
          <div class="chart-card">
            <div class="chart-title">
              <div>
                <strong>Acerto vs erro</strong>
                <span>Somente validações salvas no modo Desenvolvedor.</span>
              </div>
            </div>
            <div id="developerStatusChart" class="bar-list"></div>
          </div>
          <div class="chart-card wide">
            <div class="chart-title">
              <div>
                <strong>Matriz de confusão validada</strong>
                <span>Linhas são classes corretas; colunas são previsões da IA.</span>
              </div>
            </div>
            <div id="developerConfusionMatrix" class="matrix-wrap"></div>
          </div>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Data</th>
                <th>Imagem</th>
                <th>IA previu</th>
                <th>Classe correta</th>
                <th>Resultado</th>
                <th>Confiança</th>
                <th>Observação</th>
              </tr>
            </thead>
            <tbody id="developerValidationBody"></tbody>
          </table>
        </div>
      </div>
      <div id="dashboardGradcam" class="gradcam-card dashboard-gradcam">
        <img id="dashboardGradcamImage" alt="Último mapa Grad-CAM registrado" />
        <div class="gradcam-copy">
          <strong>Último Grad-CAM</strong>
          <span>Mapa de atenção da análise mais recente registrada no histórico.</span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Grad-CAM</th>
              <th>Data</th>
              <th>Imagem</th>
              <th>Classe</th>
              <th>Confiança</th>
              <th>Prioridade</th>
            </tr>
          </thead>
          <tbody id="historyBody"></tbody>
        </table>
      </div>
    </section>

    <section id="trainingView" class="view">
      <div class="analytics-grid">
        <div class="chart-card wide">
          <div class="chart-title">
            <div>
              <strong>Resumo técnico do modelo</strong>
              <span>Parâmetros, camadas treináveis, otimizador e melhores métricas registradas.</span>
            </div>
          </div>
          <div id="modelSummary" class="model-summary"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Loss por época</strong>
              <span>Compara erro de treino e validação.</span>
            </div>
          </div>
          <div id="lossChart" class="chart"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Accuracy por época</strong>
              <span>Mostra acerto no treino e validação.</span>
            </div>
          </div>
          <div id="accuracyChart" class="chart"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Learning rate</strong>
              <span>Mostra quando o treino reduziu o passo de aprendizado.</span>
            </div>
          </div>
          <div id="learningRateChart" class="chart"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>F1-score por classe</strong>
              <span>Equilíbrio entre precisão e recall no conjunto de teste.</span>
            </div>
          </div>
          <div id="f1Chart" class="bar-list"></div>
        </div>
        <div class="chart-card">
          <div class="chart-title">
            <div>
              <strong>Precisão e recall</strong>
              <span>Desempenho individual por classe no teste.</span>
            </div>
          </div>
          <div id="precisionRecallChart" class="chart"></div>
        </div>
        <div class="chart-card wide">
          <div class="chart-title">
            <div>
              <strong>Matriz de confusão</strong>
              <span>Linhas são classes reais; colunas são classes previstas.</span>
            </div>
          </div>
          <div id="confusionMatrix" class="matrix-wrap"></div>
        </div>
      </div>
    </section>
  </div>

  <script id="app-config" type="application/json">__APP_CONFIG__</script>
  <script>
    const config = JSON.parse(document.getElementById("app-config").textContent);
    const imageInput = document.getElementById("imageInput");
    const imagePreview = document.getElementById("imagePreview");
    const uploadCopy = document.getElementById("uploadCopy");
    const fileName = document.getElementById("fileName");
    const errorBox = document.getElementById("errorBox");
    const analyzeButton = document.getElementById("analyzeButton");
    let currentAnalysis = null;

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function checkboxCard(item) {
      const label = document.createElement("label");
      label.className = "check-card";
      label.innerHTML = `<input type="checkbox" value="${escapeHtml(item.key)}" /> <span>${escapeHtml(item.label)}</span>`;
      return label;
    }

    function renderSymptoms() {
      const grid = document.getElementById("symptomGrid");
      grid.innerHTML = "";
      config.symptoms.forEach((item) => grid.appendChild(checkboxCard(item)));
    }

    function labRow(item) {
      const row = document.createElement("div");
      row.className = "lab-row";
      row.dataset.labKey = item.key;
      const value = item.default ?? "";
      row.innerHTML = `
        <label class="lab-label">
          <input class="lab-enabled" type="checkbox" />
          <span>${escapeHtml(item.label)}<span class="lab-meta">Ref. ${escapeHtml(item.range)}</span></span>
        </label>
        <input class="lab-value" type="number" step="any" value="${escapeHtml(value)}" disabled />
      `;
      const checkbox = row.querySelector(".lab-enabled");
      const input = row.querySelector(".lab-value");
      checkbox.addEventListener("change", () => {
        input.disabled = !checkbox.checked;
        if (checkbox.checked) input.focus();
      });
      return row;
    }

    function renderLabs() {
      const general = document.getElementById("generalLabList");
      const hormone = document.getElementById("hormoneLabList");
      general.innerHTML = "";
      hormone.innerHTML = "";
      config.generalLabs.forEach((item) => general.appendChild(labRow(item)));
      config.hormoneLabs.forEach((item) => hormone.appendChild(labRow(item)));
    }

    function renderDeveloperClasses() {
      const select = document.getElementById("developerCorrectClass");
      select.innerHTML = "";
      (config.classes || []).forEach((item) => {
        const option = document.createElement("option");
        option.value = item.key;
        option.textContent = item.label;
        select.appendChild(option);
      });
    }

    function selectedSymptoms() {
      return Array.from(document.querySelectorAll("#symptomGrid input:checked")).map((input) => input.value);
    }

    function selectedLabs() {
      const labs = {};
      document.querySelectorAll(".lab-row").forEach((row) => {
        const checkbox = row.querySelector(".lab-enabled");
        const input = row.querySelector(".lab-value");
        if (checkbox.checked) labs[row.dataset.labKey] = input.value;
      });
      return labs;
    }

    imageInput.addEventListener("change", () => {
      const file = imageInput.files[0];
      if (!file) {
        imagePreview.style.display = "none";
        uploadCopy.style.display = "block";
        fileName.textContent = "Nenhum arquivo selecionado";
        return;
      }
      fileName.textContent = file.name;
      imagePreview.src = URL.createObjectURL(file);
      imagePreview.style.display = "block";
      uploadCopy.style.display = "none";
    });

    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".view").forEach((view) => view.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(tab.dataset.view).classList.add("active");
        if (tab.dataset.view === "dashboardView" || tab.dataset.view === "trainingView") loadDashboard();
      });
    });

    function showError(message) {
      errorBox.textContent = message;
      errorBox.style.display = "block";
    }

    function clearError() {
      errorBox.textContent = "";
      errorBox.style.display = "none";
    }

    function renderAnalysis(data) {
      currentAnalysis = data;
      document.getElementById("resultEmpty").style.display = "none";
      document.getElementById("resultStack").style.display = "grid";
      document.getElementById("developerStatus").textContent = "";
      document.getElementById("developerNote").value = "";
      document.getElementById("developerPanel").open = false;
      const developerClassSelect = document.getElementById("developerCorrectClass");
      if (data.prediction?.className && Array.from(developerClassSelect.options).some((option) => option.value === data.prediction.className)) {
        developerClassSelect.value = data.prediction.className;
      }
      const gradcamCard = document.getElementById("gradcamCard");
      const gradcamImage = document.getElementById("gradcamImage");
      const gradcamDescription = document.getElementById("gradcamDescription");
      if (data.gradcam && data.gradcam.imageUrl) {
        gradcamImage.src = data.gradcam.imageUrl;
        gradcamImage.style.display = "block";
        gradcamDescription.textContent = data.gradcam.description;
        gradcamCard.style.display = "block";
      } else {
        gradcamImage.removeAttribute("src");
        gradcamImage.style.display = "none";
        gradcamDescription.textContent = data.gradcam?.description || "Mapa Grad-CAM indisponível para esta análise.";
        gradcamCard.style.display = "block";
      }
      document.getElementById("predictedClass").textContent = data.prediction.displayName;
      document.getElementById("confidenceValue").textContent = `${data.prediction.confidencePercent}%`;
      const tumorSummary = data.interpretation?.tumorVsNoTumor || {};
      const classMargin = data.interpretation?.classMargin || {};
      document.getElementById("tumorVsNoTumorValue").textContent = `${tumorSummary.tumorPercent ?? "-"}% tumor`;
      document.getElementById("tumorVsNoTumorDetail").textContent = `Sem tumor aparente: ${tumorSummary.noTumorPercent ?? "-"}%`;
      document.getElementById("classMarginValue").textContent = `${classMargin.marginPercent ?? "-"} p.p.`;
      document.getElementById("classMarginDetail").textContent = classMargin.description || "-";
      const reportLink = document.getElementById("downloadReportLink");
      if (data.reportUrl) {
        reportLink.href = data.reportUrl;
        reportLink.style.display = "flex";
      } else {
        reportLink.removeAttribute("href");
        reportLink.style.display = "none";
      }
      document.getElementById("priorityLabel").textContent = data.priority.label;
      document.getElementById("priorityWarning").textContent = data.priority.warning;
      document.getElementById("discordanceAlert").style.display = data.prediction.discordance ? "block" : "none";
      document.getElementById("symptomSummary").textContent = data.symptoms.summary;
      document.getElementById("labSummary").textContent = data.labs.summary;

      const probabilities = document.getElementById("probabilities");
      probabilities.innerHTML = "";
      data.probabilities.forEach((item) => {
        const row = document.createElement("div");
        row.className = "prob-row";
        row.innerHTML = `
          <span>${escapeHtml(item.label)}</span>
          <span>${escapeHtml(item.percent)}%</span>
          <div class="bar"><i style="width:${Math.max(0, Math.min(100, item.percent))}%"></i></div>
        `;
        probabilities.appendChild(row);
      });

      const reasonList = document.getElementById("reasonList");
      reasonList.innerHTML = "";
      data.priority.reasons.forEach((reason) => {
        const item = document.createElement("li");
        item.textContent = reason;
        reasonList.appendChild(item);
      });
    }

    function emptyChart(id, message = "Dados indisponíveis.") {
      const target = document.getElementById(id);
      if (!target) return;
      target.innerHTML = `<div class="result-empty" style="min-height:180px">${escapeHtml(message)}</div>`;
    }

    function numberLabel(value, digits = 1) {
      if (!Number.isFinite(value)) return "-";
      return Number(value).toFixed(digits);
    }

    function drawLineChart(id, series, options = {}) {
      const target = document.getElementById(id);
      if (!target) return;

      const cleanSeries = series.map((item) => ({
        ...item,
        values: item.values.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y)),
      })).filter((item) => item.values.length);

      if (!cleanSeries.length) {
        emptyChart(id);
        return;
      }

      const width = 640;
      const height = 260;
      const pad = { left: 54, right: 18, top: 18, bottom: 42 };
      const allPoints = cleanSeries.flatMap((item) => item.values);
      const minX = Math.min(...allPoints.map((point) => point.x));
      const maxX = Math.max(...allPoints.map((point) => point.x));
      let minY = options.minY ?? Math.min(...allPoints.map((point) => point.y));
      let maxY = options.maxY ?? Math.max(...allPoints.map((point) => point.y));
      if (minY === maxY) {
        minY = minY - 1;
        maxY = maxY + 1;
      }

      const xScale = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * (width - pad.left - pad.right);
      const yScale = (y) => height - pad.bottom - ((y - minY) / Math.max(0.000001, maxY - minY)) * (height - pad.top - pad.bottom);
      const yFormat = options.yFormat || ((value) => numberLabel(value, 2));
      const xFormat = options.xFormat || ((value) => numberLabel(value, 0));
      const ticks = [0, .25, .5, .75, 1].map((ratio) => minY + (maxY - minY) * ratio);

      const grid = ticks.map((tick) => {
        const y = yScale(tick);
        return `
          <line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#edf4fb" />
          <text x="${pad.left - 10}" y="${y + 4}" text-anchor="end" fill="#60758d" font-size="11">${escapeHtml(yFormat(tick))}</text>
        `;
      }).join("");

      const paths = cleanSeries.map((item) => {
        const points = item.values.map((point) => `${xScale(point.x)},${yScale(point.y)}`).join(" ");
        return `<polyline points="${points}" fill="none" stroke="${item.color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />`;
      }).join("");

      const endLabels = cleanSeries.map((item) => {
        const last = item.values[item.values.length - 1];
        return `<circle cx="${xScale(last.x)}" cy="${yScale(last.y)}" r="4" fill="${item.color}" />`;
      }).join("");

      target.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(options.label || "Gráfico")}">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff" />
          ${grid}
          <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#d8e7f7" />
          <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#d8e7f7" />
          <text x="${pad.left}" y="${height - 12}" fill="#60758d" font-size="11">${escapeHtml(xFormat(minX))}</text>
          <text x="${width - pad.right}" y="${height - 12}" fill="#60758d" font-size="11" text-anchor="end">${escapeHtml(xFormat(maxX))}</text>
          ${paths}
          ${endLabels}
        </svg>
        <div class="legend">
          ${cleanSeries.map((item) => `<span class="legend-item"><i class="legend-dot" style="background:${item.color}"></i>${escapeHtml(item.label)}</span>`).join("")}
        </div>
      `;
    }

    function drawBarList(id, rows, options = {}) {
      const target = document.getElementById(id);
      if (!target) return;
      const cleanRows = rows.filter((row) => Number.isFinite(Number(row.value)));
      if (!cleanRows.length) {
        emptyChart(id);
        return;
      }
      const maxValue = Math.max(...cleanRows.map((row) => Number(row.value)), 1);
      target.innerHTML = cleanRows.map((row) => {
        const value = Number(row.value);
        const width = Math.max(2, Math.min(100, (value / maxValue) * 100));
        const label = options.format ? options.format(value) : numberLabel(value, 1);
        return `
          <div class="bar-row">
            <span>${escapeHtml(row.label)}</span>
            <span>${escapeHtml(label)}</span>
            <div class="bar"><i style="width:${width}%; background:${row.color || "var(--blue)"}"></i></div>
          </div>
        `;
      }).join("");
    }

    function drawGroupedBars(id, rows, keys) {
      const target = document.getElementById(id);
      if (!target) return;
      if (!rows.length) {
        emptyChart(id);
        return;
      }
      const width = 640;
      const height = 280;
      const pad = { left: 58, right: 18, top: 22, bottom: 78 };
      const maxValue = Math.max(...rows.flatMap((row) => keys.map((key) => Number(row[key.field]) || 0)), 100);
      const groupWidth = (width - pad.left - pad.right) / Math.max(1, rows.length);
      const barWidth = Math.min(26, (groupWidth - 12) / keys.length);
      const yScale = (value) => height - pad.bottom - (value / maxValue) * (height - pad.top - pad.bottom);
      const bars = rows.map((row, rowIndex) => {
        const groupX = pad.left + rowIndex * groupWidth + groupWidth / 2;
        const label = escapeHtml(row.label);
        const rects = keys.map((key, keyIndex) => {
          const value = Number(row[key.field]) || 0;
          const x = groupX - (barWidth * keys.length) / 2 + keyIndex * barWidth;
          const y = yScale(value);
          return `<rect x="${x}" y="${y}" width="${barWidth - 3}" height="${height - pad.bottom - y}" rx="4" fill="${key.color}"><title>${label}: ${key.label} ${numberLabel(value, 1)}%</title></rect>`;
        }).join("");
        return `
          ${rects}
          <text x="${groupX}" y="${height - 48}" text-anchor="middle" fill="#60758d" font-size="11" transform="rotate(-28 ${groupX} ${height - 48})">${label}</text>
        `;
      }).join("");

      target.innerHTML = `
        <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Barras agrupadas">
          <rect x="0" y="0" width="${width}" height="${height}" fill="#fff" />
          <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#d8e7f7" />
          <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#d8e7f7" />
          <text x="${pad.left - 10}" y="${height - pad.bottom + 4}" text-anchor="end" fill="#60758d" font-size="11">0%</text>
          <text x="${pad.left - 10}" y="${pad.top + 4}" text-anchor="end" fill="#60758d" font-size="11">${numberLabel(maxValue, 0)}%</text>
          ${bars}
        </svg>
        <div class="legend">
          ${keys.map((key) => `<span class="legend-item"><i class="legend-dot" style="background:${key.color}"></i>${escapeHtml(key.label)}</span>`).join("")}
        </div>
      `;
    }

    function drawConfusionMatrix(data, targetId = "confusionMatrix") {
      const target = document.getElementById(targetId);
      if (!target) return;
      if (!data?.available || !data.matrix.length) {
        emptyChart(targetId);
        return;
      }
      const maxValue = Math.max(data.maxValue || 1, 1);
      const header = `<tr><th>Real \\ Prevista</th>${data.labels.map((label) => `<th>${escapeHtml(label)}</th>`).join("")}</tr>`;
      const rows = data.matrix.map((row, rowIndex) => {
        const cells = row.map((value) => {
          const intensity = Math.max(0.08, Number(value) / maxValue);
          const color = `rgba(23, 105, 209, ${intensity})`;
          const textColor = intensity > .55 ? "#fff" : "#07131f";
          return `<td class="matrix-cell" style="background:${color}; color:${textColor}">${escapeHtml(value)}</td>`;
        }).join("");
        return `<tr><th>${escapeHtml(data.labels[rowIndex])}</th>${cells}</tr>`;
      }).join("");
      target.innerHTML = `<table class="matrix-table">${header}${rows}</table>`;
    }

    function renderModelSummary(data) {
      const target = document.getElementById("modelSummary");
      if (!target) return;
      const training = data.training?.summary || {};
      const model = data.model || {};
      const report = data.evaluation?.classificationReport || {};
      target.innerHTML = [
        ["Modelo", model.name || "-"],
        ["Otimizador", model.optimizer || "Adam"],
        ["Parâmetros", model.totalParams ? model.totalParams.toLocaleString("pt-BR") : "-"],
        ["Camadas treináveis", model.trainableLayers ?? "-"],
        ["Camadas congeladas", model.frozenLayers ?? "-"],
        ["Épocas rodadas", training.totalEpochs ?? "-"],
        ["Melhor val_loss", training.bestValLoss ? `${training.bestValLoss} ep. ${training.bestValLossEpoch}` : "-"],
        ["Melhor val_accuracy", training.bestValAccuracy ? `${training.bestValAccuracy}% ep. ${training.bestValAccuracyEpoch}` : "-"],
        ["Accuracy teste", report.accuracy ? `${report.accuracy}%` : "-"],
      ].map(([label, value]) => `<div class="mini-stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("");
    }

    function renderTrainingAnalytics(data) {
      renderModelSummary(data);

      const history = data.training?.history || [];
      drawLineChart("lossChart", [
        { label: "loss", color: "#1769d1", values: history.map((row) => ({ x: Number(row.epoch), y: Number(row.loss) })) },
        { label: "val_loss", color: "#c24141", values: history.map((row) => ({ x: Number(row.epoch), y: Number(row.val_loss) })) },
      ], { label: "Loss por época", xFormat: (value) => `Ep. ${numberLabel(value, 0)}` });

      drawLineChart("accuracyChart", [
        { label: "accuracy", color: "#1f9d6a", values: history.map((row) => ({ x: Number(row.epoch), y: Number(row.accuracy) * 100 })) },
        { label: "val_accuracy", color: "#1769d1", values: history.map((row) => ({ x: Number(row.epoch), y: Number(row.val_accuracy) * 100 })) },
      ], { label: "Accuracy por época", minY: 0, maxY: 100, yFormat: (value) => `${numberLabel(value, 0)}%`, xFormat: (value) => `Ep. ${numberLabel(value, 0)}` });

      drawLineChart("learningRateChart", [
        { label: "learning_rate", color: "#c47a16", values: history.map((row) => ({ x: Number(row.epoch), y: Number(row.learning_rate) })) },
      ], { label: "Learning rate", yFormat: (value) => value.toExponential(1), xFormat: (value) => `Ep. ${numberLabel(value, 0)}` });

      const reportRows = data.evaluation?.classificationReport?.classes || [];
      drawBarList("f1Chart", reportRows.map((row) => ({ label: row.label, value: row.f1, color: "#1769d1" })), { format: (value) => `${numberLabel(value, 1)}%` });
      drawGroupedBars("precisionRecallChart", reportRows, [
        { field: "precision", label: "Precisão", color: "#1769d1" },
        { field: "recall", label: "Recall", color: "#1f9d6a" },
      ]);
      drawConfusionMatrix(data.evaluation?.confusionMatrix);
    }

    function renderUsageAnalytics(data) {
      drawBarList("classDistributionChart", (data.classDistribution || []).map((row) => ({ label: row.classe_prevista, value: Number(row.total), color: "#1769d1" })), { format: (value) => numberLabel(value, 0) });
      drawBarList("confidenceDistributionChart", (data.confidenceDistribution || []).map((row) => ({ label: row.faixa_confianca, value: Number(row.total), color: "#1f9d6a" })), { format: (value) => numberLabel(value, 0) });
      drawBarList("priorityDistributionChart", (data.usage?.priorityDistribution || []).map((row) => ({ label: row.label, value: Number(row.total), color: "#c47a16" })), { format: (value) => numberLabel(value, 0) });

      const trend = data.usage?.confidenceTrend || [];
      drawLineChart("confidenceTrendChart", [
        { label: "confiança", color: "#1769d1", values: trend.map((row, index) => ({ x: index + 1, y: Number(row.confianca_percent) })) },
      ], { label: "Confiança recente", minY: 0, maxY: 100, yFormat: (value) => `${numberLabel(value, 0)}%`, xFormat: (value) => `#${numberLabel(value, 0)}` });
    }

    function renderDeveloperValidation(validation) {
      const summary = validation?.summary || {};
      document.getElementById("devTotal").textContent = summary.total ?? 0;
      document.getElementById("devCorrect").textContent = summary.correct ?? 0;
      document.getElementById("devIncorrect").textContent = summary.incorrect ?? 0;
      document.getElementById("devAccuracy").textContent = summary.accuracy === null || summary.accuracy === undefined ? "-" : `${summary.accuracy}%`;

      drawBarList("developerStatusChart", (validation?.statusDistribution || []).map((row) => ({
        label: row.label,
        value: Number(row.total),
        color: row.label === "Acertos" ? "#1f9d6a" : "#c24141",
      })), { format: (value) => numberLabel(value, 0) });
      drawConfusionMatrix(validation?.confusionMatrix, "developerConfusionMatrix");

      const body = document.getElementById("developerValidationBody");
      const history = validation?.history || [];
      body.innerHTML = "";
      if (!history.length) {
        body.innerHTML = `<tr><td colspan="7">Nenhuma validação do Desenvolvedor registrada.</td></tr>`;
        return;
      }

      history.slice().reverse().forEach((row) => {
        const tr = document.createElement("tr");
        const badgeClass = row.result === "acerto" ? "acerto" : "erro";
        tr.innerHTML = `
          <td>${escapeHtml(row.date || "-")}</td>
          <td>${escapeHtml(row.imageName || "-")}</td>
          <td>${escapeHtml(row.predictedLabel || "-")}</td>
          <td>${escapeHtml(row.trueLabel || "-")}</td>
          <td><span class="status-badge ${badgeClass}">${escapeHtml(row.resultLabel || "-")}</span></td>
          <td>${Number(row.confidencePercent || 0).toFixed(1)}%</td>
          <td>${escapeHtml(row.note || "-")}</td>
        `;
        body.appendChild(tr);
      });
    }

    document.getElementById("analysisForm").addEventListener("submit", async (event) => {
      event.preventDefault();
      clearError();
      const file = imageInput.files[0];
      if (!file) {
        showError("Selecione uma imagem MRI antes de analisar.");
        return;
      }

      analyzeButton.disabled = true;
      analyzeButton.textContent = "Analisando...";
      const formData = new FormData();
      formData.append("image", file);
      formData.append("case", JSON.stringify({
        symptoms: selectedSymptoms(),
        labs: selectedLabs(),
      }));

      try {
        const response = await fetch("/api/analyze", { method: "POST", body: formData });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Falha ao analisar o caso.");
        renderAnalysis(payload);
        loadDashboard();
      } catch (error) {
        showError(error.message);
      } finally {
        analyzeButton.disabled = false;
        analyzeButton.textContent = "Analisar caso";
      }
    });

    document.getElementById("saveDeveloperValidation").addEventListener("click", async () => {
      const status = document.getElementById("developerStatus");
      const button = document.getElementById("saveDeveloperValidation");
      if (!currentAnalysis?.analysisId) {
        status.textContent = "Faça uma análise antes de salvar a validação.";
        return;
      }

      button.disabled = true;
      status.textContent = "Salvando validação...";
      try {
        const response = await fetch("/api/developer-validation", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            analysisId: currentAnalysis.analysisId,
            trueClass: document.getElementById("developerCorrectClass").value,
            note: document.getElementById("developerNote").value,
          }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Falha ao salvar a validação.");
        status.textContent = `Validação salva: ${payload.validation.resultLabel}.`;
        loadDashboard();
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });

    async function loadDashboard() {
      const response = await fetch("/api/dashboard");
      const data = await response.json();
      document.getElementById("dashTotal").textContent = data.total;
      document.getElementById("dashMean").textContent = data.meanConfidence === null ? "-" : `${data.meanConfidence}%`;
      document.getElementById("dashClass").textContent = data.latestClass || "-";
      document.getElementById("dashPriority").textContent = data.latestPriority || "-";
      renderUsageAnalytics(data);
      renderDeveloperValidation(data.developerValidation);
      renderTrainingAnalytics(data);
      const dashboardGradcam = document.getElementById("dashboardGradcam");
      const dashboardGradcamImage = document.getElementById("dashboardGradcamImage");
      if (data.latestGradcam) {
        dashboardGradcamImage.src = data.latestGradcam;
        dashboardGradcam.style.display = "block";
      } else {
        dashboardGradcamImage.removeAttribute("src");
        dashboardGradcam.style.display = "none";
      }

      const body = document.getElementById("historyBody");
      body.innerHTML = "";
      if (!data.history.length) {
        body.innerHTML = `<tr><td colspan="6">Nenhuma análise registrada.</td></tr>`;
        return;
      }
      data.history.slice().reverse().forEach((row) => {
        const tr = document.createElement("tr");
        const confidence = Number(row.confianca || 0) * 100;
        const gradcamCell = row.gradcam_image
          ? `<img class="history-gradcam" src="${escapeHtml(row.gradcam_image)}" alt="Grad-CAM da análise" />`
          : "-";
        tr.innerHTML = `
          <td>${gradcamCell}</td>
          <td>${escapeHtml(row.data_hora || "-")}</td>
          <td>${escapeHtml(row.nome_imagem || "-")}</td>
          <td>${escapeHtml(row.classe_prevista || "-")}</td>
          <td>${confidence.toFixed(1)}%</td>
          <td>${escapeHtml(row.prioridade_final || "-")}</td>
        `;
        body.appendChild(tr);
      });
    }

    renderSymptoms();
    renderLabs();
    renderDeveloperClasses();
    loadDashboard();
  </script>
</body>
</html>
"""


def _configure_local_server_environment() -> None:
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)


def create_app():
    app = FastAPI(title="Brain MRI Triage")
    GRADCAM_DIR.mkdir(parents=True, exist_ok=True)
    CASE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CASE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    app.mount("/gradcam", StaticFiles(directory=GRADCAM_DIR), name="gradcam")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        config_json = json.dumps(_frontend_config(), ensure_ascii=False)
        return INDEX_HTML.replace("__APP_CONFIG__", config_json)

    @app.get("/api/health")
    async def health():
        return {
            "ok": MODEL is not None,
            "modelError": MODEL_ERROR,
            "classes": CLASS_NAMES,
        }

    @app.get("/api/dashboard")
    async def dashboard():
        return _dashboard_payload()

    @app.post("/api/developer-validation")
    async def developer_validation(payload: dict[str, Any]):
        analysis_id = str(payload.get("analysisId", "") or "").strip()
        true_class_input = str(payload.get("trueClass", "") or "").strip()
        note = str(payload.get("note", "") or "")
        if not analysis_id:
            raise HTTPException(status_code=400, detail="Análise não informada.")
        if not true_class_input:
            raise HTTPException(status_code=400, detail="Classe correta não informada.")

        valid_classes = {compact_class_name(class_name): class_name for class_name in CLASS_NAMES}
        true_class = valid_classes.get(compact_class_name(true_class_input))
        if true_class is None:
            raise HTTPException(status_code=400, detail="Classe correta inválida.")

        history = read_history(HISTORY_PATH)
        if history.empty or "analysis_id" not in history:
            raise HTTPException(status_code=404, detail="Análise não encontrada no histórico.")

        matches = history[history["analysis_id"].astype(str) == analysis_id]
        if matches.empty:
            raise HTTPException(status_code=404, detail="Análise não encontrada no histórico.")

        validation = upsert_developer_validation(matches.iloc[-1].to_dict(), true_class, note)
        return {"ok": True, "validation": _validation_display_row(validation)}

    @app.get("/case-reports/{filename}")
    async def case_report(filename: str):
        safe_name = Path(filename).name
        report_path = CASE_REPORTS_DIR / safe_name
        if not report_path.exists() or report_path.suffix.lower() != ".html":
            raise HTTPException(status_code=404, detail="Relatório não encontrado.")
        return FileResponse(
            report_path,
            media_type="text/html; charset=utf-8",
            filename=safe_name,
        )

    @app.post("/api/analyze")
    async def analyze(image: UploadFile = File(...), case: str = Form("{}")):
        if MODEL is None:
            raise HTTPException(
                status_code=503,
                detail=f"Modelo ainda não carregado. Detalhe: {MODEL_ERROR}",
            )

        try:
            payload = json.loads(case or "{}")
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Dados do formulário inválidos.") from exc

        try:
            contents = await image.read()
            pil_image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Imagem inválida ou não suportada.") from exc

        symptoms = _symptom_dict(payload.get("symptoms"))
        labs = _labs_dict(payload.get("labs"))
        analysis_id = uuid4().hex
        image_name = Path(image.filename or "imagem_web").name
        saved_image_path = _save_case_image(pil_image, image_name, analysis_id)
        analysis = analyze_case(pil_image, symptoms, labs, MODEL, CLASS_NAMES)
        gradcam_payload = None
        gradcam_path = None
        try:
            pred_index = CLASS_NAMES.index(analysis.prediction.class_name)
            gradcam = save_gradcam_overlay(pil_image, MODEL, pred_index=pred_index)
            gradcam_path = gradcam.image_path
            gradcam_payload = {
                "imageUrl": gradcam.image_url,
                "layerName": gradcam.layer_name,
                "description": "Mapa de atenção do modelo; não representa segmentação clínica precisa.",
            }
        except Exception as exc:
            gradcam_payload = {
                "imageUrl": None,
                "layerName": None,
                "description": "Não foi possível gerar o mapa Grad-CAM para esta imagem.",
                "error": str(exc),
            }
        append_analysis_history(
            analysis,
            symptoms,
            labs,
            image_name=image_name,
            history_path=HISTORY_PATH,
            gradcam_image=gradcam_payload.get("imageUrl") if gradcam_payload else None,
            analysis_id=analysis_id,
            saved_image_path=saved_image_path,
        )
        response_payload = _analysis_response(analysis, gradcam=gradcam_payload, analysis_id=analysis_id)
        try:
            report_path = create_case_report(
                original_image=pil_image,
                gradcam_image_path=gradcam_path,
                image_name=image_name,
                response=response_payload,
            )
            response_payload["reportUrl"] = f"/case-reports/{report_path.name}"
        except Exception as exc:
            response_payload["reportUrl"] = None
            response_payload["reportError"] = str(exc)
        return response_payload

    return app


def main() -> None:
    import uvicorn

    _configure_local_server_environment()
    uvicorn.run(create_app(), host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
