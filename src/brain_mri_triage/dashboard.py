from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import HISTORY_PATH, display_class_name


def read_history(history_path: str | Path = HISTORY_PATH) -> pd.DataFrame:
    history_path = Path(history_path)
    if not history_path.exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


def class_distribution(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "classe_prevista" not in history:
        return pd.DataFrame(columns=["classe_prevista", "total"])
    counts = history["classe_prevista"].value_counts().rename_axis("classe_prevista").reset_index(name="total")
    counts["classe_prevista"] = counts["classe_prevista"].map(display_class_name)
    return counts


def confidence_distribution(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "confianca" not in history:
        return pd.DataFrame(columns=["faixa_confianca", "total"])

    bins = [0.0, 0.6, 0.8, 1.0]
    labels = ["< 60%", "60% a 79%", ">= 80%"]
    confidence = pd.to_numeric(history["confianca"], errors="coerce").fillna(0)
    bucket = pd.cut(confidence, bins=bins, labels=labels, include_lowest=True)
    return bucket.value_counts().sort_index().rename_axis("faixa_confianca").reset_index(name="total")


def dashboard_summary(history: pd.DataFrame) -> str:
    if history.empty:
        return "Nenhuma análise registrada ainda."

    total = len(history)
    mean_confidence = pd.to_numeric(history["confianca"], errors="coerce").mean() * 100
    latest = history.iloc[-1]
    latest_class = display_class_name(str(latest.get("classe_prevista", "")))
    latest_priority = latest.get("prioridade_final", "")

    return (
        f"Total de exames analisados: {total}\n\n"
        f"Confiança média: {mean_confidence:.1f}%\n\n"
        f"Última classe prevista: {latest_class}\n\n"
        f"Última prioridade final: {latest_priority}\n\n"
        "A matriz de confusão 4x4 fica em reports/confusion_matrix.csv após rodar brain-evaluate."
    )
