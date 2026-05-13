from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import (
    HIGH_CONFIDENCE_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    display_class_name,
    is_no_tumor_class,
    is_pituitary_class,
)


SYMPTOM_OPTIONS = [
    {"key": "dor_cabeca_persistente", "label": "Dor de cabeça persistente", "weight": 1, "group": "neuro"},
    {"key": "convulsao", "label": "Convulsão", "weight": 3, "group": "neuro"},
    {"key": "alteracao_visual", "label": "Alteração visual", "weight": 2, "group": "neuro"},
    {"key": "dificuldade_fala", "label": "Dificuldade na fala", "weight": 2, "group": "neuro"},
    {"key": "fraqueza_lado_corpo", "label": "Fraqueza em um lado do corpo", "weight": 3, "group": "neuro"},
    {"key": "confusao_mental", "label": "Confusão mental", "weight": 2, "group": "neuro"},
    {"key": "perda_equilibrio", "label": "Perda de equilíbrio", "weight": 2, "group": "neuro"},
    {"key": "alteracao_memoria", "label": "Alteração de memória", "weight": 1, "group": "neuro"},
    {"key": "nauseas_vomitos", "label": "Náuseas ou vômitos frequentes", "weight": 1, "group": "neuro"},
    {"key": "sonolencia_excessiva", "label": "Sonolência excessiva", "weight": 1, "group": "neuro"},
    {"key": "mudanca_comportamento", "label": "Mudança de comportamento", "weight": 1, "group": "neuro"},
    {"key": "perda_visao_periferica", "label": "Perda de visão periférica", "weight": 3, "group": "pituitary"},
    {"key": "alteracoes_menstruais", "label": "Alterações menstruais", "weight": 2, "group": "pituitary"},
    {"key": "infertilidade", "label": "Infertilidade", "weight": 2, "group": "pituitary"},
    {"key": "reducao_libido", "label": "Redução da libido", "weight": 1, "group": "pituitary"},
    {"key": "galactorreia", "label": "Produção de leite fora da amamentação", "weight": 2, "group": "pituitary"},
    {"key": "crescimento_anormal", "label": "Crescimento anormal de mãos, pés ou face", "weight": 2, "group": "pituitary"},
    {"key": "alteracoes_peso", "label": "Alterações de peso sem causa clara", "weight": 1, "group": "pituitary"},
    {"key": "sinais_hormonais", "label": "Sinais de alteração hormonal", "weight": 2, "group": "pituitary"},
]

SYMPTOM_BY_KEY = {item["key"]: item for item in SYMPTOM_OPTIONS}
SYMPTOM_KEY_BY_LABEL = {item["label"]: item["key"] for item in SYMPTOM_OPTIONS}

GENERAL_LAB_RANGES = {
    "leucocitos": (4000, 11000),
    "neutrofilos": (1800, 7700),
    "linfocitos": (1000, 4800),
    "monocitos": (200, 1000),
    "plaquetas": (150000, 450000),
    "hemoglobina": (12.0, 17.5),
    "albumina": (3.5, 5.2),
    "pcr": (0.0, 5.0),
    "fibrinogenio": (200, 400),
}

HORMONE_LAB_RANGES = {
    "prolactina": (4.0, 25.0),
    "gh": (0.0, 10.0),
    "igf1": (80.0, 350.0),
    "acth": (7.0, 63.0),
    "cortisol": (5.0, 25.0),
    "tsh": (0.4, 4.5),
    "t3": (80.0, 180.0),
    "t4": (4.5, 12.5),
    "lh": (1.0, 20.0),
    "fsh": (1.0, 20.0),
    "testosterona": (15.0, 1000.0),
    "estrogenio": (10.0, 400.0),
    "progesterona": (0.1, 25.0),
}


@dataclass(frozen=True)
class SymptomAssessment:
    priority: str
    neuro_score: int
    pituitary_score: int
    selected_labels: list[str]
    summary: str


@dataclass(frozen=True)
class LabAssessment:
    general_risk: str
    hormone_alert: bool
    abnormalities: list[str]
    derived_indices: dict[str, float]
    summary: str


@dataclass(frozen=True)
class PriorityAssessment:
    priority: str
    warning: str
    reasons: list[str]


def evaluate_symptoms(symptoms: Mapping[str, bool] | None) -> SymptomAssessment:
    symptoms = symptoms or {}
    selected_keys = {key for key, value in symptoms.items() if bool(value)}
    selected_labels = [SYMPTOM_BY_KEY[key]["label"] for key in selected_keys if key in SYMPTOM_BY_KEY]

    neuro_score = sum(
        item["weight"]
        for item in SYMPTOM_OPTIONS
        if item["group"] == "neuro" and item["key"] in selected_keys
    )
    pituitary_score = sum(
        item["weight"]
        for item in SYMPTOM_OPTIONS
        if item["group"] == "pituitary" and item["key"] in selected_keys
    )
    total_score = neuro_score + pituitary_score

    high_alert_keys = {"convulsao", "fraqueza_lado_corpo", "perda_visao_periferica"}
    if selected_keys & high_alert_keys or total_score >= 9:
        priority = "alta"
    elif total_score >= 4 or pituitary_score >= 3:
        priority = "media"
    else:
        priority = "baixa"

    if not selected_labels:
        summary = "Nenhum sintoma foi informado."
    elif priority == "alta":
        summary = "Alta prioridade por sintomas neurológicos, visuais ou hormonais informados."
    elif priority == "media":
        summary = "Prioridade moderada por sintomas complementares informados."
    else:
        summary = "Sintomas informados com baixa prioridade complementar."

    return SymptomAssessment(
        priority=priority,
        neuro_score=neuro_score,
        pituitary_score=pituitary_score,
        selected_labels=selected_labels,
        summary=summary,
    )


def _to_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _outside_range(value: float | None, normal_range: tuple[float, float]) -> bool:
    if value is None:
        return False
    lower, upper = normal_range
    return value < lower or value > upper


def _safe_divide(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def calculate_derived_indices(labs: Mapping[str, float | None]) -> dict[str, float]:
    neutrophils = _to_float(labs.get("neutrofilos"))
    lymphocytes = _to_float(labs.get("linfocitos"))
    monocytes = _to_float(labs.get("monocitos"))
    platelets = _to_float(labs.get("plaquetas"))
    albumin = _to_float(labs.get("albumina"))

    indices = {
        "nlr": _safe_divide(neutrophils, lymphocytes),
        "plr": _safe_divide(platelets, lymphocytes),
        "lmr": _safe_divide(lymphocytes, monocytes),
        "sii": None,
        "pni": None,
    }
    if platelets is not None and neutrophils is not None and lymphocytes not in (None, 0):
        indices["sii"] = platelets * neutrophils / lymphocytes
    if albumin is not None and lymphocytes is not None:
        indices["pni"] = albumin + 5 * (lymphocytes / 1000)

    return {key: round(value, 3) for key, value in indices.items() if value is not None}


def evaluate_labs(labs: Mapping[str, float | None] | None, predicted_class: str | None = None) -> LabAssessment:
    labs = labs or {}
    parsed = {key: _to_float(value) for key, value in labs.items()}
    derived = calculate_derived_indices(parsed)
    abnormalities: list[str] = []
    score = 0

    for key, normal_range in GENERAL_LAB_RANGES.items():
        value = parsed.get(key)
        if _outside_range(value, normal_range):
            abnormalities.append(key)
            score += 1

    if parsed.get("pcr") is not None and parsed["pcr"] > 10:
        score += 1
    if derived.get("nlr", 0) > 4:
        abnormalities.append("nlr")
        score += 2
    if derived.get("plr", 0) > 250:
        abnormalities.append("plr")
        score += 1
    if derived.get("lmr", 999) < 2:
        abnormalities.append("lmr")
        score += 1
    if derived.get("sii", 0) > 1000:
        abnormalities.append("sii")
        score += 1
    if derived.get("pni", 999) < 45:
        abnormalities.append("pni")
        score += 1

    hormone_abnormalities = [
        key
        for key, normal_range in HORMONE_LAB_RANGES.items()
        if _outside_range(parsed.get(key), normal_range)
    ]
    abnormalities.extend(hormone_abnormalities)
    hormone_alert = bool(hormone_abnormalities)

    if score <= 1:
        general_risk = "baixo"
    elif score <= 4:
        general_risk = "moderado"
    else:
        general_risk = "alto"

    supplied_values = [value for value in parsed.values() if value is not None]
    if not supplied_values:
        summary = "Nenhum exame laboratorial foi informado."
    elif predicted_class and is_pituitary_class(predicted_class) and hormone_alert:
        summary = (
            "Alterações hormonais complementares foram informadas junto a uma classe de imagem "
            "compatível com tumor pituitário."
        )
    elif hormone_alert:
        summary = "Alterações hormonais complementares foram informadas."
    elif general_risk == "alto":
        summary = "Alterações laboratoriais inflamatórias ou hematológicas relevantes foram informadas."
    elif general_risk == "moderado":
        summary = "Algumas alterações laboratoriais complementares foram informadas."
    else:
        summary = "Sem alterações laboratoriais relevantes nas faixas simplificadas do sistema."

    return LabAssessment(
        general_risk=general_risk,
        hormone_alert=hormone_alert,
        abnormalities=sorted(set(abnormalities)),
        derived_indices=derived,
        summary=summary,
    )


def define_final_priority(
    predicted_class: str,
    confidence: float,
    symptoms: SymptomAssessment,
    labs: LabAssessment,
) -> PriorityAssessment:
    reasons: list[str] = []

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        level = 2
        reasons.append("confiança baixa do modelo de imagem")
    elif not is_no_tumor_class(predicted_class) and confidence >= HIGH_CONFIDENCE_THRESHOLD:
        level = 3
        reasons.append("imagem com alta confiança para classe tumoral treinada")
    elif not is_no_tumor_class(predicted_class):
        level = 2
        reasons.append("imagem classificada em uma classe tumoral treinada")
    else:
        level = 1
        reasons.append("modelo classificou como sem tumor aparente entre as classes treinadas")

    if symptoms.priority == "alta":
        if is_no_tumor_class(predicted_class):
            level = max(level, 3)
            reasons.append("discordância entre sintomas graves e classe sem tumor aparente")
        else:
            level = max(level, 3)
            reasons.append("sintomas graves associados a imagem suspeita")
    elif symptoms.priority == "media":
        level = max(level, 2)
        reasons.append("sintomas complementares com prioridade moderada")

    if labs.general_risk == "alto":
        level = max(level, 2)
        reasons.append("exames gerais com alterações complementares relevantes")
    elif labs.general_risk == "moderado":
        level = max(level, 2)
        reasons.append("exames gerais com alterações complementares moderadas")

    if is_pituitary_class(predicted_class) and labs.hormone_alert:
        level = max(level, 3)
        reasons.append("classe pituitária com alterações hormonais informadas")

    if level >= 3:
        priority = "Alta prioridade para avaliação profissional"
    elif level == 2:
        priority = "Prioridade elevada para revisão profissional"
    else:
        priority = "Baixa prioridade relativa, mantendo revisão profissional quando indicada"

    warning = "Este sistema não realiza diagnóstico médico."
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        warning += " Resultado incerto: recomenda-se revisão por profissional qualificado."
    if is_no_tumor_class(predicted_class):
        if symptoms.priority == "alta":
            warning += (
                " Há discordância clínica: sintomas graves foram informados apesar da classe "
                "sem tumor aparente, portanto este caso deve ser revisado com alta prioridade."
            )
        warning += (
            " A classe sem tumor aparente é apenas a saída do modelo atual e não descarta AVC, hemorragia, abscesso, "
            "metástase, processos inflamatórios ou outras condições fora das classes treinadas."
        )

    return PriorityAssessment(priority=priority, warning=warning, reasons=reasons)


def compose_response(
    predicted_class: str,
    confidence: float,
    probabilities: Mapping[str, float],
    symptoms: SymptomAssessment,
    labs: LabAssessment,
    priority: PriorityAssessment,
) -> str:
    probability_lines = [
        f"- {display_class_name(class_name)}: {probability * 100:.1f}%"
        for class_name, probability in probabilities.items()
    ]

    symptom_labels = ", ".join(symptoms.selected_labels) if symptoms.selected_labels else "Nenhum"
    derived = (
        ", ".join(f"{key.upper()}={value}" for key, value in labs.derived_indices.items())
        if labs.derived_indices
        else "Não calculados"
    )
    abnormalities = ", ".join(labs.abnormalities) if labs.abnormalities else "Nenhuma relevante"

    return f"""Resultado da análise por imagem:
Classe prevista: {display_class_name(predicted_class)}
Confiança da classificação: {confidence * 100:.1f}%

Probabilidades por classe:
{chr(10).join(probability_lines)}

Triagem por sintomas:
Prioridade: {symptoms.priority}
Sintomas informados: {symptom_labels}
Observação: {symptoms.summary}

Exames laboratoriais:
Risco complementar: {labs.general_risk}
Índices derivados: {derived}
Alterações sinalizadas: {abnormalities}
Observação: {labs.summary}

Prioridade final:
{priority.priority}
Motivos: {", ".join(priority.reasons)}

Aviso:
{priority.warning}
"""
