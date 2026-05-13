from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import base64

from PIL import Image

from .config import CASE_REPORTS_DIR, display_class_name


def _image_to_data_uri(image) -> str:
    if image is None:
        return ""

    buffer = BytesIO()
    if isinstance(image, Image.Image):
        image.convert("RGB").save(buffer, format="PNG")
    else:
        Image.open(image).convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _percent(value: float | int | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}%"


def _probability_rows(probabilities: list[dict]) -> str:
    rows = []
    for item in probabilities:
        rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('label', '-')))}</td>"
            f"<td>{_percent(item.get('percent'))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _list_items(items: list[str]) -> str:
    if not items:
        return "<li>Nenhum motivo adicional registrado.</li>"
    return "\n".join(f"<li>{escape(str(item))}</li>" for item in items)


def create_case_report(
    *,
    original_image,
    gradcam_image_path: str | Path | None,
    image_name: str,
    response: dict,
    output_dir: str | Path = CASE_REPORTS_DIR,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_uri = _image_to_data_uri(original_image)
    gradcam_uri = _image_to_data_uri(gradcam_image_path) if gradcam_image_path else ""
    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    filename = f"relatorio_caso_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.html"
    output_path = output_dir / filename

    prediction = response.get("prediction", {})
    interpretation = response.get("interpretation", {})
    tumor_summary = interpretation.get("tumorVsNoTumor", {})
    margin = interpretation.get("classMargin", {})
    symptoms = response.get("symptoms", {})
    labs = response.get("labs", {})
    priority = response.get("priority", {})
    gradcam = response.get("gradcam") or {}

    html = f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório de análise MRI</title>
  <style>
    body {{
      margin: 0;
      background: #f6faff;
      color: #152c45;
      font-family: Arial, sans-serif;
      line-height: 1.45;
    }}
    .page {{
      max-width: 980px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 28px; }}
    h2 {{ font-size: 18px; margin-bottom: 10px; }}
    .muted {{ color: #60758d; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
    .card {{
      background: #fff;
      border: 1px solid #d8e7f7;
      border-radius: 12px;
      padding: 16px;
      margin-top: 14px;
    }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
    .metric {{
      border: 1px solid #edf4fb;
      border-radius: 10px;
      padding: 12px;
      background: #fbfdff;
    }}
    .metric span {{ display: block; color: #60758d; font-size: 12px; font-weight: bold; }}
    .metric strong {{ display: block; margin-top: 4px; font-size: 20px; }}
    img {{
      width: 100%;
      max-height: 420px;
      object-fit: contain;
      background: #07131f;
      border-radius: 10px;
      border: 1px solid #d8e7f7;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ text-align: left; border-bottom: 1px solid #edf4fb; padding: 8px; }}
    th {{ color: #60758d; font-size: 12px; text-transform: uppercase; }}
    ul {{ margin: 0; padding-left: 20px; }}
    .warning {{
      background: #e8f3ff;
      border: 1px solid #b9d7f7;
      border-radius: 10px;
      padding: 12px;
      color: #0e56af;
      font-weight: bold;
    }}
    @media print {{
      body {{ background: #fff; }}
      .page {{ padding: 0; }}
      .card {{ break-inside: avoid; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <h1>Relatório de análise MRI cerebral</h1>
      <p class="muted">Gerado em {escape(generated_at)} | Imagem: {escape(image_name)}</p>
    </header>

    <section class="card">
      <h2>Resultado principal</h2>
      <div class="metric-grid">
        <div class="metric"><span>Classe prevista</span><strong>{escape(str(prediction.get("displayName", "-")))}</strong></div>
        <div class="metric"><span>Confiança</span><strong>{_percent(prediction.get("confidencePercent"))}</strong></div>
        <div class="metric"><span>Prioridade</span><strong>{escape(str(priority.get("label", "-")))}</strong></div>
      </div>
    </section>

    <section class="grid">
      <div class="card">
        <h2>Imagem enviada</h2>
        {"<img alt='Imagem MRI enviada' src='" + original_uri + "' />" if original_uri else "<p>Imagem indisponível.</p>"}
      </div>
      <div class="card">
        <h2>Grad-CAM</h2>
        {"<img alt='Mapa Grad-CAM' src='" + gradcam_uri + "' />" if gradcam_uri else "<p>Mapa Grad-CAM indisponível.</p>"}
        <p class="muted">{escape(str(gradcam.get("description", "Mapa de atenção do modelo; não representa segmentação clínica precisa.")))}</p>
      </div>
    </section>

    <section class="card">
      <h2>Interpretação do resultado</h2>
      <div class="metric-grid">
        <div class="metric"><span>Tumor agrupado</span><strong>{_percent(tumor_summary.get("tumorPercent"))}</strong></div>
        <div class="metric"><span>Sem tumor aparente</span><strong>{_percent(tumor_summary.get("noTumorPercent"))}</strong></div>
        <div class="metric"><span>Margem 1ª vs 2ª</span><strong>{_percent(margin.get("marginPercent"))}</strong></div>
      </div>
      <p class="muted" style="margin-top:10px;">{escape(str(margin.get("description", "")))}</p>
    </section>

    <section class="card">
      <h2>Probabilidades por classe</h2>
      <table>
        <thead><tr><th>Classe</th><th>Probabilidade</th></tr></thead>
        <tbody>
          {_probability_rows(response.get("probabilities", []))}
        </tbody>
      </table>
    </section>

    <section class="card">
      <h2>Triagem complementar</h2>
      <p><strong>Sintomas:</strong> {escape(str(symptoms.get("summary", "-")))}</p>
      <p><strong>Exames:</strong> {escape(str(labs.get("summary", "-")))}</p>
    </section>

    <section class="card">
      <h2>Motivos e observações</h2>
      <div class="warning">{escape(str(priority.get("warning", "-")))}</div>
      <h3 style="margin-top:12px;">Motivos</h3>
      <ul>{_list_items(priority.get("reasons", []))}</ul>
    </section>

    <section class="card">
      <h2>Aviso</h2>
      <p>Este sistema não realiza diagnóstico médico. Use o resultado como apoio para revisão profissional.</p>
    </section>
  </main>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path
