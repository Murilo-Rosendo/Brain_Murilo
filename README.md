# Brain MRI Triage - Render

Versao enxuta do projeto para deploy no Render.

## Render

Build command:

```bash
pip install -r requirements.txt && pip install -e .
```

Start command:

```bash
uvicorn brain_mri_triage.app:create_app --factory --host 0.0.0.0 --port $PORT
```

Variavel de ambiente recomendada:

```text
PYTHON_VERSION=3.12.10
```

## Arquivos incluidos

- `src/brain_mri_triage/`: codigo do site e inferencia.
- `artifacts/modelo_mendeley_multiclasse.keras`: modelo treinado usado pelo site.
- `artifacts/class_names.json`: nomes das classes do modelo.
- `artifacts/training_history.csv`: historico de treino para os graficos.
- `reports/classification_report.txt`: metricas por classe.
- `reports/confusion_matrix.csv`: matriz de confusao da avaliacao.

## Nao incluidos

- Ambiente virtual.
- Dataset.
- Logs.
- Relatorios e Grad-CAMs gerados em uso local.
- Historico antigo do dashboard.
