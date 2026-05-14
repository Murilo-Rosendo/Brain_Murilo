FROM python:3.12-slim

# Dependências de sistema para opencv e outros
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o projeto
COPY . .

# Instala o pacote local
RUN pip install --no-cache-dir -e .

# Hugging Face Spaces usa porta 7860
EXPOSE 7860

CMD ["uvicorn", "brain_mri_triage.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "7860"]
