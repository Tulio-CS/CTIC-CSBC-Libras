# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Imagem de produção para o servidor LIBRAS (FastAPI + uvicorn).
#
# Build local (teste):
#   docker build -t libras-web .
#   docker run -p 8000:8000 -v $(pwd)/models:/app/models libras-web
#
# No Coolify: aponta para este Dockerfile, porta 8000, volume /app/models
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Evita prompts interativos durante o apt
ENV DEBIAN_FRONTEND=noninteractive

# Dependências de sistema mínimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependências Python ───────────────────────────────────────────────────────
COPY web/requirements.txt ./web/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r web/requirements.txt

# ── Código do projeto ─────────────────────────────────────────────────────────
# Utilitários compartilhados (importados por web/predictor.py via sys.path)
COPY config.py      ./config.py
COPY data_utils.py  ./data_utils.py

# Servidor web
COPY web/app.py       ./web/app.py
COPY web/predictor.py ./web/predictor.py
COPY web/frontend/    ./web/frontend/

# ── Modelos (montados via volume no Coolify) ──────────────────────────────────
# /app/models/<nome>/model.keras       — NÃO está no git (grande demais)
# /app/models/<nome>/norm_stats.json   — versionado no git
# /app/models/<nome>/actions.npy       — versionado no git
# O volume do Coolify sobrescreve /app/models com os arquivos reais.

ENV LIBRAS_MODEL=bilstm_attn

EXPOSE 8000

# Roda a partir de /app/web para que 'from predictor import Predictor' funcione
WORKDIR /app/web

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
