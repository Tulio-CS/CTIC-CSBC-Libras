# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Imagem de produção para o servidor LIBRAS (FastAPI + uvicorn).
#
# Build local (teste):
#   docker build -t libras-web .
#   docker run -p 8000:8000 libras-web
#
# No Coolify: aponta para este Dockerfile, porta 8000.
# Para trocar o modelo: variável de ambiente LIBRAS_MODEL=bilstm_attn_combined
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

# ── Modelos de produção ───────────────────────────────────────────────────────
# Os modelos de produção estão versionados no git (model.keras ≤ 31 MB cada).
# Modelos de ablação (models/T*/, models/attn_*/, etc.) ficam fora via .dockerignore.
COPY models/bilstm_attn_best/     ./models/bilstm_attn_best/
COPY models/bilstm_attn_combined/ ./models/bilstm_attn_combined/
COPY models/bilstm_attn_minds/    ./models/bilstm_attn_minds/
COPY models/bilstm_attn/          ./models/bilstm_attn/
COPY models/lstm/                 ./models/lstm/

# Modelo padrão — pode ser sobrescrito via variável de ambiente no Coolify:
#   LIBRAS_MODEL=bilstm_attn_combined
ENV LIBRAS_MODEL=bilstm_attn_best

EXPOSE 8000

# Roda a partir de /app/web para que 'from predictor import Predictor' funcione
WORKDIR /app/web

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
