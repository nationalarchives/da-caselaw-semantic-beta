FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_URL_PATH=beta \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -m -d /home/appuser appuser && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl=8.* \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Fetch the sentence-transformer model (~90MB) as root, so the non-root
# runtime user never needs network access to get it. `hf download` just
# fetches the repo's files — no torch/tokenizer construction needed here.
RUN hf download sentence-transformers/all-MiniLM-L6-v2

# Only go offline once the model is already cached above.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Load the model with the offline flags active, so the build fails now if
# the cache above is ever incomplete, rather than surfacing at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

USER appuser:appgroup

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8501/${BASE_URL_PATH}/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run beta_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.baseUrlPath=${BASE_URL_PATH}"]

