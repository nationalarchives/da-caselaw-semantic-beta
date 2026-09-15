FROM python:3.13-slim@sha256:9d2e5553305c7c7b0097999bb17187c69b921ccd6bc9d40e4bb5ebe652c00285

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BASE_URL_PATH=beta

WORKDIR /app

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup appuser && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl=8.* \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser:appgroup

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 \
  CMD curl -fsS http://localhost:8501/${BASE_URL_PATH}/_stcore/health || exit 1

CMD ["sh", "-c", "streamlit run beta_app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --server.baseUrlPath=${BASE_URL_PATH}"]

