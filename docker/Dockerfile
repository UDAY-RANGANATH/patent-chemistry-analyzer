# Patent Chemistry Analyzer — production-ish container image.
# Build context is the repository root:
#   docker build -f docker/Dockerfile -t patent-chemistry-analyzer .

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: OpenMP runtime (rdkit/numpy). fonts-dejavu-core: RDKit PNG structure
# rendering. tesseract-ocr: optional OCR fallback for scanned patents.
# curl: container healthchecks.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fonts-dejavu-core \
        tesseract-ocr \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps first (better layer caching).
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --retries 5 --timeout 120 -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# Secrets are injected at runtime via environment (compose env_file), never baked in.
CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
