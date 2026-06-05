# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="HireReady AI"
LABEL description="Multi-agent LangGraph placement evaluation API (FastAPI + Claude)"

# ── System dependencies ───────────────────────────────────────────────────────
# gcc/g++ are needed by some pdfplumber/cryptography native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpoppler-cpp-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Runtime config ────────────────────────────────────────────────────────────
# PORT is injected by Railway at runtime; default to 8000 for local builds
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE ${PORT}

# ── Entrypoint ────────────────────────────────────────────────────────────────
# Run from project root so relative imports (graph/, agents/) resolve correctly
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
