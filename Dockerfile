FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by some pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (layer caching)
COPY environment.yml .

# Install pip dependencies directly from environment.yml
# (conda not used in Docker — pip install is faster and lighter)
RUN pip install --no-cache-dir \
    fastapi==0.111.0 \
    "uvicorn[standard]==0.29.0" \
    pydantic==2.7.1 \
    requests==2.31.0 \
    httpx==0.27.0 \
    "duckduckgo-search==5.3.1b1" \
    wikipedia-api==0.6.0 \
    beautifulsoup4==4.12.3 \
    chromadb==0.5.0 \
    sentence-transformers==2.7.0 \
    aiosqlite==0.20.0 \
    prometheus-client==0.20.0 \
    pyyaml==6.0.1 \
    rich==13.7.1 \
    python-multipart==0.0.9 \
    tenacity==8.3.0 \
    "numpy<2.0"

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p logs chroma_data db data/files

EXPOSE 8000

# num_workers=1 — Ollama is single-threaded anyway
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]