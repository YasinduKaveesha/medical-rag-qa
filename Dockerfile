FROM python:3.11-slim

WORKDIR /app

# Build tools required by some wheels (sentence-transformers, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first — separate layer so code changes don't retrigger pip
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application source
COPY src/ src/
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
