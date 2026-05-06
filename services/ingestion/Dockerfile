# Build context: dss/ root
# docker build -f services/ingestion/Dockerfile .

FROM python:3.12-slim

# System deps for httpx / motor (no native extensions needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install shared package first (layer cached until shared/ changes)
COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared/

# Install service package
COPY services/ingestion/ ./services/ingestion/
RUN pip install --no-cache-dir ./services/ingestion/

# Copy runtime config (non-secret; secrets come from .env / environment)
COPY config/config.yaml ./config/config.yaml

# Copy utility scripts (index creation, client registration, etc.)
COPY scripts/ ./scripts/

ENV DSS_CONFIG_PATH=/app/config/config.yaml \
    DSS_ENV=docker \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "services.ingestion.main"]
