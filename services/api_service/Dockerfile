# Build context: dss/ root
# docker build -f services/api_service/Dockerfile .

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install shared package
COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared/

# Install service package (fastapi, uvicorn, jose, passlib)
COPY services/api_service/ ./services/api_service/
RUN pip install --no-cache-dir ./services/api_service/

# Copy runtime config
COPY config/config.yaml ./config/config.yaml

# Copy utility scripts
COPY scripts/ ./scripts/

ENV DSS_CONFIG_PATH=/app/config/config.yaml \
    DSS_ENV=docker \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Run as non-root
RUN groupadd -r dss && useradd -r -g dss -d /app -s /sbin/nologin dss \
 && chown -R dss:dss /app
USER dss

CMD ["python", "-m", "services.api_service.main"]
