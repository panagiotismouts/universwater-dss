# Build context: dss/ root
# docker build -f services/ml_engine/Dockerfile .

FROM python:3.12-slim

# Build deps for numpy / scikit-learn / xgboost native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install shared package
COPY shared/ ./shared/
RUN pip install --no-cache-dir ./shared/

# Install service package (heavy: sklearn, xgboost, shap, numpy)
COPY services/ml_engine/ ./services/ml_engine/
RUN pip install --no-cache-dir ./services/ml_engine/

# Copy runtime config
COPY config/config.yaml ./config/config.yaml

ENV DSS_CONFIG_PATH=/app/config/config.yaml \
    DSS_ENV=docker \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

CMD ["python", "-m", "services.ml_engine.main"]
