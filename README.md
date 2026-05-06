# DSS — Decision Support System

Environmental data ingestion, preprocessing, model training, prediction, and XAI for water and soil pipelines.

## Services

| Service | Folder | Responsibility |
|---|---|---|
| `ingestion` | `services/ingestion/` | API polling → normalization → preprocessing → feature engineering → persistence |
| `ml_engine` | `services/ml_engine/` | Bootstrap training, recalibration, prediction cycles, XAI |
| `api_service` | `services/api_service/` | External API: token issuance, result serving |
| `mongo` | Docker only | MongoDB persistence layer |

## Shared Library

`shared/dss_shared/` — installable Python package shared by all services.

Install for local development:

```bash
pip install -e ./shared
```

## Quick Start

```bash
cp config/.env.example .env
# edit .env with your secrets and connection strings
docker compose up
```

## Structure

```
dss/
├── shared/          # dss_shared installable package
├── services/
│   ├── ingestion/   # data-worker
│   ├── ml_engine/   # ml-worker
│   └── api_service/ # result API
├── config/          # config.yaml + .env.example
├── scripts/         # operational CLI utilities
└── tests/           # unit + integration tests
```

## Blueprints

Design documents are in `docs/`. The Amendment v1 note supersedes earlier blueprints where they conflict.
