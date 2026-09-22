# UNIVERSWATER — Decision Support System (DSS)

A continuous, automated pipeline for environmental sensor data acquisition, preprocessing, machine learning-based prediction, and explainability (XAI) for water and soil quality monitoring.

> Full technical description: [DSS Data Flow Description](docs/DSS_Data_Flow_Description.md)

---

## Overview

The DSS ingests real-time observations from heterogeneous sensor networks, transforms them into structured feature representations, trains and recalibrates supervised regression models, and delivers predictions together with SHAP-based explanations through a REST API.

### Data Sources

| Station | Interface | Domain | Variables |
|---|---|---|---|
| Aquaread Water Quality Sensor (252186) | WINGS SensorThings API | Water | pH, Dissolved Oxygen, Turbidity, Conductivity, Temperature, ORP, TDS, Salinity, Ammonia, Nitrate, Chlorophyll-a, Ammonium, Blue-Green Algae, CDOM, DO Saturation, Sigma-t |
| HCMR Water Quality Monitoring Station | WINGS SensorThings API | Water | pH, Water Temperature, Conductivity, Dissolved Oxygen, ORP |
| Soil Sensor Station | WINGS SensorThings API | Soil | Soil Temperature, N, P, K, Soil Conductivity |
| Universwater Meteorological Station | Local REST API (UOWM) | Met | Air Temperature, Precipitation, Humidity, Wind Speed, Wind Direction, Solar Radiation, Infrared Temperature |

### Prediction Targets

| Variable | Domain | Unit |
|---|---|---|
| Dissolved Oxygen | Water | mg/L |
| Chlorophyll-a | Water | μg/L |
| Turbidity | Water | NTU |
| pH | Water | pH units |
| Conductivity | Water | μS/cm |
| Volumetric Water Content | Soil | m³/m³ |

### Machine Learning Models

| Category | Models |
|---|---|
| White-box | Linear Regression, Elastic Net, Decision Tree |
| Black-box | Random Forest, XGBoost, LightGBM, CatBoost, SVR |
| AutoML (soil only) | FLAML |

Explainability is provided by **SHAP** (TreeExplainer for tree-based models, LinearExplainer for linear models), delivered per-prediction via the API.

---

## Services

| Service | Folder | Responsibility |
|---|---|---|
| `ingestion` | `services/ingestion/` | API polling → normalization → preprocessing → feature engineering → persistence |
| `ml_engine` | `services/ml_engine/` | Bootstrap training, recalibration, prediction cycles, XAI |
| `api_service` | `services/api_service/` | External API: token issuance, result serving |
| `mongo` | Docker only | MongoDB persistence layer |

## Shared Library

`shared/dss_shared/` — installable Python package shared by all services.

```bash
pip install -e ./shared
```

## Quick Start

```bash
cp config/.env.example .env
# edit .env with your secrets and connection strings
docker compose up
```

## Repository Structure

```
dss/
├── shared/          # dss_shared installable package
├── services/
│   ├── ingestion/   # data-worker: clients, normalizer, preprocessor, feature engineering
│   ├── ml_engine/   # ml-worker: training, recalibration, prediction, XAI
│   └── api_service/ # REST API: auth, prediction delivery, delivery logging
├── config/          # config.yaml + .env.example
├── scripts/         # operational CLI utilities (bootstrap, discovery, healthcheck)
├── docs/            # technical documentation
└── tests/           # unit + integration tests
```

## Documentation

- [DSS Data Flow Description](docs/DSS_Data_Flow_Description.md) — end-to-end technical description of the pipeline: data acquisition, normalization, preprocessing, feature engineering, model training, XAI, and API delivery.

## Deployment

The DSS runs as four Docker containers (three services + MongoDB). A typical
deployment uses ssh port-forwarding from each team member's workstation to the
VM; the VM should not expose port 8000 to the open internet.

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- 16 GB RAM minimum on the host (12 GB allocated to containers, 4 GB for the host)
- WINGS OAuth2 `client_id` and `client_secret` (obtain from project coordinator)
- SSH access to the VM for each team member who will use the API

### First-time setup

1. Clone the repository on the VM.
2. Copy `config/.env.example` to `.env` (in the repo root, NOT inside `config/`)
   and fill in:
   - `DSS_ENV=server`
   - `MONGO_INITDB_ROOT_USERNAME` / `MONGO_INITDB_ROOT_PASSWORD` — MongoDB admin
     credentials (operator-chosen; must match the credentials embedded in
     `DSS_MONGO_URI`)
   - `DSS_MONGO_URI=mongodb://<root>:<password>@mongo:27017/?authSource=admin`
   - `DSS_WINGS_CLIENT_ID` / `DSS_WINGS_CLIENT_SECRET`
   - `DSS_WINGS_API_BASE_URL` / `DSS_WINGS_SSO_URL` — production URLs (the
     default `.staging.` URLs are rejected at startup when `DSS_ENV=server`)
   - `DSS_JWT_SECRET_KEY` — generate with `openssl rand -base64 48`
   - `DSS_ADMIN_API_KEY` — generate with `openssl rand -base64 48`
3. Set `DSS_BOOTSTRAP_MODE=true` for the first ingestion cycle only.
4. Run `docker compose up -d`.
5. Watch the logs:
   ```bash
   docker compose logs -f ingestion
   docker compose logs -f ml_engine
   ```
   Wait until `ml_engine` logs `bootstrap_best_model_activated` for each enabled
   pipeline (this can take many minutes the first time).
6. Set `DSS_BOOTSTRAP_MODE=false` in `.env` and `docker compose restart ingestion`.

### Team access (ssh tunnel)

Each team member forwards port 8000 over ssh from their own workstation:

```bash
ssh -L 8000:localhost:8000 <vm-hostname>
```

Then open `http://localhost:8000/health` in a browser or use any HTTP client
against `http://localhost:8000`. Each member must register their own
`client_id` via `python scripts/register_client.py` and obtain a bearer token
via `POST /auth/token`.

For a future browser dashboard, a reverse proxy with TLS termination is
recommended over exposing port 8000 directly.

### Backup and recovery

```bash
# MongoDB (run from repo root on the VM)
docker compose exec mongo mongodump \
  --uri="$DSS_MONGO_URI" \
  --out=/backup/$(date +%F)

# Model artifacts
docker compose cp ml_engine:/app/artifacts ./artifacts-$(date +%F)
```

Restore by mounting the backup into a fresh `mongo_data` volume and dropping
the artifact directory into the `model_artifacts` volume. There is no
built-in automation; run the commands above on a schedule (cron / systemd
timer) to keep backups current.

### Security notes

- **WINGS `client_secret` history**: a WINGS OAuth client secret was previously
  committed to the repository (commit `e03524d`). It cannot be rotated by the
  maintainers of this repo (no admin access to the WINGS SSO). Treat the
  value as permanently public; rely on the WINGS side for rate limiting and
  on ssh-port-forwarding for access control to this deployment.
- **All secrets are loaded from `.env`** (excluded from version control by
  `.gitignore`). Never commit secrets to the repository or to issue trackers.
- **MongoDB is not exposed on the host.** Only the three application services
  can reach it on the internal docker network. For MongoDB Compass access,
  run a one-off container on the same network rather than opening a host port.
- **Port 8000 is exposed for ssh-tunnel access.** The compose file does not
  rely on this port being unreachable from outside the VM. The operator is
  responsible for keeping it behind a firewall or VPN.
