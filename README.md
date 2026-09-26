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

### Rolling out to an existing deployment

The first-time steps above assume an empty MongoDB volume. A VM that is
already running an older revision (unauthenticated Mongo, root containers,
per-container `DSS_MONGO_URI`) needs the migration below **before**
`docker compose up -d` with the current compose file. Skipping it locks every
service out of the database.

Why: the `mongo` image adds `--auth` whenever `MONGO_INITDB_ROOT_*` are set,
but it only *creates* that root user when `/data/db` is empty. On a populated
volume Mongo comes up requiring auth with no users defined.

**Pre-flight**

- [ ] `git fetch && git checkout deploy` on the VM (or the tag being rolled out).
- [ ] Backup: `docker compose exec mongo mongodump --db dss --archive --gzip > ~/backups/dss_$(date -u +%FT%TZ).archive.gz`
      and confirm with `mongorestore --archive --gzip --dryRun < file`.
- [ ] Free disk on the Docker root (`df -h /`): the `ml_engine` image is
      ~2.7 GB and any change under `shared/` rebuilds its dependency layer.
      Need ≥ 4 GB free, or move the Docker data root first (see below).
- [ ] Decide `DSS_ENV`. `server` enables the startup validators, which reject
      the `.staging.` WINGS URLs. If staging is the only WINGS environment
      available to this deployment, keep `DSS_ENV=local` and note it here;
      the validators are skipped and nothing else changes.

**1. Create the Mongo root user while auth is still off** (localhost
exception; safe to run before anything else, has no effect until `--auth`):

```bash
MONGO_PW=$(openssl rand -hex 24)   # hex: no URI-reserved characters
docker compose exec mongo mongosh --quiet admin --eval \
  "db.createUser({user:'dss_root', pwd:'$MONGO_PW', roles:[{role:'root', db:'admin'}]})"
```

**2. Add the new required variables to `.env`** (they are ignored by the old
compose file, so this is also safe ahead of time):

```bash
cat >> .env <<EOF
MONGO_INITDB_ROOT_USERNAME=dss_root
MONGO_INITDB_ROOT_PASSWORD=$MONGO_PW
DSS_MONGO_URI=mongodb://dss_root:$MONGO_PW@mongo:27017/?authSource=admin
EOF
```

**3. Rebuild and switch** (short outage: Mongo restarts with `--auth`, the
services are recreated with the new URI, limits and log rotation):

```bash
docker compose build ingestion api_service      # ml_engine too if disk allows
docker compose up -d
```

**4. Fix volume ownership for the non-root user.** Images now run as `dss`.
Volumes created by earlier root containers are root-owned, so the first write
fails. Only `model_artifacts` is written by the services (by `ml_engine`).
Run this the first time `ml_engine` starts from a non-root image:

```bash
docker compose run --rm --user root --entrypoint sh ml_engine \
  -c 'chown -R dss:dss /app/artifacts'
```

**5. Verify**

```bash
docker compose ps                                   # all Up, mongo + api healthy
curl -s localhost:8000/health                       # {"status":"ok",...,"mongo":"ok"}
docker compose logs --tail 20 ingestion | grep -E 'db_bootstrap_complete|scheduler_started'
docker compose exec mongo mongosh --quiet --eval 'db.adminCommand({listDatabases:1})' \
  ; echo "expect: Unauthorized"                     # auth is enforced
ss -tln | grep 27017 ; echo "expect: no output"     # host port is gone
```

**Rollback**: `git checkout <previous-rev> && docker compose up -d`. The root
user in Mongo and the extra `.env` lines are inert under the old compose
file, so nothing needs to be undone in the database.

**Moving the Docker data root to a bigger disk** (when `/` is small):

```bash
sudo systemctl stop docker docker.socket
sudo rsync -aHAX /var/lib/docker/ /home/docker/
sudo tee /etc/docker/daemon.json <<'EOF'
{ "data-root": "/home/docker" }
EOF
sudo systemctl start docker && docker compose up -d
# once verified: sudo rm -rf /var/lib/docker
```

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
