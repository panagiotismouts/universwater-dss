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
