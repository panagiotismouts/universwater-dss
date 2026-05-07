# Data Acquisition, Processing, and Prediction Delivery in the UNIVERSWATER Decision Support System

## 1. Introduction

The predictive component of the UNIVERSWATER Decision Support System (DSS) implements a continuous, automated pipeline that encompasses the acquisition of heterogeneous environmental sensor data, its transformation into structured feature representations, the training and operational deployment of supervised machine learning models, and the delivery of predictions together with their interpretability artifacts through a standardized application programming interface. The pipeline is designed to operate without manual intervention, maintaining temporal continuity through watermark-based incremental ingestion, and ensuring prediction quality through a governed model lifecycle that includes scheduled recalibration and metric-gated activation. The following sections describe each stage of this pipeline in technical detail.

---

## 2. Data Sources and Acquisition

### 2.1 Sensor Infrastructure

Environmental observations are collected from four distinct physical monitoring installations, accessed through two separate remote interfaces. Table 1 summarizes the sensor inventory, the variables acquired from each station, the observed refresh rates, and the temporal coverage available at the time of system development.

**Table 1. Sensor inventory and observation characteristics.**

| Station | Interface | Variable | Unit | Records | Median Refresh Rate | Coverage |
|---|---|---|---|---|---|---|
| Aquaread Water Quality Sensor (252186) | WINGS SensorThings API | pH, Dissolved Oxygen, Dissolved Oxygen Saturation, Water Temperature, Conductivity, Total Dissolved Solids, Salinity, ORP (RedOx), Ammonia, Nitrate, Turbidity, Chlorophyll-a, Ammonium, Blue-Green Algae (Phycocyanin), CDOM, Sigma-t | various | 2,034 per variable | 2 hours | Sep 2025 – Mar 2026 |
| Aquaread Water Quality Sensor (252251) | WINGS SensorThings API | same 16 variables as above | various | 1,523 per variable | 5 minutes | Dec 2025 – Mar 2026 |
| HCMR Water Quality Monitoring Station | WINGS SensorThings API | pH, Water Temperature, Conductivity, Dissolved Oxygen, ORP | various | 1,303 per variable | 1 hour | Jan 2026 – Mar 2026 |
| Soil Sensor Station | WINGS SensorThings API | Volumetric Water Content, Soil Temperature, Nitrogen (N), Phosphorus (P), Potassium (K), Soil Conductivity | various | 1,030 per variable | 4 hours | Sep 2025 – Mar 2026 |
| Universwater Meteorological Station | Universwater Local REST API | Air Temperature, Precipitation, Relative Humidity, Wind Speed, Wind Direction, Solar Radiation (Pyranometer), Infrared Surface Temperature | various | 7,660 per variable | 30 minutes | Oct 2025 – Mar 2026 |

It is noted that the Universwater meteorological station is also registered as a data source within the WINGS platform; however, this stream was excluded from the operational pipeline due to observed data staleness, with the last available WINGS observation recorded in January 2026. The meteorological variables are therefore acquired exclusively through the station's dedicated local REST interface.

### 2.2 WINGS SensorThings Application Programming Interface

The WINGS platform exposes sensor observations through an implementation of the OGC SensorThings Application Programming Interface, a specification defined by the Open Geospatial Consortium for the interoperable access to Internet-of-Things sensor data. Authentication is performed via the OAuth 2.0 client credentials authorization grant: the system exchanges a `client_id` and `client_secret` against a Security Service (SSO) token endpoint, receiving a short-lived Bearer token in JSON Web Token (JWT) format. Token lifecycle management is handled by a dedicated in-process component that caches the active token, monitors its scheduled expiration with a 60-second safety margin to prevent clock-edge invalidations, and performs transparent re-authentication in the event of token revocation or the reception of an HTTP 401 Unauthorized response.

Observations are retrieved from the `/api/v1/collections/sensor_things:observations` endpoint through HTTP GET requests. Each request includes a structured JSON filter object specifying the target datastream identifier and a lower-bound temporal constraint corresponding to the timestamp of the last successfully ingested observation — a mechanism referred to throughout the system as the **watermark cursor**. This incremental retrieval strategy ensures that each polling cycle fetches only new observations, avoiding redundant data transfer. Pagination is implemented using the `limit`/`offset` query parameters, with each page containing up to 1,000 records; iteration continues until a partial page signals the exhaustion of available data. Each observation record exposes two fields: `phenomenon_time_start`, representing the UTC-normalized ISO 8601 observation timestamp, and `result_number`, carrying the floating-point measurement value.

### 2.3 Universwater Local REST Interface

The local meteorological station interface is a proprietary REST (Representational State Transfer) service that requires no authentication. Observations are retrieved via HTTP GET requests to the endpoint `/sensors/{sensor_id}/data`, parameterized with Unix epoch timestamps defining the start and end of the requested temporal window and a maximum record count of 10,000 per request. Pagination is cursor-based: upon receipt of each batch, the start timestamp is advanced to the maximum observed timestamp in the batch incremented by one second, and the request is repeated until an empty response is returned. Each record carries a Unix epoch integer timestamp and a floating-point measurement value.

All network communication throughout the acquisition layer is performed asynchronously using the `httpx` library operating under the Python `asyncio` concurrency model. Each (source, variable) pair is assigned an independent scheduled polling job managed by the APScheduler library, ensuring that latency or failure in the retrieval of one variable does not block the ingestion of others.

---

## 3. Data Normalization and Preprocessing

### 3.1 Source Normalization

Upon retrieval, raw API responses are immediately processed by a normalization component that constitutes the sole location in the system where source-specific field names, timestamp formats, and unit conventions are resolved. This design ensures that all subsequent processing stages operate exclusively on a canonical, source-agnostic internal schema. Each normalized observation record carries: the pipeline assignment (water quality, soil quality, or meteorological), the source system identifier, the sensor identifier, the canonical variable name, the measurement value, the physical unit, the UTC-normalized observation timestamp, and the timestamp of data retrieval.

Meteorological observations are duplicated at this stage: each record acquired from the meteorological station produces two internal records — one assigned to the water quality pipeline and one to the soil quality pipeline. This duplication ensures that meteorological covariates, which are relevant to both domains, are available to each machine learning pipeline independently without requiring cross-pipeline data joins at training time.

### 3.2 Preprocessing Pipeline

Following normalization, observations pass through a multi-stage preprocessing pipeline before being persisted in the database. In the first stage, values outside physically plausible bounds — defined per variable in a centralized variable registry — are flagged with a quality indicator of "suspect" and excluded from downstream feature computation. In the second stage, statistical outliers are identified and removed using the interquartile range (IQR) method, applied independently per variable: observations falling below Q1 − 1.5 × IQR or above Q3 + 1.5 × IQR are treated as anomalous. In the third stage, duplicate observations, identified by coincident (sensor identifier, variable name, observation timestamp) tuples, are removed. In the fourth stage, observations from sensors operating at differing sampling frequencies are aligned to a regular temporal grid through resampling and interpolation, resolving the misalignment inherent in the heterogeneous sensor network. In the fifth stage, short temporal gaps are filled through forward-fill imputation using the most recently valid observation; gaps exceeding a per-variable maximum fill duration are left unfilled to avoid introducing spurious temporal structure. Imputed records are tagged accordingly to allow downstream components to assess data completeness. Processed records are finally persisted in the database through upsert operations keyed on the compound tuple (pipeline, sensor identifier, variable name, observation timestamp), guaranteeing idempotency under repeated execution.

---

## 4. Feature Engineering

Preprocessed observations are transformed into fixed-length numerical feature vectors suitable for supervised machine learning through a feature engineering layer. The water quality feature schema constructs features from a six-hour retrospective sliding window relative to each prediction timestamp. Rolling mean and standard deviation statistics are computed over one-hour and three-hour sub-windows for the primary water quality variables. Lag features — the values of the most recent one or two valid observations — are extracted for each variable to capture short-term temporal autocorrelation. Rainfall accumulation sums are computed over one-hour and three-hour windows as meteorological covariates. For the oxidation-reduction potential (ORP) variable, which is available from both the Aquaread stations and the HCMR station, a one-hour rolling mean and a first-order lag feature are included. Additionally, composite water quality indices — including the Trophic State Index (TSI) and the Water Quality Index (WQI) — are computed as engineered features from combinations of measured variables. Temporal context is encoded through the sine and cosine transformations of hour-of-day, day-of-week, and month-of-year, following the standard circular encoding convention for cyclical time variables, which avoids the ordinal discontinuities introduced by raw integer representations.

Each computed feature vector is persisted as a document in the feature store collection of the database, indexed by pipeline, sensor identifier, feature timestamp, and feature schema version. The schema version field enables non-destructive evolution of the feature set: new schema versions coexist with historical vectors, and each model is trained and evaluated exclusively against the schema version present at its training time.

Feature dimensionality is subsequently reduced through Recursive Feature Elimination (RFE) using cross-validated performance as the selection criterion, followed by a confirmatory ranking pass using SHAP-based feature importance, as described in Section 6. This two-stage selection process reduces collinearity and improves model generalization.

---

## 5. Target Variables and Model Training

### 5.1 Target Variables

The supervised learning formulation defines prediction targets at the level of individual variables within each pipeline. For the water quality domain, the primary target variables are dissolved oxygen (mg/L), chlorophyll-a (μg/L), turbidity (NTU), pH, and conductivity (μS/cm), all sourced from the Aquaread sensor network. For the soil quality domain, the primary target is volumetric water content (m³/m³). The macronutrient variables — nitrogen, phosphorus, and potassium — are acquired and stored but are excluded from the primary predictive models in the current implementation due to approximately 55% missing observations attributable to documented sensor malfunction; they are retained for future model inclusion as data quality improves.

**Table 2. Prediction target variables.**

| Variable | Domain | Unit | Source Station | Notes |
|---|---|---|---|---|
| Dissolved Oxygen | Water | mg/L | Aquaread (252186, 252251) | Primary aquatic ecosystem health indicator |
| Chlorophyll-a | Water | μg/L | Aquaread (252186, 252251) | Phytoplankton / algal bloom indicator |
| Turbidity | Water | NTU | Aquaread (252186, 252251) | Water clarity and suspended solids proxy |
| pH | Water | pH units | Aquaread (252186, 252251) | Acidity-alkalinity balance |
| Conductivity | Water | μS/cm | Aquaread (252186, 252251) | Ionic concentration proxy |
| Volumetric Water Content | Soil | m³/m³ | Soil Sensor Station | Key crop health and irrigation indicator |
| Nitrogen, Phosphorus, Potassium | Soil | mg/kg | Soil Sensor Station | Deferred — 55% missing data |

### 5.2 Machine Learning Models

A diverse ensemble of supervised regression algorithms is evaluated for each target variable, spanning both interpretable white-box and high-capacity black-box model families, as detailed in Table 3. This multi-model strategy enables a systematic comparison of predictive performance against interpretability, a trade-off that is central to the explainability objectives of the DSS.

**Table 3. Machine learning models evaluated per domain.**

| Model | Category | Algorithm Family | Library |
|---|---|---|---|
| Linear Regression | White-box | Ordinary Least Squares | scikit-learn |
| Elastic Net | White-box | Regularized Linear (L1 + L2) | scikit-learn |
| Decision Tree | White-box | Recursive Binary Partitioning | scikit-learn |
| Random Forest | Black-box | Bagged Decision Tree Ensemble | scikit-learn |
| XGBoost | Black-box | Gradient Boosted Tree Ensemble | xgboost |
| LightGBM | Black-box | Histogram-based Gradient Boosting | lightgbm |
| CatBoost | Black-box | Ordered Gradient Boosting | catboost |
| Support Vector Regression | Black-box | Kernel-based Margin Regression | scikit-learn |
| FLAML (AutoML) | AutoML | Automated Model Selection | flaml |

For the soil quality pipeline, automated machine learning via the FLAML framework is employed as the primary training strategy, given the high degree of missingness in the soil sensor data, which complicates manual model selection and hyperparameter tuning.

### 5.3 Training Protocol and Model Lifecycle

On initial deployment, the system detects the absence of any active model for each pipeline and initiates a bootstrap training procedure. All available feature documents from the configured historical start date are loaded from the feature store, and an 80/20 chronological train-validation split is applied. Chronological splitting is used in preference to random splitting to prevent the information leakage from future observations into the training set that would otherwise inflate performance estimates.

Each candidate model is evaluated against a metric gate before activation. The gate requires the coefficient of determination (R²) to exceed a minimum threshold and the mean absolute error (MAE) — expressed as a proportion of the target variable's observed range — to remain below a maximum threshold. Models satisfying both conditions are designated as active and recorded in the model registry; those failing are rejected and logged. The target performance criterion across all models is R² > 0.85, selected to ensure predictions of sufficient quality for operational decision support.

A scheduled recalibration procedure executes on a weekly cron schedule. It retrains all model types using either an expanding window from the historical start date or a rolling window of a configurable number of most recent days, depending on configuration. Each newly trained candidate is compared against both the metric gate and the performance of the currently active model; a candidate that outperforms the incumbent on the validation set is activated, otherwise the incumbent is retained. This governed recalibration mechanism ensures that model quality can only improve over the operational lifetime of the system.

---

## 6. Explainability via SHAP

The interpretability layer of the DSS is built on the SHAP (SHapley Additive exPlanations) framework, a game-theoretic approach to feature attribution that assigns each input feature a contribution value corresponding to its marginal impact on the model's output relative to the expected prediction over the training distribution. SHAP values satisfy the desirable axiomatic properties of local accuracy, consistency, and missingness, making them a rigorous basis for both local (per-prediction) and global (model-level) explanation.

SHAP is employed at two distinct stages of the pipeline. During the preprocessing phase, a SHAP-based feature importance ranking is used to validate and cross-check the results of the Recursive Feature Elimination step, ensuring that the features retained for training are those with genuine predictive contribution rather than those that merely survive the RFE criterion by chance. During the operational prediction phase, a SHAP explainer is applied at each prediction cycle to attribute the contribution of every input feature to the individual prediction.

The choice of SHAP explainer variant is conditioned on the model type. For tree-based ensemble models (Random Forest, XGBoost, LightGBM, CatBoost), the TreeExplainer algorithm is used, which exploits the recursive structure of decision trees to compute exact Shapley values in polynomial time without sampling approximations. For linear models (Linear Regression, Elastic Net), the LinearExplainer is applied with interventional feature perturbation, computing exact linear Shapley values. The resulting explanation artifact for each prediction includes the signed SHAP value for every feature, the feature's actual input value, and the base value (the model's expected output over the background distribution). These artifacts are persisted in the database and exposed through the API alongside the prediction itself.

The SHAP output is visualized through summary plots (global feature importance via beeswarm and bar representations), waterfall plots (per-prediction decomposition), and dependence plots (feature-level non-linear relationship and interaction analysis), providing decision-makers with a multi-scale view of model behavior.

---

## 7. Model Evaluation

Model performance is assessed using four quantitative metrics: the coefficient of determination (R²), the Mean Absolute Error (MAE), the Root Mean Squared Error (RMSE), and a training stability score measuring the consistency of R² across cross-validation folds. Table 4 summarizes the evaluation framework.

**Table 4. Evaluation metrics.**

| Metric | Full Name | Preferred Direction | Description |
|---|---|---|---|
| R² | Coefficient of Determination | Higher (target > 0.85) | Proportion of variance in the target explained by the model |
| MAE | Mean Absolute Error | Lower | Mean absolute deviation of predictions from observations; same unit as target |
| RMSE | Root Mean Squared Error | Lower | Square root of mean squared prediction error; penalizes large deviations more heavily than MAE |
| Stability Score | Cross-Validation Stability | Higher | Consistency of R² across cross-validation folds; high values indicate robustness to data partitioning |

Cross-validation convergence is additionally monitored as a qualitative diagnostic, tracking R² across successive folds to detect trends indicative of overfitting or underfitting.

---

## 8. Persistence Architecture

All intermediate and final data products are stored in a MongoDB document-oriented database management system, accessed through the Motor asynchronous driver. The database comprises nine logically distinct collections: raw ingested measurements, preprocessed measurements, feature store documents, model registry entries, model evaluation metrics, predictions, XAI result artifacts, API client credentials, and an API delivery log. Compound indexes are defined on all collections against the field combinations expected by the most frequent query patterns, ensuring sub-millisecond retrieval latency as collection sizes grow. Each collection enforces a document schema validated through Pydantic version 2 at the application boundary.

Serialized model artifacts — the fitted estimator objects produced by scikit-learn, XGBoost, LightGBM, or other training libraries — are stored on the local filesystem using the `joblib` serialization library, with the artifact file path recorded in the corresponding model registry document. This separation of structured metadata from binary artifacts follows established MLOps conventions and avoids placing large binary objects in the database.

---

## 9. Prediction Delivery

### 9.1 API Service Architecture

Predictions, together with their associated explainability artifacts, are made available to external consumers through a RESTful application programming interface implemented using FastAPI, an asynchronous Python web framework built on the ASGI (Asynchronous Server Gateway Interface) standard, served by the Uvicorn server. The API service is deployed as an independent containerized process and operates without dependency on the operational state of the ingestion or machine learning engine services, returning structured error responses if no predictions are yet available.

### 9.2 Authentication and Authorization

All prediction endpoints are protected by Bearer token authentication based on the JSON Web Token standard (RFC 7519), with tokens signed using the HMAC-SHA256 (HS256) algorithm. A client system wishing to access the API first submits its `client_id` and `client_secret` as an `application/x-www-form-urlencoded` POST request to the `/auth/token` endpoint. The server verifies the submitted secret against a bcrypt-hashed credential stored in the database, issues a signed JWT with a configurable time-to-live (default: 24 hours), and records a SHA-256 digest of the token's `jti` (JWT Identifier) claim to support token revocation without storing the token itself. Administrative endpoints exposing model registry and system health information are protected by a separate API key scheme conveyed in the `X-Admin-Key` request header.

### 9.3 Endpoints and Response Structure

The API exposes three operationally relevant endpoint groups. The authentication endpoint (`POST /auth/token`) issues Bearer tokens in exchange for valid client credentials. The latest results endpoint (`GET /results/{pipeline}/latest`) returns the most recent prediction for each active sensor within the requested pipeline. The historical results endpoint (`GET /results/{pipeline}/history`) returns a paginated, optionally time-filtered sequence of past predictions. Each prediction response object includes the target variable, the predicted value, the generation timestamp, the model identifier, the model type, the feature schema version, and the top-N SHAP feature contributions ranked by absolute magnitude (N configurable, default: 5). All responses are serialized as JSON (JavaScript Object Notation).

### 9.4 Integration with WINGS and Third-Party Platforms

External platforms, including the WINGS environmental monitoring dashboard and any authorized third-party decision support system, may integrate with the DSS API in pull mode: the external system periodically queries the latest results endpoint at an interval aligned with the prediction cycle frequency, receiving the most recent prediction along with its SHAP-based explanation. Historical prediction retrieval is available for backfill operations or audit purposes. Every API interaction is recorded in the delivery log collection by a middleware component operating on the Starlette ASGI layer, capturing the client identifier, the requested endpoint, the HTTP response status code, and the end-to-end response latency. This logging layer provides a complete, non-repudiable audit trail of data consumption without affecting response latency, as the log write is executed as a non-blocking, best-effort background operation.
