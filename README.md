# 🚀 End-to-End MLOps Pipeline — Complete Tutorial

**Project:** Customer Churn Prediction  
**Stack:** XGBoost · MLflow · FastAPI · Docker · DVC · GitHub Actions · Evidently AI  
**Dataset:** Telecom Customer Churn (5000 rows, generated synthetic)  
**Difficulty:** Intermediate–Advanced | **Duration:** 3–4 weeks

---

## 📁 Project Structure

```
mlops_pipeline/
├── data/
│   ├── generate_dataset.py     # Step 1: Creates churn.csv
│   └── churn.csv               # Generated dataset (5000 rows)
│
├── src/
│   ├── train.py                # Step 2: Preprocessing + MLflow training
│   └── api.py                  # Step 3: FastAPI serving endpoints
│
├── models/                     # Auto-created by train.py
│   ├── model.pkl               # Trained XGBoost model
│   ├── encoders.pkl            # Label encoders
│   ├── scaler.pkl              # StandardScaler
│   └── metrics.json            # Best model metrics
│
├── monitoring/
│   ├── monitor.py              # Step 4: Evidently drift detection
│   └── reports/                # HTML monitoring reports (auto-created)
│
├── tests/
│   └── test_pipeline.py        # Step 5: 23 automated tests
│
├── .github/
│   └── workflows/
│       └── mlops_pipeline.yml  # Step 6: CI/CD automation
│
├── Dockerfile                  # Step 7: Containerization
├── requirements.txt
└── .dvcignore
```

---

## 🧠 What You'll Build & Learn

| Component | Tool | Concept Learned |
|---|---|---|
| Dataset Generation | Python / NumPy | Feature engineering, realistic data simulation |
| Model Training | XGBoost + scikit-learn | Classification, preprocessing pipelines |
| Experiment Tracking | MLflow | Comparing runs, artifact versioning |
| REST API | FastAPI + Pydantic | Model serving, input validation, Swagger docs |
| Containerization | Docker | Reproducible environments, image health checks |
| Data Versioning | DVC | Tracking data like code in Git |
| CI/CD Automation | GitHub Actions | Auto train → test → build on every push |
| Monitoring | Evidently AI | Data drift detection, model performance tracking |

---

## ⚙️ Prerequisites

```bash
# Python 3.9+ required
python --version

# Install all dependencies
pip install -r requirements.txt

# Optional: Install Docker Desktop for containerization step
# https://www.docker.com/products/docker-desktop/

# Optional: Install DVC for data versioning
pip install dvc
```

---

## 📦 STEP 1 — Dataset Generation

**File:** `data/generate_dataset.py`

### What it does
Creates a realistic telecom customer churn dataset with **5000 rows** and **12 features**.  
Churn probability is computed from real-world drivers (contract type, tenure, payment method etc.)

### Features Explained

| Feature | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months customer has been with the company |
| `monthly_charges` | Numeric | Current monthly bill (USD) |
| `total_charges` | Numeric | Total amount paid over lifetime |
| `senior_citizen` | Binary | Is the customer a senior citizen? |
| `dependents` | Binary | Does the customer have dependents? |
| `tech_support` | Binary | Is tech support subscribed? |
| `online_security` | Binary | Is online security subscribed? |
| `num_services` | Numeric | Count of subscribed add-ons (1–8) |
| `contract` | Categorical | Month-to-month / One year / Two year |
| `internet_service` | Categorical | DSL / Fiber optic / No |
| `payment_method` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `churn` | Target (0/1) | Did the customer leave? |

### Run it
```bash
python data/generate_dataset.py
```

### Expected Output
```
Dataset saved → data/churn.csv
Shape        : (5000, 12)
Churn rate   : 23.38%
```

### Key Concept: Why Simulate Data?
Real churn data is confidential. Simulating realistic data teaches you:
- How features relate to target (causal modeling)
- Class imbalance handling (23% churn is realistic)
- Feature correlation understanding

---

## 🏋️ STEP 2 — Model Training with MLflow

**File:** `src/train.py`

### What it does
1. Loads and preprocesses the dataset (encodes categoricals, scales numerics)
2. Runs **3 experiments** with different XGBoost hyperparameters
3. Logs every run to **MLflow** (params, metrics, model artifact)
4. Saves the best model artifacts to `models/`

### Preprocessing Pipeline

```
Raw CSV → LabelEncoder (categoricals) → StandardScaler (numerics) → XGBoost
```

**Why LabelEncoder?** XGBoost handles ordinal encodings well, and our categoricals
have a natural order (One year > Month-to-month for customer value).

**Why StandardScaler?** Even though XGBoost is tree-based and scale-invariant,
scaling helps the API endpoint process mixed features consistently.

### Run Training
```bash
python src/train.py
```

### View MLflow Experiments
```bash
# Start MLflow UI (open http://localhost:5000 in browser)
mlflow ui

# You'll see:
# - 3 runs side by side
# - Metrics chart (accuracy, F1, ROC-AUC)
# - Parameter comparison table
# - Downloadable model artifacts
```

### MLflow Key Concepts

| MLflow Concept | What it does |
|---|---|
| `mlflow.set_experiment()` | Groups related runs together |
| `mlflow.start_run()` | Opens a new experiment run |
| `mlflow.log_params()` | Records hyperparameters |
| `mlflow.log_metrics()` | Records evaluation scores |
| `mlflow.sklearn.log_model()` | Saves model as versioned artifact |

### Metrics You'll Track
- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted churners, how many actually churned?
- **Recall** — Of actual churners, how many did we catch?
- **F1 Score** — Harmonic mean of precision & recall (best single metric for imbalanced data)
- **ROC-AUC** — Discriminative ability across all thresholds

---

## 🌐 STEP 3 — FastAPI Model Serving

**File:** `src/api.py`

### What it does
Wraps the trained model in a production-ready REST API with:
- Input validation (Pydantic schemas)
- Auto-generated Swagger documentation
- Single and batch prediction endpoints
- Risk level categorization + business recommendations

### Run the API
```bash
uvicorn src.api:app --reload --port 8000
```

### Open Swagger UI
```
http://localhost:8000/docs
```
This gives you a fully interactive interface to test every endpoint!

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System status + model metrics |
| GET | `/info` | Feature info + valid input values |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Up to 100 customers at once |

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "senior_citizen": 1,
    "dependents": 0,
    "tech_support": 0,
    "online_security": 0,
    "num_services": 1,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
  }'
```

### Expected Response
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7842,
  "risk_level": "HIGH",
  "recommendation": "Immediate retention call + discount offer recommended"
}
```

### Key Concept: Pydantic Validation
FastAPI uses Pydantic to validate every input automatically:
- Wrong data type → 422 Unprocessable Entity returned
- Invalid enum value → Clear error message with allowed values
- Out of range numbers → Rejected before reaching model
This prevents garbage-in-garbage-out problems in production.

---

## 📊 STEP 4 — Monitoring with Evidently AI

**File:** `monitoring/monitor.py`

### What it does
Detects **data drift** — when production data starts looking different from training data.
This is critical because models degrade silently when real-world patterns shift.

### Run Monitoring
```bash
python monitoring/monitor.py
```

### Output Reports
```
monitoring/reports/drift_report.html       ← Open in browser
monitoring/reports/performance_report.html ← Open in browser
```

### How Drift is Simulated
The script artificially drifts the "current" data:
- Monthly charges increased 15% (pricing change)
- More Fiber optic customers (product push)
- More Month-to-month contracts (market trend)

This mimics real production scenarios where business conditions change.

### What to Look For in Reports
- **Dataset Drift** — Overall drift score (>0.5 = significant)
- **Column Drift** — Which specific features drifted most
- **Missing Values** — Did new data introduce nulls?
- **Performance Drop** — Did accuracy/F1 degrade?

### When to Retrain?
- Drift share > 30% of columns → Retrain
- ROC-AUC drops > 5 points → Retrain
- Precision/Recall shifts significantly → Investigate + possibly retrain

---

## 🧪 STEP 5 — Automated Testing with Pytest

**File:** `tests/test_pipeline.py`

### Run All Tests
```bash
pytest tests/ -v
```

### Test Categories (23 tests total)

```
TestDataset         (7 tests)  — Shape, nulls, value ranges, churn rate
TestModelArtifacts  (6 tests)  — Model loads, metrics valid, encoders present
TestAPI             (10 tests) — All endpoints, edge cases, batch limits
```

### Key Testing Concepts
- **Unit tests** — Each function tested in isolation
- **Integration tests** — API endpoints tested end-to-end
- **Edge case tests** — Invalid inputs, boundary conditions, batch limits
- **Threshold tests** — Accuracy must be above minimum (guards against broken models)

---

## 🐳 STEP 6 — Dockerization

**File:** `Dockerfile`

### Build the Docker Image
```bash
docker build -t churn-api .
```

### Run the Container
```bash
docker run -p 8000:8000 churn-api
```

### Test the Containerized API
```bash
curl http://localhost:8000/health
```

### Docker Concepts Used
- **Multi-layer caching** — requirements.txt copied before source code for faster rebuilds
- **Slim base image** — `python:3.11-slim` reduces image size by ~60%
- **HEALTHCHECK** — Docker automatically restarts unhealthy containers
- **ENV variables** — Configuration without hardcoding values

### View Container Logs
```bash
docker logs <container-id>
```

### Stop the Container
```bash
docker stop <container-id>
```

---

## 📂 STEP 7 — Data Versioning with DVC

### Why DVC?
Git can't track large files like datasets and model weights.
DVC gives Git-like version control for data files.

### Initialize DVC
```bash
# Initialize git first (if not done)
git init
git add .
git commit -m "Initial commit"

# Initialize DVC
dvc init
git commit -m "Initialize DVC"
```

### Track Dataset with DVC
```bash
# Tell DVC to track the dataset
dvc add data/churn.csv

# This creates data/churn.csv.dvc (tiny pointer file)
# Track the pointer in Git
git add data/churn.csv.dvc data/.gitignore
git commit -m "Track dataset with DVC"
```

### Track Model Artifacts
```bash
dvc add models/model.pkl models/scaler.pkl models/encoders.pkl
git add models/.gitignore models/*.dvc
git commit -m "Track model artifacts with DVC"
```

### Create DVC Pipeline
```bash
# Define the full training pipeline
dvc run -n train \
  -d data/churn.csv \
  -d src/train.py \
  -o models/model.pkl \
  -o models/scaler.pkl \
  -o models/encoders.pkl \
  python src/train.py
```

### Reproduce Pipeline (Re-runs only changed steps)
```bash
dvc repro
```

### Add Remote Storage (Google Drive Example)
```bash
dvc remote add -d myremote gdrive://<YOUR_DRIVE_FOLDER_ID>
dvc push    # Upload data to Drive
dvc pull    # Download data from Drive
```

### Key DVC Concept
When a teammate clones your repo:
```bash
git clone <your-repo>
dvc pull               # Downloads the exact data version you worked with
python src/train.py    # Reproduces your exact results
```

---

## 🤖 STEP 8 — CI/CD with GitHub Actions

**File:** `.github/workflows/mlops_pipeline.yml`

### How to Set Up

1. Create a GitHub repository
2. Push your code:
```bash
git remote add origin https://github.com/YOUR_USERNAME/mlops-pipeline.git
git push -u origin main
```
3. GitHub automatically detects `.github/workflows/` and runs the pipeline!

### Pipeline Jobs (Runs in order)

```
push to main
    │
    ▼
[1] Lint Code (flake8)
    │ passes
    ▼
[2] Train Model + Run 23 Tests
    │ passes
    ├──────────────────────────────┐
    ▼                              ▼
[3] Build Docker Image       [4] Run Monitoring
    │ health check passes
    ▼
✅ All artifacts uploaded to GitHub
```

### View Pipeline Results
- Go to your GitHub repo → Click **"Actions"** tab
- See real-time logs for each job
- Download uploaded artifacts (model files, test results, monitoring reports)

---

## 🔁 Full Pipeline in One Shot

```bash
# Complete pipeline from scratch:

# 1. Generate data
python data/generate_dataset.py

# 2. Train model (tracks to MLflow)
python src/train.py

# 3. View experiments
mlflow ui                            # → http://localhost:5000

# 4. Start API
uvicorn src.api:app --reload         # → http://localhost:8000/docs

# 5. Run monitoring
python monitoring/monitor.py         # → monitoring/reports/*.html

# 6. Run all tests
pytest tests/ -v

# 7. Build & run Docker container
docker build -t churn-api . && docker run -p 8000:8000 churn-api
```

---

## 📈 What This Project Proves to Employers / PhD Supervisors

| Skill Demonstrated | How |
|---|---|
| ML Engineering | XGBoost training, preprocessing pipeline, model evaluation |
| Software Engineering | FastAPI, Pydantic validation, clean code structure |
| MLOps | MLflow tracking, DVC versioning, Docker deployment |
| Production Mindset | Health checks, batch APIs, error handling |
| Testing | 23 tests covering dataset, model, and API |
| Automation | GitHub Actions CI/CD pipeline |
| Monitoring | Evidently drift detection, automated reports |

---

## 🔮 Extend This Project (Level Up Ideas)

| Extension | What to Add |
|---|---|
| **Database** | Store predictions in PostgreSQL with SQLAlchemy |
| **Auth** | Add API key authentication with FastAPI security |
| **Dashboard** | Build a Streamlit dashboard to visualize predictions |
| **Retraining** | Auto-trigger retraining when drift is detected |
| **Feature Store** | Add Feast or Hopsworks for feature management |
| **A/B Testing** | Serve two models and compare performance |
| **Cloud Deploy** | Deploy to AWS ECS, GCP Cloud Run, or Azure AKS |
| **Kubernetes** | Write K8s manifests for auto-scaling |

---

*Built as part of AI/ML learning roadmap | Jr. AI Engineer → MLOps Engineer path*
