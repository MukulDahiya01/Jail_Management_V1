# Indian Jail Management — Synthetic AI/ML Cloud Dashboard

A complete **FastAPI + HTML/CSS/JavaScript** educational demonstration, compatible with **Render** or **Azure App Service**. This is **not Streamlit**.

> **Warning:** All records, locations, models and outcomes are **synthetic**. `Case_Review_Priority_Synthetic` is not a legal bailability label. Do not use predictions for real custody, bail, sentencing, resource allocation or legal advice.

## Included

- Responsive browser dashboard with monthly population and capacity trends, state aggregates and metrics.
- All six notebook classification model families: Logistic Regression, Decision Tree, KNN, SVM, Naive Bayes and Random Forest. Pipeline transformations are saved within each model.
- Five regression models for next-month population forecasting; trained with an ordered month split.
- K-Means facility clusters (offline computed), and synthetic facility association-rule tables.
- The **original 145-cell notebook** in `notebooks/original_workflow.ipynb`, retaining its EDA, PCA, t-SNE, hierarchical and DBSCAN sections. Those notebook-only analytical steps are not live API endpoints.
- FastAPI OpenAPI documentation at `/docs`, readiness check at `/health`, and smoke tests.
- Expanded synthetic Excel workbook in `data/` and saved models in `models/`.

## Run locally

Python **3.11** is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000 (API documentation: http://localhost:8000/docs).

**Optional retraining** after installing dependencies:

```bash
python train_models.py
```

Retraining overwrites pre-built joblib model artifacts and dashboard snapshots. The artifacts shipped in the ZIP are already trained from the included synthetic dataset. Never load untrusted `.joblib` files.

## Deploy to Render (recommended for demonstration)

1. Create a GitHub repository and upload **all files and folders**, including `models/`, `templates/`, `static/`, `app/`, `data/` and `requirements.txt`.
2. Log in to [Render](https://dashboard.render.com), select **New → Blueprint**, and select the repository containing `render.yaml` (or create a Web Service manually).
3. If setting up manually, use build command `pip install -r requirements.txt` and start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
4. Select a plan that supports the application's memory requirements. Free services can sleep, have resource limits, and may change in availability; consult Render pricing.
5. Once deployment completes, visit the `.onrender.com` URL and `/health`.

The dataset is included for reproducibility, but **training is not performed at each startup**. Saved artifacts are loaded on demand. Commit the complete `models/` directory to the repo or arrange to provision it at build time.

## Deploy to Azure App Service (Linux)

1. Create an **Azure App Service** running Linux / supported Python 3.11, with a plan suitable for your requirements.
2. Upload/push this repository (Deployment Center → GitHub or Azure CLI ZIP deployment).
3. Set **Startup Command** to:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

   Azure's Python App Service reverse proxy typically targets port 8000; adapt to your platform's documented port configuration if it differs.
4. Ensure `requirements.txt` is installed via the App Service build process. For Azure ZIP deployments, configure build automation as required.
5. Verify `/health` and `/docs`, and view application logs if startup fails.

Keep the app running on one worker initially to reduce memory consumption. Do not store private jail or inmate records in this public demo.

## Public API

- `GET /health`
- `GET /api/dashboard`
- `GET /api/facilities?state=...`
- `GET /api/metadata`
- `POST /api/predict/review`: fictional high/standard case review priority, accepts documented input schema
- `POST /api/predict/forecast`: synthetic next-month population forecast

Request schemas and interactive API calls: `/docs`.

## Evaluation notes

- Classification: 25% stratified holdout; target and direct legal Act–Section lookup identifiers excluded from input. A majority-class baseline is included.
- Regression: train/test split by `Feature_Month`, with later months reserved for testing.
- These evaluations only establish performance on a **generated data process**. They do not indicate real-world usefulness.
- The demo priority model is a **research-only toy task**, not a proxy for bailability or any real case outcome.

## Security and production limits

This is an open read-only prototype: it has no authentication, rate limiting, audit logging, database, or production risk review. Add these before any production use, and never use it for consequential legal or custodial decisions. Model artifacts are locally trusted and should not be accepted from visitors.
