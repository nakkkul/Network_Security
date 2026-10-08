# Phishing Website Detection System

End-to-end machine learning system for **phishing website classification** with a modular training pipeline and a FastAPI-based batch inference application.

> **V1 scope:** the deployed web application accepts a CSV containing the model's 30 engineered input features and returns batch predictions. It does **not** accept raw URLs in V1.

## Live Demo

Add the deployed Render URL here after deployment:

`https://<your-service>.onrender.com`

## Project Overview

This project demonstrates the lifecycle of a tabular machine-learning application rather than a notebook-only model:

```text
Training
MongoDB / training data
        ↓
Data ingestion
        ↓
Data validation
        ↓
Data transformation
        ↓
Model training + experiment tracking
        ↓
Persisted preprocessing + model artifacts

Inference
CSV feature dataset
        ↓
FastAPI
        ↓
Schema validation
        ↓
Saved preprocessor
        ↓
Saved classifier
        ↓
Prediction results
```

## What the application does

The application classifies phishing-website observations from the feature representation used by the underlying dataset.

The public V1 flow is:

1. Download the sample prediction CSV or prepare a compatible feature dataset.
2. Upload the CSV through the web interface.
3. Validate the 30 required inference features.
4. Apply the persisted preprocessing pipeline.
5. Generate predictions with the persisted scikit-learn classifier.
6. Review the batch prediction table.
7. Download the prediction output CSV.

### Prediction labels

The project maps the original dataset target convention to the persisted model representation:

- `0` → Phishing
- `1` → Legitimate

The `Result` target column is used during training and is **not required for inference**.

## Dataset

The project uses the UCI Machine Learning Repository **Phishing Websites** dataset (Dataset ID 327), which contains 11,055 instances and 30 input features for classification.

Dataset citation:

> Mohammad, R. & McCluskey, L. (2012). Phishing Websites [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51W2X

Source: https://archive.ics.uci.edu/dataset/327/phishing

The dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0); attribution is retained here for the project documentation.

## Machine Learning Pipeline

### Training

The existing training architecture contains separate components for:

- data ingestion
- data validation
- data transformation
- model training
- experiment tracking
- model artifact persistence
- optional S3 synchronization

The current persisted V1 inference artifacts are:

```text
final_model/model.pkl
final_model/preprocessor.pkl
```

These artifacts are treated as frozen release artifacts for V1.

### Inference

The inference application intentionally does not retrain the model.

```text
CSV upload
   ↓
30-feature schema validation
   ↓
Persisted preprocessor
   ↓
Decision Tree classifier
   ↓
0 / 1 prediction
   ↓
HTML results + downloadable CSV
```

## Why batch inference?

The model expects structured engineered features rather than a raw URL string. Batch CSV inference therefore provides a natural interface for scoring multiple observations at once while keeping the trained ML pipeline unchanged.

## Project Structure

```text
Network_Security/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── README.md
├── data_schema/
│   └── schema.yaml
├── final_model/
│   ├── model.pkl
│   └── preprocessor.pkl
├── sample_data/
│   └── sample_prediction.csv
├── templates/
│   ├── index.html
│   └── table.html
├── static/
│   └── style.css
├── tests/
│   └── test_inference.py
├── network_security/
│   ├── cloud/
│   ├── components/
│   ├── constant/
│   ├── entity/
│   ├── exception/
│   ├── logging/
│   ├── pipeline/
│   └── utils/
└── .github/
    └── workflows/
        └── main.yml
```

## Local Setup

### 1. Create and activate an environment

The current model artifacts were created with Python 3.11 and scikit-learn 1.9.1. Use a Python 3.11 environment for reproducible artifact loading.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the application

```bash
uvicorn app:app --reload
```

Or:

```bash
python app.py
```

The application will normally be available at:

`http://127.0.0.1:8000`

Swagger/OpenAPI remains available at:

`http://127.0.0.1:8000/docs`

## Test the inference pipeline

Run the built-in smoke tests:

```bash
python -m unittest discover -s tests -v
```

The test suite verifies:

- persisted artifact/schema compatibility
- sample CSV schema
- successful sample prediction
- rejection of the training target column
- rejection of missing features
- rejection of unexpected features

## Environment Variables

Inference does not require database or MLflow credentials.

Training/cloud functionality may use these variables:

```text
MONGO_DB_URL=
MLFLOW_TRACKING_URI=
MLFLOW_TRACKING_USERNAME=
MLFLOW_TRACKING_PASSWORD=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
```

Never commit real credentials. Use environment variables or the hosting provider's secret/environment-variable settings.

## Docker

Build:

```bash
docker build -t phishing-website-detection .
```

Run locally:

```bash
docker run --rm -p 8000:8000 -e PORT=8000 phishing-website-detection
```

The Docker image runs Uvicorn on `0.0.0.0` and respects the hosting platform's `PORT` environment variable.

## Free Deployment on Render

Render currently supports free Python web services and provides a public `onrender.com` URL. Free web services can spin down after 15 minutes of inactivity, so the first request after idle time may take longer.

Recommended V1 deployment settings:

### Using the repository directly

- **Service type:** Web Service
- **Language:** Python 3
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Plan:** Free

The included Dockerfile can also be used if you prefer Render's Docker runtime.

Do not add the training secrets to Render for inference-only deployment unless you later enable a protected training workflow.

## Limitations of V1

- The application predicts from engineered feature columns rather than accepting a raw URL.
- The public application is designed for synchronous batch inference through a web request, not a background job queue.
- The trained artifact is frozen for this resume release; the training pipeline is intentionally not retriggered by public users.
- The current ML training code is preserved rather than fully refactored in V1.

## Future Improvements

Possible V2 enhancements include:

- raw-URL feature extraction and single-URL inference
- improved classification-metric-based model selection
- stronger automated training validation
- explainability with feature-level model insights
- asynchronous batch jobs for very large datasets
- expanded automated tests and CI quality gates
- evaluation against newer phishing datasets

## Technologies

Python • Pandas • NumPy • Scikit-learn • FastAPI • Jinja2 • MongoDB • MLflow • DagsHub • Docker • AWS S3 • GitHub Actions

## Screenshots

Add the following screenshots after deployment:

- landing page
- CSV upload flow
- prediction results
- GitHub repository / live demo

## Resume Positioning

This project is best presented as an **end-to-end phishing-website classification and batch inference system** demonstrating machine-learning engineering, API development, validation, model persistence, containerization, and deployment—not as a raw-URL real-time phishing detector.
