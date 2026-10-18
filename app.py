import hashlib
import json
import os
import re
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from uvicorn import run as app_run

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.utils.main_utils.utils import load_object
from network_security.utils.ml_utils.model.estimator import NetworkModel


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
FINAL_MODEL_DIR = BASE_DIR / "final_model"
# The Phase 7 artifacts are local, verified outputs and are not bundled in Git.
# Keep V1 as the default; opt into the pinned Phase 7 model using
# NETWORK_SECURITY_MODEL_VERSION=phase7 before starting the application.
PHASE7_MODEL_DIR = Path(
    os.getenv(
        "NETWORK_SECURITY_PHASE7_ARTIFACT_DIR",
        str(BASE_DIR / "Artifacts" / "phase7_final_fit_20261018T145029Z_07fc8d2c"),
    )
).expanduser().resolve()
MODEL_VERSION_ENV = "NETWORK_SECURITY_MODEL_VERSION"
PHASE7_EXPECTED_ARTIFACT_SHA256 = {
    "fit_report.json": "4ac345827cde00412828f92f7c106b65a648c0ad5fb5107da223b9c341204ed3",
    "selected_pipeline.joblib": "ea810723e15af5f4f17df39f913eb7bdf4cd752b7e263834c2e83ce3b4c403db",
    "preprocessor.joblib": "3bab3417907555b052fc26760ad3b35b3f70fc0d0d66f3ca3a0f4c1c2f85f8ae",
    "classifier.joblib": "628cc23f0f8a31297105316c010ef798bc36c1f6b8249f1878d3fae1a81cf05e",
    "outer_training_source_row_indices.csv": "4be3b605a2585bf239aac1f4b929eef18a7940f25bfc0bc5747c43e6e0134265",
}
PREDICTION_OUTPUT_DIR = BASE_DIR / "prediction_output"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
PREDICTION_RESULT_RETENTION_SECONDS = 24 * 60 * 60
PREDICTION_RESULT_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")

PREDICTION_FEATURES = (
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
)

app = FastAPI(
    title="Phishing Website Detection System",
    description="Batch inference service for phishing-website classification.",
    version="1.0.0",
)

# Same-origin Jinja pages do not require CORS, but the existing API behavior is preserved.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Preserve the original API CORS behavior for clients that may call /predict directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256 without deserializing its contents."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_phase7_inference_model() -> NetworkModel:
    """Load the approved Phase 7 artifacts only after hash/provenance checks."""
    fit_report_path = PHASE7_MODEL_DIR / "fit_report.json"
    artifact_paths = {
        name: PHASE7_MODEL_DIR / name
        for name in PHASE7_EXPECTED_ARTIFACT_SHA256
        if name != "fit_report.json"
    }
    paths_to_verify = {"fit_report.json": fit_report_path, **artifact_paths}

    # Hash-check every artifact before loading any joblib/pickle content.
    for name, path in paths_to_verify.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"The configured Phase 7 model artifact is missing: {path}"
            )
        if _sha256_file(path) != PHASE7_EXPECTED_ARTIFACT_SHA256[name]:
            raise ValueError(
                f"SHA-256 mismatch for Phase 7 artifact {name}; refusing to load it."
            )

    with fit_report_path.open("r", encoding="utf-8") as file_obj:
        fit_report = json.load(file_obj)

    expected_report_values = {
        "phase": 7,
        "selected_candidate_by_inner_cv_only": "Gradient Boosting",
        "outer_training_rows_fitted": 8843,
        "reserved_holdout_rows": 2212,
        "fitted_estimator_saved": True,
        "holdout_scored": False,
        "holdout_predictions_generated": False,
        "final_model_directory_modified": False,
        "dataset_sha256": "e20bf913b6c2e669f4e875e5d223be2542cc7c79a7b62d284ae9d9627c3d9678",
        "schema_sha256": "0fe172eb670067a0048c62b601d5b63d311be827a5ee81c65f0fbd3dc480ad83",
        "split_manifest_sha256": "3dc82b79e2feb8f7b5fd23045be577be92d73d14aeed0005d3577782749f3ac7",
        "inner_cv_assignment_sha256": "9540bfa60ee0955455e8ba8ae3e039f6c83741f7b2760bac1db9a10fb1291cc9",
        "selection_source_git_commit": "76482641d41c300660be597224208a14090607e5",
        "implementation_git_commit": "27f44fe950e92196ad8fe88a8500d5a6a89fa8ff",
    }
    for key, expected in expected_report_values.items():
        if fit_report.get(key) != expected:
            raise ValueError(
                f"The Phase 7 fit report field {key!r} does not match the approved run."
            )

    reported_hashes = fit_report.get("artifacts_sha256", {})
    for artifact_name in (
        "selected_pipeline.joblib",
        "preprocessor.joblib",
        "classifier.joblib",
        "outer_training_source_row_indices.csv",
    ):
        if reported_hashes.get(artifact_name) != PHASE7_EXPECTED_ARTIFACT_SHA256[artifact_name]:
            raise ValueError(
                f"The Phase 7 fit report has an unexpected hash for {artifact_name}."
            )

    # These are trusted-local artifacts; fixed hashes and report fields are verified first.
    preprocessor = joblib.load(artifact_paths["preprocessor.joblib"])
    final_model = joblib.load(artifact_paths["classifier.joblib"])

    expected_features = tuple(
        getattr(preprocessor, "feature_names_in_", PREDICTION_FEATURES)
    )
    if expected_features != PREDICTION_FEATURES:
        raise ValueError(
            "The Phase 7 preprocessor feature order does not match the application's expected schema."
        )
    if getattr(preprocessor, "n_features_in_", len(PREDICTION_FEATURES)) != len(PREDICTION_FEATURES):
        raise ValueError("The Phase 7 preprocessor does not expect the required 30 inference features.")
    if getattr(final_model, "n_features_in_", len(PREDICTION_FEATURES)) != len(PREDICTION_FEATURES):
        raise ValueError("The Phase 7 classifier does not expect the required 30 inference features.")
    model_classes = {int(value) for value in getattr(final_model, "classes_", [])}
    if model_classes != {0, 1}:
        raise ValueError("The Phase 7 classifier does not expose the expected encoded classes 0 and 1.")

    return NetworkModel(preprocessor=preprocessor, model=final_model)


@lru_cache(maxsize=1)
def load_inference_model() -> NetworkModel:
    """Load V1 by default, or the pinned Phase 7 model when explicitly selected."""
    model_version = os.getenv(MODEL_VERSION_ENV, "legacy").strip().lower()
    if model_version not in {"legacy", "phase7"}:
        raise ValueError(
            f"Unsupported {MODEL_VERSION_ENV} value {model_version!r}; use 'legacy' or 'phase7'."
        )

    try:
        if model_version == "phase7":
            return _load_phase7_inference_model()

        # Preserve legacy V1 behavior unless V2 is explicitly enabled.
        preprocessor = load_object(str(FINAL_MODEL_DIR / "preprocessor.pkl"))
        final_model = load_object(str(FINAL_MODEL_DIR / "model.pkl"))
        expected_features = tuple(getattr(preprocessor, "feature_names_in_", PREDICTION_FEATURES))

        if tuple(expected_features) != PREDICTION_FEATURES:
            raise ValueError(
                "The persisted preprocessor feature order does not match the application's expected schema."
            )

        if getattr(final_model, "n_features_in_", len(PREDICTION_FEATURES)) != len(PREDICTION_FEATURES):
            raise ValueError("The persisted model does not expect the required 30 inference features.")

        return NetworkModel(preprocessor=preprocessor, model=final_model)
    except Exception as exc:
        raise NetworkSecurityException(exc, sys) from exc


def render_index(request: Request, *, error: str | None = None, success: str | None = None):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "error": error,
            "success": success,
            "feature_count": len(PREDICTION_FEATURES),
        },
    )


def validate_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the CSV schema before model inference."""
    if df.empty:
        raise ValueError("The uploaded CSV is empty. Please upload a CSV containing prediction rows.")

    uploaded_columns = list(df.columns)
    expected = list(PREDICTION_FEATURES)

    if "Result" in uploaded_columns:
        raise ValueError(
            "The Result column is the training target and is not required for prediction. "
            "Please upload the 30 feature columns only."
        )

    missing_columns = [column for column in expected if column not in df.columns]
    extra_columns = [column for column in df.columns if column not in expected]

    if missing_columns or extra_columns:
        details = []
        if missing_columns:
            details.append(f"Missing columns: {', '.join(missing_columns)}")
        if extra_columns:
            details.append(f"Unexpected columns: {', '.join(extra_columns)}")
        raise ValueError("Invalid prediction schema. " + " | ".join(details))

    normalized = df.loc[:, expected].copy()

    non_numeric_columns = [column for column in expected if not pd.api.types.is_numeric_dtype(normalized[column])]
    if non_numeric_columns:
        raise ValueError(
            "The following columns must contain numeric values: " + ", ".join(non_numeric_columns)
        )

    return normalized


def training_endpoint_enabled() -> bool:
    """Allow training only when explicitly enabled in a development environment."""
    app_environment = os.getenv("APP_ENV", "").strip().lower()
    enabled_value = os.getenv("ENABLE_TRAINING_ENDPOINT", "").strip().lower()
    return app_environment == "development" and enabled_value in {"1", "true", "yes", "on"}


def create_prediction_result_id() -> str:
    """Create an unpredictable identifier for one prediction result."""
    return uuid.uuid4().hex


def prediction_result_path(result_id: str) -> Path:
    """Resolve a request-specific result ID without accepting path components."""
    if not isinstance(result_id, str) or not PREDICTION_RESULT_ID_PATTERN.fullmatch(result_id):
        raise HTTPException(status_code=404, detail="Prediction result is unavailable or expired.")
    return PREDICTION_OUTPUT_DIR / f"prediction_{result_id}.csv"


def cleanup_expired_prediction_results(*, now: float | None = None) -> None:
    """Remove expired per-request result CSVs, leaving unrelated files untouched."""
    if not PREDICTION_OUTPUT_DIR.is_dir():
        return

    current_time = time.time() if now is None else now
    expiry_cutoff = current_time - PREDICTION_RESULT_RETENTION_SECONDS

    for result_file in PREDICTION_OUTPUT_DIR.glob("prediction_*.csv"):
        try:
            if result_file.is_file() and result_file.stat().st_mtime < expiry_cutoff:
                result_file.unlink()
        except OSError:
            logging.warning(
                "Could not remove expired prediction result file: %s",
                result_file,
                exc_info=True,
            )


@app.get("/", tags=["application"])
async def index(request: Request):
    return render_index(request)


@app.get("/sample-data", tags=["application"], name="sample_data")
async def sample_data():
    sample_file = SAMPLE_DATA_DIR / "sample_prediction.csv"
    if not sample_file.exists():
        raise HTTPException(status_code=404, detail="Sample CSV is not available.")
    return FileResponse(
        path=str(sample_file),
        filename="sample_prediction.csv",
        media_type="text/csv",
    )


@app.get("/download-result", tags=["application"], name="download_result")
async def download_result(result_id: str | None = None):
    """Download only the prediction result identified by this request's opaque ID."""
    if result_id is None:
        raise HTTPException(status_code=404, detail="Prediction result is unavailable or expired.")

    result_file = prediction_result_path(result_id)
    if not result_file.is_file():
        raise HTTPException(status_code=404, detail="Prediction result is unavailable or expired.")

    return FileResponse(
        path=str(result_file),
        filename="phishing_predictions.csv",
        media_type="text/csv",
    )


@app.get("/train", include_in_schema=False)
async def train_route():
    """Run training only when explicitly enabled for a controlled development environment."""
    if not training_endpoint_enabled():
        raise HTTPException(status_code=404, detail="Training endpoint is disabled.")

    try:
        from network_security.pipeline.training_pipeline import TrainingPipeline

        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        return Response("Training is successful")
    except Exception as exc:
        raise NetworkSecurityException(exc, sys) from exc


@app.post("/predict", tags=["application"])
async def predict_route(request: Request, file: UploadFile = File(...)):
    try:
        filename = (file.filename or "").lower()
        if not filename.endswith(".csv"):
            return templates.TemplateResponse(
                request=request,
                name="index.html",
                context={
                    "request": request,
                    "error": "Please upload a CSV file.",
                    "success": None,
                    "feature_count": len(PREDICTION_FEATURES),
                },
                status_code=400,
            )

        df = pd.read_csv(file.file)
        inference_df = validate_prediction_dataframe(df)
        network_model = load_inference_model()
        y_pred = network_model.predict(inference_df)

        result_df = inference_df.copy()
        result_df["predicted_column"] = y_pred

        PREDICTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cleanup_expired_prediction_results()
        result_id = create_prediction_result_id()
        result_file = prediction_result_path(result_id)
        result_df.to_csv(result_file, index=False)

        predictions = [int(value) for value in y_pred]
        phishing_count = sum(1 for value in predictions if value == 0)
        legitimate_count = sum(1 for value in predictions if value == 1)

        table_html = result_df.to_html(
            classes="prediction-table",
            index=False,
            border=0,
        )

        return templates.TemplateResponse(
            request=request,
            name="table.html",
            context={
                "request": request,
                "table": table_html,
                "rows_analyzed": len(result_df),
                "phishing_count": phishing_count,
                "legitimate_count": legitimate_count,
                "download_available": result_file.is_file(),
                "result_id": result_id,
            },
        )

    except ValueError as exc:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": str(exc),
                "success": None,
                "feature_count": len(PREDICTION_FEATURES),
            },
            status_code=400,
        )
    except Exception as exc:
        logging.exception("Prediction request failed")
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "request": request,
                "error": "Prediction could not be completed. Please verify the CSV format and try again.",
                "success": None,
                "feature_count": len(PREDICTION_FEATURES),
            },
            status_code=500,
        )
    finally:
        await file.close()


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app_run(app, host="0.0.0.0", port=port)
