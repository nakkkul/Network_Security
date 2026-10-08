import os
import sys
from functools import lru_cache
from pathlib import Path

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
PREDICTION_OUTPUT_DIR = BASE_DIR / "prediction_output"
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
OUTPUT_FILE = PREDICTION_OUTPUT_DIR / "output.csv"

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


@lru_cache(maxsize=1)
def load_inference_model() -> NetworkModel:
    """Load the persisted preprocessing pipeline and model once per process."""
    try:
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
async def download_result():
    if not OUTPUT_FILE.exists():
        raise HTTPException(status_code=404, detail="No prediction result is available yet.")
    return FileResponse(
        path=str(OUTPUT_FILE),
        filename="phishing_predictions.csv",
        media_type="text/csv",
    )


@app.get("/train", include_in_schema=False)
async def train_route():
    """Preserve the existing training entry point without loading training dependencies at startup."""
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
        result_df.to_csv(OUTPUT_FILE, index=False)

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
                "download_available": OUTPUT_FILE.exists(),
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
