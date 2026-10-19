# Phase 11 — GitHub, Docker, and Render preparation (no AWS dependency)

## Decision

Use GitHub for source/version control, Docker for packaging, and a separate Render Free web service for V2. Do not create an S3 bucket or configure AWS credentials for this deployment. MongoDB is not required for the current CSV inference flow.

## Deployment dependency split

- `requirements.txt` remains unchanged for training, data ingestion, MLflow/DagsHub, MongoDB integration, and development.
- `requirements-deploy.txt` contains only the pinned libraries required to import and run the HTTP inference application, including its current utility-module import chain.
- Docker installs `requirements-deploy.txt`; `/train` remains disabled in the deployment configuration.
- Validate the optimized image by rebuilding it, running the container with `NETWORK_SECURITY_MODEL_VERSION=phase7`, exercising prediction and download routes, and checking memory before deploying.

## Approved artifact bundle

With the user's approval, include only the exact five hash-pinned Phase 7 artifacts in `Artifacts/phase7_final_fit_20261018T145029Z_07fc8d2c/`:

- `fit_report.json`
- `selected_pipeline.joblib`
- `preprocessor.joblib`
- `classifier.joblib`
- `outer_training_source_row_indices.csv`

The `.gitignore` and `.dockerignore` exceptions are restricted to these files. Datasets, holdout/evaluation records, logs, secrets, caches, archives, and all other generated runs remain excluded. Do not use `git add -f` unless the allowlist is re-reviewed first.

## Startup behavior

`artifact_bootstrap.py` is a no-op when `NETWORK_SECURITY_MODEL_VERSION=legacy`. In `phase7` mode it requires every approved file to exist and verifies each SHA-256 before application startup. It performs no network download, overwrite, repair, model fitting, or evaluation. Missing or mismatched files stop startup.

The Dockerfile runs this verification before Uvicorn. Configure the separate V2 Render service with `NETWORK_SECURITY_MODEL_VERSION=phase7`. Keep the V1 service and settings unchanged.

## Cost and interview framing

No AWS bucket, IAM user/role, cloud credential, MongoDB service, Render service, or paid resource is created by these local preparation steps. Keep the claims aligned with what is actually implemented: leakage-aware grouped cross-validation, a one-time reserved holdout evaluation, pinned model provenance, fail-closed artifact verification, regression tests, Docker packaging, CI, and a separate inference deployment.

## Not performed

No Render service has been created or deployed. No artifacts have been uploaded externally. No commit or push has been performed. The Phase 8 holdout must not be evaluated or used for further tuning.
