FROM python:3.11-slim-bookworm

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements-deploy.txt

EXPOSE 8000

# Verify the bundled Phase 7 hashes before starting the web application.
CMD ["sh", "-c", "python artifact_bootstrap.py && exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
