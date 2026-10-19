"""Verify the approved, repository-bundled Phase 7 model artifacts before startup.

The legacy/V1 path is intentionally a no-op. V2 accepts only the exact pinned
local bundle; it never downloads, overwrites, or repairs artifacts automatically.
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
RUN_ID = "20261018T145029Z_07fc8d2c"
DEFAULT_ARTIFACT_RELATIVE = Path("Artifacts") / f"phase7_final_fit_{RUN_ID}"
MODEL_VERSION_ENV = "NETWORK_SECURITY_MODEL_VERSION"
ARTIFACT_DIR_ENV = "NETWORK_SECURITY_PHASE7_ARTIFACT_DIR"

EXPECTED_SHA256: dict[str, str] = {
    "fit_report.json": "4ac345827cde00412828f92f7c106b65a648c0ad5fb5107da223b9c341204ed3",
    "selected_pipeline.joblib": "ea810723e15af5f4f17df39f913eb7bdf4cd752b7e263834c2e83ce3b4c403db",
    "preprocessor.joblib": "3bab3417907555b052fc26760ad3b35b3f70fc0d0d66f3ca3a0f4c1c2f85f8ae",
    "classifier.joblib": "628cc23f0f8a31297105316c010ef798bc36c1f6b8249f1878d3fae1a81cf05e",
    "outer_training_source_row_indices.csv": "4be3b605a2585bf239aac1f4b929eef18a7940f25bfc0bc5747c43e6e0134265",
}


def sha256_file(path: Path) -> str:
    """Hash a file without deserializing it."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_directory(env: Mapping[str, str], base_dir: Path) -> Path:
    """Resolve the artifact directory using the same override as the app."""
    configured = env.get(ARTIFACT_DIR_ENV, "").strip()
    if not configured:
        return (base_dir / DEFAULT_ARTIFACT_RELATIVE).resolve()
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def ensure_phase7_artifacts(
    *,
    env: Mapping[str, str] | None = None,
    base_dir: Path | None = None,
    expected_sha256: Mapping[str, str] | None = None,
) -> str:
    """No-op for V1; for V2 require every artifact and verify its pinned hash.

    There is intentionally no network, S3, file creation, replacement, or repair
    logic here. Missing/partial/mismatched model bundles fail closed.
    """
    environment = dict(os.environ if env is None else env)
    model_version = environment.get(MODEL_VERSION_ENV, "legacy").strip().lower()
    if model_version == "legacy":
        return "legacy_skipped"
    if model_version != "phase7":
        raise RuntimeError(
            f"Unsupported {MODEL_VERSION_ENV} value {model_version!r}; use 'legacy' or 'phase7'."
        )

    project_dir = (base_dir or BASE_DIR).resolve()
    directory = artifact_directory(environment, project_dir)
    expected = dict(EXPECTED_SHA256 if expected_sha256 is None else expected_sha256)

    missing = [name for name in expected if not (directory / name).is_file()]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            "The approved Phase 7 artifact bundle is missing required file(s): "
            f"{names}. Include the verified bundle in the deployment source; "
            "automatic download is disabled."
        )

    for filename, expected_hash in expected.items():
        actual_hash = sha256_file(directory / filename)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for Phase 7 artifact {filename}; refusing to use it."
            )
    return "verified_existing"


def main() -> int:
    """CLI entry point used before starting the web server."""
    load_dotenv()
    try:
        action = ensure_phase7_artifacts()
    except Exception as exc:
        print(f"Phase 7 artifact verification failed: {exc}", file=sys.stderr)
        return 1

    if action == "legacy_skipped":
        print("Legacy/V1 selected; Phase 7 artifact verification skipped.")
    else:
        print("Pinned Phase 7 artifact bundle verified; starting application.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
