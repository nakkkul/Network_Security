"""Command-line read-only checks for the frozen Phase 2 evaluation inputs.

Safe command: ``python -m v2_phase6_evaluation.preflight``
This does not fit a model, predict, score the holdout, or write output files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .protocol import PROJECT_ROOT, verify_phase2_protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root containing Network_Data, data_schema, and v2_phase2_evaluation.",
    )
    args = parser.parse_args()
    evidence = verify_phase2_protocol(args.root)
    result = {
        "phase": 6,
        "purpose": "read-only Phase 2 protocol preflight",
        "project_root": str(evidence.root),
        "status": evidence.summary["status"],
        "hashes": evidence.hashes,
        **evidence.summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
