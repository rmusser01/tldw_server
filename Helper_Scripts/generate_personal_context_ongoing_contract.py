"""Generate the vendorable Personal Context ongoing-sync schema and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldw_Server_API.app.core.Sync.v2.personal_context_ongoing_contract import (  # noqa: E402
    export_personal_context_ongoing_contract,
)

CONTRACT_NAME = "personal-context-ongoing-v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "tldw_Server_API/app/core/Sync/v2/contracts"
)


def _canonical_json_bytes(value: object) -> bytes:
    """Return stable UTF-8 JSON suitable for exact client vendoring."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def generate_contract_artifacts(*, source_commit: str, output_dir: Path) -> tuple[Path, Path]:
    """Write the versioned schema and a digest-bound provenance manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    schema_bytes = _canonical_json_bytes(export_personal_context_ongoing_contract())
    schema_path = output_dir / f"{CONTRACT_NAME}.schema.json"
    manifest_path = output_dir / f"{CONTRACT_NAME}.manifest.json"
    manifest = {
        "contract": CONTRACT_NAME,
        "schema_version": 1,
        "server_source_commit": source_commit,
        "sha256": f"sha256:{hashlib.sha256(schema_bytes).hexdigest()}",
    }
    schema_path.write_bytes(schema_bytes)
    manifest_path.write_bytes(_canonical_json_bytes(manifest))
    return schema_path, manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Personal Context ongoing-sync contract artifacts."
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """Generate contract artifacts from the current checked-out source."""

    args = _parse_args()
    generate_contract_artifacts(
        source_commit=args.source_commit,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
