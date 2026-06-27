from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def build_openapi_export_command(output_path: Path) -> list[str]:
    return [
        "python",
        "-m",
        "Helper_Scripts.cats_fuzz.openapi_export",
        "--output",
        str(output_path),
    ]


def export_openapi(output_path: Path) -> str:
    from tldw_Server_API.app.main import app

    app.openapi_schema = None
    openapi_schema = app.openapi()
    json_bytes = json.dumps(
        openapi_schema,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(json_bytes)
    return hashlib.sha256(json_bytes).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export deterministic OpenAPI JSON.")
    parser.add_argument("--output", required=True, type=Path, help="Output OpenAPI JSON path.")
    args = parser.parse_args(argv)

    digest = export_openapi(args.output)
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_openapi_export_command", "export_openapi", "main"]
