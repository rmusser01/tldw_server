#!/usr/bin/env python3
"""Export the FastAPI OpenAPI schema deterministically + a drift fingerprint.

The frontend hand-writes ``page.route()`` mocks and API clients with no link to
the backend contract, so a Pydantic model change can silently break the wired
system while both suites stay green (audits/2026-07-04-test-suite-audit-round2.md
RF1, the #2590 class). This script is the backend half of the drift gate:

* ``--out openapi.json``  — write the full canonical schema (feeds
  ``openapi-typescript`` codegen).
* ``--fingerprint PATH``  — write a tiny fingerprint (sha256 + counts) that is
  checked in; CI recomputes it and fails on mismatch. A field rename changes
  the sha256, so the gate fires.

Determinism: a PINNED canonical environment is applied internally (the app has
env-driven route toggles — MINIMAL_TEST_APP, route enable/disable, _TEST_MODE
middleware — that would otherwise make the schema differ between machines), and
the schema is serialized with ``sort_keys=True``. The version field is
normalized so a package bump does not read as API drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Import THIS checkout's code, not a stray editable install that may point at a
# different worktree. The repo root is two levels up (Helper_Scripts/<this>).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Pinned canonical environment for a complete, reproducible schema. Applied
# BEFORE importing the app so settings/route-toggles resolve identically
# everywhere. Keep this list in sync with any new env-driven route gating.
_CANONICAL_ENV = {
    "AUTH_MODE": "single_user",
    "SINGLE_USER_API_KEY": "openapi-export-fixed-key-0123456789",
    "TEST_MODE": "true",
    "DISABLE_HEAVY_STARTUP": "1",
    # explicitly UNSET the minimizers so every router is registered
    "MINIMAL_TEST_APP": "",
    "ULTRA_MINIMAL_APP": "",
    "ROUTES_ENABLE": "",
    "ROUTES_DISABLE": "",
    # pin the deployment profile so environment-derived toggles are stable
    "ENVIRONMENT": "test",
    "APP_ENV": "test",
}

# The schema version is intentionally excluded from the fingerprint: a package
# version bump is not an API contract change.
_NORMALIZED_VERSION = "0.0.0-fingerprint"


def _apply_canonical_env() -> None:
    for key, value in _CANONICAL_ENV.items():
        if value == "":
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_schema() -> dict:
    _apply_canonical_env()
    # Import only after the env is pinned.
    from tldw_Server_API.app.main import app

    schema = app.openapi()
    # Normalize the version so a bump is not read as drift.
    if isinstance(schema.get("info"), dict):
        schema = {**schema, "info": {**schema["info"], "version": _NORMALIZED_VERSION}}
    return schema


def _canonical_json(schema: dict) -> str:
    return json.dumps(schema, sort_keys=True, separators=(",", ":"))


def _fingerprint(schema: dict) -> dict:
    canonical = _canonical_json(schema)
    return {
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "openapi_version": schema.get("openapi"),
        "path_count": len(schema.get("paths", {})),
        "schema_count": len(schema.get("components", {}).get("schemas", {})),
        "note": (
            "Regenerate with `make openapi-fingerprint`. A change here means the "
            "backend API contract drifted; regenerate the frontend types "
            "(`bun run generate:api-types` in apps/tldw-frontend) and review."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=None, help="Write the full canonical OpenAPI JSON here.")
    parser.add_argument("--fingerprint", default=None, help="Write the drift fingerprint JSON here.")
    parser.add_argument(
        "--check",
        default=None,
        help="Compare a freshly computed fingerprint against this checked-in file; exit 1 on drift.",
    )
    args = parser.parse_args(argv)

    schema = _load_schema()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(schema, fh, sort_keys=True, indent=2)
            fh.write("\n")
        print(f"[openapi-export] wrote schema -> {args.out} "
              f"({len(schema.get('paths', {}))} paths, "
              f"{len(schema.get('components', {}).get('schemas', {}))} schemas)")

    fp = _fingerprint(schema)
    fp_json = json.dumps(fp, indent=2, sort_keys=True) + "\n"

    if args.fingerprint:
        with open(args.fingerprint, "w", encoding="utf-8") as fh:
            fh.write(fp_json)
        print(f"[openapi-export] wrote fingerprint -> {args.fingerprint} (sha256={fp['sha256'][:12]}...)")

    if args.check:
        try:
            with open(args.check, encoding="utf-8") as fh:
                checked_in = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[openapi-export] FAIL — cannot read checked-in fingerprint {args.check}: {exc}", file=sys.stderr)
            return 2
        if checked_in.get("sha256") != fp["sha256"]:
            print(
                "[openapi-export] FAIL — OpenAPI contract drift detected.\n"
                f"  checked-in sha256: {checked_in.get('sha256')}\n"
                f"  current    sha256: {fp['sha256']}\n"
                f"  checked-in counts: paths={checked_in.get('path_count')} schemas={checked_in.get('schema_count')}\n"
                f"  current    counts: paths={fp['path_count']} schemas={fp['schema_count']}\n"
                "  Run `make openapi-fingerprint` to update the snapshot, then regenerate the\n"
                "  frontend types (`bun run generate:api-types` in apps/tldw-frontend) and review.",
                file=sys.stderr,
            )
            return 1
        print("[openapi-export] OK — OpenAPI fingerprint matches the checked-in snapshot.")

    if not any((args.out, args.fingerprint, args.check)):
        # default: print the fingerprint
        sys.stdout.write(fp_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
