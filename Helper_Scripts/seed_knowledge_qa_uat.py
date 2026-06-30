"""Seed deterministic Knowledge QA live-UAT sources through public APIs.

This helper is intentionally inert unless called directly. It creates a small
personal-library fixture for live browser tests and writes a manifest that those
tests can use for exact source IDs and expected phrases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tldw_Server_API.tests.RAG.knowledge_qa_uat_fixtures import (
    FIXTURE_SOURCES,
    KnowledgeQaUatSource,
    build_fixture_manifest,
)


DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_MANIFEST_FILENAME = "knowledge-qa-uat.json"
DEFAULT_TIMEOUT_SECONDS = 30.0


class KnowledgeQaUatSeedError(RuntimeError):
    """Raised when live Knowledge QA UAT seeding cannot complete."""


def build_dry_run_manifest() -> dict[str, Any]:
    """Return the deterministic fixture manifest without touching a backend."""

    return build_fixture_manifest()


def _resolve_server_url(value: str | None) -> str:
    server_url = (
        value
        or os.getenv("TLDW_SERVER_URL")
        or os.getenv("TLDW_E2E_SERVER_URL")
        or DEFAULT_SERVER_URL
    ).strip()
    if not server_url:
        raise KnowledgeQaUatSeedError("A non-empty server URL is required.")
    if not server_url.startswith(("http://", "https://")):
        server_url = f"http://{server_url}"
    return server_url.rstrip("/")


def _resolve_api_key(value: str | None) -> str:
    api_key = (
        value
        or os.getenv("TLDW_API_KEY")
        or os.getenv("TLDW_E2E_API_KEY")
        or os.getenv("SINGLE_USER_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise KnowledgeQaUatSeedError(
            "Set --api-key, TLDW_API_KEY, TLDW_E2E_API_KEY, or SINGLE_USER_API_KEY."
        )
    return api_key


def _resolve_manifest_path(value: str | None) -> Path:
    default_manifest_path = Path(tempfile.gettempdir()) / DEFAULT_MANIFEST_FILENAME
    return Path(
        value
        or os.getenv("TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST")
        or default_manifest_path
    ).expanduser()


def _write_manifest(manifest: dict[str, Any], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _extract_media_id(payload: Any, source: KnowledgeQaUatSource) -> int:
    if not isinstance(payload, dict):
        raise KnowledgeQaUatSeedError(
            f"POST /api/v1/media/add for {source.key} returned a non-object payload."
        )

    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise KnowledgeQaUatSeedError(
            f"POST /api/v1/media/add for {source.key} returned no results: {payload!r}"
        )

    first = results[0]
    if not isinstance(first, dict):
        raise KnowledgeQaUatSeedError(
            f"POST /api/v1/media/add for {source.key} returned an invalid result: {first!r}"
        )

    status = str(first.get("status") or "").lower()
    if status not in {"success", "warning", "skipped"}:
        raise KnowledgeQaUatSeedError(
            f"POST /api/v1/media/add for {source.key} failed: {first!r}"
        )

    db_id = first.get("db_id")
    if isinstance(db_id, int):
        return db_id
    if isinstance(db_id, str) and db_id.isdigit():
        return int(db_id)

    message = str(first.get("message") or first.get("db_message") or "")
    marker = "ID:"
    if marker in message:
        candidate = message.split(marker, 1)[1].split(")", 1)[0].strip()
        if candidate.isdigit():
            return int(candidate)

    raise KnowledgeQaUatSeedError(
        f"POST /api/v1/media/add for {source.key} did not return a usable db_id: {first!r}"
    )


def _extract_note_id(payload: Any, source: KnowledgeQaUatSource) -> str:
    if isinstance(payload, dict) and payload.get("id"):
        return str(payload["id"])
    raise KnowledgeQaUatSeedError(
        f"POST /api/v1/notes/ for {source.key} did not return a usable id: {payload!r}"
    )


def _seed_media_source(client: httpx.Client, source: KnowledgeQaUatSource) -> int:
    response = client.post(
        "/api/v1/media/add",
        data={
            "media_type": "document",
            "title": source.title,
            "author": "Knowledge QA UAT",
            "keywords": f"knowledge-qa-uat,{source.key}",
            "overwrite_existing": "true",
            "perform_analysis": "false",
            "perform_chunking": "true",
            "chunk_method": "words",
            "chunk_size": "300",
            "chunk_overlap": "0",
            "generate_embeddings": "false",
        },
        files={
            "files": (
                f"{source.key}.txt",
                source.body.encode("utf-8"),
                "text/plain",
            )
        },
    )
    _raise_for_seed_status(response, f"POST /api/v1/media/add for {source.key}")
    return _extract_media_id(response.json(), source)


def _seed_note_source(client: httpx.Client, source: KnowledgeQaUatSource) -> str:
    response = client.post(
        "/api/v1/notes/",
        json={
            "title": source.title,
            "content": source.body,
            "keywords": ["knowledge-qa-uat", source.key],
        },
    )
    _raise_for_seed_status(response, f"POST /api/v1/notes/ for {source.key}")
    return _extract_note_id(response.json(), source)


def _raise_for_seed_status(response: httpx.Response, label: str) -> None:
    if response.is_success:
        return

    detail = response.text[:1000]
    raise KnowledgeQaUatSeedError(
        f"{label} returned HTTP {response.status_code}: {detail}"
    )


def seed_knowledge_qa_uat(
    *,
    server_url: str | None = None,
    api_key: str | None = None,
    manifest_path: str | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Create live-UAT sources and write the manifest consumed by browser tests."""

    resolved_server_url = _resolve_server_url(server_url)
    resolved_api_key = _resolve_api_key(api_key)
    resolved_manifest_path = _resolve_manifest_path(manifest_path)
    created_ids: dict[str, str | int] = {}

    with httpx.Client(
        base_url=resolved_server_url,
        headers={"X-API-KEY": resolved_api_key},
        timeout=timeout_seconds,
    ) as client:
        health_response = client.get("/api/v1/health")
        _raise_for_seed_status(health_response, "GET /api/v1/health")

        for source in FIXTURE_SOURCES:
            if source.source_type == "media_db":
                created_ids[source.key] = _seed_media_source(client, source)
            elif source.source_type == "notes":
                created_ids[source.key] = _seed_note_source(client, source)
            else:
                raise KnowledgeQaUatSeedError(
                    f"Unsupported fixture source type {source.source_type!r} for {source.key}."
                )

    manifest = build_fixture_manifest(created_ids)
    _write_manifest(manifest, resolved_manifest_path)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed deterministic Knowledge QA live-UAT sources."
    )
    parser.add_argument("--server-url", help="Backend URL, defaulting to TLDW_SERVER_URL or TLDW_E2E_SERVER_URL.")
    parser.add_argument("--api-key", help="API key, defaulting to TLDW_API_KEY, TLDW_E2E_API_KEY, or SINGLE_USER_API_KEY.")
    parser.add_argument("--manifest", help="Output manifest path, defaulting to TLDW_KNOWLEDGE_QA_FIXTURE_MANIFEST or the system temporary directory.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a manifest with null IDs without calling the backend.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = _resolve_manifest_path(args.manifest)

    if args.dry_run:
        manifest = build_dry_run_manifest()
        _write_manifest(manifest, manifest_path)
    else:
        manifest = seed_knowledge_qa_uat(
            server_url=args.server_url,
            api_key=args.api_key,
            manifest_path=str(manifest_path),
            timeout_seconds=args.timeout,
        )

    print(json.dumps({"manifest": str(manifest_path), "sources": manifest["sources"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
