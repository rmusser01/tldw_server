"""Export helpers for traceable workspace artifact versions."""
from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from html import escape
from typing import Any

import markdown

from tldw_Server_API.app.core.exceptions import WorkspaceArtifactExportStateError

ALLOWED_WORKSPACE_ARTIFACT_EXPORT_FORMATS = ("md", "html", "json")


def _utc_now_iso() -> str:
    """Return a second-precision UTC timestamp for deterministic export metadata."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _artifact_identity(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Build the stable artifact identity block shared by all export formats."""
    artifact_id = str(artifact.get("id") or artifact.get("artifact_id") or "")
    version = int(artifact.get("version") or 1)
    return {
        "id": artifact_id,
        "workspace_id": artifact.get("workspace_id"),
        "artifact_type": artifact.get("artifact_type"),
        "title": artifact.get("title"),
        "root_artifact_id": artifact.get("root_artifact_id") or artifact_id,
        "artifact_version_id": artifact.get("artifact_version_id") or f"{artifact_id}:v{version}",
        "previous_version_id": artifact.get("previous_version_id"),
        "content_type": artifact.get("content_type") or "text/markdown",
        "schema_version": artifact.get("schema_version") or 1,
        "version": version,
    }


def _export_metadata(artifact: Mapping[str, Any], *, export_format: str, generated_at: str) -> dict[str, Any]:
    """Collect traceability metadata that travels with an artifact export."""
    return {
        "artifact": _artifact_identity(artifact),
        "review_state": artifact.get("review_state") or "draft",
        "producer_metadata": artifact.get("producer_metadata") or {},
        "source_lineage": artifact.get("source_lineage") or {},
        "review_metadata": artifact.get("review_metadata") or {},
        "version_metadata": artifact.get("version_metadata") or {},
        "redaction": artifact.get("redaction") or {"support_safe": True, "redacted": False},
        "export": {
            "format": export_format,
            "generated_at": generated_at,
        },
    }


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    """Serialize export metadata with stable ordering for reproducible payloads."""
    return json.dumps(metadata, ensure_ascii=True, sort_keys=True)


def _metadata_base64(metadata: Mapping[str, Any]) -> str:
    """Encode metadata for HTML-comment-safe Markdown embedding."""
    return base64.b64encode(_metadata_json(metadata).encode("utf-8")).decode("ascii")


def _render_markdown_body_html(body: Any) -> str:
    """Render artifact Markdown to HTML while escaping raw inline HTML."""
    safe_markdown = escape(str(body or ""))
    return markdown.markdown(safe_markdown, extensions=["extra", "sane_lists"], output_format="html5")


def _render_markdown_export(artifact: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Render an accepted artifact version as Markdown with trace metadata."""
    identity = metadata["artifact"]
    body = artifact.get("content") or ""
    return (
        "---\n"
        f"artifact_id: {identity['id']}\n"
        f"artifact_version_id: {identity['artifact_version_id']}\n"
        f"workspace_id: {identity['workspace_id']}\n"
        f"review_state: {metadata['review_state']}\n"
        f"format: {metadata['export']['format']}\n"
        f"generated_at: {metadata['export']['generated_at']}\n"
        "---\n\n"
        "<!-- tldw-artifact-metadata-base64: "
        f"{_metadata_base64(metadata)}"
        " -->\n\n"
        f"{body}"
    )


def _render_html_export(artifact: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Render an accepted artifact version as standalone HTML."""
    identity = metadata["artifact"]
    body = artifact.get("content") or ""
    metadata_json = _metadata_json(metadata)
    script_metadata_json = (
        metadata_json
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )
    title = identity.get("title") or identity["id"]
    return (
        "<!doctype html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        f'  <meta name="tldw-artifact-id" content="{escape(str(identity["id"]), quote=True)}">\n'
        f'  <meta name="tldw-artifact-version-id" content="{escape(str(identity["artifact_version_id"]), quote=True)}">\n'
        f"  <title>{escape(str(title))}</title>\n"
        "</head>\n"
        "<body>\n"
        "  <article"
        f' data-artifact-id="{escape(str(identity["id"]), quote=True)}"'
        f' data-artifact-version-id="{escape(str(identity["artifact_version_id"]), quote=True)}"'
        f' data-workspace-id="{escape(str(identity["workspace_id"]), quote=True)}"'
        f' data-review-state="{escape(str(metadata["review_state"]), quote=True)}"'
        ">\n"
        f"    <h1>{escape(str(title))}</h1>\n"
        f"    <section class=\"artifact-content\">\n{_render_markdown_body_html(body)}\n    </section>\n"
        "  </article>\n"
        '  <script type="application/json" data-tldw-artifact-metadata>'
        f"{script_metadata_json}"
        "</script>\n"
        "</body>\n"
        "</html>\n"
    )


def _render_json_export(artifact: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    """Render an accepted artifact version as a structured JSON document."""
    return json.dumps(
        {
            "artifact": metadata["artifact"],
            "metadata": metadata,
            "content": artifact.get("content") or "",
        },
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    )


def export_workspace_artifact_version(
    artifact: Mapping[str, Any],
    *,
    export_format: str,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Render an accepted workspace artifact version into an export payload."""
    if export_format not in ALLOWED_WORKSPACE_ARTIFACT_EXPORT_FORMATS:
        raise ValueError(f"Unsupported workspace artifact export format '{export_format}'.")

    review_state = str(artifact.get("review_state") or "draft")
    if review_state != "accepted":
        raise WorkspaceArtifactExportStateError("workspace_artifact_not_accepted")

    generated_at = generated_at or _utc_now_iso()
    metadata = _export_metadata(artifact, export_format=export_format, generated_at=generated_at)
    content_by_format = {
        "md": _render_markdown_export,
        "html": _render_html_export,
        "json": _render_json_export,
    }
    content = content_by_format[export_format](artifact, metadata)
    content_type = {
        "md": "text/markdown",
        "html": "text/html",
        "json": "application/json",
    }[export_format]
    encoded_length = len(content.encode("utf-8"))
    identity = metadata["artifact"]
    export_ref = {
        "source": "workspace_artifact_export",
        "format": export_format,
        "workspace_id": identity["workspace_id"],
        "artifact_id": identity["id"],
        "artifact_version_id": identity["artifact_version_id"],
        "content_type": content_type,
        "bytes": encoded_length,
        "exported_at": generated_at,
    }
    return {
        "workspace_id": identity["workspace_id"],
        "artifact_id": identity["id"],
        "artifact_version_id": identity["artifact_version_id"],
        "review_state": review_state,
        "format": export_format,
        "content_type": content_type,
        "content": content,
        "bytes": encoded_length,
        "metadata": metadata,
        "export_ref": export_ref,
        "generated_at": generated_at,
    }
