"""Export helpers for Audio Studio packages."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Audio_Studio.render import (
    audio_studio_artifact_manifest_entry,
    _is_url_storage_path,
    load_pinned_audio_studio_artifacts,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths


_EXPORT_AUDIO_ARTIFACT_TYPES = {
    "clip_audio",
    "generated_audio",
    "normalized_audio",
    "preview_mix",
    "final_mix",
    "alternate_format",
}
_SUPPORTED_EXPORT_TYPES = {"single_audio", "zip_package", "narration_package", "package"}


@dataclass(frozen=True)
class AudioStudioExportPackageResult:
    """Package file emitted for an Audio Studio export."""

    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    content_hash: str
    size_bytes: int
    mime_type: str
    export_type: str


@dataclass(frozen=True)
class AudioStudioRecordedExport:
    """Artifact ids recorded for an export."""

    package_artifact_id: str
    manifest_artifact_id: str


def create_audio_studio_export_manifest(
    *,
    collections_db: Any,
    project: Any,
    export_id: str,
    export_type: str,
    target_revision_id: str,
    artifact_refs: list[dict[str, Any] | str],
    source_render_id: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a secret-free export manifest with source artifact provenance."""

    if str(getattr(project, "current_revision_id", "")) != str(target_revision_id):
        raise ValueError("stale_target_revision")
    export_type = _normalize_export_type(export_type)
    source_artifacts = _resolve_export_artifacts(
        collections_db=collections_db,
        project=project,
        artifact_refs=artifact_refs,
        source_render_id=source_render_id,
    )
    return {
        "schema_version": 1,
        "kind": "audio_studio_export_manifest",
        "project_id": project.project_id,
        "project_title": project.title,
        "workflow": project.workflow,
        "export_id": export_id,
        "export_type": export_type,
        "target_revision_id": target_revision_id,
        "source_render_id": source_render_id,
        "settings": settings or {},
        "source_artifacts": [audio_studio_artifact_manifest_entry(item.row) for item in source_artifacts],
    }


def package_audio_studio_export(
    *,
    manifest: dict[str, Any],
    source_artifacts: list[Any],
    export_type: str,
    output_dir: str | Path,
    collections_db: Any | None = None,
) -> AudioStudioExportPackageResult:
    """Create one of the MVP export package shapes."""

    export_type = _normalize_export_type(export_type)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    export_id = _safe_name(str(manifest.get("export_id") or "export"))
    manifest_path = output_base / f"{export_id}.manifest.json"
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")

    if export_type == "single_audio":
        if len(source_artifacts) != 1:
            raise ValueError("single_audio_export_requires_one_artifact")
        source_path = _artifact_path_from_row(source_artifacts[0], collections_db=collections_db)
        package_path = output_base / f"{export_id}{source_path.suffix or '.audio'}"
        shutil.copyfile(source_path, package_path)
        mime_type = str(getattr(source_artifacts[0], "mime_type", "") or "application/octet-stream")
    else:
        package_path = output_base / f"{export_id}.zip"
        with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(manifest_path, "manifest.json")
            for artifact in source_artifacts:
                source_path = _artifact_path_from_row(artifact, collections_db=collections_db)
                archive.write(source_path, f"audio/{_safe_name(artifact.artifact_id)}{source_path.suffix}")
            if export_type == "narration_package":
                archive.writestr("audiobook.json", _json_dumps(_narration_package_descriptor(manifest)))
        mime_type = "application/zip"

    content = package_path.read_bytes()
    return AudioStudioExportPackageResult(
        path=package_path,
        manifest_path=manifest_path,
        manifest={
            **manifest,
            "output": {
                "mime_type": mime_type,
                "size_bytes": len(content),
                "content_hash": hashlib.sha256(content).hexdigest(),
            },
        },
        content_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        mime_type=mime_type,
        export_type=export_type,
    )


def record_audio_studio_export_artifact(
    *,
    collections_db: Any,
    project: Any,
    manifest: dict[str, Any],
    package_result: AudioStudioExportPackageResult,
    artifact_id_prefix: str | None = None,
) -> AudioStudioRecordedExport:
    """Record export package and manifest artifacts separately from source artifacts."""

    export_id = str(manifest.get("export_id") or "export")
    artifact_prefix = _safe_name(artifact_id_prefix or export_id)
    export_name = _safe_name(export_id)
    package_artifact_id = f"art_{artifact_prefix}_{export_name}_package"
    manifest_artifact_id = f"art_{artifact_prefix}_{export_name}_manifest"
    metadata = {
        "export_id": export_id,
        "export_type": package_result.export_type,
        "target_revision_id": manifest.get("target_revision_id"),
        "source_render_id": manifest.get("source_render_id"),
        "source_artifacts": manifest.get("source_artifacts") or [],
        "source": "audio_studio_export",
    }
    collections_db.create_audio_studio_artifact(
        project_row_id=project.id,
        artifact_id=package_artifact_id,
        artifact_type="package",
        provider="audio_studio",
        output_id=None,
        storage_path=str(package_result.path),
        mime_type=package_result.mime_type,
        size_bytes=package_result.size_bytes,
        source_resource_kind="export",
        source_resource_id=export_id,
        source_revision_id=str(manifest.get("target_revision_id") or ""),
        content_hash=package_result.content_hash,
        metadata_json=_json_dumps(metadata),
    )
    manifest_payload = _json_dumps(package_result.manifest).encode("utf-8")
    package_result.manifest_path.write_text(_json_dumps(package_result.manifest), encoding="utf-8")
    collections_db.create_audio_studio_artifact(
        project_row_id=project.id,
        artifact_id=manifest_artifact_id,
        artifact_type="export_manifest",
        provider="audio_studio",
        output_id=None,
        storage_path=str(package_result.manifest_path),
        mime_type="application/json",
        size_bytes=len(manifest_payload),
        source_resource_kind="export",
        source_resource_id=export_id,
        source_revision_id=str(manifest.get("target_revision_id") or ""),
        content_hash=hashlib.sha256(manifest_payload).hexdigest(),
        metadata_json=_json_dumps(metadata),
    )
    return AudioStudioRecordedExport(
        package_artifact_id=package_artifact_id,
        manifest_artifact_id=manifest_artifact_id,
    )


def resolve_audio_studio_export_artifact_rows(
    *,
    collections_db: Any,
    project: Any,
    artifact_refs: list[dict[str, Any] | str],
    source_render_id: str | None = None,
) -> list[Any]:
    """Return rows for export packaging after the same provenance checks as manifest creation."""

    return [
        item.row
        for item in _resolve_export_artifacts(
            collections_db=collections_db,
            project=project,
            artifact_refs=artifact_refs,
            source_render_id=source_render_id,
        )
    ]


def _resolve_export_artifacts(
    *,
    collections_db: Any,
    project: Any,
    artifact_refs: list[dict[str, Any] | str],
    source_render_id: str | None,
):
    refs = list(artifact_refs or [])
    if not refs and source_render_id:
        rows = collections_db.list_audio_studio_artifacts(project_row_id=project.id, limit=500)
        refs = [
            {
                "artifact_id": row.artifact_id,
                "source_revision_id": row.source_revision_id,
                "content_hash": row.content_hash,
            }
            for row in rows
            if row.artifact_type in {"preview_mix", "final_mix"}
            and _metadata_value(row, "render_id") == source_render_id
        ]
    if not refs:
        raise ValueError("audio_studio_export_requires_artifacts")
    return load_pinned_audio_studio_artifacts(
        collections_db=collections_db,
        project=project,
        artifact_refs=refs,
        allowed_artifact_types=_EXPORT_AUDIO_ARTIFACT_TYPES,
    )


def _artifact_path_from_row(artifact: Any, *, collections_db: Any | None) -> Path:
    storage_path = str(getattr(artifact, "storage_path", "") or "")
    if not storage_path:
        raise ValueError("audio_studio_artifact_file_not_available")
    if _is_url_storage_path(storage_path):
        raise ValueError("invalid_audio_studio_artifact_storage_path")
    path = Path(storage_path)
    if ".." in path.parts:
        raise ValueError("invalid_audio_studio_artifact_storage_path")
    if not path.is_absolute() and collections_db is not None:
        path = DatabasePaths.get_user_outputs_dir(int(collections_db.user_id)) / path
    if not path.exists():
        raise ValueError("audio_studio_artifact_file_not_found")
    return path


def _metadata_value(row: Any, key: str) -> Any:
    try:
        parsed = json.loads(row.metadata_json or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        parsed = {}
    return parsed.get(key) if isinstance(parsed, dict) else None


def _narration_package_descriptor(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "audio_studio_narration_package",
        "project_id": manifest.get("project_id"),
        "title": manifest.get("project_title"),
        "source_artifacts": manifest.get("source_artifacts") or [],
    }


def _normalize_export_type(value: str) -> str:
    export_type = str(value or "").strip().lower()
    if export_type == "package":
        export_type = "zip_package"
    if export_type not in _SUPPORTED_EXPORT_TYPES:
        raise ValueError("unsupported_audio_studio_export_type")
    return export_type


def _safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip()]
    name = "".join(chars).strip("_")
    return name[:120] or "audio_studio"


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
