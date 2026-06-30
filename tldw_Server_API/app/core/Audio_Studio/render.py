"""Render helpers for Audio Studio timeline artifacts."""

from __future__ import annotations

import hashlib
import json
import ntpath
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.TTS.audio_converter import AudioConverter


_RENDERABLE_ARTIFACT_TYPES = {
    "clip_audio",
    "generated_audio",
    "normalized_audio",
    "reference_audio",
    "alternate_format",
}
_MIME_BY_FORMAT = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "m4a": "audio/mp4",
    "m4b": "audio/mp4",
}


def _is_url_storage_path(storage_path: str) -> bool:
    """Return True when a storage pointer is a URL, not a local filesystem path."""

    parsed = urlparse(storage_path)
    if not parsed.scheme and not parsed.netloc:
        return False
    if len(parsed.scheme) == 1 and not parsed.netloc and ntpath.splitdrive(storage_path)[0]:
        return False
    return True


@dataclass(frozen=True)
class AudioStudioResolvedArtifact:
    """Artifact row plus validated local file path for render/export work."""

    row: Any
    path: Path


@dataclass(frozen=True)
class AudioStudioRenderPlan:
    """Validated render plan with revision-pinned source artifacts."""

    project_id: str
    project_row_id: int
    render_id: str
    render_type: str
    target_revision_id: str
    clip_artifact_ids: list[str]
    output_format: str
    loudness_normalize: bool
    source_artifacts: list[AudioStudioResolvedArtifact]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class AudioStudioRenderResult:
    """Files created by a render operation."""

    path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    content_hash: str
    size_bytes: int
    mime_type: str


@dataclass(frozen=True)
class AudioStudioRecordedRender:
    """Artifact ids recorded for a render."""

    mix_artifact_id: str
    manifest_artifact_id: str


def build_render_plan(
    *,
    collections_db: Any,
    project: Any,
    render_id: str,
    target_revision_id: str,
    artifact_refs: list[dict[str, Any] | str],
    output_format: str = "wav",
    loudness_normalize: bool = False,
    render_type: str = "preview_mix",
) -> AudioStudioRenderPlan:
    """Build a render plan after validating project and artifact provenance."""

    _validate_project_revision(project, target_revision_id)
    output_format = _normalize_audio_format(output_format)
    if render_type not in {"preview_mix", "final_mix"}:
        raise ValueError("unsupported_render_type")
    source_artifacts = load_pinned_audio_studio_artifacts(
        collections_db=collections_db,
        project=project,
        artifact_refs=artifact_refs,
        allowed_artifact_types=_RENDERABLE_ARTIFACT_TYPES,
    )
    if not source_artifacts:
        raise ValueError("audio_studio_render_requires_artifacts")
    manifest = {
        "schema_version": 1,
        "kind": "audio_studio_render_manifest",
        "project_id": project.project_id,
        "render_id": render_id,
        "render_type": render_type,
        "target_revision_id": target_revision_id,
        "output_format": output_format,
        "loudness_normalize": bool(loudness_normalize),
        "source_artifacts": [audio_studio_artifact_manifest_entry(item.row) for item in source_artifacts],
    }
    return AudioStudioRenderPlan(
        project_id=project.project_id,
        project_row_id=project.id,
        render_id=render_id,
        render_type=render_type,
        target_revision_id=target_revision_id,
        clip_artifact_ids=[item.row.artifact_id for item in source_artifacts],
        output_format=output_format,
        loudness_normalize=bool(loudness_normalize),
        source_artifacts=source_artifacts,
        manifest=manifest,
    )


async def render_audio_studio_mix(
    plan: AudioStudioRenderPlan,
    *,
    output_dir: str | Path,
) -> AudioStudioRenderResult:
    """Render a simple MVP mix by concatenating validated source artifacts."""

    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)
    output_path = output_base / f"{_safe_name(plan.render_id)}.{plan.output_format}"
    input_paths = [item.path for item in plan.source_artifacts]
    if (
        plan.output_format == "wav"
        and not plan.loudness_normalize
        and all(path.suffix.lower() == ".wav" for path in input_paths)
    ):
        _concat_wav_files(input_paths, output_path)
    else:
        converted = await AudioConverter.concat_audio_files(
            input_paths,
            output_path.with_suffix(""),
            plan.output_format,
        )
        if not converted:
            raise RuntimeError("audio_studio_render_failed")
        output_path = output_path.with_suffix(f".{plan.output_format}")
        if plan.loudness_normalize:
            normalized_path = output_path.with_name(f"{output_path.stem}_normalized{output_path.suffix}")
            normalized = await AudioConverter.normalize_audio(output_path, normalized_path)
            if not normalized:
                raise RuntimeError("audio_studio_render_normalize_failed")
            output_path = normalized_path
    content = output_path.read_bytes()
    manifest = {
        **plan.manifest,
        "output": {
            "format": plan.output_format,
            "mime_type": _mime_for_format(plan.output_format),
            "size_bytes": len(content),
            "content_hash": hashlib.sha256(content).hexdigest(),
        },
    }
    manifest_path = output_base / f"{_safe_name(plan.render_id)}.manifest.json"
    manifest_path.write_text(_json_dumps(manifest), encoding="utf-8")
    return AudioStudioRenderResult(
        path=output_path,
        manifest_path=manifest_path,
        manifest=manifest,
        content_hash=manifest["output"]["content_hash"],
        size_bytes=len(content),
        mime_type=manifest["output"]["mime_type"],
    )


def record_audio_studio_render_artifact(
    *,
    collections_db: Any,
    project: Any,
    plan: AudioStudioRenderPlan,
    render_result: AudioStudioRenderResult,
    artifact_id_prefix: str | None = None,
) -> AudioStudioRecordedRender:
    """Record render mix and manifest as artifacts distinct from generation outputs."""

    artifact_prefix = _safe_name(artifact_id_prefix or plan.render_id)
    render_name = _safe_name(plan.render_id)
    mix_artifact_id = f"art_{artifact_prefix}_{render_name}_mix"
    manifest_artifact_id = f"art_{artifact_prefix}_{render_name}_manifest"
    common_metadata = {
        "render_id": plan.render_id,
        "render_type": plan.render_type,
        "target_revision_id": plan.target_revision_id,
        "source_artifacts": plan.manifest["source_artifacts"],
        "source": "audio_studio_render",
    }
    collections_db.create_audio_studio_artifact(
        project_row_id=project.id,
        artifact_id=mix_artifact_id,
        artifact_type=plan.render_type,
        provider="audio_studio",
        output_id=None,
        storage_path=str(render_result.path),
        mime_type=render_result.mime_type,
        size_bytes=render_result.size_bytes,
        source_resource_kind="render",
        source_resource_id=plan.render_id,
        source_revision_id=plan.target_revision_id,
        content_hash=render_result.content_hash,
        metadata_json=_json_dumps(common_metadata),
    )
    manifest_bytes = render_result.manifest_path.read_bytes()
    collections_db.create_audio_studio_artifact(
        project_row_id=project.id,
        artifact_id=manifest_artifact_id,
        artifact_type="render_manifest",
        provider="audio_studio",
        output_id=None,
        storage_path=str(render_result.manifest_path),
        mime_type="application/json",
        size_bytes=len(manifest_bytes),
        source_resource_kind="render",
        source_resource_id=plan.render_id,
        source_revision_id=plan.target_revision_id,
        content_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        metadata_json=_json_dumps(common_metadata),
    )
    return AudioStudioRecordedRender(
        mix_artifact_id=mix_artifact_id,
        manifest_artifact_id=manifest_artifact_id,
    )


def load_pinned_audio_studio_artifacts(
    *,
    collections_db: Any,
    project: Any,
    artifact_refs: list[dict[str, Any] | str],
    allowed_artifact_types: set[str] | None = None,
) -> list[AudioStudioResolvedArtifact]:
    """Return project-owned artifacts after checking caller-supplied revision pins."""

    resolved: list[AudioStudioResolvedArtifact] = []
    for ref in artifact_refs:
        artifact_id, expected_revision, expected_hash = _artifact_pin(ref)
        rows = collections_db.list_audio_studio_artifacts(
            project_row_id=project.id,
            limit=1,
            artifact_id=artifact_id,
        )
        if not rows:
            raise ValueError("audio_studio_artifact_not_found")
        row = rows[0]
        if allowed_artifact_types is not None and row.artifact_type not in allowed_artifact_types:
            raise ValueError("unsupported_audio_studio_artifact_type")
        if expected_revision is not None and row.source_revision_id != expected_revision:
            raise ValueError("stale_artifact_revision")
        if expected_hash is not None and row.content_hash != expected_hash:
            raise ValueError("stale_artifact_revision")
        resolved.append(
            AudioStudioResolvedArtifact(
                row=row,
                path=resolve_audio_studio_artifact_path(collections_db, row),
            )
        )
    return resolved


def resolve_audio_studio_artifact_path(collections_db: Any, artifact: Any) -> Path:
    """Resolve an Audio Studio artifact storage pointer to a safe local path."""

    storage_path = str(getattr(artifact, "storage_path", "") or "").strip()
    if not storage_path:
        raise ValueError("audio_studio_artifact_file_not_available")
    if _is_url_storage_path(storage_path):
        raise ValueError("invalid_audio_studio_artifact_storage_path")
    path = Path(storage_path)
    if path.is_absolute():
        resolved = path.resolve(strict=False)
    else:
        if ".." in path.parts:
            raise ValueError("invalid_audio_studio_artifact_storage_path")
        outputs_dir = DatabasePaths.get_user_outputs_dir(int(collections_db.user_id))
        resolved = (outputs_dir / path).resolve(strict=False)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError("audio_studio_artifact_file_not_found")
    return resolved


def audio_studio_artifact_manifest_entry(artifact: Any) -> dict[str, Any]:
    """Return secret-free provenance for a source artifact."""

    metadata = _parse_json_object(getattr(artifact, "metadata_json", None))
    return {
        "artifact_id": artifact.artifact_id,
        "artifact_type": artifact.artifact_type,
        "provider": artifact.provider,
        "mime_type": artifact.mime_type,
        "size_bytes": artifact.size_bytes,
        "source_resource_kind": artifact.source_resource_kind,
        "source_resource_id": artifact.source_resource_id,
        "source_revision_id": artifact.source_revision_id,
        "content_hash": artifact.content_hash,
        "metadata": metadata,
    }


def _validate_project_revision(project: Any, target_revision_id: str) -> None:
    if str(getattr(project, "current_revision_id", "")) != str(target_revision_id):
        raise ValueError("stale_target_revision")


def _artifact_pin(ref: dict[str, Any] | str) -> tuple[str, str | None, str | None]:
    if isinstance(ref, str):
        artifact_id = ref.strip()
        if not artifact_id:
            raise ValueError("audio_studio_artifact_not_found")
        return artifact_id, None, None
    artifact_id = str(ref.get("artifact_id") or "").strip()
    if not artifact_id:
        raise ValueError("audio_studio_artifact_not_found")
    revision = ref.get("source_revision_id")
    content_hash = ref.get("content_hash")
    return (
        artifact_id,
        str(revision).strip() if revision not in (None, "") else None,
        str(content_hash).strip() if content_hash not in (None, "") else None,
    )


def _concat_wav_files(input_paths: list[Path], output_path: Path) -> None:
    if not input_paths:
        raise ValueError("audio_studio_render_requires_artifacts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = None
    with wave.open(str(output_path), "wb") as output_wav:
        for input_path in input_paths:
            with wave.open(str(input_path), "rb") as input_wav:
                current = input_wav.getparams()
                comparable = current[:3]
                if params is None:
                    params = comparable
                    output_wav.setnchannels(current.nchannels)
                    output_wav.setsampwidth(current.sampwidth)
                    output_wav.setframerate(current.framerate)
                elif comparable != params:
                    raise ValueError("audio_studio_wav_params_mismatch")
                output_wav.writeframes(input_wav.readframes(input_wav.getnframes()))


def _normalize_audio_format(value: str) -> str:
    fmt = str(value or "wav").strip().lower().lstrip(".")
    if fmt not in AudioConverter.AUDIO_CODECS:
        raise ValueError("unsupported_audio_studio_output_format")
    return fmt


def _mime_for_format(value: str) -> str:
    return _MIME_BY_FORMAT.get(value, f"audio/{value}")


def _safe_name(value: str) -> str:
    chars = [ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value or "").strip()]
    name = "".join(chars).strip("_")
    return name[:120] or "audio_studio"


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
