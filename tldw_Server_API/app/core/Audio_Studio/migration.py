"""Legacy Audiobook Studio migration helpers."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from tldw_Server_API.app.core.Audio_Studio.security import sanitize_audio_studio_payload


MIGRATION_NAMESPACE = "audio_studio_migration:audiobook_commit"


@dataclass(frozen=True)
class AudioStudioMigrationPreviewResult:
    """Preview counts for a sanitized legacy Audiobook Studio payload."""

    preview_id: str
    fingerprint: str
    workflow: str
    project_count: int
    section_count: int
    audio_reference_count: int
    needs_regeneration_count: int
    warnings: list[str]


@dataclass(frozen=True)
class AudioStudioMigrationCommitResult:
    """Commit result for a legacy Audiobook Studio import."""

    project: Any
    imported_section_count: int
    audio_reference_count: int
    needs_regeneration_count: int
    fingerprint: str
    replayed: bool


def preview_audio_studio_audiobook_migration(
    *,
    project_payload: dict[str, Any],
    legacy_project_id: str | None = None,
    user_id: str | int | None = None,
) -> AudioStudioMigrationPreviewResult:
    """Return import counts without writing any Audio Studio project records."""

    normalized = normalize_legacy_audiobook_payload(
        project_payload=project_payload,
        legacy_project_id=legacy_project_id,
        user_id=user_id,
    )
    return AudioStudioMigrationPreviewResult(
        preview_id=f"migprev_{normalized['fingerprint'][:16]}",
        fingerprint=normalized["fingerprint"],
        workflow="narration",
        project_count=1,
        section_count=len(normalized["chapters"]),
        audio_reference_count=sum(1 for chapter in normalized["chapters"] if chapter.get("audio_upload_ref")),
        needs_regeneration_count=sum(1 for chapter in normalized["chapters"] if not chapter.get("audio_upload_ref")),
        warnings=normalized["warnings"],
    )


def commit_audio_studio_audiobook_migration(
    *,
    collections_db: Any,
    project_payload: dict[str, Any],
    legacy_project_id: str | None = None,
    idempotency_key: str | None = None,
    user_id: str | int | None = None,
) -> AudioStudioMigrationCommitResult:
    """Create a narration Audio Studio project from sanitized legacy Dexie data."""

    normalized = normalize_legacy_audiobook_payload(
        project_payload=project_payload,
        legacy_project_id=legacy_project_id,
        user_id=user_id or getattr(collections_db, "user_id", None),
    )
    key = str(idempotency_key or normalized["fingerprint"]).strip()
    if not key:
        key = normalized["fingerprint"]
    request_hash = _sha256_json(normalized["request_fingerprint_payload"])
    existing = collections_db.get_audio_studio_idempotency_record(MIGRATION_NAMESPACE, key)
    if existing is not None:
        if existing.request_hash != request_hash:
            raise ValueError("audio_studio_idempotency_conflict")
        response = _parse_json_object(existing.response_json)
        project_id = str(response.get("project_id") or "")
        if not project_id:
            raise ValueError("audio_studio_migration_response_invalid")
        project = collections_db.get_audio_studio_project_by_project_id(project_id, include_archived=True)
        return AudioStudioMigrationCommitResult(
            project=project,
            imported_section_count=int(response.get("imported_section_count") or 0),
            audio_reference_count=int(response.get("audio_reference_count") or 0),
            needs_regeneration_count=int(response.get("needs_regeneration_count") or 0),
            fingerprint=normalized["fingerprint"],
            replayed=True,
        )

    project_id = f"ast_mig_{normalized['fingerprint'][:12]}"
    revision_id = f"rev_mig_{normalized['fingerprint'][:12]}"
    project = collections_db.create_audio_studio_project(
        project_id=project_id,
        title=normalized["title"],
        workflow="narration",
        revision_id=revision_id,
        mutation_kind="migration.audiobook.create",
        resource_kind="migration",
        resource_id=normalized["legacy_project_id"],
        content_hash=request_hash,
        payload_json=_json_dumps(normalized["request_fingerprint_payload"]),
        settings_json=_json_dumps(
            {
                "settings": {
                    "voice": normalized["voice"],
                    "speed": normalized["speed"],
                    "migration": {
                        "source": "legacy_audiobook_dexie",
                        "legacy_project_id": normalized["legacy_project_id"],
                        "fingerprint": normalized["fingerprint"],
                    },
                },
                "metadata": normalized["metadata"],
                "description": normalized["description"],
            }
        ),
    )
    current_revision = project.current_revision_id
    track_id = "trk_narration"
    track_revision = f"rev_mig_{normalized['fingerprint'][:8]}_track"
    track = collections_db.upsert_audio_studio_track(
        project_row_id=project.id,
        track_id=track_id,
        base_revision_id=current_revision,
        revision_id=track_revision,
        name="Narration",
        kind="speech",
        order_index=0,
        muted=False,
        solo=False,
        volume=1.0,
        settings_json=_json_dumps({"settings": {}, "metadata": {"migration": True}}),
        content_hash=_sha256_json({"track_id": track_id, "kind": "speech"}),
        payload_json=_json_dumps({"track_id": track_id, "kind": "speech"}),
    )
    current_revision = track.current_revision_id
    audio_reference_count = 0
    for index, chapter in enumerate(normalized["chapters"]):
        section_id = f"sec_{_safe_id(chapter['id'] or str(index + 1))}"
        section_revision = f"rev_mig_{normalized['fingerprint'][:8]}_sec_{index:04d}"
        section = collections_db.upsert_audio_studio_section(
            project_row_id=project.id,
            section_id=section_id,
            base_revision_id=current_revision,
            revision_id=section_revision,
            workflow="narration",
            title=chapter["title"],
            body_text=chapter["text"],
            speaker_id=None,
            order_index=index,
            settings_json=_json_dumps(
                {
                    "settings": {
                        "voice": chapter.get("voice") or normalized["voice"],
                        "speed": chapter.get("speed") or normalized["speed"],
                    },
                    "metadata": {
                        "legacy_chapter_id": chapter["id"],
                        "migration": True,
                    },
                }
            ),
            content_hash=_sha256_json(chapter),
            payload_json=_json_dumps(chapter),
        )
        current_revision = section.current_revision_id
        artifact_id = None
        if chapter.get("audio_upload_ref"):
            audio_reference_count += 1
            artifact_id = f"art_mig_{normalized['fingerprint'][:8]}_{index:04d}"
            collections_db.create_audio_studio_artifact(
                project_row_id=project.id,
                artifact_id=artifact_id,
                artifact_type="clip_audio",
                provider="legacy_audiobook_migration",
                output_id=None,
                storage_path=None,
                mime_type=chapter.get("audio_mime_type") or "audio/mpeg",
                size_bytes=chapter.get("audio_size_bytes"),
                source_resource_kind="section",
                source_resource_id=section_id,
                source_revision_id=current_revision,
                content_hash=chapter.get("audio_sha256") or _sha256_json({"upload_ref": chapter["audio_upload_ref"]}),
                metadata_json=_json_dumps(
                    {
                        "upload_ref": chapter["audio_upload_ref"],
                        "legacy_chapter_id": chapter["id"],
                        "source": "legacy_audiobook_migration",
                    }
                ),
            )
        clip_revision = f"rev_mig_{normalized['fingerprint'][:8]}_clip_{index:04d}"
        clip_payload = {
            "section_id": section_id,
            "track_id": track_id,
            "artifact_id": artifact_id,
            "migration": True,
        }
        clip = collections_db.upsert_audio_studio_clip(
            project_row_id=project.id,
            clip_id=f"clip_{section_id}",
            base_revision_id=current_revision,
            revision_id=clip_revision,
            section_id=section_id,
            track_id=track_id,
            title=chapter["title"],
            clip_type="speech",
            start_ms=0,
            duration_ms=None,
            volume=1.0,
            fade_in_ms=0,
            fade_out_ms=0,
            muted=False,
            artifact_id=artifact_id,
            settings_json=_json_dumps({"settings": {}, "metadata": {"migration": True}}),
            content_hash=_sha256_json(clip_payload),
            payload_json=_json_dumps(clip_payload),
        )
        current_revision = clip.current_revision_id

    final_project = collections_db.get_audio_studio_project(project.id, include_archived=True)
    needs_regeneration_count = sum(1 for chapter in normalized["chapters"] if not chapter.get("audio_upload_ref"))
    response_payload = {
        "project_id": final_project.project_id,
        "imported_section_count": len(normalized["chapters"]),
        "audio_reference_count": audio_reference_count,
        "needs_regeneration_count": needs_regeneration_count,
        "fingerprint": normalized["fingerprint"],
    }
    collections_db.put_audio_studio_idempotency_record(
        namespace=MIGRATION_NAMESPACE,
        key=key,
        project_row_id=final_project.id,
        request_hash=request_hash,
        response_json=_json_dumps(response_payload),
    )
    return AudioStudioMigrationCommitResult(
        project=final_project,
        imported_section_count=len(normalized["chapters"]),
        audio_reference_count=audio_reference_count,
        needs_regeneration_count=needs_regeneration_count,
        fingerprint=normalized["fingerprint"],
        replayed=False,
    )


def normalize_legacy_audiobook_payload(
    *,
    project_payload: dict[str, Any],
    legacy_project_id: str | None = None,
    user_id: str | int | None = None,
) -> dict[str, Any]:
    """Normalize and validate a legacy Audiobook Studio Dexie project payload."""

    if not isinstance(project_payload, dict) or not project_payload:
        raise ValueError("legacy_audiobook_payload_invalid")
    sanitized = sanitize_audio_studio_payload(project_payload)
    if sanitized != project_payload:
        raise ValueError("legacy_audiobook_payload_contains_forbidden_fields")
    legacy_id = _clean_text(legacy_project_id or project_payload.get("id") or project_payload.get("project_id"), 200)
    if not legacy_id:
        raise ValueError("legacy_audiobook_project_id_required")
    title = _clean_text(project_payload.get("title") or project_payload.get("name"), 200) or "Imported Audiobook"
    description = _clean_text(project_payload.get("description"), 2000)
    chapters_raw = project_payload.get("chapters")
    if chapters_raw is None:
        chapters_raw = project_payload.get("sections")
    if not isinstance(chapters_raw, list) or not chapters_raw:
        raise ValueError("legacy_audiobook_chapters_required")
    chapters: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, chapter in enumerate(chapters_raw):
        if not isinstance(chapter, dict):
            warnings.append(f"chapter_{index}_skipped_invalid")
            continue
        chapter_id = _clean_text(chapter.get("id") or chapter.get("chapter_id") or f"chapter-{index + 1}", 120)
        chapter_text = _clean_text(chapter.get("text") or chapter.get("body_text") or chapter.get("content"), None)
        title_value = _clean_text(chapter.get("title") or f"Chapter {index + 1}", 200) or f"Chapter {index + 1}"
        upload_ref = _clean_upload_ref(
            chapter.get("audio_upload_ref")
            or chapter.get("upload_ref")
            or chapter.get("audio_ref")
        )
        audio_sha256 = _clean_sha256(chapter.get("audio_sha256") or chapter.get("checksum"))
        chapters.append(
            {
                "id": chapter_id,
                "title": title_value,
                "text": chapter_text or "",
                "voice": _clean_text(chapter.get("voice") or chapter.get("voice_id"), 120),
                "speed": _clean_float(chapter.get("speed")),
                "audio_upload_ref": upload_ref,
                "audio_sha256": audio_sha256,
                "audio_mime_type": _clean_text(chapter.get("audio_mime_type") or chapter.get("mime_type"), 120),
                "audio_size_bytes": _clean_int(chapter.get("audio_size_bytes") or chapter.get("size_bytes")),
            }
        )
    if not chapters:
        raise ValueError("legacy_audiobook_chapters_required")
    metadata = {
        "legacy_project_id": legacy_id,
        "updated_at": _clean_text(project_payload.get("updated_at") or project_payload.get("updatedAt"), 120),
    }
    fingerprint_payload = {
        "user_id": str(user_id or ""),
        "legacy_project_id": legacy_id,
        "title": title,
        "description": description,
        "voice": _clean_text(project_payload.get("voice") or project_payload.get("default_voice"), 120),
        "speed": _clean_float(project_payload.get("speed") or project_payload.get("default_speed")),
        "updated_at": metadata["updated_at"],
        "chapters": chapters,
    }
    fingerprint = _sha256_json(fingerprint_payload)
    return {
        "legacy_project_id": legacy_id,
        "title": title,
        "description": description,
        "voice": fingerprint_payload["voice"],
        "speed": fingerprint_payload["speed"],
        "chapters": chapters,
        "metadata": metadata,
        "warnings": warnings,
        "fingerprint": fingerprint,
        "request_fingerprint_payload": fingerprint_payload,
    }


def _clean_text(value: Any, max_length: int | None) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\x00", "").strip()
    if max_length is not None:
        text = text[:max_length]
    return text or None


def _clean_upload_ref(value: Any) -> str | None:
    text = _clean_text(value, 200)
    if not text:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", text):
        raise ValueError("legacy_audiobook_upload_ref_invalid")
    return text


def _clean_sha256(value: Any) -> str | None:
    text = _clean_text(value, 128)
    if not text:
        return None
    if not re.fullmatch(r"[A-Fa-f0-9]{64}", text):
        return None
    return text.lower()


def _clean_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _clean_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _safe_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return safe[:80] or "chapter"


def _parse_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dumps(payload).encode("utf-8")).hexdigest()


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
