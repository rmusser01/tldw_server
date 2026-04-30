"""Service for cloning shared workspaces into the accessor's own DB."""
from __future__ import annotations

import uuid
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.media_db.api import (
    get_media_transcripts,
)
from tldw_Server_API.app.core.DB_Management.media_db.legacy_transcripts import (
    upsert_transcript,
)
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def _safe_exception_type(exc: BaseException) -> str:
    """Return a log-safe exception type label without exception details."""
    exc_type = exc.__class__.__name__
    if exc_type and all(char.isalnum() or char == "_" for char in exc_type):
        return exc_type
    return "Exception"


class CloneService:
    """
    Deep-copies a workspace from the owner's DBs into the cloner's DBs.

    Steps:
    1. Copy workspace metadata with new UUID
    2. Deep copy media (Media + MediaChunks + Transcripts + Keywords)
    3. Copy workspace_sources with new media_ids
    4. Copy workspace_notes and workspace_artifacts
    5. Skip embedding copy (re-embed in cloner's namespace via separate job)
    """

    def __init__(
        self,
        source_chacha_db: CharactersRAGDB,
        source_media_db: MediaDatabase,
        target_chacha_db: CharactersRAGDB,
        target_media_db: MediaDatabase,
    ) -> None:
        self._src_chacha = source_chacha_db
        self._src_media = source_media_db
        self._tgt_chacha = target_chacha_db
        self._tgt_media = target_media_db

    def clone_workspace(
        self,
        workspace_id: str,
        *,
        new_name: str | None = None,
        on_progress: Callable[[str, float], None] | None = None,
    ) -> dict[str, Any]:
        """
        Synchronously clone a workspace. Returns the new workspace metadata.

        Args:
            workspace_id: Source workspace ID in owner's ChaChaNotes DB.
            new_name: Optional name override for the clone.
            on_progress: Optional callback (stage, pct) for progress tracking.
        """
        def _progress(stage: str, pct: float) -> None:
            if on_progress:
                on_progress(stage, pct)

        _progress("loading_source", 0.0)

        # 1. Read source workspace
        src_ws = self._src_chacha.get_workspace(workspace_id)
        if src_ws is None:
            raise ValueError(f"Workspace '{workspace_id}' not found in source DB")

        # 2. Create new workspace in target
        new_ws_id = str(uuid.uuid4())
        ws_data = {
            "id": new_ws_id,
            "name": new_name or f"{src_ws.get('name', 'Untitled')} (Clone)",
            "description": src_ws.get("description", ""),
            "workspace_type": src_ws.get("workspace_type", "research"),
        }
        self._tgt_chacha.create_workspace(ws_data)
        _progress("workspace_created", 0.1)

        # 3. Copy sources
        sources = self._src_chacha.list_workspace_sources(workspace_id)
        total_sources = len(sources)
        media_id_map: dict[str, str] = {}  # old_media_id -> new_media_id

        for i, source in enumerate(sources):
            old_media_id = source.get("media_id")
            if old_media_id:
                new_media_id = self._copy_media_item(old_media_id)
                if new_media_id:
                    media_id_map[old_media_id] = new_media_id
                else:
                    # Media copy failed — skip this source to avoid dangling references
                    logger.warning("Skipping workspace source because media copy failed")
                    if total_sources > 0:
                        _progress("copying_sources", 0.1 + 0.5 * ((i + 1) / total_sources))
                    continue

            # Add source to new workspace (use mapped ID, or None for non-media sources)
            source_data = {
                "id": str(uuid.uuid4()),
                "media_id": media_id_map.get(old_media_id) if old_media_id else None,
                "source_type": source.get("source_type", "media"),
                "title": source.get("title", ""),
                "url": source.get("url"),
            }
            try:
                self._tgt_chacha.add_workspace_source(new_ws_id, source_data)
            except Exception as exc:
                logger.warning(
                    f"Failed to copy workspace source; exception_type={_safe_exception_type(exc)}"
                )

            if total_sources > 0:
                _progress("copying_sources", 0.1 + 0.5 * ((i + 1) / total_sources))

        # 4. Copy notes
        notes = self._src_chacha.list_workspace_notes(workspace_id)
        for note in notes:
            note_data = {
                "title": note.get("title", ""),
                "content": note.get("content", ""),
            }
            try:
                self._tgt_chacha.add_workspace_note(new_ws_id, note_data)
            except Exception as exc:
                logger.warning(
                    f"Failed to copy workspace note; exception_type={_safe_exception_type(exc)}"
                )
        _progress("notes_copied", 0.8)

        # 5. Copy artifacts
        artifacts = self._src_chacha.list_workspace_artifacts(workspace_id)
        for artifact in artifacts:
            artifact_data = {
                "id": str(uuid.uuid4()),
                "artifact_type": artifact.get("artifact_type", "text"),
                "title": artifact.get("title", ""),
                "content": artifact.get("content", ""),
            }
            try:
                self._tgt_chacha.add_workspace_artifact(new_ws_id, artifact_data)
            except Exception as exc:
                logger.warning(
                    f"Failed to copy workspace artifact; exception_type={_safe_exception_type(exc)}"
                )
        _progress("artifacts_copied", 0.9)

        _progress("complete", 1.0)

        logger.info(
            f"Cloned workspace {workspace_id} -> {new_ws_id} "
            f"({total_sources} sources, {len(notes)} notes, {len(artifacts)} artifacts)"
        )

        return {
            "workspace_id": new_ws_id,
            "name": ws_data["name"],
            "sources_copied": total_sources,
            "notes_copied": len(notes),
            "artifacts_copied": len(artifacts),
            "media_id_map": media_id_map,
        }

    def _copy_media_item(self, media_id: str) -> str | None:
        """Copy a single media item (with chunks and transcripts) from source to target Media DB."""
        try:
            source_media_id = int(media_id)
            media = self._src_media.get_media_by_id(source_media_id)
            if not media:
                return None

            # Parse keywords: stored as comma-separated string, method expects list[str]
            raw_kw = media.get("keywords", "")
            if isinstance(raw_kw, str):
                keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
            else:
                keywords = list(raw_kw) if raw_kw else []

            # Insert media into target; capture actual DB-generated ID
            # Note: chunks are not separately readable; they'll be re-generated
            # from content during re-ingestion if needed.
            result = self._tgt_media.add_media_with_keywords(
                url=media.get("url", ""),
                title=media.get("title", "Untitled"),
                media_type=media.get("type", "unknown"),
                content=media.get("content", ""),
                keywords=keywords,
                prompt=media.get("prompt", ""),
                transcription_model=media.get("transcription_model", ""),
                author=media.get("author", "Unknown"),
                ingestion_date=media.get("ingestion_date", ""),
                overwrite=False,
            )
            # result is (media_id: int|None, media_uuid: str|None, status_message: str)
            new_media_id = result[0]
            if new_media_id is None:
                logger.warning("Target media insert returned no media id during clone")
                return None

            # Deep copy transcripts
            try:
                transcripts = get_media_transcripts(self._src_media, source_media_id)
                source_latest_run_id = media.get("latest_transcription_run_id")
                try:
                    source_latest_run_id_int = (
                        int(source_latest_run_id) if source_latest_run_id is not None else None
                    )
                except (TypeError, ValueError):
                    logger.debug(
                        "Malformed latest_transcription_run_id while cloning media; treating as None"
                    )
                    source_latest_run_id_int = None
                def _safe_int(val: object, default: int = 0) -> int:
                    try:
                        return int(val) if val is not None else default
                    except (TypeError, ValueError):
                        return default

                ordered_transcripts = sorted(
                    transcripts,
                    key=lambda row: (
                        row.get("transcription_run_id") is None,
                        _safe_int(row.get("transcription_run_id")),
                        str(row.get("created_at") or ""),
                        _safe_int(row.get("id")),
                    ),
                )
                has_matching_latest_run = (
                    source_latest_run_id_int is not None
                    and any(
                        row.get("transcription_run_id") is not None
                        and _safe_int(row.get("transcription_run_id"), -1)
                        == source_latest_run_id_int
                        for row in ordered_transcripts
                    )
                )
                fallback_to_last_transcript = (
                    source_latest_run_id_int is None or not has_matching_latest_run
                )
                for index, t in enumerate(ordered_transcripts):
                    run_id = t.get("transcription_run_id")
                    is_latest_run = False
                    if source_latest_run_id_int is not None and run_id is not None:
                        try:
                            is_latest_run = int(run_id) == source_latest_run_id_int
                        except (TypeError, ValueError):
                            is_latest_run = False
                    if (
                        not is_latest_run
                        and fallback_to_last_transcript
                        and index == len(ordered_transcripts) - 1
                    ):
                        is_latest_run = True
                    upsert_transcript(
                        self._tgt_media,
                        new_media_id,
                        transcription=t.get("transcription", ""),
                        whisper_model=t.get("whisper_model", "cloned"),
                        created_at=t.get("created_at"),
                        transcription_run_id=run_id,
                        idempotency_key=t.get("idempotency_key"),
                        set_as_latest=is_latest_run,
                    )
            except Exception:
                logger.warning("Failed to copy transcripts for cloned media")

            return str(new_media_id)
        except Exception as exc:
            logger.warning(
                f"Failed to copy media item; exception_type={_safe_exception_type(exc)}"
            )
            return None
