"""Service entrypoint for deep research sessions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import (
    ResearchArtifactManifestEntry,
    ResearchCheckpointSummary,
    ResearchRunResponse,
    ResearchRunSnapshotResponse,
)
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import (
    ResearchArtifactRow,
    ResearchChatLinkedRunRow,
    ResearchCheckpointRow,
    ResearchRunEventRow,
    ResearchSessionRow,
    ResearchSessionsDB,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env
from tldw_Server_API.app.core.config import settings

from .artifact_store import ResearchArtifactStore
from .checkpoint_service import apply_checkpoint_patch
from .exporter import build_final_package
from .jobs import enqueue_research_phase_job

_EXECUTABLE_PHASES = {"drafting_plan", "collecting", "synthesizing", "packaging"}
_CHECKPOINT_PHASES = {
    "awaiting_plan_review",
    "awaiting_source_review",
    "awaiting_outline_review",
}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_CHECKPOINT_BLOCKED_CONTROL_STATES = {"paused", "pause_requested", "cancel_requested", "cancelled"}


class ResearchService:
    """Create research sessions and enqueue the next executable slice."""

    _ALLOWED_ARTIFACT_NAMES = {
        "plan.json",
        "approved_plan.json",
        "approved_sources.json",
        "approved_outline.json",
        "provider_config.json",
        "source_registry.json",
        "evidence_notes.jsonl",
        "collection_summary.json",
        "outline_v1.json",
        "claims.json",
        "report_v1.md",
        "synthesis_summary.json",
        "verification_summary.json",
        "unsupported_claims.json",
        "contradictions.json",
        "source_trust.json",
        "bundle.json",
    }

    def __init__(
        self,
        *,
        research_db_path: str | Path | None,
        outputs_dir: str | Path | None,
        job_manager: Any | None,
    ):
        self._research_db_path = Path(research_db_path) if research_db_path is not None else None
        self._outputs_dir = Path(outputs_dir) if outputs_dir is not None else None
        self._job_manager = job_manager

    def _db_for_user(self, owner_user_id: str) -> ResearchSessionsDB:
        if self._research_db_path is not None:
            return ResearchSessionsDB(self._research_db_path)
        return ResearchSessionsDB(DatabasePaths.get_research_sessions_db_path(owner_user_id))

    def _outputs_dir_for_user(self, owner_user_id: str) -> Path:
        if self._outputs_dir is not None:
            return self._outputs_dir
        return DatabasePaths.get_user_outputs_dir(owner_user_id)

    def _job_manager_for_session(self) -> Any:
        if self._job_manager is not None:
            return self._job_manager
        return jobs_manager_from_env()

    @staticmethod
    def _normalized_event_payload_json(payload: dict[str, Any]) -> str:
        return json.dumps(payload or {}, sort_keys=True)

    @staticmethod
    def _status_event_payload(
        *,
        session_id: str,
        status: str,
        phase: str,
        control_state: str,
        active_job_id: str | None,
        latest_checkpoint_id: str | None,
        completed_at: str | None,
    ) -> dict[str, Any]:
        return {
            "id": session_id,
            "status": status,
            "phase": phase,
            "control_state": control_state,
            "active_job_id": active_job_id,
            "latest_checkpoint_id": latest_checkpoint_id,
            "completed_at": completed_at,
        }

    def get_job_manager(self) -> Any:
        """Return the Jobs manager used for research phase execution and progress reads."""
        return self._job_manager_for_session()

    @staticmethod
    def _job_identifier(job: dict[str, Any]) -> str | None:
        job_id = job.get("id")
        if job_id is not None:
            return str(job_id)
        job_uuid = job.get("uuid")
        return str(job_uuid) if job_uuid else None

    @staticmethod
    def _normalize_json_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}
        return json.loads(json.dumps(value, sort_keys=True))

    @staticmethod
    def _has_explicit_chat_db_base() -> bool:
        configured = settings.get("USER_DB_BASE_DIR")
        if configured:
            return True
        from os import getenv

        return bool(getenv("USER_DB_BASE_DIR"))

    @staticmethod
    def _validate_chat_handoff_chat_id(*, owner_user_id: str, chat_id: str) -> None:
        if ResearchService._resolve_chat_handoff_chat_id(
            owner_user_id=owner_user_id,
            chat_id=chat_id,
        ):
            return
        raise ValueError("chat_handoff.chat_id is not owned by the current user or was not found")

    @staticmethod
    def _resolve_chat_handoff_chat_id(*, owner_user_id: str, chat_id: str) -> str | None:
        chat_db_path = DatabasePaths.get_chacha_db_path(owner_user_id)
        if not chat_db_path.exists() and not ResearchService._has_explicit_chat_db_base():
            return None
        request_user_id = str(owner_user_id).strip()
        try:
            with sqlite3.connect(str(chat_db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT client_id FROM conversations WHERE id = ? AND deleted = 0",
                    (chat_id,),
                ).fetchone()
        except sqlite3.Error:
            return None
        if row is None:
            return None
        stored_client_id = str(row["client_id"] or "").strip()
        if stored_client_id != request_user_id:
            return None
        return str(chat_id).strip()

    @staticmethod
    def _is_executable_phase(phase: str) -> bool:
        return phase in _EXECUTABLE_PHASES

    @staticmethod
    def _is_checkpoint_phase(phase: str) -> bool:
        return phase in _CHECKPOINT_PHASES

    @staticmethod
    def _is_terminal(session: ResearchSessionRow) -> bool:
        return session.status in _TERMINAL_STATUSES or session.control_state == "cancelled"

    @staticmethod
    def _numeric_job_id(job_id: str | None) -> int | None:
        if job_id is None:
            return None
        if not str(job_id).isdigit():
            return None
        return int(job_id)

    def _cancel_active_job_best_effort(self, session: ResearchSessionRow, *, reason: str) -> None:
        numeric_job_id = self._numeric_job_id(session.active_job_id)
        if numeric_job_id is None:
            return
        manager = self._job_manager_for_session()
        cancel_job = getattr(manager, "cancel_job", None)
        if cancel_job is None:
            return
        try:
            cancel_job(numeric_job_id, reason=reason)
        except Exception:
            return

    @staticmethod
    def _next_phase_for_checkpoint(checkpoint_type: str) -> tuple[str, bool]:
        if checkpoint_type == "plan_review":
            return ("collecting", True)
        if checkpoint_type == "sources_review":
            return ("synthesizing", True)
        if checkpoint_type == "outline_review":
            return ("packaging", True)
        raise ValueError(f"unsupported checkpoint type: {checkpoint_type}")

    @staticmethod
    def _artifact_name_for_checkpoint(checkpoint_type: str) -> str:
        if checkpoint_type == "plan_review":
            return "approved_plan.json"
        if checkpoint_type == "sources_review":
            return "approved_sources.json"
        if checkpoint_type == "outline_review":
            return "approved_outline.json"
        raise ValueError(f"unsupported checkpoint type: {checkpoint_type}")

    @staticmethod
    def _checkpoint_event_payload(
        *,
        checkpoint: ResearchCheckpointRow,
        phase: str | None,
    ) -> dict[str, Any]:
        return {
            "checkpoint_id": checkpoint.id,
            "checkpoint_type": checkpoint.checkpoint_type,
            "status": checkpoint.status,
            "resolution": checkpoint.resolution,
            "phase": phase,
            "has_proposed_payload": bool(checkpoint.proposed_payload),
        }

    def create_session(
        self,
        *,
        owner_user_id: str,
        query: str,
        source_policy: str,
        autonomy_mode: str,
        limits_json: dict[str, Any] | None = None,
        provider_overrides: dict[str, Any] | None = None,
        chat_handoff: dict[str, Any] | None = None,
        follow_up: dict[str, Any] | None = None,
    ) -> ResearchSessionRow:
        """Create a research session and enqueue the planning phase."""
        db = self._db_for_user(owner_user_id)
        normalized_follow_up = self._normalize_json_mapping(follow_up)
        if chat_handoff:
            chat_id = str(chat_handoff.get("chat_id") or "").strip()
            if not chat_id:
                raise ValueError("chat_handoff.chat_id is required")
            self._validate_chat_handoff_chat_id(owner_user_id=owner_user_id, chat_id=chat_id)

        session = db.create_session(
            owner_user_id=owner_user_id,
            query=query,
            source_policy=source_policy,
            autonomy_mode=autonomy_mode,
            limits_json=limits_json or {},
            provider_overrides_json=provider_overrides or {},
            follow_up_json=normalized_follow_up,
        )
        if chat_handoff:
            launch_message_id = chat_handoff.get("launch_message_id")
            db.create_chat_handoff(
                session_id=session.id,
                owner_user_id=owner_user_id,
                chat_id=chat_id,
                launch_message_id=(
                    str(launch_message_id).strip()
                    if isinstance(launch_message_id, str) and launch_message_id.strip()
                    else None
                ),
            )

        try:
            job = enqueue_research_phase_job(
                jm=self._job_manager_for_session(),
                session_id=session.id,
                phase=session.phase,
                owner_user_id=session.owner_user_id,
            )
        except Exception:
            try:
                db.delete_session_cascade(session.id)
            except Exception as cleanup_exc:
                logger.warning(
                    "Failed to clean up research session {} after enqueue failure: {}",
                    session.id,
                    cleanup_exc,
                )
            raise
        return db.attach_active_job(
            session.id,
            self._job_identifier(job),
        )

    def record_run_event(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        event_type: str,
        event_payload: dict[str, Any],
        phase: str | None = None,
        job_id: str | None = None,
    ) -> ResearchRunEventRow:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None or session.owner_user_id != str(owner_user_id):
            raise KeyError(session_id)

        normalized_payload = self._normalized_event_payload_json(event_payload)
        latest = db.get_latest_run_event(
            owner_user_id=str(owner_user_id),
            session_id=session_id,
            event_type=event_type,
        )
        if (
            latest is not None
            and latest.phase == phase
            and latest.job_id == job_id
            and self._normalized_event_payload_json(latest.event_payload) == normalized_payload
        ):
            return latest

        return db.record_run_event(
            owner_user_id=str(owner_user_id),
            session_id=session_id,
            event_type=event_type,
            event_payload=event_payload,
            phase=phase,
            job_id=job_id,
        )

    def list_run_events_after(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        after_id: int,
        limit: int | None = None,
    ) -> list[ResearchRunEventRow]:
        db = self._db_for_user(owner_user_id)
        return db.list_run_events_after(
            owner_user_id=str(owner_user_id),
            session_id=session_id,
            after_id=after_id,
            limit=limit,
        )

    def approve_checkpoint(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        checkpoint_id: str,
        patch_payload: dict[str, Any] | None = None,
    ) -> ResearchSessionRow:
        """Resolve a review checkpoint and advance the session to the next phase."""
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.control_state in _CHECKPOINT_BLOCKED_CONTROL_STATES:
            raise ValueError("checkpoint_approval_not_allowed")

        checkpoint = db.get_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint.session_id != session_id:
            raise KeyError(checkpoint_id)

        patch_result = apply_checkpoint_patch(
            checkpoint_type=checkpoint.checkpoint_type,
            proposed_payload=checkpoint.proposed_payload,
            patch_payload=patch_payload or {},
        )
        db.resolve_checkpoint(
            checkpoint_id,
            resolution="patched" if patch_payload else "approved",
            user_patch_payload=patch_payload or {},
        )
        self.record_run_event(
            owner_user_id=owner_user_id,
            session_id=session_id,
            event_type="checkpoint",
            event_payload=self._checkpoint_event_payload(
                checkpoint=db.get_checkpoint(checkpoint_id) or checkpoint,
                phase=session.phase,
            ),
            phase=session.phase,
            job_id=None,
        )

        next_phase, should_enqueue = self._next_phase_for_checkpoint(checkpoint.checkpoint_type)
        enqueue_payload_overrides: dict[str, Any] = {}
        if checkpoint.checkpoint_type == "sources_review":
            recollect = patch_result.artifact_payload.get("recollect")
            if isinstance(recollect, dict) and bool(recollect.get("enabled")):
                next_phase = "collecting"
        elif checkpoint.checkpoint_type == "outline_review" and patch_payload:
            next_phase = "synthesizing"
            enqueue_payload_overrides["approved_outline_locked"] = True

        artifact_store = ResearchArtifactStore(
            base_dir=self._outputs_dir_for_user(owner_user_id),
            db=db,
        )
        artifact_store.write_json(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name=self._artifact_name_for_checkpoint(checkpoint.checkpoint_type),
            payload=patch_result.artifact_payload,
            phase=next_phase,
            job_id=None,
        )

        job = enqueue_research_phase_job(
            jm=self._job_manager_for_session(),
            session_id=session.id,
            phase=next_phase,
            owner_user_id=session.owner_user_id,
            checkpoint_id=checkpoint.id,
            payload_overrides=enqueue_payload_overrides,
        )
        updated, _ = db.update_phase_with_event(
            session_id,
            phase=next_phase,
            status="queued",
            control_state=session.control_state,
            active_job_id=self._job_identifier(job),
            owner_user_id=owner_user_id,
            event_type="status",
            event_payload=self._status_event_payload(
                session_id=session.id,
                status="queued",
                phase=next_phase,
                control_state=session.control_state,
                active_job_id=self._job_identifier(job),
                latest_checkpoint_id=session.latest_checkpoint_id,
                completed_at=session.completed_at,
            ),
            event_phase=next_phase,
            event_job_id=self._job_identifier(job),
        )
        return updated

    def build_package(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        brief: dict[str, Any],
        outline: dict[str, Any],
        report_markdown: str,
        claims: list[dict[str, Any]],
        source_inventory: list[dict[str, Any]],
        unresolved_questions: list[str] | None = None,
        verification_summary: dict[str, Any] | None = None,
        unsupported_claims: list[dict[str, Any]] | None = None,
        contradictions: list[dict[str, Any]] | None = None,
        source_trust: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build and persist the final deep research package."""
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        package = build_final_package(
            brief=brief,
            outline=outline,
            report_markdown=report_markdown,
            claims=claims,
            source_inventory=source_inventory,
            unresolved_questions=unresolved_questions,
            verification_summary=verification_summary,
            unsupported_claims=unsupported_claims,
            contradictions=contradictions,
            source_trust=source_trust,
        )
        artifact_store = ResearchArtifactStore(
            base_dir=self._outputs_dir_for_user(owner_user_id),
            db=db,
        )
        artifact_store.write_json(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name="bundle.json",
            payload=package,
            phase="packaging",
            job_id=None,
        )
        return package

    @staticmethod
    def _latest_artifact_manifest(artifacts: list[ResearchArtifactRow]) -> list[ResearchArtifactManifestEntry]:
        latest_by_name: dict[str, ResearchArtifactManifestEntry] = {}
        for artifact in artifacts:
            if artifact.artifact_name in latest_by_name:
                continue
            latest_by_name[artifact.artifact_name] = ResearchArtifactManifestEntry(
                artifact_name=artifact.artifact_name,
                artifact_version=artifact.artifact_version,
                content_type=artifact.content_type,
                phase=artifact.phase,
                job_id=artifact.job_id,
            )
        return list(latest_by_name.values())

    def get_stream_snapshot(
        self,
        *,
        owner_user_id: str,
        session_id: str,
    ) -> ResearchRunSnapshotResponse:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)

        checkpoint_summary: ResearchCheckpointSummary | None = None
        if session.latest_checkpoint_id:
            checkpoint = db.get_checkpoint(session.latest_checkpoint_id)
            if checkpoint is not None:
                checkpoint_summary = ResearchCheckpointSummary(
                    checkpoint_id=checkpoint.id,
                    checkpoint_type=checkpoint.checkpoint_type,
                    status=checkpoint.status,
                    proposed_payload=checkpoint.proposed_payload,
                    resolution=checkpoint.resolution,
                )

        return ResearchRunSnapshotResponse(
            run=ResearchRunResponse.model_validate(session),
            latest_event_id=db.get_latest_run_event_id(
                owner_user_id=owner_user_id,
                session_id=session_id,
            ),
            checkpoint=checkpoint_summary,
            artifacts=self._latest_artifact_manifest(db.list_artifacts(session_id)),
        )

    def list_sessions(
        self,
        *,
        owner_user_id: str,
        limit: int = 25,
        offset: int = 0,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[ResearchSessionRow]:
        db = self._db_for_user(owner_user_id)
        return db.list_sessions(
            owner_user_id,
            limit=limit,
            offset=offset,
            status=status,
            session_id=session_id,
        )

    def list_chat_linked_runs(
        self,
        *,
        owner_user_id: str,
        chat_id: str,
        terminal_limit: int = 10,
    ) -> list[ResearchChatLinkedRunRow]:
        db = self._db_for_user(owner_user_id)
        return db.list_chat_linked_runs(
            owner_user_id=str(owner_user_id),
            chat_id=str(chat_id),
            terminal_limit=terminal_limit,
        )

    def get_session(self, *, owner_user_id: str, session_id: str) -> ResearchSessionRow:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        handoff = db.get_chat_handoff(session_id)
        chat_id = None
        if handoff is not None:
            chat_id = self._resolve_chat_handoff_chat_id(
                owner_user_id=owner_user_id,
                chat_id=handoff.chat_id,
            )
        if session.chat_id != chat_id:
            session = replace(session, chat_id=chat_id)
        numeric_job_id = self._numeric_job_id(session.active_job_id)
        if numeric_job_id is None:
            return session

        manager = self._job_manager if self._job_manager is not None else self._job_manager_for_session()
        get_job = getattr(manager, "get_job", None)
        if get_job is None:
            return session

        job = get_job(numeric_job_id)
        if not isinstance(job, dict):
            return session

        progress_percent = session.progress_percent
        progress_message = session.progress_message
        if progress_percent is None and job.get("progress_percent") is not None:
            progress_percent = float(job["progress_percent"])
        if progress_message is None and job.get("progress_message") is not None:
            progress_message = str(job["progress_message"])
        if progress_percent == session.progress_percent and progress_message == session.progress_message:
            return session
        return replace(
            session,
            progress_percent=progress_percent,
            progress_message=progress_message,
        )

    def delete_session(self, *, owner_user_id: str, session_id: str) -> bool:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None or session.owner_user_id != str(owner_user_id):
            raise KeyError(session_id)
        numeric_job_id = self._numeric_job_id(session.active_job_id)
        if numeric_job_id is not None and not self._is_terminal(session):
            manager = self._job_manager if self._job_manager is not None else self._job_manager_for_session()
            cancel_job = getattr(manager, "cancel_job", None)
            if callable(cancel_job):
                try:
                    cancel_job(numeric_job_id, reason="research_run_deleted")
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel active research job {} while deleting session {}: {}",
                        numeric_job_id,
                        session_id,
                        exc,
                    )
        return db.delete_session_cascade(session_id)

    def delete_run(self, *, owner_user_id: str, session_id: str) -> bool:
        return self.delete_session(owner_user_id=owner_user_id, session_id=session_id)

    def pause_run(self, *, owner_user_id: str, session_id: str) -> ResearchSessionRow:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if self._is_terminal(session):
            raise ValueError("pause_not_allowed")
        if session.control_state in {"paused", "pause_requested"}:
            return session
        if self._is_executable_phase(session.phase) and session.active_job_id:
            updated, _ = db.update_control_state_with_event(
                session.id,
                control_state="pause_requested",
                owner_user_id=owner_user_id,
                event_type="status",
                event_payload=self._status_event_payload(
                    session_id=session.id,
                    status=session.status,
                    phase=session.phase,
                    control_state="pause_requested",
                    active_job_id=session.active_job_id,
                    latest_checkpoint_id=session.latest_checkpoint_id,
                    completed_at=session.completed_at,
                ),
                event_phase=session.phase,
                event_job_id=session.active_job_id,
            )
            return updated
        updated, _ = db.update_control_state_with_event(
            session.id,
            control_state="paused",
            owner_user_id=owner_user_id,
            event_type="status",
            event_payload=self._status_event_payload(
                session_id=session.id,
                status=session.status,
                phase=session.phase,
                control_state="paused",
                active_job_id=session.active_job_id,
                latest_checkpoint_id=session.latest_checkpoint_id,
                completed_at=session.completed_at,
            ),
            event_phase=session.phase,
            event_job_id=session.active_job_id,
        )
        return updated

    def resume_run(self, *, owner_user_id: str, session_id: str) -> ResearchSessionRow:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if session.control_state != "paused":
            raise ValueError("resume_not_allowed")
        if self._is_terminal(session):
            raise ValueError("resume_not_allowed")
        if self._is_checkpoint_phase(session.phase):
            updated, _ = db.update_phase_with_event(
                session.id,
                phase=session.phase,
                status="waiting_human",
                control_state="running",
                active_job_id=None,
                owner_user_id=owner_user_id,
                event_type="status",
                event_payload=self._status_event_payload(
                    session_id=session.id,
                    status="waiting_human",
                    phase=session.phase,
                    control_state="running",
                    active_job_id=None,
                    latest_checkpoint_id=session.latest_checkpoint_id,
                    completed_at=session.completed_at,
                ),
                event_phase=session.phase,
                event_job_id=None,
            )
            return updated
        if not self._is_executable_phase(session.phase):
            raise ValueError("resume_not_allowed")
        if session.active_job_id:
            updated, _ = db.update_control_state_with_event(
                session.id,
                control_state="running",
                owner_user_id=owner_user_id,
                event_type="status",
                event_payload=self._status_event_payload(
                    session_id=session.id,
                    status=session.status,
                    phase=session.phase,
                    control_state="running",
                    active_job_id=session.active_job_id,
                    latest_checkpoint_id=session.latest_checkpoint_id,
                    completed_at=session.completed_at,
                ),
                event_phase=session.phase,
                event_job_id=session.active_job_id,
            )
            return updated

        job = enqueue_research_phase_job(
            jm=self._job_manager_for_session(),
            session_id=session.id,
            phase=session.phase,
            owner_user_id=session.owner_user_id,
        )
        job_id = self._job_identifier(job)
        updated, _ = db.update_phase_with_event(
            session.id,
            phase=session.phase,
            status="queued",
            control_state="running",
            active_job_id=job_id,
            owner_user_id=owner_user_id,
            event_type="status",
            event_payload=self._status_event_payload(
                session_id=session.id,
                status="queued",
                phase=session.phase,
                control_state="running",
                active_job_id=job_id,
                latest_checkpoint_id=session.latest_checkpoint_id,
                completed_at=session.completed_at,
            ),
            event_phase=session.phase,
            event_job_id=job_id,
        )
        return updated

    def cancel_run(self, *, owner_user_id: str, session_id: str) -> ResearchSessionRow:
        db = self._db_for_user(owner_user_id)
        session = db.get_session(session_id)
        if session is None:
            raise KeyError(session_id)
        if self._is_terminal(session):
            raise ValueError("cancel_not_allowed")
        if session.control_state == "cancel_requested":
            return session
        if self._is_executable_phase(session.phase) and session.active_job_id:
            self._cancel_active_job_best_effort(session, reason="research_cancel_requested")
            updated, _ = db.update_control_state_with_event(
                session.id,
                control_state="cancel_requested",
                owner_user_id=owner_user_id,
                event_type="status",
                event_payload=self._status_event_payload(
                    session_id=session.id,
                    status=session.status,
                    phase=session.phase,
                    control_state="cancel_requested",
                    active_job_id=session.active_job_id,
                    latest_checkpoint_id=session.latest_checkpoint_id,
                    completed_at=session.completed_at,
                ),
                event_phase=session.phase,
                event_job_id=session.active_job_id,
            )
            return updated
        updated, _ = db.update_status_with_event(
            session.id,
            status="cancelled",
            owner_user_id=owner_user_id,
            event_type="status",
            event_payload=self._status_event_payload(
                session_id=session.id,
                status="cancelled",
                phase=session.phase,
                control_state="cancelled",
                active_job_id=None,
                latest_checkpoint_id=session.latest_checkpoint_id,
                completed_at=session.completed_at,
            ),
            phase=session.phase,
            job_id=None,
            control_state="cancelled",
            active_job_id=None,
        )
        return updated

    def get_bundle(self, *, owner_user_id: str, session_id: str) -> dict[str, Any]:
        artifact = self.get_artifact(
            owner_user_id=owner_user_id,
            session_id=session_id,
            artifact_name="bundle.json",
        )
        content = artifact["content"]
        if not isinstance(content, dict):
            raise KeyError("bundle.json")
        return content

    def get_artifact(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        artifact_name: str,
    ) -> dict[str, Any]:
        if artifact_name not in self._ALLOWED_ARTIFACT_NAMES:
            raise ValueError("artifact_not_allowed")

        db = self._db_for_user(owner_user_id)
        artifact_row = self._get_latest_artifact_row(db=db, session_id=session_id, artifact_name=artifact_name)
        if artifact_row is None:
            raise KeyError(artifact_name)
        path = Path(artifact_row.storage_path)
        if not path.exists():
            raise KeyError(artifact_name)

        content: Any
        if artifact_row.content_type == "application/json":
            content = json.loads(path.read_text(encoding="utf-8"))
        elif artifact_row.content_type == "application/x-ndjson":
            content = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            content = path.read_text(encoding="utf-8")

        return {
            "artifact_name": artifact_row.artifact_name,
            "content_type": artifact_row.content_type,
            "content": content,
        }

    @staticmethod
    def _get_latest_artifact_row(
        *,
        db: ResearchSessionsDB,
        session_id: str,
        artifact_name: str,
    ) -> ResearchArtifactRow | None:
        matches = [
            artifact
            for artifact in db.list_artifacts(session_id)
            if artifact.artifact_name == artifact_name
        ]
        if not matches:
            return None
        return max(matches, key=lambda artifact: artifact.artifact_version)


__all__ = ["ResearchService"]
