"""Core Jobs helpers for deep research execution slices."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
from tldw_Server_API.app.core.Research.broker import ResearchBroker
from tldw_Server_API.app.core.Research.chat_handoff import deliver_research_chat_handoff
from tldw_Server_API.app.core.Research.exporter import build_final_package
from tldw_Server_API.app.core.Research.limits import ResearchLimits, ensure_limit_available
from tldw_Server_API.app.core.Research.models import (
    ResearchEvidenceNote,
    ResearchPlan,
    ResearchSourceRecord,
)
from tldw_Server_API.app.core.Research.planner import build_initial_plan
from tldw_Server_API.app.core.Research.providers.academic import AcademicResearchProvider
from tldw_Server_API.app.core.Research.providers.config import resolve_provider_config
from tldw_Server_API.app.core.Research.providers.local import LocalResearchProvider
from tldw_Server_API.app.core.Research.providers.web import WebResearchProvider
from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

RESEARCH_DOMAIN = "research"
RESEARCH_JOB_TYPE = "research_phase"
RESEARCH_QUEUE = "default"
_PHASE_PROGRESS = {
    "drafting_plan": (10.0, "planning research"),
    "collecting": (45.0, "collecting sources"),
    "synthesizing": (75.0, "synthesizing report"),
    "packaging": (95.0, "packaging results"),
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


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


def _progress_event_payload(
    *,
    session_id: str,
    progress_percent: float | None,
    progress_message: str | None,
) -> dict[str, Any]:
    return {
        "id": session_id,
        "progress_percent": progress_percent,
        "progress_message": progress_message,
    }


def _checkpoint_event_payload(*, checkpoint: Any, phase: str | None) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint.id,
        "checkpoint_type": checkpoint.checkpoint_type,
        "status": checkpoint.status,
        "resolution": checkpoint.resolution,
        "phase": phase,
        "has_proposed_payload": bool(checkpoint.proposed_payload),
    }


def _limits_from_session(session: Any) -> ResearchLimits | None:
    payload = session.limits_json if isinstance(getattr(session, "limits_json", None), dict) else {}
    required = ("max_searches", "max_fetched_docs", "max_runtime_seconds")
    if not any(key in payload for key in required):
        return None
    try:
        unlimited = 2_147_483_647
        return ResearchLimits(
            max_searches=int(payload["max_searches"]) if "max_searches" in payload else unlimited,
            max_fetched_docs=int(payload["max_fetched_docs"]) if "max_fetched_docs" in payload else unlimited,
            max_runtime_seconds=int(payload["max_runtime_seconds"]) if "max_runtime_seconds" in payload else unlimited,
        )
    except (TypeError, ValueError):
        raise ValueError("research_limit_invalid") from None


def _enforce_limit_available(limits: ResearchLimits | None, usage: dict[str, int], key: str) -> None:
    if limits is None:
        return
    error = ensure_limit_available(limits, usage, key)
    if error is not None:
        raise ValueError(f"{error.code}:{error.limit_key}")


def _update_runtime_usage(*, usage: dict[str, int], started_at: float) -> None:
    usage["runtime_seconds"] = int(max(0, time.monotonic() - started_at))


async def _write_json_artifact(artifact_store: ResearchArtifactStore, **kwargs: Any) -> Any:
    return await asyncio.to_thread(artifact_store.write_json, **kwargs)


async def _write_jsonl_artifact(artifact_store: ResearchArtifactStore, **kwargs: Any) -> Any:
    return await asyncio.to_thread(artifact_store.write_jsonl, **kwargs)


async def _write_text_artifact(artifact_store: ResearchArtifactStore, **kwargs: Any) -> Any:
    return await asyncio.to_thread(artifact_store.write_text, **kwargs)


def enqueue_research_phase_job(
    *,
    jm: Any,
    session_id: str,
    phase: str,
    owner_user_id: str,
    checkpoint_id: str | None = None,
    payload_overrides: dict[str, Any] | None = None,
    policy_version: int = 1,
    priority: int = 5,
) -> dict[str, Any]:
    """Create a core Jobs entry for a research session phase."""
    payload = {
        "session_id": session_id,
        "phase": phase,
        "checkpoint_id": checkpoint_id,
        "owner_user_id": str(owner_user_id),
        "policy_version": int(policy_version),
    }
    if isinstance(payload_overrides, dict):
        payload.update(payload_overrides)
    return jm.create_job(
        domain=RESEARCH_DOMAIN,
        queue=RESEARCH_QUEUE,
        job_type=RESEARCH_JOB_TYPE,
        payload=payload,
        owner_user_id=str(owner_user_id),
        priority=priority,
        idempotency_key=f"research:{session_id}:{phase}:{checkpoint_id or 'none'}:{policy_version}",
    )


async def handle_research_phase_job(
    job: dict[str, Any],
    *,
    research_db_path: str | Path,
    outputs_dir: str | Path,
    broker: ResearchBroker | None = None,
    synthesizer: ResearchSynthesizer | None = None,
) -> dict[str, Any]:
    """Advance a single research phase job."""
    payload = job.get("payload") or {}
    job_type = str(job.get("job_type") or payload.get("job_type") or RESEARCH_JOB_TYPE).strip().lower()
    if job_type != RESEARCH_JOB_TYPE:
        raise ValueError(f"unsupported research job type: {job_type}")

    session_id = str(payload.get("session_id") or "").strip()
    if not session_id:
        raise ValueError("missing research session_id")

    phase = str(payload.get("phase") or "").strip().lower()

    db = ResearchSessionsDB(research_db_path)
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(session_id)

    artifact_store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    job_id = str(job.get("id")) if job.get("id") is not None else None

    if phase == "drafting_plan":
        return await _handle_planning_phase(
            session=session,
            db=db,
            artifact_store=artifact_store,
            job_id=job_id,
        )
    if phase == "collecting":
        return await _handle_collecting_phase(
            session=session,
            db=db,
            artifact_store=artifact_store,
            job_id=job_id,
            broker=broker or ResearchBroker(
                local_provider=LocalResearchProvider(),
                academic_provider=AcademicResearchProvider(),
                web_provider=WebResearchProvider(),
            ),
        )
    if phase == "synthesizing":
        return await _handle_synthesizing_phase(
            session=session,
            db=db,
            artifact_store=artifact_store,
            job_id=job_id,
            approved_outline_locked=bool(payload.get("approved_outline_locked")),
            synthesizer=synthesizer or ResearchSynthesizer(),
        )
    if phase == "packaging":
        return await _handle_packaging_phase(
            session=session,
            db=db,
            artifact_store=artifact_store,
            job_id=job_id,
        )
    raise ValueError(f"unsupported research phase: {phase}")


async def _handle_planning_phase(
    *,
    session: Any,
    db: ResearchSessionsDB,
    artifact_store: ResearchArtifactStore,
    job_id: str | None,
) -> dict[str, Any]:
    halted = _halt_for_control_before_phase(db=db, session_id=session.id)
    if halted is not None:
        return halted
    _set_phase_progress(db=db, session=session, phase="drafting_plan", job_id=job_id)

    follow_up_background = None
    if isinstance(getattr(session, "follow_up_json", None), dict):
        background = session.follow_up_json.get("background")
        if isinstance(background, dict):
            follow_up_background = background

    plan = build_initial_plan(
        query=session.query,
        source_policy=session.source_policy,
        autonomy_mode=session.autonomy_mode,
        follow_up_background=follow_up_background,
    )

    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="plan.json",
        payload=asdict(plan),
        phase=session.phase,
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="provider_config.json",
        payload=resolve_provider_config(session.provider_overrides_json),
        phase=session.phase,
        job_id=job_id,
    )

    next_phase = "collecting"
    next_status = "queued"
    checkpoint_id: str | None = None
    if session.autonomy_mode == "checkpointed":
        checkpoint = db.create_checkpoint(
            session_id=session.id,
            checkpoint_type="plan_review",
            proposed_payload=asdict(plan),
        )
        db.record_run_event(
            owner_user_id=session.owner_user_id,
            session_id=session.id,
            event_type="checkpoint",
            event_payload=_checkpoint_event_payload(
                checkpoint=checkpoint,
                phase="awaiting_plan_review",
            ),
            phase="awaiting_plan_review",
            job_id=job_id,
        )
        checkpoint_id = checkpoint.id
        next_phase = "awaiting_plan_review"
        next_status = "waiting_human"

    return _finalize_phase_transition(
        db=db,
        session_id=session.id,
        next_phase=next_phase,
        next_status=next_status,
        checkpoint_id=checkpoint_id,
        artifacts_written=2,
    )


async def _handle_collecting_phase(
    *,
    session: Any,
    db: ResearchSessionsDB,
    artifact_store: ResearchArtifactStore,
    job_id: str | None,
    broker: ResearchBroker,
) -> dict[str, Any]:
    halted = _halt_for_control_before_phase(db=db, session_id=session.id)
    if halted is not None:
        return halted
    _set_phase_progress(db=db, session=session, phase="collecting", job_id=job_id)

    plan = _load_effective_plan(session=session, artifact_store=artifact_store)
    provider_config = _load_provider_config(session=session, artifact_store=artifact_store)
    approved_sources = _load_approved_sources(artifact_store=artifact_store, session=session)
    dropped_source_ids = set(approved_sources.get("dropped_source_ids", []))
    pinned_source_ids = set(approved_sources.get("pinned_source_ids", []))
    recollect = approved_sources.get("recollect", {}) if isinstance(approved_sources.get("recollect"), dict) else {}
    recollect_enabled = bool(recollect.get("enabled"))
    limits = _limits_from_session(session)
    usage = {"searches": 0, "fetched_docs": 0, "runtime_seconds": 0}
    started_at = time.monotonic()

    sources_by_fingerprint: dict[str, dict[str, Any]] = {}
    evidence_notes: list[dict[str, Any]] = []
    evidence_note_ids: set[str] = set()
    remaining_gaps: list[str] = []
    gap_set: set[str] = set()
    lane_counts = {"local": 0, "academic": 0, "web": 0}
    lane_attempts = {"local": 0, "academic": 0, "web": 0}
    lane_errors: list[dict[str, str]] = []
    deduped_sources = 0
    if recollect_enabled:
        existing_sources_payload = artifact_store.read_json(session_id=session.id, artifact_name="source_registry.json")
        existing_evidence_notes_payload = artifact_store.read_jsonl(session_id=session.id, artifact_name="evidence_notes.jsonl")
        existing_sources = [
            record
            for record in (existing_sources_payload or {}).get("sources", [])
            if isinstance(record, dict)
        ]
        existing_notes = [
            record
            for record in (existing_evidence_notes_payload or [])
            if isinstance(record, dict)
        ]
        for record in existing_sources:
            source_id = str(record.get("source_id") or "").strip()
            fingerprint = str(record.get("fingerprint") or "").strip()
            if source_id in pinned_source_ids and source_id not in dropped_source_ids and fingerprint:
                sources_by_fingerprint[fingerprint] = dict(record)
        for note in existing_notes:
            source_id = str(note.get("source_id") or "").strip()
            note_id = str(note.get("note_id") or "").strip()
            if source_id in pinned_source_ids and source_id not in dropped_source_ids and note_id and note_id not in evidence_note_ids:
                evidence_note_ids.add(note_id)
                evidence_notes.append(dict(note))

    for focus_area in plan.focus_areas:
        halted = _halt_for_cancel_during_phase(db=db, session_id=session.id)
        if halted is not None:
            return halted
        _update_runtime_usage(usage=usage, started_at=started_at)
        _enforce_limit_available(limits, usage, "runtime_seconds")
        _enforce_limit_available(limits, usage, "searches")
        _enforce_limit_available(limits, usage, "fetched_docs")
        result = await broker.collect_focus_area(
            session_id=session.id,
            owner_user_id=session.owner_user_id,
            focus_area=focus_area,
            plan=plan,
            provider_config=provider_config,
            context={
                "approved_sources": approved_sources,
                "recollect": dict(recollect),
            },
        )
        usage["searches"] += 1
        usage["fetched_docs"] += len(result.sources)
        _update_runtime_usage(usage=usage, started_at=started_at)
        _enforce_limit_available(limits, usage, "runtime_seconds")
        if limits is not None and usage["fetched_docs"] > limits.max_fetched_docs:
            raise ValueError("research_limit_exceeded:fetched_docs")
        for source in result.sources:
            serialized = asdict(source)
            if serialized["source_id"] in dropped_source_ids:
                continue
            fingerprint = str(serialized["fingerprint"])
            if fingerprint in sources_by_fingerprint:
                deduped_sources += 1
                continue
            sources_by_fingerprint[fingerprint] = serialized
        for note in result.evidence_notes:
            serialized_note = asdict(note)
            if serialized_note["source_id"] in dropped_source_ids:
                continue
            note_id = str(serialized_note["note_id"])
            if note_id in evidence_note_ids:
                continue
            evidence_note_ids.add(note_id)
            evidence_notes.append(serialized_note)
        metrics = result.collection_metrics or {}
        lane_metrics = metrics.get("lane_counts") if isinstance(metrics.get("lane_counts"), dict) else {}
        for lane in lane_counts:
            lane_counts[lane] += int(lane_metrics.get(lane, 0) or 0)
        lane_attempt_metrics = metrics.get("lane_attempts") if isinstance(metrics.get("lane_attempts"), dict) else {}
        for lane in lane_attempts:
            lane_attempts[lane] += int(lane_attempt_metrics.get(lane, 0) or 0)
        lane_error_records = metrics.get("lane_errors")
        if isinstance(lane_error_records, list):
            lane_errors.extend(
                item
                for item in lane_error_records
                if isinstance(item, dict)
                and str(item.get("lane") or "").strip()
                and str(item.get("message") or "").strip()
            )
        deduped_sources += int(metrics.get("deduped_sources", 0) or 0)
        for gap in result.remaining_gaps:
            if gap not in gap_set:
                gap_set.add(gap)
                remaining_gaps.append(gap)
        halted = _halt_for_cancel_during_phase(db=db, session_id=session.id)
        if halted is not None:
            return halted

    source_registry = list(sources_by_fingerprint.values())
    attempted_lane_total = sum(lane_attempts.values())
    if not source_registry and attempted_lane_total > 0 and len(lane_errors) == attempted_lane_total:
        raise ValueError("all attempted collection lanes failed")

    collection_summary = {
        "query": plan.query,
        "focus_areas": plan.focus_areas,
        "source_policy": plan.source_policy,
        "source_count": len(source_registry),
        "evidence_note_count": len(evidence_notes),
        "remaining_gaps": remaining_gaps,
        "lane_errors": lane_errors,
        "collection_metrics": {
            "lane_counts": lane_counts,
            "lane_attempts": lane_attempts,
            "deduped_sources": deduped_sources,
        },
        "review_directives": approved_sources if approved_sources else None,
    }

    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": source_registry,
        },
        phase="collecting",
        job_id=job_id,
    )
    await _write_jsonl_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=evidence_notes,
        phase="collecting",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload=collection_summary,
        phase="collecting",
        job_id=job_id,
    )

    next_phase = "synthesizing"
    next_status = "queued"
    checkpoint_id: str | None = None
    if session.autonomy_mode == "checkpointed":
        checkpoint = db.create_checkpoint(
            session_id=session.id,
            checkpoint_type="sources_review",
            proposed_payload={
                "query": plan.query,
                "focus_areas": plan.focus_areas,
                "source_inventory": source_registry,
                "collection_summary": collection_summary,
            },
        )
        db.record_run_event(
            owner_user_id=session.owner_user_id,
            session_id=session.id,
            event_type="checkpoint",
            event_payload=_checkpoint_event_payload(
                checkpoint=checkpoint,
                phase="awaiting_source_review",
            ),
            phase="awaiting_source_review",
            job_id=job_id,
        )
        checkpoint_id = checkpoint.id
        next_phase = "awaiting_source_review"
        next_status = "waiting_human"

    return _finalize_phase_transition(
        db=db,
        session_id=session.id,
        next_phase=next_phase,
        next_status=next_status,
        checkpoint_id=checkpoint_id,
        artifacts_written=3,
    )


def _load_provider_config(
    *,
    session: Any,
    artifact_store: ResearchArtifactStore,
) -> dict[str, Any]:
    provider_config = artifact_store.read_json(session_id=session.id, artifact_name="provider_config.json")
    if isinstance(provider_config, dict):
        return provider_config
    return resolve_provider_config(session.provider_overrides_json)


def _load_approved_sources(*, artifact_store: ResearchArtifactStore, session: Any) -> dict[str, Any]:
    payload = artifact_store.read_json(session_id=session.id, artifact_name="approved_sources.json")
    if not isinstance(payload, dict):
        return {
            "pinned_source_ids": [],
            "dropped_source_ids": [],
            "prioritized_source_ids": [],
            "recollect": {
                "enabled": False,
                "need_primary_sources": False,
                "need_contradictions": False,
                "guidance": "",
            },
        }
    recollect_payload = payload.get("recollect")
    if not isinstance(recollect_payload, dict):
        recollect_payload = {}
    return {
        "pinned_source_ids": [
            str(source_id).strip()
            for source_id in payload.get("pinned_source_ids", [])
            if str(source_id).strip()
        ],
        "dropped_source_ids": [
            str(source_id).strip()
            for source_id in payload.get("dropped_source_ids", [])
            if str(source_id).strip()
        ],
        "prioritized_source_ids": [
            str(source_id).strip()
            for source_id in payload.get("prioritized_source_ids", [])
            if str(source_id).strip()
        ],
        "recollect": {
            "enabled": bool(recollect_payload.get("enabled")),
            "need_primary_sources": bool(recollect_payload.get("need_primary_sources")),
            "need_contradictions": bool(recollect_payload.get("need_contradictions")),
            "guidance": str(recollect_payload.get("guidance") or "").strip(),
        },
    }


def _apply_source_review_to_records(
    *,
    source_registry: list[dict[str, Any]],
    evidence_notes: list[dict[str, Any]],
    approved_sources: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dropped_source_ids = set(approved_sources.get("dropped_source_ids", []))
    prioritized_source_ids = list(approved_sources.get("prioritized_source_ids", []))
    pinned_source_ids = list(approved_sources.get("pinned_source_ids", []))

    filtered_sources = [
        dict(record)
        for record in source_registry
        if str(record.get("source_id") or "").strip()
        and str(record.get("source_id") or "").strip() not in dropped_source_ids
    ]
    filtered_notes = [
        dict(record)
        for record in evidence_notes
        if str(record.get("source_id") or "").strip()
        and str(record.get("source_id") or "").strip() not in dropped_source_ids
    ]

    if not filtered_sources:
        return filtered_sources, filtered_notes

    priority_order: dict[str, int] = {}
    for index, source_id in enumerate(prioritized_source_ids):
        priority_order[source_id] = index
    next_index = len(priority_order)
    for source_id in pinned_source_ids:
        if source_id not in priority_order:
            priority_order[source_id] = next_index
            next_index += 1

    def _sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
        source_id = str(record.get("source_id") or "").strip()
        if source_id in priority_order:
            return (0, priority_order[source_id], source_id)
        return (1, len(priority_order), source_id)

    filtered_sources.sort(key=_sort_key)
    return filtered_sources, filtered_notes


async def _handle_synthesizing_phase(
    *,
    session: Any,
    db: ResearchSessionsDB,
    artifact_store: ResearchArtifactStore,
    job_id: str | None,
    approved_outline_locked: bool,
    synthesizer: ResearchSynthesizer,
) -> dict[str, Any]:
    halted = _halt_for_control_before_phase(db=db, session_id=session.id)
    if halted is not None:
        return halted
    _set_phase_progress(db=db, session=session, phase="synthesizing", job_id=job_id)

    plan = _load_effective_plan(session=session, artifact_store=artifact_store)
    provider_config = _load_provider_config(session=session, artifact_store=artifact_store)
    source_registry_payload = artifact_store.read_json(session_id=session.id, artifact_name="source_registry.json")
    if source_registry_payload is None:
        raise ValueError(f"missing source registry artifact for session {session.id}")
    evidence_notes_payload = artifact_store.read_jsonl(session_id=session.id, artifact_name="evidence_notes.jsonl")
    if evidence_notes_payload is None:
        raise ValueError(f"missing evidence notes artifact for session {session.id}")
    collection_summary = artifact_store.read_json(session_id=session.id, artifact_name="collection_summary.json")
    if collection_summary is None:
        raise ValueError(f"missing collection summary artifact for session {session.id}")

    approved_sources = _load_approved_sources(artifact_store=artifact_store, session=session)
    source_registry_records, evidence_note_records = _apply_source_review_to_records(
        source_registry=[
            record
            for record in source_registry_payload.get("sources", [])
            if isinstance(record, dict)
        ],
        evidence_notes=[
            record
            for record in evidence_notes_payload
            if isinstance(record, dict)
        ],
        approved_sources=approved_sources,
    )
    source_registry = [
        ResearchSourceRecord(**record)
        for record in source_registry_records
    ]
    evidence_notes = [
        ResearchEvidenceNote(**record)
        for record in evidence_note_records
    ]
    approved_outline = artifact_store.read_json(session_id=session.id, artifact_name="approved_outline.json")
    outline_seed = approved_outline.get("sections") if isinstance(approved_outline, dict) else None

    result = await synthesizer.synthesize(
        plan=plan,
        source_registry=source_registry,
        evidence_notes=evidence_notes,
        collection_summary=collection_summary,
        provider_config=provider_config,
        outline_seed=outline_seed if approved_outline_locked else None,
        approved_outline_locked=approved_outline_locked,
    )
    halted = _halt_for_control_before_phase(db=db, session_id=session.id)
    if halted is not None:
        return halted

    outline_payload = {
        "query": plan.query,
        "sections": [asdict(section) for section in result.outline_sections],
        "unresolved_questions": list(result.unresolved_questions),
    }
    claims_payload = {
        "claims": [asdict(claim) for claim in result.claims],
    }

    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="outline_v1.json",
        payload=outline_payload,
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="claims.json",
        payload=claims_payload,
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_text_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="report_v1.md",
        content=result.report_markdown,
        phase="synthesizing",
        job_id=job_id,
        content_type="text/markdown",
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="synthesis_summary.json",
        payload=result.synthesis_summary,
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="verification_summary.json",
        payload=result.verification_summary,
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="unsupported_claims.json",
        payload={"claims": result.unsupported_claims},
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="contradictions.json",
        payload={"contradictions": result.contradictions},
        phase="synthesizing",
        job_id=job_id,
    )
    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="source_trust.json",
        payload={"sources": result.source_trust},
        phase="synthesizing",
        job_id=job_id,
    )

    next_phase = "packaging"
    next_status = "queued"
    checkpoint_id: str | None = None
    if session.autonomy_mode == "checkpointed" and not approved_outline_locked:
        checkpoint = db.create_checkpoint(
            session_id=session.id,
            checkpoint_type="outline_review",
            proposed_payload={
                "outline": outline_payload,
                "claim_count": len(result.claims),
                "report_preview": "\n".join(result.report_markdown.splitlines()[:8]),
                "focus_areas": list(plan.focus_areas),
            },
        )
        db.record_run_event(
            owner_user_id=session.owner_user_id,
            session_id=session.id,
            event_type="checkpoint",
            event_payload=_checkpoint_event_payload(
                checkpoint=checkpoint,
                phase="awaiting_outline_review",
            ),
            phase="awaiting_outline_review",
            job_id=job_id,
        )
        checkpoint_id = checkpoint.id
        next_phase = "awaiting_outline_review"
        next_status = "waiting_human"

    return _finalize_phase_transition(
        db=db,
        session_id=session.id,
        next_phase=next_phase,
        next_status=next_status,
        checkpoint_id=checkpoint_id,
        artifacts_written=8,
    )


async def _handle_packaging_phase(
    *,
    session: Any,
    db: ResearchSessionsDB,
    artifact_store: ResearchArtifactStore,
    job_id: str | None,
) -> dict[str, Any]:
    halted = _halt_for_control_before_phase(db=db, session_id=session.id)
    if halted is not None:
        return halted
    _set_phase_progress(db=db, session=session, phase="packaging", job_id=job_id)

    plan = _load_effective_plan(session=session, artifact_store=artifact_store)
    outline = artifact_store.read_json(session_id=session.id, artifact_name="outline_v1.json")
    if outline is None:
        raise ValueError(f"missing outline artifact for session {session.id}")
    claims_payload = artifact_store.read_json(session_id=session.id, artifact_name="claims.json")
    if claims_payload is None:
        raise ValueError(f"missing claims artifact for session {session.id}")
    report_markdown = artifact_store.read_text(session_id=session.id, artifact_name="report_v1.md")
    if report_markdown is None:
        raise ValueError(f"missing report artifact for session {session.id}")
    source_registry_payload = artifact_store.read_json(session_id=session.id, artifact_name="source_registry.json")
    if source_registry_payload is None:
        raise ValueError(f"missing source registry artifact for session {session.id}")
    synthesis_summary = artifact_store.read_json(session_id=session.id, artifact_name="synthesis_summary.json")
    if synthesis_summary is None:
        raise ValueError(f"missing synthesis summary artifact for session {session.id}")
    verification_summary = artifact_store.read_json(session_id=session.id, artifact_name="verification_summary.json") or {}
    unsupported_claims_payload = artifact_store.read_json(session_id=session.id, artifact_name="unsupported_claims.json") or {}
    contradictions_payload = artifact_store.read_json(session_id=session.id, artifact_name="contradictions.json") or {}
    source_trust_payload = artifact_store.read_json(session_id=session.id, artifact_name="source_trust.json") or {}

    package = build_final_package(
        brief={"query": plan.query},
        outline=outline,
        report_markdown=report_markdown,
        claims=list(claims_payload.get("claims", [])),
        source_inventory=list(source_registry_payload.get("sources", [])),
        unresolved_questions=list(synthesis_summary.get("unresolved_questions", [])),
        verification_summary=verification_summary,
        unsupported_claims=list(unsupported_claims_payload.get("claims", [])),
        contradictions=list(contradictions_payload.get("contradictions", [])),
        source_trust=list(source_trust_payload.get("sources", [])),
    )

    await _write_json_artifact(
        artifact_store,
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="bundle.json",
        payload=package,
        phase="packaging",
        job_id=job_id,
    )

    db.update_progress_with_event(
        session.id,
        progress_percent=100.0,
        progress_message="packaging results",
        owner_user_id=session.owner_user_id,
        event_type="progress",
        event_payload=_progress_event_payload(
            session_id=session.id,
            progress_percent=100.0,
            progress_message="packaging results",
        ),
        event_phase="packaging",
        event_job_id=job_id,
    )
    transition = _finalize_phase_transition(
        db=db,
        session_id=session.id,
        next_phase="completed",
        next_status="completed",
        checkpoint_id=None,
        artifacts_written=1,
        completed_at=_utc_now(),
    )
    if transition.get("phase") == "completed":
        try:
            deliver_research_chat_handoff(
                db=db,
                artifact_store=artifact_store,
                session_id=session.id,
            )
        except Exception as exc:
            handoff = db.get_chat_handoff(session.id)
            if handoff is not None and handoff.handoff_status == "pending":
                db.mark_chat_handoff_failed(session.id, last_error=str(exc))
    return transition


def _set_phase_progress(*, db: ResearchSessionsDB, session: Any, phase: str, job_id: str | None) -> None:
    progress = _PHASE_PROGRESS.get(phase)
    if progress is None:
        return
    db.update_progress_with_event(
        session.id,
        progress_percent=progress[0],
        progress_message=progress[1],
        owner_user_id=session.owner_user_id,
        event_type="progress",
        event_payload=_progress_event_payload(
            session_id=session.id,
            progress_percent=progress[0],
            progress_message=progress[1],
        ),
        event_phase=phase,
        event_job_id=job_id,
    )


def _halt_for_control_before_phase(*, db: ResearchSessionsDB, session_id: str) -> dict[str, Any] | None:
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(session_id)
    if session.control_state == "cancel_requested":
        return _cancel_session(db=db, session_id=session_id)
    if session.control_state in {"pause_requested", "paused"}:
        updated, _ = db.update_control_state_with_event(
            session_id,
            control_state="paused",
            active_job_id=None,
            owner_user_id=session.owner_user_id,
            event_type="status",
            event_payload=_status_event_payload(
                session_id=session.id,
                status=session.status,
                phase=session.phase,
                control_state="paused",
                active_job_id=None,
                latest_checkpoint_id=session.latest_checkpoint_id,
                completed_at=session.completed_at,
            ),
            event_phase=session.phase,
            event_job_id=None,
        )
        return {
            "session_id": updated.id,
            "phase": updated.phase,
            "checkpoint_id": updated.latest_checkpoint_id,
            "artifacts_written": 0,
        }
    return None


def _halt_for_cancel_during_phase(*, db: ResearchSessionsDB, session_id: str) -> dict[str, Any] | None:
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(session_id)
    if session.control_state == "cancel_requested":
        return _cancel_session(db=db, session_id=session_id)
    return None


def _cancel_session(*, db: ResearchSessionsDB, session_id: str) -> dict[str, Any]:
    session = db.get_session(session_id)
    if session is None:
        raise KeyError(session_id)
    updated, _ = db.update_status_with_event(
        session_id,
        status="cancelled",
        owner_user_id=session.owner_user_id,
        event_type="status",
        event_payload=_status_event_payload(
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
    db.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="terminal",
        event_payload=_status_event_payload(
            session_id=updated.id,
            status=updated.status,
            phase=updated.phase,
            control_state=updated.control_state,
            active_job_id=updated.active_job_id,
            latest_checkpoint_id=updated.latest_checkpoint_id,
            completed_at=updated.completed_at,
        ),
        phase=updated.phase,
        job_id=None,
    )
    return {
        "session_id": updated.id,
        "phase": updated.phase,
        "checkpoint_id": updated.latest_checkpoint_id,
        "artifacts_written": 0,
    }


def _finalize_phase_transition(
    *,
    db: ResearchSessionsDB,
    session_id: str,
    next_phase: str,
    next_status: str,
    checkpoint_id: str | None,
    artifacts_written: int,
    completed_at: str | None = None,
) -> dict[str, Any]:
    current = db.get_session(session_id)
    if current is None:
        raise KeyError(session_id)
    if current.control_state == "cancel_requested":
        return _cancel_session(db=db, session_id=session_id)
    if current.control_state in {"pause_requested", "paused"} and next_phase != "completed":
        updated, _ = db.update_phase_with_event(
            session_id,
            phase=next_phase,
            status=next_status,
            control_state="paused",
            active_job_id=None,
            completed_at=completed_at,
            owner_user_id=current.owner_user_id,
            event_type="status",
            event_payload=_status_event_payload(
                session_id=current.id,
                status=next_status,
                phase=next_phase,
                control_state="paused",
                active_job_id=None,
                latest_checkpoint_id=current.latest_checkpoint_id,
                completed_at=completed_at,
            ),
            event_phase=next_phase,
            event_job_id=None,
        )
        return {
            "session_id": updated.id,
            "phase": updated.phase,
            "checkpoint_id": checkpoint_id,
            "artifacts_written": artifacts_written,
        }
    next_control_state = (
        "running" if next_phase == "completed" and current.control_state in {"pause_requested", "paused"} else current.control_state
    )
    updated, _ = db.update_phase_with_event(
        session_id,
        phase=next_phase,
        status=next_status,
        control_state=next_control_state,
        active_job_id=None,
        completed_at=completed_at,
        owner_user_id=current.owner_user_id,
        event_type="status",
        event_payload=_status_event_payload(
            session_id=current.id,
            status=next_status,
            phase=next_phase,
            control_state=next_control_state,
            active_job_id=None,
            latest_checkpoint_id=current.latest_checkpoint_id,
            completed_at=completed_at,
        ),
        event_phase=next_phase,
        event_job_id=None,
    )
    if next_status in {"completed", "failed", "cancelled"}:
        db.record_run_event(
            owner_user_id=updated.owner_user_id,
            session_id=updated.id,
            event_type="terminal",
            event_payload=_status_event_payload(
                session_id=updated.id,
                status=updated.status,
                phase=updated.phase,
                control_state=updated.control_state,
                active_job_id=updated.active_job_id,
                latest_checkpoint_id=updated.latest_checkpoint_id,
                completed_at=updated.completed_at,
            ),
            phase=updated.phase,
            job_id=None,
        )
    return {
        "session_id": updated.id,
        "phase": updated.phase,
        "checkpoint_id": checkpoint_id,
        "artifacts_written": artifacts_written,
    }


def _load_effective_plan(*, session: Any, artifact_store: ResearchArtifactStore) -> ResearchPlan:
    plan_payload = artifact_store.read_json(session_id=session.id, artifact_name="approved_plan.json")
    if plan_payload is None:
        plan_payload = artifact_store.read_json(session_id=session.id, artifact_name="plan.json")
    if plan_payload is None:
        raise ValueError(f"missing research plan artifact for session {session.id}")

    focus_areas = [
        str(area).strip()
        for area in plan_payload.get("focus_areas", [])
        if str(area).strip()
    ]
    if not focus_areas:
        focus_areas = [session.query]

    return ResearchPlan(
        query=str(plan_payload.get("query") or session.query),
        focus_areas=focus_areas,
        source_policy=str(plan_payload.get("source_policy") or session.source_policy),
        autonomy_mode=str(plan_payload.get("autonomy_mode") or session.autonomy_mode),
        stop_criteria=(
            dict(plan_payload.get("stop_criteria"))
            if isinstance(plan_payload.get("stop_criteria"), dict)
            else {}
        ),
    )


__all__ = [
    "RESEARCH_DOMAIN",
    "RESEARCH_JOB_TYPE",
    "RESEARCH_QUEUE",
    "enqueue_research_phase_job",
    "handle_research_phase_job",
]
