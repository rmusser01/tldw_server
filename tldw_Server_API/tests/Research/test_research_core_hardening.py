import hashlib
import json
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


class _RecordingJobs:
    def __init__(self):
        self.created_jobs: list[dict[str, object]] = []
        self.cancelled_jobs: list[tuple[int, str | None]] = []

    def create_job(self, **kwargs):
        job_id = len(self.created_jobs) + 100
        job = {"id": job_id, "uuid": f"job-{job_id}", "status": "queued", **kwargs}
        self.created_jobs.append(job)
        return job

    def cancel_job(self, job_id: int, *, reason: str | None = None):
        self.cancelled_jobs.append((job_id, reason))
        return True


def _service_bundle(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    db_path = tmp_path / "research.db"
    outputs_dir = tmp_path / "outputs"
    jobs = _RecordingJobs()
    service = ResearchService(
        research_db_path=db_path,
        outputs_dir=outputs_dir,
        job_manager=jobs,
    )
    db = ResearchSessionsDB(db_path)
    store = ResearchArtifactStore(base_dir=outputs_dir, db=db)
    return service, db, store, jobs


def _create_session(db, *, owner_user_id: str = "1", phase: str = "drafting_plan", status: str = "queued", limits=None):
    return db.create_session(
        owner_user_id=owner_user_id,
        query=f"Research query for owner {owner_user_id}",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json=limits or {},
        phase=phase,
        status=status,
    )


def _write_bundle_artifact(store, session):
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="bundle.json",
        payload={"question": session.query, "claims": []},
        phase="completed",
        job_id=None,
    )


def test_service_rejects_foreign_owner_for_reads_artifacts_and_controls(tmp_path):
    service, db, store, _jobs = _service_bundle(tmp_path)

    readable = _create_session(db, phase="completed", status="completed")
    _write_bundle_artifact(store, readable)
    active = _create_session(db, phase="collecting", status="queued")
    db.attach_active_job(active.id, "41")
    paused = _create_session(db, phase="synthesizing", status="queued")
    db.update_control_state(paused.id, control_state="paused")

    with pytest.raises(KeyError):
        service.get_session(owner_user_id="2", session_id=readable.id)
    with pytest.raises(KeyError):
        service.get_stream_snapshot(owner_user_id="2", session_id=readable.id)
    with pytest.raises(KeyError):
        service.get_artifact(owner_user_id="2", session_id=readable.id, artifact_name="bundle.json")
    with pytest.raises(KeyError):
        service.get_bundle(owner_user_id="2", session_id=readable.id)
    with pytest.raises(KeyError):
        service.pause_run(owner_user_id="2", session_id=active.id)
    with pytest.raises(KeyError):
        service.resume_run(owner_user_id="2", session_id=paused.id)
    with pytest.raises(KeyError):
        service.cancel_run(owner_user_id="2", session_id=active.id)
    with pytest.raises(KeyError):
        service.build_package(
            owner_user_id="2",
            session_id=readable.id,
            brief={"query": readable.query},
            outline={},
            report_markdown="# Report",
            claims=[],
            source_inventory=[],
        )


def test_service_rejects_foreign_owner_checkpoint_approval(tmp_path):
    service, db, _store, jobs = _service_bundle(tmp_path)
    session = _create_session(db, phase="awaiting_plan_review", status="waiting_human")
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )

    with pytest.raises(KeyError):
        service.approve_checkpoint(
            owner_user_id="2",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
        )

    assert jobs.created_jobs == []


def test_checkpoint_approval_rejects_resolved_stale_and_wrong_phase_checkpoints(tmp_path):
    service, db, _store, jobs = _service_bundle(tmp_path)
    session = _create_session(db, phase="awaiting_plan_review", status="waiting_human")
    approved = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )
    service.approve_checkpoint(owner_user_id="1", session_id=session.id, checkpoint_id=approved.id)

    with pytest.raises(ValueError, match="checkpoint"):
        service.approve_checkpoint(owner_user_id="1", session_id=session.id, checkpoint_id=approved.id)

    wrong_phase = _create_session(db, phase="awaiting_source_review", status="waiting_human")
    wrong_phase_checkpoint = db.create_checkpoint(
        session_id=wrong_phase.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": wrong_phase.query,
            "focus_areas": ["background"],
            "source_policy": wrong_phase.source_policy,
            "autonomy_mode": wrong_phase.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )
    with pytest.raises(ValueError, match="checkpoint"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=wrong_phase.id,
            checkpoint_id=wrong_phase_checkpoint.id,
        )

    stale_session = _create_session(db, phase="awaiting_plan_review", status="waiting_human")
    stale_checkpoint = db.create_checkpoint(
        session_id=stale_session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": stale_session.query,
            "focus_areas": ["old"],
            "source_policy": stale_session.source_policy,
            "autonomy_mode": stale_session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )
    db.create_checkpoint(
        session_id=stale_session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": stale_session.query,
            "focus_areas": ["new"],
            "source_policy": stale_session.source_policy,
            "autonomy_mode": stale_session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )
    with pytest.raises(ValueError, match="checkpoint"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=stale_session.id,
            checkpoint_id=stale_checkpoint.id,
        )

    assert len(jobs.created_jobs) == 1


def test_artifact_versions_use_distinct_immutable_files(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = _create_session(db)
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)

    first = store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["old"]},
        phase="drafting_plan",
        job_id=None,
    )
    second = store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["new"]},
        phase="drafting_plan",
        job_id=None,
    )

    assert first.storage_path != second.storage_path
    for artifact, expected in ((first, {"focus_areas": ["old"]}), (second, {"focus_areas": ["new"]})):
        content = Path(artifact.storage_path).read_bytes()
        assert json.loads(content.decode("utf-8")) == expected
        assert hashlib.sha256(content).hexdigest() == artifact.checksum


def test_artifact_storage_paths_do_not_collide_when_version_allocator_repeats(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = _create_session(db)
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    monkeypatch.setattr(store, "_next_version", lambda session_id, artifact_name: 1)

    first = store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["first"]},
        phase="drafting_plan",
        job_id=None,
    )
    second = store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["second"]},
        phase="drafting_plan",
        job_id=None,
    )

    assert first.artifact_version == second.artifact_version == 1
    assert first.storage_path != second.storage_path
    assert json.loads(Path(first.storage_path).read_text(encoding="utf-8")) == {"focus_areas": ["first"]}
    assert json.loads(Path(second.storage_path).read_text(encoding="utf-8")) == {"focus_areas": ["second"]}


@pytest.mark.asyncio
async def test_planning_phase_offloads_artifact_writes_from_event_loop(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research import jobs
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db_path = tmp_path / "research.db"
    db = ResearchSessionsDB(db_path)
    session = _create_session(db, phase="drafting_plan", status="queued")
    offloaded: list[str] = []

    async def fake_to_thread(func, /, *args, **kwargs):
        offloaded.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(jobs.asyncio, "to_thread", fake_to_thread)

    result = await handle_research_phase_job(
        {"id": 5, "payload": {"session_id": session.id, "phase": "drafting_plan"}},
        research_db_path=db_path,
        outputs_dir=tmp_path / "outputs",
    )

    assert result["phase"] == "awaiting_plan_review"
    assert offloaded == ["write_json", "write_json"]


@pytest.mark.asyncio
async def test_research_worker_resolves_per_user_paths_when_no_override(monkeypatch, tmp_path):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.core.Research import jobs_worker

    previous_user_db_base = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(tmp_path / "user_dbs")
    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    monkeypatch.delenv("RESEARCH_SESSIONS_DB_PATH", raising=False)
    monkeypatch.delenv("RESEARCH_OUTPUTS_DIR", raising=False)
    captured: dict[str, Path] = {}

    class FakeSDK:
        def __init__(self, jm, cfg):
            self.jm = jm
            self.cfg = cfg

        async def run(self, *, handler, **kwargs):
            await handler(
                {
                    "id": 1,
                    "owner_user_id": "7",
                    "payload": {"session_id": "rs_1", "phase": "drafting_plan"},
                }
            )

        def stop(self):
            return None

    async def fake_handle(job, *, research_db_path, outputs_dir):
        captured["research_db_path"] = Path(research_db_path)
        captured["outputs_dir"] = Path(outputs_dir)
        return {"ok": True}

    monkeypatch.setattr(jobs_worker, "WorkerSDK", FakeSDK)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object())
    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", fake_handle)

    expected_research_db_path = DatabasePaths.get_research_sessions_db_path("7")
    expected_outputs_dir = DatabasePaths.get_user_outputs_dir("7")
    try:
        await jobs_worker.run_research_jobs_worker()
    finally:
        if previous_user_db_base is not None:
            settings.USER_DB_BASE_DIR = previous_user_db_base
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass

    assert captured["research_db_path"] == expected_research_db_path
    assert captured["outputs_dir"] == expected_outputs_dir


@pytest.mark.asyncio
async def test_collecting_phase_stops_between_focus_areas_when_cancel_requested(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )

    db_path = tmp_path / "research.db"
    db = ResearchSessionsDB(db_path)
    session = _create_session(db, phase="collecting", status="queued")
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["first", "second"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="drafting_plan",
        job_id=None,
    )

    class CancelAfterFirstBroker:
        def __init__(self):
            self.calls: list[str] = []

        async def collect_focus_area(self, **kwargs):
            focus_area = str(kwargs["focus_area"])
            self.calls.append(focus_area)
            db.update_control_state(session.id, control_state="cancel_requested")
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id=f"src_{focus_area}",
                        focus_area=focus_area,
                        source_type="local_document",
                        provider="local_corpus",
                        title=f"Source {focus_area}",
                        url=None,
                        snippet="Evidence",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint=f"fp_{focus_area}",
                        trust_tier="internal",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id=f"note_{focus_area}",
                        source_id=f"src_{focus_area}",
                        focus_area=focus_area,
                        kind="supporting",
                        text="Evidence",
                        citation_locator=None,
                        confidence=0.8,
                        metadata={},
                    )
                ],
                collection_metrics={
                    "lane_counts": {"local": 1, "academic": 0, "web": 0},
                    "lane_attempts": {"local": 1, "academic": 0, "web": 0},
                    "deduped_sources": 0,
                },
                remaining_gaps=[],
            )

    broker = CancelAfterFirstBroker()

    result = await handle_research_phase_job(
        {"id": 5, "payload": {"session_id": session.id, "phase": "collecting"}},
        research_db_path=db_path,
        outputs_dir=tmp_path / "outputs",
        broker=broker,
    )

    assert broker.calls == ["first"]
    assert result["phase"] == "collecting"
    assert db.get_session(session.id).status == "cancelled"


@pytest.mark.asyncio
async def test_collecting_phase_enforces_search_limit_before_provider_call(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db_path = tmp_path / "research.db"
    db = ResearchSessionsDB(db_path)
    session = _create_session(
        db,
        phase="collecting",
        status="queued",
        limits={"max_searches": 0, "max_fetched_docs": 10, "max_runtime_seconds": 300},
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["first"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="drafting_plan",
        job_id=None,
    )

    class Broker:
        def __init__(self):
            self.calls = 0

        async def collect_focus_area(self, **kwargs):
            self.calls += 1
            raise AssertionError("provider should not be called after budget exhaustion")

    broker = Broker()

    with pytest.raises(ValueError, match="research_limit_exceeded"):
        await handle_research_phase_job(
            {"id": 5, "payload": {"session_id": session.id, "phase": "collecting"}},
            research_db_path=db_path,
            outputs_dir=tmp_path / "outputs",
            broker=broker,
        )

    assert broker.calls == 0


@pytest.mark.asyncio
async def test_collecting_phase_enforces_fetched_doc_limit_before_next_provider_call(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )

    db_path = tmp_path / "research.db"
    db = ResearchSessionsDB(db_path)
    session = _create_session(
        db,
        phase="collecting",
        status="queued",
        limits={"max_searches": 10, "max_fetched_docs": 1, "max_runtime_seconds": 300},
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["first", "second"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="drafting_plan",
        job_id=None,
    )

    class Broker:
        def __init__(self):
            self.calls: list[str] = []

        async def collect_focus_area(self, **kwargs):
            focus_area = str(kwargs["focus_area"])
            self.calls.append(focus_area)
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id=f"src_{focus_area}",
                        focus_area=focus_area,
                        source_type="local_document",
                        provider="local_corpus",
                        title=f"Source {focus_area}",
                        url=None,
                        snippet="Evidence",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint=f"fp_{focus_area}",
                        trust_tier="internal",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id=f"note_{focus_area}",
                        source_id=f"src_{focus_area}",
                        focus_area=focus_area,
                        kind="supporting",
                        text="Evidence",
                        citation_locator=None,
                        confidence=0.8,
                        metadata={},
                    )
                ],
                collection_metrics={
                    "lane_counts": {"local": 1, "academic": 0, "web": 0},
                    "lane_attempts": {"local": 1, "academic": 0, "web": 0},
                    "deduped_sources": 0,
                },
                remaining_gaps=[],
            )

    broker = Broker()

    with pytest.raises(ValueError, match="research_limit_exceeded:fetched_docs"):
        await handle_research_phase_job(
            {"id": 5, "payload": {"session_id": session.id, "phase": "collecting"}},
            research_db_path=db_path,
            outputs_dir=tmp_path / "outputs",
            broker=broker,
        )

    assert broker.calls == ["first"]
