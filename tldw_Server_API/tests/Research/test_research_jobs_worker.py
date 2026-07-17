from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _assert_artifacts_exist(
    db,
    base_dir: Path,
    session_id: str,
    artifact_names: set[str],
) -> None:
    artifacts = {
        artifact.artifact_name: artifact
        for artifact in db.list_artifacts(session_id)
    }
    assert artifact_names <= artifacts.keys()
    research_root = (base_dir / "research").resolve()
    for name in artifact_names:
        path = Path(artifacts[name].storage_path).resolve()
        path.relative_to(research_root)
        assert path.is_file()


class _ProviderStub:
    def __init__(self, records=None, *, error: Exception | None = None):
        self._records = list(records or [])
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def search(self, *, focus_area: str, query: str, owner_user_id: str, config: dict[str, object]):
        self.calls.append(
            {
                "focus_area": focus_area,
                "query": query,
                "owner_user_id": owner_user_id,
                "config": dict(config),
            }
        )
        if self._error is not None:
            raise self._error
        return list(self._records)


class _SynthesisProviderStub:
    def __init__(self, response=None, *, error: Exception | None = None):
        self._response = response
        self._error = error
        self.calls: list[dict[str, object]] = []

    async def summarize(
        self,
        *,
        plan,
        source_registry,
        evidence_notes,
        collection_summary,
        config,
    ):
        self.calls.append(
            {
                "plan": plan,
                "source_registry": list(source_registry),
                "evidence_notes": list(evidence_notes),
                "collection_summary": dict(collection_summary or {}),
                "config": dict(config),
            }
        )
        if self._error is not None:
            raise self._error
        return self._response


def _seed_packaging_artifacts(*, store, session) -> None:
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="outline_v1.json",
        payload={
            "query": session.query,
            "sections": [
                {
                    "title": "Evidence Alignment",
                    "focus_area": "evidence alignment",
                    "source_ids": ["src_1"],
                    "note_ids": ["note_1"],
                }
            ],
            "unresolved_questions": [],
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="claims.json",
        payload={
            "claims": [
                {
                    "claim_id": "clm_1",
                    "text": "Evidence is aligned.",
                    "focus_area": "evidence alignment",
                    "source_ids": ["src_1"],
                    "citations": [{"source_id": "src_1"}],
                    "confidence": 0.8,
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )
    store.write_text(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="report_v1.md",
        content="# Research Report\n\n## Evidence Alignment\n\n- Evidence is aligned. [Sources: src_1]",
        phase="packaging",
        job_id=None,
        content_type="text/markdown",
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "focus_area": "evidence alignment",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Internal source",
                    "url": None,
                    "snippet": "Internal note summary",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_1",
                    "trust_tier": "internal",
                    "metadata": {},
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="synthesis_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "section_count": 1,
            "claim_count": 1,
            "source_count": 1,
            "unresolved_questions": [],
            "coverage": {"covered_focus_areas": ["evidence alignment"], "missing_focus_areas": []},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="verification_summary.json",
        payload={
            "supported_claim_count": 1,
            "unsupported_claim_count": 0,
            "contradiction_count": 0,
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="unsupported_claims.json",
        payload={"claims": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="contradictions.json",
        payload={"contradictions": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="source_trust.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "snapshot_policy": "full_artifact",
                    "trust_label": "local_corpus",
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )


def _set_user_db_base(monkeypatch: pytest.MonkeyPatch, base_dir) -> str | None:
    from tldw_Server_API.app.core.config import settings

    previous = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))
    return previous


def _restore_user_db_base(previous: str | None) -> None:
    from tldw_Server_API.app.core.config import settings

    if previous is not None:
        settings.USER_DB_BASE_DIR = previous
        return
    try:
        del settings.USER_DB_BASE_DIR
    except AttributeError:
        pass


def _seed_chat_thread(*, owner_user_id: str, chat_db_path) -> tuple[str, str]:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    chat_db = CharactersRAGDB(str(chat_db_path), client_id=owner_user_id)
    character_id = chat_db.add_character_card(
        {
            "name": DEFAULT_CHARACTER_NAME,
            "description": "Research thread helper",
            "personality": "Helpful",
            "scenario": "Research completion testing",
            "system_prompt": "You are a research assistant.",
            "first_message": "Hello",
            "creator_notes": "Created by deep research tests",
        }
    )
    chat_id = chat_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Deep Research Chat",
            "client_id": owner_user_id,
        }
    )
    launch_message_id = chat_db.add_message(
        {
            "conversation_id": chat_id,
            "sender": "user",
            "content": "Please run deep research on this topic.",
            "client_id": owner_user_id,
        }
    )
    assert chat_id is not None
    assert launch_message_id is not None
    return chat_id, launch_message_id


@pytest.mark.asyncio
async def test_planning_job_writes_plan_and_opens_checkpoint(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test planning",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )

    result = await handle_research_phase_job(
        {
            "id": 5,
            "payload": {
                "session_id": session.id,
                "phase": "drafting_plan",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "awaiting_plan_review"
    assert updated.latest_checkpoint_id is not None
    assert result["phase"] == "awaiting_plan_review"
    assert result["artifacts_written"] >= 1
    _assert_artifacts_exist(db, tmp_path / "outputs", session.id, {"plan.json"})


@pytest.mark.asyncio
async def test_planning_job_records_progress_artifact_checkpoint_and_status_events(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Planning event log coverage",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )

    await handle_research_phase_job(
        {
            "id": 55,
            "payload": {
                "session_id": session.id,
                "phase": "drafting_plan",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    events = db.list_run_events_after(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        after_id=0,
    )

    assert [event.event_type for event in events] == [
        "progress",
        "artifact",
        "artifact",
        "checkpoint",
        "status",
    ]
    assert events[-1].event_payload["phase"] == "awaiting_plan_review"
    assert events[-1].event_payload["status"] == "waiting_human"


@pytest.mark.asyncio
async def test_collecting_job_writes_artifacts_and_opens_source_review(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test collecting",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )

    class StubBroker:
        async def collect_focus_area(self, **kwargs):
            _ = kwargs
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id="src_1",
                        focus_area="evidence alignment",
                        source_type="local_document",
                        provider="local_corpus",
                        title="Internal source",
                        url=None,
                        snippet="Internal note summary",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_1",
                        trust_tier="internal",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id="note_1",
                        source_id="src_1",
                        focus_area="evidence alignment",
                        kind="summary",
                        text="Internal note summary",
                        citation_locator=None,
                        confidence=0.8,
                        metadata={},
                    )
                ],
                collection_metrics={"lane_counts": {"local": 1, "academic": 0, "web": 0}, "deduped_sources": 0},
                remaining_gaps=[],
            )

    result = await handle_research_phase_job(
        {
            "id": 6,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=StubBroker(),
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "awaiting_source_review"
    assert updated.latest_checkpoint_id is not None
    assert result["phase"] == "awaiting_source_review"
    assert result["artifacts_written"] >= 3
    _assert_artifacts_exist(
        db,
        tmp_path / "outputs",
        session.id,
        {"source_registry.json", "evidence_notes.jsonl", "collection_summary.json"},
    )


@pytest.mark.asyncio
async def test_collecting_job_advances_autonomous_run_to_synthesizing(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import ResearchCollectionResult

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test collecting",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )

    class StubBroker:
        async def collect_focus_area(self, **kwargs):
            _ = kwargs
            return ResearchCollectionResult(
                sources=[],
                evidence_notes=[],
                collection_metrics={"lane_counts": {"local": 0, "academic": 0, "web": 0}, "deduped_sources": 0},
                remaining_gaps=["no_sources_collected"],
            )

    result = await handle_research_phase_job(
        {
            "id": 7,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=StubBroker(),
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "synthesizing"
    assert result["phase"] == "synthesizing"


@pytest.mark.asyncio
async def test_collecting_job_parks_paused_before_phase_start(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Pause before collecting",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    db.update_control_state(session.id, control_state="pause_requested")
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )

    class StubBroker:
        async def collect_focus_area(self, **kwargs):
            raise AssertionError(f"phase should have paused before broker call: {kwargs}")

    await handle_research_phase_job(
        {
            "id": 700,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=StubBroker(),
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.control_state == "paused"
    assert updated.phase == "collecting"
    assert updated.active_job_id is None
    assert updated.progress_percent is None
    assert updated.progress_message is None


@pytest.mark.asyncio
async def test_collecting_job_advances_phase_and_parks_paused_when_pause_requested_during_work(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Pause after collecting work",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )

    class StubBroker:
        async def collect_focus_area(self, **kwargs):
            db.update_control_state(session.id, control_state="pause_requested")
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id="src_pause",
                        focus_area="background",
                        source_type="local_document",
                        provider="local_corpus",
                        title="Internal source",
                        url=None,
                        snippet="Internal summary",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_pause",
                        trust_tier="internal",
                        metadata={},
                    )
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id="note_pause",
                        source_id="src_pause",
                        focus_area="background",
                        kind="summary",
                        text="Internal summary",
                        citation_locator=None,
                        confidence=0.9,
                        metadata={},
                    )
                ],
                collection_metrics={"lane_counts": {"local": 1, "academic": 0, "web": 0}, "deduped_sources": 0},
                remaining_gaps=[],
            )

    await handle_research_phase_job(
        {
            "id": 701,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=StubBroker(),
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "synthesizing"
    assert updated.status == "queued"
    assert updated.control_state == "paused"
    assert updated.active_job_id is None
    assert updated.progress_percent == 45.0
    assert updated.progress_message == "collecting sources"


@pytest.mark.asyncio
async def test_collecting_job_reads_provider_config_and_passes_lane_settings(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.broker import ResearchBroker
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test collecting config",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
        payload={
            "local": {"top_k": 4, "sources": ["media_db"]},
            "academic": {"providers": ["arxiv"], "max_results": 2},
            "web": {"engine": "kagi", "result_count": 3},
            "synthesis": {"provider": None, "model": None, "temperature": 0.2},
        },
        phase="collecting",
        job_id=None,
    )

    local_provider = _ProviderStub(
        [{"id": "doc-1", "title": "Internal source", "content": "Internal note summary", "provider": "local_corpus"}]
    )
    academic_provider = _ProviderStub(
        [{"doi": "10.1000/example", "title": "Academic source", "summary": "paper", "provider": "arxiv"}]
    )
    web_provider = _ProviderStub(
        [{"url": "https://example.com/report", "title": "Web source", "snippet": "report", "provider": "kagi"}]
    )

    result = await handle_research_phase_job(
        {
            "id": 70,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=ResearchBroker(
            local_provider=local_provider,
            academic_provider=academic_provider,
            web_provider=web_provider,
        ),
    )

    assert result["phase"] == "synthesizing"
    assert local_provider.calls[0]["config"] == {"top_k": 4, "sources": ["media_db"]}
    assert academic_provider.calls[0]["config"] == {"providers": ["arxiv"], "max_results": 2}
    assert web_provider.calls[0]["config"] == {"engine": "kagi", "result_count": 3}


@pytest.mark.asyncio
async def test_collecting_job_records_lane_errors_and_still_advances_when_one_lane_succeeds(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.broker import ResearchBroker
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test collecting lane errors",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
        payload={
            "local": {"top_k": 5, "sources": ["media_db"]},
            "academic": {"providers": ["arxiv"], "max_results": 2},
            "web": {"engine": "duckduckgo", "result_count": 3},
            "synthesis": {"provider": None, "model": None, "temperature": 0.2},
        },
        phase="collecting",
        job_id=None,
    )

    broker = ResearchBroker(
        local_provider=_ProviderStub(
            [{"id": "doc-1", "title": "Internal source", "content": "Internal note summary", "provider": "local_corpus"}]
        ),
        academic_provider=_ProviderStub([]),
        web_provider=_ProviderStub(error=RuntimeError("web search failed")),
    )

    result = await handle_research_phase_job(
        {
            "id": 71,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=broker,
    )

    collection_summary = store.read_json(session_id=session.id, artifact_name="collection_summary.json")

    assert result["phase"] == "synthesizing"
    assert collection_summary is not None
    assert collection_summary["lane_errors"] == [
        {
            "focus_area": "evidence alignment",
            "lane": "web",
            "message": "web search failed",
        }
    ]


@pytest.mark.asyncio
async def test_collecting_job_recollects_with_pinned_and_dropped_source_directives(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.models import (
        ResearchCollectionResult,
        ResearchEvidenceNote,
        ResearchSourceRecord,
    )

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Refresh evidence coverage",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="collecting",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_keep",
                    "focus_area": "evidence alignment",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Pinned source",
                    "url": None,
                    "snippet": "Keep this source",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_keep",
                    "trust_tier": "internal",
                    "metadata": {},
                },
                {
                    "source_id": "src_drop",
                    "focus_area": "evidence alignment",
                    "source_type": "web_result",
                    "provider": "kagi",
                    "title": "Dropped source",
                    "url": "https://example.com/drop",
                    "snippet": "Drop this source",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_drop",
                    "trust_tier": "medium",
                    "metadata": {},
                },
            ]
        },
        phase="collecting",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[
            {
                "note_id": "note_keep",
                "source_id": "src_keep",
                "focus_area": "evidence alignment",
                "kind": "summary",
                "text": "Pinned evidence remains useful.",
                "citation_locator": None,
                "confidence": 0.8,
                "metadata": {},
            },
            {
                "note_id": "note_drop",
                "source_id": "src_drop",
                "focus_area": "evidence alignment",
                "kind": "summary",
                "text": "Dropped evidence should disappear.",
                "citation_locator": None,
                "confidence": 0.4,
                "metadata": {},
            },
        ],
        phase="collecting",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_sources.json",
        payload={
            "pinned_source_ids": ["src_keep"],
            "dropped_source_ids": ["src_drop"],
            "prioritized_source_ids": ["src_keep"],
            "recollect": {
                "enabled": True,
                "need_primary_sources": True,
                "need_contradictions": True,
                "guidance": "Refresh with contradictory primary sources.",
            },
        },
        phase="collecting",
        job_id=None,
    )

    class StubBroker:
        def __init__(self):
            self.calls: list[dict[str, object]] = []

        async def collect_focus_area(self, **kwargs):
            self.calls.append(kwargs)
            return ResearchCollectionResult(
                sources=[
                    ResearchSourceRecord(
                        source_id="src_new",
                        focus_area="evidence alignment",
                        source_type="academic_paper",
                        provider="arxiv",
                        title="New source",
                        url="https://arxiv.org/abs/1234.5678",
                        snippet="New source snippet",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_new",
                        trust_tier="high",
                        metadata={},
                    ),
                    ResearchSourceRecord(
                        source_id="src_drop",
                        focus_area="evidence alignment",
                        source_type="web_result",
                        provider="kagi",
                        title="Dropped source duplicate",
                        url="https://example.com/drop",
                        snippet="Should still be filtered",
                        published_at=None,
                        retrieved_at="2026-03-07T00:00:00+00:00",
                        fingerprint="fp_drop_2",
                        trust_tier="medium",
                        metadata={},
                    ),
                ],
                evidence_notes=[
                    ResearchEvidenceNote(
                        note_id="note_new",
                        source_id="src_new",
                        focus_area="evidence alignment",
                        kind="summary",
                        text="New contradictory evidence.",
                        citation_locator=None,
                        confidence=0.9,
                        metadata={},
                    ),
                    ResearchEvidenceNote(
                        note_id="note_drop_new",
                        source_id="src_drop",
                        focus_area="evidence alignment",
                        kind="summary",
                        text="Dropped evidence should remain filtered.",
                        citation_locator=None,
                        confidence=0.2,
                        metadata={},
                    ),
                ],
                collection_metrics={"lane_counts": {"local": 0, "academic": 1, "web": 0}, "deduped_sources": 0},
                remaining_gaps=[],
            )

    broker = StubBroker()

    result = await handle_research_phase_job(
        {
            "id": 72,
            "payload": {
                "session_id": session.id,
                "phase": "collecting",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        broker=broker,
    )

    updated = db.get_session(session.id)
    source_registry = store.read_json(session_id=session.id, artifact_name="source_registry.json")
    evidence_notes = store.read_jsonl(session_id=session.id, artifact_name="evidence_notes.jsonl")
    collection_summary = store.read_json(session_id=session.id, artifact_name="collection_summary.json")

    assert updated is not None
    assert updated.phase == "awaiting_source_review"
    assert result["phase"] == "awaiting_source_review"
    assert broker.calls[0]["context"]["recollect"]["enabled"] is True
    assert broker.calls[0]["context"]["recollect"]["guidance"] == "Refresh with contradictory primary sources."
    assert source_registry is not None
    assert [item["source_id"] for item in source_registry["sources"]] == ["src_keep", "src_new"]
    assert evidence_notes is not None
    assert [item["note_id"] for item in evidence_notes] == ["note_keep", "note_new"]
    assert collection_summary is not None
    assert collection_summary["review_directives"]["recollect"]["need_primary_sources"] is True


@pytest.mark.asyncio
async def test_synthesizing_job_writes_artifacts_and_opens_outline_review(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test synthesizing",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "focus_area": "evidence alignment",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Internal source",
                    "url": None,
                    "snippet": "Internal note summary",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_1",
                    "trust_tier": "internal",
                    "metadata": {},
                }
            ]
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[
            {
                "note_id": "note_1",
                "source_id": "src_1",
                "focus_area": "evidence alignment",
                "kind": "summary",
                "text": "Internal evidence remains aligned.",
                "citation_locator": None,
                "confidence": 0.8,
                "metadata": {},
            }
        ],
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_count": 1,
            "evidence_note_count": 1,
            "remaining_gaps": [],
            "collection_metrics": {"lane_counts": {"local": 1, "academic": 0, "web": 0}, "deduped_sources": 0},
        },
        phase="synthesizing",
        job_id=None,
    )

    result = await handle_research_phase_job(
        {
            "id": 8,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "awaiting_outline_review"
    assert updated.latest_checkpoint_id is not None
    assert result["phase"] == "awaiting_outline_review"
    assert result["artifacts_written"] >= 4
    _assert_artifacts_exist(
        db,
        tmp_path / "outputs",
        session.id,
        {
            "outline_v1.json",
            "claims.json",
            "report_v1.md",
            "synthesis_summary.json",
            "verification_summary.json",
            "unsupported_claims.json",
            "contradictions.json",
            "source_trust.json",
        },
    )


@pytest.mark.asyncio
async def test_synthesizing_job_advances_autonomous_run_to_packaging(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test synthesizing",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={"sources": []},
        phase="synthesizing",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[],
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_count": 0,
            "evidence_note_count": 0,
            "remaining_gaps": ["no_sources_collected"],
            "collection_metrics": {"lane_counts": {"local": 0, "academic": 0, "web": 0}, "deduped_sources": 0},
        },
        phase="synthesizing",
        job_id=None,
    )

    result = await handle_research_phase_job(
        {
            "id": 9,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "packaging"
    assert result["phase"] == "packaging"


@pytest.mark.asyncio
async def test_synthesizing_job_uses_provider_config_and_writes_llm_backed_summary(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Provider-backed synthesizing",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
        payload={
            "local": {"top_k": 5, "sources": ["media_db"]},
            "academic": {"providers": ["arxiv"], "max_results": 2},
            "web": {"engine": "duckduckgo", "result_count": 3},
            "synthesis": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "focus_area": "background",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Internal source",
                    "url": None,
                    "snippet": "Internal note summary",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_1",
                    "trust_tier": "internal",
                    "metadata": {},
                }
            ]
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[
            {
                "note_id": "note_1",
                "source_id": "src_1",
                "focus_area": "background",
                "kind": "summary",
                "text": "Internal evidence remains aligned.",
                "citation_locator": None,
                "confidence": 0.8,
                "metadata": {},
            }
        ],
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_count": 1,
            "evidence_note_count": 1,
            "remaining_gaps": [],
            "collection_metrics": {"lane_counts": {"local": 1, "academic": 0, "web": 0}, "deduped_sources": 0},
        },
        phase="synthesizing",
        job_id=None,
    )

    provider = _SynthesisProviderStub(
        {
            "outline_sections": [
                {
                    "title": "Background",
                    "focus_area": "background",
                    "source_ids": ["src_1"],
                    "note_ids": ["note_1"],
                }
            ],
            "claims": [
                {
                    "text": "Supported claim",
                    "focus_area": "background",
                    "source_ids": ["src_1"],
                    "citations": [{"source_id": "src_1"}],
                    "confidence": 0.81,
                }
            ],
            "report_sections": [
                {"title": "Background", "markdown": "Evidence-backed section text."}
            ],
            "unresolved_questions": [],
            "summary": {"mode": "llm_backed"},
        }
    )

    result = await handle_research_phase_job(
        {
            "id": 81,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        synthesizer=ResearchSynthesizer(synthesis_provider=provider),
    )

    summary = store.read_json(session_id=session.id, artifact_name="synthesis_summary.json")

    assert result["phase"] == "packaging"
    assert provider.calls[0]["config"] == {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2}
    assert summary is not None
    assert summary["mode"] == "llm_backed"


@pytest.mark.asyncio
async def test_synthesizing_job_filters_dropped_sources_before_provider_backed_synthesis(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.synthesizer import ResearchSynthesizer

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Curated synthesis",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_keep",
                    "focus_area": "background",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Pinned source",
                    "url": None,
                    "snippet": "Keep this source",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_keep",
                    "trust_tier": "internal",
                    "metadata": {},
                },
                {
                    "source_id": "src_drop",
                    "focus_area": "background",
                    "source_type": "web_result",
                    "provider": "kagi",
                    "title": "Dropped source",
                    "url": "https://example.com/drop",
                    "snippet": "Drop this source",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_drop",
                    "trust_tier": "medium",
                    "metadata": {},
                },
            ]
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[
            {
                "note_id": "note_keep",
                "source_id": "src_keep",
                "focus_area": "background",
                "kind": "summary",
                "text": "Grounded evidence.",
                "citation_locator": None,
                "confidence": 0.8,
                "metadata": {},
            },
            {
                "note_id": "note_drop",
                "source_id": "src_drop",
                "focus_area": "background",
                "kind": "summary",
                "text": "Dropped evidence should be filtered.",
                "citation_locator": None,
                "confidence": 0.4,
                "metadata": {},
            },
        ],
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_count": 2,
            "evidence_note_count": 2,
            "remaining_gaps": [],
            "collection_metrics": {"lane_counts": {"local": 1, "academic": 0, "web": 1}, "deduped_sources": 0},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
        payload={"synthesis": {"provider": "openai", "model": "gpt-4.1-mini", "temperature": 0.2}},
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_sources.json",
        payload={
            "pinned_source_ids": ["src_keep"],
            "dropped_source_ids": ["src_drop"],
            "prioritized_source_ids": ["src_keep"],
            "recollect": {
                "enabled": False,
                "need_primary_sources": False,
                "need_contradictions": False,
                "guidance": "",
            },
        },
        phase="synthesizing",
        job_id=None,
    )

    provider = _SynthesisProviderStub(
        {
            "outline_sections": [
                {
                    "title": "Background",
                    "focus_area": "background",
                    "source_ids": ["src_keep"],
                    "note_ids": ["note_keep"],
                }
            ],
            "claims": [
                {
                    "text": "Supported claim",
                    "focus_area": "background",
                    "source_ids": ["src_keep"],
                    "citations": [{"source_id": "src_keep"}],
                    "confidence": 0.9,
                }
            ],
            "report_sections": [{"title": "Background", "markdown": "Evidence-backed."}],
            "unresolved_questions": [],
            "summary": {"mode": "llm_backed"},
        }
    )

    await handle_research_phase_job(
        {
            "id": 90,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        synthesizer=ResearchSynthesizer(synthesis_provider=provider),
    )

    assert [source.source_id for source in provider.calls[0]["source_registry"]] == ["src_keep"]
    assert [note.note_id for note in provider.calls[0]["evidence_notes"]] == ["note_keep"]


@pytest.mark.asyncio
async def test_synthesizing_job_with_locked_outline_skips_outline_review_and_uses_outline_seed(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Locked outline synthesis",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background", "counterevidence"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_background",
                    "focus_area": "background",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Background source",
                    "url": None,
                    "snippet": "Background note",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_background",
                    "trust_tier": "internal",
                    "metadata": {},
                },
                {
                    "source_id": "src_counter",
                    "focus_area": "counterevidence",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Counter source",
                    "url": None,
                    "snippet": "Counter note",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_counter",
                    "trust_tier": "internal",
                    "metadata": {},
                },
            ]
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[
            {
                "note_id": "note_background",
                "source_id": "src_background",
                "focus_area": "background",
                "kind": "summary",
                "text": "Background evidence.",
                "citation_locator": None,
                "confidence": 0.8,
                "metadata": {},
            },
            {
                "note_id": "note_counter",
                "source_id": "src_counter",
                "focus_area": "counterevidence",
                "kind": "summary",
                "text": "Counterevidence note.",
                "citation_locator": None,
                "confidence": 0.7,
                "metadata": {},
            },
        ],
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="collection_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["background", "counterevidence"],
            "source_count": 2,
            "evidence_note_count": 2,
            "remaining_gaps": [],
            "collection_metrics": {"lane_counts": {"local": 2, "academic": 0, "web": 0}, "deduped_sources": 0},
        },
        phase="synthesizing",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="approved_outline.json",
        payload={
            "sections": [
                {"title": "Counterevidence First", "focus_area": "counterevidence"},
                {"title": "Background Context", "focus_area": "background"},
            ]
        },
        phase="synthesizing",
        job_id=None,
    )

    result = await handle_research_phase_job(
        {
            "id": 91,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
                "approved_outline_locked": True,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    outline_payload = store.read_json(session_id=session.id, artifact_name="outline_v1.json")
    report_markdown = store.read_text(session_id=session.id, artifact_name="report_v1.md")
    synthesis_summary = store.read_json(session_id=session.id, artifact_name="synthesis_summary.json")

    assert updated is not None
    assert updated.phase == "packaging"
    assert updated.latest_checkpoint_id is None
    assert result["phase"] == "packaging"
    assert outline_payload is not None
    assert [section["title"] for section in outline_payload["sections"]] == [
        "Counterevidence First",
        "Background Context",
    ]
    assert report_markdown is not None
    assert "## Counterevidence First" in report_markdown
    assert "## Background Context" in report_markdown
    assert synthesis_summary is not None
    assert synthesis_summary["mode"] == "deterministic_outline_locked"


@pytest.mark.asyncio
async def test_packaging_job_writes_bundle_and_completes_session(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test packaging",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="packaging",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="outline_v1.json",
        payload={
            "query": session.query,
            "sections": [
                {
                    "title": "Evidence Alignment",
                    "focus_area": "evidence alignment",
                    "source_ids": ["src_1"],
                    "note_ids": ["note_1"],
                }
            ],
            "unresolved_questions": [],
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="claims.json",
        payload={
            "claims": [
                {
                    "claim_id": "clm_1",
                    "text": "Evidence is aligned.",
                    "focus_area": "evidence alignment",
                    "source_ids": ["src_1"],
                    "citations": [{"source_id": "src_1"}],
                    "confidence": 0.8,
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )
    store.write_text(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="report_v1.md",
        content="# Research Report\n\n## Evidence Alignment\n\n- Evidence is aligned. [Sources: src_1]",
        phase="packaging",
        job_id=None,
        content_type="text/markdown",
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "focus_area": "evidence alignment",
                    "source_type": "local_document",
                    "provider": "local_corpus",
                    "title": "Internal source",
                    "url": None,
                    "snippet": "Internal note summary",
                    "published_at": None,
                    "retrieved_at": "2026-03-07T00:00:00+00:00",
                    "fingerprint": "fp_1",
                    "trust_tier": "internal",
                    "metadata": {},
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="synthesis_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "section_count": 1,
            "claim_count": 1,
            "source_count": 1,
            "unresolved_questions": [],
            "coverage": {"covered_focus_areas": ["evidence alignment"], "missing_focus_areas": []},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="verification_summary.json",
        payload={
            "supported_claim_count": 1,
            "unsupported_claim_count": 0,
            "contradiction_count": 0,
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="unsupported_claims.json",
        payload={"claims": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="contradictions.json",
        payload={"contradictions": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_trust.json",
        payload={
            "sources": [
                {
                    "source_id": "src_1",
                    "snapshot_policy": "full_artifact",
                    "trust_label": "local_corpus",
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )

    result = await handle_research_phase_job(
        {
            "id": 10,
            "payload": {
                "session_id": session.id,
                "phase": "packaging",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.phase == "completed"
    assert updated.status == "completed"
    assert updated.completed_at is not None
    assert updated.progress_percent == 100.0
    assert updated.progress_message == "packaging results"
    assert result["phase"] == "completed"
    assert result["artifacts_written"] >= 1

    bundle = store.read_json(session_id=session.id, artifact_name="bundle.json")
    assert bundle is not None
    assert bundle["question"] == session.query
    assert bundle["claims"][0]["citations"][0]["source_id"] == "src_1"
    assert bundle["verification_summary"]["supported_claim_count"] == 1
    assert bundle["unsupported_claims"] == []
    assert bundle["contradictions"] == []
    assert bundle["source_trust"][0]["snapshot_policy"] == "full_artifact"


@pytest.mark.asyncio
async def test_packaging_job_inserts_completion_message_into_linked_chat(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_databases")
    try:
        owner_user_id = "1"
        chat_db_path = DatabasePaths.get_chacha_db_path(int(owner_user_id))
        chat_id, launch_message_id = _seed_chat_thread(
            owner_user_id=owner_user_id,
            chat_db_path=chat_db_path,
        )

        db = ResearchSessionsDB(tmp_path / "research.db")
        session = db.create_session(
            owner_user_id=owner_user_id,
            query="Test packaging",
            source_policy="balanced",
            autonomy_mode="autonomous",
            limits_json={},
            phase="packaging",
            status="queued",
        )
        db.create_chat_handoff(
            session_id=session.id,
            owner_user_id=owner_user_id,
            chat_id=chat_id,
            launch_message_id=launch_message_id,
        )
        store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
        _seed_packaging_artifacts(store=store, session=session)

        await handle_research_phase_job(
            {
                "id": 1010,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )
        await handle_research_phase_job(
            {
                "id": 1011,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )

        chat_db = CharactersRAGDB(str(chat_db_path), client_id=owner_user_id)
        messages = chat_db.get_messages_for_conversation(chat_id, order_by_timestamp="ASC")
        handoff = db.get_chat_handoff(session.id)

        assistant_messages = [message for message in messages if message.get("sender") == "assistant"]
        assert len(assistant_messages) == 1
        assert session.query in str(assistant_messages[0].get("content") or "")
        assert f"/research?run={session.id}" in str(assistant_messages[0].get("content") or "")
        message_metadata = chat_db.get_message_metadata(assistant_messages[0]["id"])
        assert handoff is not None
        assert handoff.handoff_status == "chat_inserted"
        assert handoff.delivered_chat_message_id == assistant_messages[0]["id"]
        assert handoff.delivered_notification_id is None
        assert message_metadata is not None
        assert message_metadata.get("extra") == {
            "deep_research_completion": {
                "run_id": session.id,
                "query": session.query,
                "kind": "completion_handoff",
            }
        }
    finally:
        _restore_user_db_base(previous_user_db_base)


@pytest.mark.asyncio
async def test_packaging_job_falls_back_to_deduped_notification_when_chat_missing(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_databases")
    try:
        owner_user_id = "1"
        db = ResearchSessionsDB(tmp_path / "research.db")
        session = db.create_session(
            owner_user_id=owner_user_id,
            query="Test packaging fallback",
            source_policy="balanced",
            autonomy_mode="autonomous",
            limits_json={},
            phase="packaging",
            status="queued",
        )
        db.create_chat_handoff(
            session_id=session.id,
            owner_user_id=owner_user_id,
            chat_id="missing-chat",
            launch_message_id="missing-message",
        )
        store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
        _seed_packaging_artifacts(store=store, session=session)

        await handle_research_phase_job(
            {
                "id": 1020,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )
        await handle_research_phase_job(
            {
                "id": 1021,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )

        notifications_db = CollectionsDatabase.for_user(user_id=int(owner_user_id))
        notifications = notifications_db.list_user_notifications(limit=10, offset=0)
        handoff = db.get_chat_handoff(session.id)

        assert len(notifications) == 1
        assert notifications[0].kind == "deep_research_completed"
        assert notifications[0].dedupe_key == f"deep_research_completed:{session.id}"
        assert notifications[0].link_url == f"/research?run={session.id}"
        assert handoff is not None
        assert handoff.handoff_status == "notification_only"
        assert handoff.delivered_notification_id == notifications[0].id
        assert handoff.delivered_chat_message_id is None
    finally:
        _restore_user_db_base(previous_user_db_base)


@pytest.mark.asyncio
async def test_packaging_job_bridge_failure_does_not_fail_completed_run(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research import jobs as research_jobs
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_databases")
    try:
        calls: list[str] = []

        def _boom(*, session_id: str, **kwargs):
            _ = kwargs
            calls.append(session_id)
            raise RuntimeError("handoff bridge exploded")

        monkeypatch.setattr(research_jobs, "deliver_research_chat_handoff", _boom, raising=False)

        db = ResearchSessionsDB(tmp_path / "research.db")
        session = db.create_session(
            owner_user_id="1",
            query="Test packaging bridge failure",
            source_policy="balanced",
            autonomy_mode="autonomous",
            limits_json={},
            phase="packaging",
            status="queued",
        )
        db.create_chat_handoff(
            session_id=session.id,
            owner_user_id="1",
            chat_id="chat_123",
            launch_message_id="msg_456",
        )
        store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
        _seed_packaging_artifacts(store=store, session=session)

        result = await handle_research_phase_job(
            {
                "id": 1030,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )

        updated = db.get_session(session.id)
        handoff = db.get_chat_handoff(session.id)

        assert calls == [session.id]
        assert updated is not None
        assert updated.status == "completed"
        assert result["phase"] == "completed"
        assert handoff is not None
        assert handoff.handoff_status == "failed"
        assert "handoff bridge exploded" in str(handoff.last_error or "")
    finally:
        _restore_user_db_base(previous_user_db_base)


@pytest.mark.asyncio
async def test_packaging_job_records_terminal_and_artifact_events(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Packaging event log coverage",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="packaging",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["background"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="outline_v1.json",
        payload={"query": session.query, "sections": [], "unresolved_questions": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="claims.json",
        payload={"claims": []},
        phase="packaging",
        job_id=None,
    )
    store.write_text(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="report_v1.md",
        content="# Report",
        phase="packaging",
        job_id=None,
        content_type="text/markdown",
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={"sources": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="synthesis_summary.json",
        payload={"unresolved_questions": []},
        phase="packaging",
        job_id=None,
    )
    before_id = db.get_latest_run_event_id(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
    )

    await handle_research_phase_job(
        {
            "id": 510,
            "payload": {
                "session_id": session.id,
                "phase": "packaging",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    events = db.list_run_events_after(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        after_id=before_id,
    )

    assert [event.event_type for event in events] == [
        "progress",
        "artifact",
        "progress",
        "status",
        "terminal",
    ]
    assert events[-1].event_payload["status"] == "completed"


@pytest.mark.asyncio
async def test_synthesizing_job_cancels_before_phase_start(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Cancel before synthesizing",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    db.update_control_state(session.id, control_state="cancel_requested")

    await handle_research_phase_job(
        {
            "id": 702,
            "payload": {
                "session_id": session.id,
                "phase": "synthesizing",
                "checkpoint_id": None,
                "policy_version": 1,
            },
        },
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
    )

    updated = db.get_session(session.id)
    assert updated is not None
    assert updated.status == "cancelled"
    assert updated.control_state == "cancelled"
    assert updated.active_job_id is None
    assert updated.phase == "synthesizing"


@pytest.mark.asyncio
async def test_packaging_job_rejects_uncited_claims(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Test packaging",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="packaging",
        status="queued",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="outline_v1.json",
        payload={"query": session.query, "sections": [], "unresolved_questions": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="claims.json",
        payload={
            "claims": [
                {
                    "claim_id": "clm_1",
                    "text": "Evidence is aligned.",
                    "focus_area": "evidence alignment",
                    "source_ids": ["src_1"],
                    "citations": [],
                    "confidence": 0.8,
                }
            ]
        },
        phase="packaging",
        job_id=None,
    )
    store.write_text(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="report_v1.md",
        content="# Research Report",
        phase="packaging",
        job_id=None,
        content_type="text/markdown",
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="source_registry.json",
        payload={"sources": []},
        phase="packaging",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="synthesis_summary.json",
        payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "section_count": 0,
            "claim_count": 1,
            "source_count": 0,
            "unresolved_questions": [],
            "coverage": {"covered_focus_areas": [], "missing_focus_areas": ["evidence alignment"]},
        },
        phase="packaging",
        job_id=None,
    )

    with pytest.raises(ValueError, match="claim_missing_citations"):
        await handle_research_phase_job(
            {
                "id": 11,
                "payload": {
                    "session_id": session.id,
                    "phase": "packaging",
                    "checkpoint_id": None,
                    "policy_version": 1,
                },
            },
            research_db_path=tmp_path / "research.db",
            outputs_dir=tmp_path / "outputs",
        )
