import pytest


pytestmark = pytest.mark.unit


class _RecordingJobs:
    def __init__(self):
        self.created_jobs: list[dict[str, object]] = []
        self.cancelled_jobs: list[tuple[int, str | None]] = []
        self.job_reads: list[int] = []
        self.job_payloads: dict[int, dict[str, object]] = {}

    def create_job(self, **kwargs):
        job_id = len(self.created_jobs) + 100
        job = {"id": job_id, "uuid": f"job-{job_id}", "status": "queued", **kwargs}
        self.created_jobs.append(job)
        return job

    def get_job(self, job_id: int):
        self.job_reads.append(job_id)
        return self.job_payloads.get(job_id)

    def cancel_job(self, job_id: int, *, reason: str | None = None):
        self.cancelled_jobs.append((job_id, reason))
        return True


def _seed_pending_authority(db, session):
    from tldw_Server_API.app.core.Research.phase_dispatch import new_phase_dispatch

    db.compare_and_set_phase(
        session,
        phase=session.phase,
        status=session.status,
        control_state=session.control_state,
        active_job_id=None,
        phase_dispatch_json=new_phase_dispatch(session.phase),
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


def _seed_chat_thread(*, owner_user_id: str, chat_db_path, chat_id: str | None = None) -> str:
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import DEFAULT_CHARACTER_NAME
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

    chat_db = CharactersRAGDB(str(chat_db_path), client_id=owner_user_id)
    character_id = chat_db.add_character_card(
        {
            "name": DEFAULT_CHARACTER_NAME,
            "description": "Research thread helper",
            "personality": "Helpful",
            "scenario": "Research follow-up testing",
            "system_prompt": "You are a research assistant.",
            "first_message": "Hello",
            "creator_notes": "Created by deep research tests",
        }
    )
    chat_id = chat_db.add_conversation(
        {
            "id": chat_id,
            "character_id": character_id,
            "title": "Deep Research Chat",
            "client_id": owner_user_id,
        }
    )
    assert chat_id is not None
    return chat_id


def test_create_session_enqueues_planning_job(tmp_path):
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 9, "uuid": "job-9", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    session = service.create_session(
        owner_user_id="1",
        query="Map evidence gaps between internal notes and public filings",
        source_policy="balanced",
        autonomy_mode="checkpointed",
    )

    assert session.phase == "drafting_plan"
    assert session.active_job_id == "9"
    assert captured["domain"] == "research"
    assert captured["job_type"] == "research_phase"
    assert captured["payload"]["session_id"] == session.id


def test_create_session_persists_chat_handoff_linkage(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 13, "uuid": "job-13", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    try:
        _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=DatabasePaths.get_chacha_db_path("1"),
            chat_id="chat_123",
        )
        session = service.create_session(
            owner_user_id="1",
            query="Investigate regulatory timeline",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={
                "chat_id": "chat_123",
                "launch_message_id": "msg_456",
            },
        )
    finally:
        _restore_user_db_base(previous_user_db_base)

    db = ResearchSessionsDB(tmp_path / "research.db")
    handoff = db.get_chat_handoff(session.id)

    assert handoff is not None
    assert handoff.session_id == session.id
    assert handoff.owner_user_id == "1"
    assert handoff.chat_id == "chat_123"
    assert handoff.launch_message_id == "msg_456"
    assert handoff.handoff_status == "pending"
    assert handoff.delivered_chat_message_id is None
    assert handoff.delivered_notification_id is None


def test_get_session_returns_validated_chat_id_for_linked_run(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 14, "uuid": "job-14", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    try:
        _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=DatabasePaths.get_chacha_db_path("1"),
            chat_id="chat_123",
        )
        created = service.create_session(
            owner_user_id="1",
            query="Investigate regulatory timeline",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": "chat_123"},
        )
        loaded = service.get_session(owner_user_id="1", session_id=created.id)
    finally:
        _restore_user_db_base(previous_user_db_base)

    assert loaded.chat_id == "chat_123"


def test_create_session_retains_pending_authority_when_enqueue_ack_fails(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class FailingJobs:
        def create_job(self, **_kwargs):
            raise RuntimeError("queue saturated")

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=FailingJobs(),
    )

    try:
        _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=DatabasePaths.get_chacha_db_path("1"),
            chat_id="chat_123",
        )
        with pytest.raises(RuntimeError, match="queue saturated"):
            service.create_session(
                owner_user_id="1",
                query="Investigate regulatory timeline",
                source_policy="balanced",
                autonomy_mode="checkpointed",
                chat_handoff={"chat_id": "chat_123"},
            )
    finally:
        _restore_user_db_base(previous_user_db_base)

    db = ResearchSessionsDB(tmp_path / "research.db")
    sessions = db.list_sessions("1")
    assert len(sessions) == 1  # nosec B101 -- Unknown admission acknowledgement retains one owned run.
    assert sessions[0].active_job_id is None  # nosec B101 -- The pending run must not invent a Job binding.
    assert sessions[0].phase_dispatch_json["dispatch_id"]  # nosec B101 -- Recovery retains one stable dispatch nonce.
    assert db.get_chat_handoff(sessions[0].id).chat_id == "chat_123"  # nosec B101 -- Unknown acknowledgement preserves the originating Chat association.


def test_get_session_returns_null_chat_id_for_unlinked_run(tmp_path):
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 15, "uuid": "job-15", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    created = service.create_session(
        owner_user_id="1",
        query="Investigate regulatory timeline",
        source_policy="balanced",
        autonomy_mode="checkpointed",
    )
    loaded = service.get_session(owner_user_id="1", session_id=created.id)

    assert loaded.chat_id is None


def test_get_session_nulls_stale_chat_handoff_linkage(tmp_path, monkeypatch):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 16, "uuid": "job-16", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    try:
        chat_db_path = DatabasePaths.get_chacha_db_path("1")
        _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=chat_db_path,
            chat_id="chat_123",
        )
        created = service.create_session(
            owner_user_id="1",
            query="Investigate regulatory timeline",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": "chat_123"},
        )
        with sqlite3.connect(chat_db_path) as conn:
            conn.execute(
                "UPDATE conversations SET deleted = 1 WHERE id = ?",
                ("chat_123",),
            )
            conn.commit()
        loaded = service.get_session(owner_user_id="1", session_id=created.id)
    finally:
        _restore_user_db_base(previous_user_db_base)

    assert loaded.chat_id is None


def test_create_session_persists_follow_up_json(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 15, "uuid": "job-15", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    session = service.create_session(
        owner_user_id="1",
        query="Investigate regulatory timeline",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        follow_up={
            "question": "What should the next research run check?",
            "background": {
                "question": "What did the attached research conclude?",
                "outline": [{"title": "Evidence gap", "focus_area": "background"}],
                "key_claims": [
                    {
                        "claim_id": "clm_1",
                        "text": "The attached run did not resolve the gap.",
                    }
                ],
                "unresolved_questions": ["Which claim still needs verification?"],
                "verification_summary": {
                    "supported_claim_count": 1,
                    "unsupported_claim_count": 0,
                },
                "source_trust_summary": {
                    "high_trust_count": 1,
                    "low_trust_count": 0,
                },
            },
        },
    )

    db = ResearchSessionsDB(tmp_path / "research.db")
    stored = db.get_session(session.id)

    assert stored is not None
    assert stored.follow_up_json["question"] == "What should the next research run check?"
    assert stored.follow_up_json["background"]["outline"][0]["title"] == "Evidence gap"


def test_create_session_rejects_invalid_chat_handoff_chat_id(tmp_path):
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 16, "uuid": "job-16", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    with pytest.raises(ValueError):
        service.create_session(
            owner_user_id="1",
            query="Investigate regulatory timeline",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": ""},
        )


def test_create_session_rejects_foreign_chat_handoff_chat_id(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 17, "uuid": "job-17", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    try:
        owned_chat_id = _seed_chat_thread(
            owner_user_id="1",
            chat_db_path=DatabasePaths.get_chacha_db_path("1"),
        )
        foreign_chat_id = _seed_chat_thread(
            owner_user_id="2",
            chat_db_path=DatabasePaths.get_chacha_db_path("2"),
        )

        owned_session = service.create_session(
            owner_user_id="1",
            query="Investigate regulatory timeline",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": owned_chat_id},
        )
        assert owned_session.owner_user_id == "1"

        with pytest.raises(ValueError):
            service.create_session(
                owner_user_id="1",
                query="Investigate regulatory timeline",
                source_policy="balanced",
                autonomy_mode="checkpointed",
                chat_handoff={"chat_id": foreign_chat_id},
            )
    finally:
        _restore_user_db_base(previous_user_db_base)


def test_create_session_rejects_unknown_chat_handoff_chat_id(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 18, "uuid": "job-18", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    try:
        with pytest.raises(ValueError):
            service.create_session(
                owner_user_id="1",
                query="Investigate regulatory timeline",
                source_policy="balanced",
                autonomy_mode="checkpointed",
                chat_handoff={"chat_id": "chat_missing"},
            )
    finally:
        _restore_user_db_base(previous_user_db_base)


def test_create_session_without_chat_handoff_has_no_linkage(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 14, "uuid": "job-14", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    session = service.create_session(
        owner_user_id="1",
        query="Investigate regulatory timeline",
        source_policy="balanced",
        autonomy_mode="checkpointed",
    )

    db = ResearchSessionsDB(tmp_path / "research.db")

    assert db.get_chat_handoff(session.id) is None


def test_list_chat_linked_runs_returns_compact_bounded_rows_ordered_by_session_update(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    previous_user_db_base = _set_user_db_base(monkeypatch, tmp_path / "user_dbs")

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 21, "uuid": "job-21", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")

    try:
        _seed_chat_thread(
            owner_user_id="owner-1",
            chat_db_path=DatabasePaths.get_chacha_db_path("owner-1"),
            chat_id="chat_123",
        )
        _seed_chat_thread(
            owner_user_id="owner-2",
            chat_db_path=DatabasePaths.get_chacha_db_path("owner-2"),
            chat_id="chat_123",
        )
        terminal_ids: list[str] = []
        for index in range(12):
            session = service.create_session(
                owner_user_id="owner-1",
                query=f"terminal query {index}",
                source_policy="balanced",
                autonomy_mode="checkpointed",
                chat_handoff={"chat_id": "chat_123"},
            )
            terminal_ids.append(session.id)
            db.update_phase(
                session.id,
                phase="completed",
                status="completed",
                completed_at=f"2026-03-08T00:00:{index:02d}+00:00",
                active_job_id=None,
            )

        active = service.create_session(
            owner_user_id="owner-1",
            query="active query",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": "chat_123"},
        )
        db.update_phase(
            active.id,
            phase="collecting",
            status="running",
            active_job_id="22",
        )

        other_user = service.create_session(
            owner_user_id="owner-2",
            query="other user query",
            source_policy="balanced",
            autonomy_mode="checkpointed",
            chat_handoff={"chat_id": "chat_123"},
        )
        db.update_phase(
            other_user.id,
            phase="completed",
            status="completed",
            completed_at="2026-03-08T00:02:00+00:00",
            active_job_id=None,
        )

        runs = service.list_chat_linked_runs(owner_user_id="owner-1", chat_id="chat_123")
    finally:
        _restore_user_db_base(previous_user_db_base)

    assert [run.run_id for run in runs] == [
        active.id,
        terminal_ids[-1],
        terminal_ids[-2],
        terminal_ids[-3],
        terminal_ids[-4],
        terminal_ids[-5],
        terminal_ids[-6],
        terminal_ids[-7],
        terminal_ids[-8],
        terminal_ids[-9],
        terminal_ids[-10],
    ]
    assert all(run.run_id != terminal_ids[0] for run in runs)
    assert all(run.run_id != other_user.id for run in runs)
    assert all(
        set(run.__dict__) == {
            "run_id",
            "query",
            "status",
            "phase",
            "control_state",
            "latest_checkpoint_id",
            "updated_at",
        }
        for run in runs
    )


def test_list_chat_linked_runs_returns_empty_list_when_chat_has_no_runs(tmp_path):
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )

    assert service.list_chat_linked_runs(owner_user_id="owner-1", chat_id="missing-chat") == []


def test_approve_plan_review_enqueues_collecting_job(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 10, "uuid": "job-10", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
            "stop_criteria": {"min_cited_sections": 1},
        },
    )

    updated = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={"focus_areas": ["evidence alignment", "contradictions"]},
    )

    assert updated.phase == "collecting"
    assert updated.active_job_id == "10"
    assert captured["payload"]["phase"] == "collecting"
    assert captured["payload"]["session_id"] == session.id
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    assert store.read_json(session_id=session.id, artifact_name="approved_plan.json") == {
        "query": session.query,
        "focus_areas": ["evidence alignment", "contradictions"],
        "source_policy": session.source_policy,
        "autonomy_mode": session.autonomy_mode,
        "stop_criteria": {"min_cited_sections": 1},
    }


def test_approve_plan_review_rejects_read_only_session_fields(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            raise AssertionError(f"unexpected job enqueue: {kwargs}")

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_source_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="sources_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_policy": session.source_policy,
            "autonomy_mode": session.autonomy_mode,
        },
    )

    with pytest.raises(ValueError, match="query"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=session.id,
            checkpoint_id=checkpoint.id,
            patch_payload={"query": "Changed query"},
        )


def test_approve_sources_review_enqueues_synthesizing_job_and_writes_review_artifact(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 11, "uuid": "job-11", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_source_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="sources_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_inventory": [
                {"source_id": "src_1", "title": "Primary memo"},
                {"source_id": "src_2", "title": "Counterevidence note"},
                {"source_id": "src_3", "title": "Background summary"},
            ],
            "collection_summary": {"source_count": 3},
        },
    )

    updated = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={
            "pinned_source_ids": ["src_1"],
            "dropped_source_ids": ["src_3"],
            "prioritized_source_ids": ["src_2"],
            "recollect": {
                "enabled": False,
                "need_primary_sources": False,
                "need_contradictions": False,
                "guidance": "",
            },
        },
    )

    assert updated.phase == "synthesizing"
    assert updated.active_job_id == "11"
    assert captured["payload"]["phase"] == "synthesizing"
    assert captured["payload"]["session_id"] == session.id
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    assert store.read_json(session_id=session.id, artifact_name="approved_sources.json") == {
        "pinned_source_ids": ["src_1"],
        "dropped_source_ids": ["src_3"],
        "prioritized_source_ids": ["src_2"],
        "recollect": {
            "enabled": False,
            "need_primary_sources": False,
            "need_contradictions": False,
            "guidance": "",
        },
    }


def test_approve_sources_review_with_recollect_enqueues_collecting_job(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 12, "uuid": "job-12", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_source_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="sources_review",
        proposed_payload={
            "query": session.query,
            "focus_areas": ["evidence alignment"],
            "source_inventory": [
                {"source_id": "src_1", "title": "Primary memo"},
                {"source_id": "src_2", "title": "Counterevidence note"},
            ],
            "collection_summary": {"source_count": 2},
        },
    )

    updated = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={
            "pinned_source_ids": ["src_1"],
            "dropped_source_ids": [],
            "prioritized_source_ids": ["src_1"],
            "recollect": {
                "enabled": True,
                "need_primary_sources": True,
                "need_contradictions": True,
                "guidance": "Find newer contradictory primary evidence.",
            },
        },
    )

    assert updated.phase == "collecting"
    assert updated.active_job_id == "12"
    assert captured["payload"]["phase"] == "collecting"
    assert captured["payload"]["session_id"] == session.id
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    assert store.read_json(session_id=session.id, artifact_name="approved_sources.json") == {
        "pinned_source_ids": ["src_1"],
        "dropped_source_ids": [],
        "prioritized_source_ids": ["src_1"],
        "recollect": {
            "enabled": True,
            "need_primary_sources": True,
            "need_contradictions": True,
            "guidance": "Find newer contradictory primary evidence.",
        },
    }


def test_approve_outline_review_enqueues_packaging_job(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 12, "uuid": "job-12", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_outline_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="outline_review",
        proposed_payload={
            "outline": {"sections": [{"title": "Background"}]},
            "claim_count": 1,
            "report_preview": "# Research Report",
        },
    )

    updated = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
    )

    assert updated.phase == "packaging"
    assert updated.active_job_id == "12"
    assert captured["payload"]["phase"] == "packaging"
    assert captured["payload"]["session_id"] == session.id


def test_approve_outline_review_patch_enqueues_resynthesis_with_locked_outline(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    captured: dict[str, object] = {}

    class DummyJobs:
        def create_job(self, **kwargs):
            captured.update(kwargs)
            return {"id": 13, "uuid": "job-13", "status": "queued"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Map evidence gaps",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_outline_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="outline_review",
        proposed_payload={
            "outline": {
                "sections": [
                    {"title": "Background", "focus_area": "history"},
                    {"title": "Evidence", "focus_area": "evidence"},
                ]
            },
            "claim_count": 1,
            "report_preview": "# Research Report",
            "focus_areas": ["history", "evidence"],
        },
    )

    updated = service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={
            "sections": [
                {"title": "Evidence first", "focus_area": "evidence"},
                {"title": "Historical context", "focus_area": "history"},
            ]
        },
    )

    assert updated.phase == "synthesizing"
    assert updated.active_job_id == "13"
    assert captured["payload"]["phase"] == "synthesizing"
    assert captured["payload"]["approved_outline_locked"] is True
    assert captured["payload"]["session_id"] == session.id
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    assert store.read_json(session_id=session.id, artifact_name="approved_outline.json") == {
        "sections": [
            {"title": "Evidence first", "focus_area": "evidence"},
            {"title": "Historical context", "focus_area": "history"},
        ]
    }


def test_get_session_bundle_and_allowlisted_artifacts(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Read artifacts",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="completed",
        status="completed",
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="bundle.json",
        payload={"question": session.query, "claims": []},
        phase="completed",
        job_id=None,
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="claims.json",
        payload={"claims": [{"claim_id": "clm_1"}]},
        phase="completed",
        job_id=None,
    )
    store.write_jsonl(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="evidence_notes.jsonl",
        records=[{"note_id": "note_1", "text": "Evidence"}],
        phase="completed",
        job_id=None,
    )
    store.write_text(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="report_v1.md",
        content="# Research Report",
        phase="completed",
        job_id=None,
        content_type="text/markdown",
    )

    loaded_session = service.get_session(owner_user_id="1", session_id=session.id)
    bundle = service.get_bundle(owner_user_id="1", session_id=session.id)
    claims = service.get_artifact(owner_user_id="1", session_id=session.id, artifact_name="claims.json")
    notes = service.get_artifact(owner_user_id="1", session_id=session.id, artifact_name="evidence_notes.jsonl")
    report = service.get_artifact(owner_user_id="1", session_id=session.id, artifact_name="report_v1.md")

    assert loaded_session.id == session.id
    assert bundle["question"] == session.query
    assert claims["content"]["claims"][0]["claim_id"] == "clm_1"
    assert notes["content"][0]["note_id"] == "note_1"
    assert report["content"] == "# Research Report"
    assert report["content_type"] == "text/markdown"


@pytest.mark.asyncio
async def test_create_session_persists_provider_overrides_and_planning_writes_provider_config(tmp_path):
    from tldw_Server_API.app.core.Research.jobs import handle_research_phase_job
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 21, "uuid": "job-21", "status": "queued", **kwargs}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )

    session = service.create_session(
        owner_user_id="1",
        query="Hybrid provider config test",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        provider_overrides={
            "local": {"top_k": 4, "sources": ["media_db"]},
            "web": {"engine": "duckduckgo", "result_count": 3},
        },
    )

    await handle_research_phase_job(
        {
            "id": 21,
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

    provider_config = service.get_artifact(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
    )

    assert provider_config["content"]["web"]["engine"] == "duckduckgo"
    assert provider_config["content"]["local"]["top_k"] == 4


def test_record_run_event_dedupes_identical_payloads_and_writes_new_row_for_changes(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Deduplicate research status events",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )

    first = service.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="status",
        event_payload={"status": "queued", "phase": "drafting_plan"},
        phase="drafting_plan",
        job_id="301",
    )
    duplicate = service.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="status",
        event_payload={"phase": "drafting_plan", "status": "queued"},
        phase="drafting_plan",
        job_id="301",
    )
    changed = service.record_run_event(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        event_type="status",
        event_payload={"status": "running", "phase": "drafting_plan"},
        phase="drafting_plan",
        job_id="301",
    )

    assert first.id == duplicate.id
    assert changed.id > first.id
    assert [
        event.id
        for event in service.list_run_events_after(
            owner_user_id=session.owner_user_id,
            session_id=session.id,
            after_id=0,
        )
    ] == [first.id, changed.id]


def test_checkpoint_event_payload_is_compact_metadata_only(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Compact checkpoint payload",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="sources_review",
        proposed_payload={
            "source_inventory": [{"source_id": "src_1", "title": "Primary source"}],
            "collection_summary": {"source_count": 1},
        },
    )

    payload = service._checkpoint_event_payload(checkpoint=checkpoint, phase="awaiting_source_review")

    assert payload == {
        "checkpoint_id": checkpoint.id,
        "checkpoint_type": "sources_review",
        "status": "pending",
        "resolution": None,
        "phase": "awaiting_source_review",
        "has_proposed_payload": True,
    }


def test_update_status_with_event_rolls_back_when_event_insert_fails(tmp_path, monkeypatch):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Transactional event writer",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
    )

    def fail_record_run_event_with_conn(self, conn, **kwargs):
        raise sqlite3.OperationalError("event insert failed")

    monkeypatch.setattr(
        ResearchSessionsDB,
        "_record_run_event_with_conn",
        fail_record_run_event_with_conn,
    )

    with pytest.raises(sqlite3.OperationalError):
        db.update_status_with_event(
            session_id=session.id,
            status="completed",
            owner_user_id=session.owner_user_id,
            event_type="status",
            event_payload={"status": "completed", "phase": session.phase},
            phase=session.phase,
            job_id=None,
        )

    reloaded = db.get_session(session.id)
    assert reloaded is not None
    assert reloaded.status == "queued"


def test_research_session_defaults_include_control_and_progress_fields(tmp_path):
    from tldw_Server_API.app.api.v1.schemas.research_runs_schemas import ResearchRunResponse
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Defaults test",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
    )

    assert session.control_state == "running"
    assert session.progress_percent is None
    assert session.progress_message is None

    payload = ResearchRunResponse.model_validate(session)
    assert payload.control_state == "running"
    assert payload.progress_percent is None
    assert payload.progress_message is None


def test_get_stream_snapshot_includes_checkpoint_and_latest_artifact_manifest(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Track live research state",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )
    checkpoint = db.create_checkpoint(
        session_id=session.id,
        checkpoint_type="plan_review",
        proposed_payload={
            "focus_areas": ["background", "counterevidence"],
            "stop_criteria": {"min_cited_sections": 2},
        },
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["background"]},
        phase="drafting_plan",
        job_id="11",
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="plan.json",
        payload={"focus_areas": ["background", "counterevidence"]},
        phase="drafting_plan",
        job_id="12",
    )
    store.write_json(
        owner_user_id="1",
        session_id=session.id,
        artifact_name="provider_config.json",
        payload={"web": {"engine": "kagi"}},
        phase="drafting_plan",
        job_id="12",
    )

    snapshot = service.get_stream_snapshot(owner_user_id="1", session_id=session.id)

    assert snapshot.run.id == session.id
    assert snapshot.run.phase == "awaiting_plan_review"
    assert snapshot.checkpoint is not None
    assert snapshot.checkpoint.checkpoint_id == checkpoint.id
    assert snapshot.checkpoint.checkpoint_type == "plan_review"
    assert snapshot.checkpoint.proposed_payload["focus_areas"] == ["background", "counterevidence"]
    assert {item.artifact_name for item in snapshot.artifacts} == {"plan.json", "provider_config.json"}
    assert next(
        item for item in snapshot.artifacts if item.artifact_name == "plan.json"
    ).artifact_version == 2


def test_list_sessions_returns_newly_created_runs_for_owner(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    db = ResearchSessionsDB(tmp_path / "research.db")
    older = db.create_session(
        owner_user_id="1",
        query="Older service run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )
    newer = db.create_session(
        owner_user_id="1",
        query="Newer service run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
    )
    db.create_session(
        owner_user_id="2",
        query="Other owner service run",
        source_policy="external_only",
        autonomy_mode="autonomous",
        limits_json={},
    )

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )

    sessions = service.list_sessions(owner_user_id="1", limit=10)

    assert [session.id for session in sessions] == [newer.id, older.id]
    assert [session.query for session in sessions] == ["Newer service run", "Older service run"]
    assert all(session.owner_user_id == "1" for session in sessions)


def test_pause_run_marks_active_executable_session_pause_requested(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=_RecordingJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Pause active run",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    db.attach_active_job(session.id, "22")

    updated = service.pause_run(owner_user_id="1", session_id=session.id)

    assert updated.control_state == "pause_requested"
    assert updated.active_job_id == "22"
    assert updated.phase == "collecting"


def test_pause_run_marks_idle_and_checkpoint_sessions_paused(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=_RecordingJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    queued = db.create_session(
        owner_user_id="1",
        query="Pause queued run",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    checkpoint = db.create_session(
        owner_user_id="1",
        query="Pause checkpoint run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )

    paused_queued = service.pause_run(owner_user_id="1", session_id=queued.id)
    paused_checkpoint = service.pause_run(owner_user_id="1", session_id=checkpoint.id)

    assert paused_queued.control_state == "paused"
    assert paused_queued.active_job_id is None
    assert paused_checkpoint.control_state == "paused"
    assert paused_checkpoint.status == "waiting_human"


def test_resume_run_reenqueues_executable_phase_and_restores_checkpoint_wait(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    jobs = _RecordingJobs()
    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=jobs,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    queued = db.create_session(
        owner_user_id="1",
        query="Resume queued run",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    _seed_pending_authority(db, queued)
    db.update_control_state(queued.id, control_state="paused")
    checkpoint = db.create_session(
        owner_user_id="1",
        query="Resume checkpoint run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_outline_review",
        status="waiting_human",
    )
    db.update_control_state(checkpoint.id, control_state="paused")

    resumed_queued = service.resume_run(owner_user_id="1", session_id=queued.id)
    resumed_checkpoint = service.resume_run(owner_user_id="1", session_id=checkpoint.id)

    assert resumed_queued.control_state == "running"
    assert resumed_queued.active_job_id == "100"
    assert jobs.created_jobs[0]["payload"]["phase"] == "synthesizing"
    assert resumed_checkpoint.control_state == "running"
    assert resumed_checkpoint.status == "waiting_human"
    assert resumed_checkpoint.active_job_id is None


def test_cancel_run_requests_active_work_and_terminalizes_idle_sessions(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    jobs = _RecordingJobs()
    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=jobs,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    active = db.create_session(
        owner_user_id="1",
        query="Cancel active run",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    db.attach_active_job(active.id, "41")
    checkpoint = db.create_session(
        owner_user_id="1",
        query="Cancel checkpoint run",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )

    cancel_requested = service.cancel_run(owner_user_id="1", session_id=active.id)
    cancelled = service.cancel_run(owner_user_id="1", session_id=checkpoint.id)

    assert cancel_requested.control_state == "cancel_requested"
    assert cancel_requested.status == "queued"
    assert cancel_requested.active_job_id == "41"
    assert jobs.cancelled_jobs == [(41, "research_cancel_requested")]
    assert cancelled.control_state == "cancelled"
    assert cancelled.status == "cancelled"
    assert cancelled.active_job_id is None

    with pytest.raises(ValueError, match="resume_not_allowed"):
        service.resume_run(owner_user_id="1", session_id=checkpoint.id)

    cancel_requested_events = service.list_run_events_after(
        owner_user_id="1",
        session_id=active.id,
        after_id=0,
    )
    cancelled_events = service.list_run_events_after(
        owner_user_id="1",
        session_id=checkpoint.id,
        after_id=0,
    )

    assert [event.event_type for event in cancel_requested_events] == ["status"]
    assert cancel_requested_events[0].event_payload["control_state"] == "cancel_requested"
    assert [event.event_type for event in cancelled_events] == ["status"]
    assert cancelled_events[0].event_payload["status"] == "cancelled"


def test_pause_and_resume_record_status_events(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    jobs = _RecordingJobs()
    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=jobs,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    paused = db.create_session(
        owner_user_id="1",
        query="Pause event logging",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    db.attach_active_job(paused.id, "41")
    resumed = db.create_session(
        owner_user_id="1",
        query="Resume event logging",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    _seed_pending_authority(db, resumed)
    db.update_control_state(resumed.id, control_state="paused")

    resume_before_id = db.get_latest_run_event_id(owner_user_id="1", session_id=resumed.id)
    service.pause_run(owner_user_id="1", session_id=paused.id)
    service.resume_run(owner_user_id="1", session_id=resumed.id)

    pause_events = service.list_run_events_after(
        owner_user_id="1",
        session_id=paused.id,
        after_id=0,
    )
    resume_events = service.list_run_events_after(
        owner_user_id="1",
        session_id=resumed.id,
        after_id=resume_before_id,
    )

    assert [event.event_type for event in pause_events] == ["status"]
    assert pause_events[0].event_payload["control_state"] == "pause_requested"
    assert [event.event_type for event in resume_events] == ["status", "status"]  # nosec B101 -- Resume records its pending and exact bound state once each.
    assert resume_events[0].event_payload["control_state"] == "running"

    assert resume_events[-1].event_payload["active_job_id"] == "100"  # nosec B101 -- The final event exposes the actual admitted binding.


def test_approve_plan_review_records_checkpoint_artifact_and_status_events(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    class DummyJobs:
        def create_job(self, **kwargs):
            return {"id": 111, "uuid": "job-111", "status": "queued", **kwargs}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=DummyJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Approval event logging",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )
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
    before_id = db.get_latest_run_event_id(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
    )

    service.approve_checkpoint(
        owner_user_id="1",
        session_id=session.id,
        checkpoint_id=checkpoint.id,
        patch_payload={"focus_areas": ["background", "contradictions"]},
    )

    events = service.list_run_events_after(
        owner_user_id="1",
        session_id=session.id,
        after_id=before_id,
    )

    assert [event.event_type for event in events] == ["checkpoint", "artifact", "status", "status"]  # nosec B101 -- Atomic approval retains event order plus the subsequent exact Job binding.
    assert events[0].event_payload["status"] == "resolved"
    assert events[1].event_payload["artifact_name"] == "approved_plan.json"
    assert events[2].event_payload["phase"] == "collecting"


def test_artifact_store_write_json_records_artifact_event(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.artifact_store import ResearchArtifactStore

    db = ResearchSessionsDB(tmp_path / "research.db")
    session = db.create_session(
        owner_user_id="1",
        query="Artifact write event",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
    )
    store = ResearchArtifactStore(base_dir=tmp_path / "outputs", db=db)

    artifact = store.write_json(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        artifact_name="plan.json",
        payload={"query": session.query},
        phase="drafting_plan",
        job_id="77",
    )

    events = db.list_run_events_after(
        owner_user_id=session.owner_user_id,
        session_id=session.id,
        after_id=0,
    )

    assert artifact.artifact_version == 1
    assert [event.event_type for event in events] == ["artifact"]
    assert events[0].event_payload == {
        "artifact_name": "plan.json",
        "artifact_version": 1,
        "content_type": "application/json",
        "phase": "drafting_plan",
        "job_id": "77",
    }


def test_approve_checkpoint_rejects_paused_or_cancellation_pending_sessions(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=_RecordingJobs(),
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    paused_session = db.create_session(
        owner_user_id="1",
        query="Paused checkpoint",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_plan_review",
        status="waiting_human",
    )
    paused_checkpoint = db.create_checkpoint(
        session_id=paused_session.id,
        checkpoint_type="plan_review",
        proposed_payload={"focus_areas": ["background"]},
    )
    db.update_control_state(paused_session.id, control_state="paused")

    cancelling_session = db.create_session(
        owner_user_id="1",
        query="Cancelling checkpoint",
        source_policy="balanced",
        autonomy_mode="checkpointed",
        limits_json={},
        phase="awaiting_source_review",
        status="waiting_human",
    )
    cancelling_checkpoint = db.create_checkpoint(
        session_id=cancelling_session.id,
        checkpoint_type="sources_review",
        proposed_payload={"source_inventory": []},
    )
    db.update_control_state(cancelling_session.id, control_state="cancel_requested")

    with pytest.raises(ValueError, match="checkpoint_approval_not_allowed"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=paused_session.id,
            checkpoint_id=paused_checkpoint.id,
        )

    with pytest.raises(ValueError, match="checkpoint_approval_not_allowed"):
        service.approve_checkpoint(
            owner_user_id="1",
            session_id=cancelling_session.id,
            checkpoint_id=cancelling_checkpoint.id,
        )


def test_get_session_uses_session_progress_as_primary_and_job_progress_as_fallback(tmp_path):
    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
    from tldw_Server_API.app.core.Research.service import ResearchService

    jobs = _RecordingJobs()
    jobs.job_payloads[77] = {"id": 77, "progress_percent": 52.5, "progress_message": "collecting sources"}
    jobs.job_payloads[78] = {"id": 78, "progress_percent": 90.0, "progress_message": "job overlay"}

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=jobs,
    )
    db = ResearchSessionsDB(tmp_path / "research.db")
    fallback_session = db.create_session(
        owner_user_id="1",
        query="Fallback progress",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="collecting",
        status="queued",
    )
    db.attach_active_job(fallback_session.id, "77")
    primary_session = db.create_session(
        owner_user_id="1",
        query="Primary progress",
        source_policy="balanced",
        autonomy_mode="autonomous",
        limits_json={},
        phase="synthesizing",
        status="queued",
    )
    db.attach_active_job(primary_session.id, "78")
    db.update_progress(
        primary_session.id,
        progress_percent=75.0,
        progress_message="synthesizing report",
    )

    loaded_fallback = service.get_session(owner_user_id="1", session_id=fallback_session.id)
    loaded_primary = service.get_session(owner_user_id="1", session_id=primary_session.id)

    assert loaded_fallback.progress_percent == 52.5
    assert loaded_fallback.progress_message == "collecting sources"
    assert loaded_primary.progress_percent == 75.0
    assert loaded_primary.progress_message == "synthesizing report"
    assert jobs.job_reads == [77, 78]


def test_research_sessions_db_migrates_provider_overrides_and_run_control_columns(tmp_path):
    import sqlite3

    from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB

    db_path = tmp_path / "research.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE research_sessions (
                id TEXT PRIMARY KEY,
                owner_user_id TEXT NOT NULL,
                status TEXT NOT NULL,
                phase TEXT NOT NULL,
                query TEXT NOT NULL,
                source_policy TEXT NOT NULL,
                autonomy_mode TEXT NOT NULL,
                limits_json TEXT NOT NULL DEFAULT '{}',
                active_job_id TEXT,
                latest_checkpoint_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            );
            """
        )

    ResearchSessionsDB(db_path)

    with sqlite3.connect(db_path) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info('research_sessions')").fetchall()}

    assert "provider_overrides_json" in cols
    assert "control_state" in cols
    assert "progress_percent" in cols
    assert "progress_message" in cols


def test_get_artifact_rejects_disallowed_name(tmp_path):
    from tldw_Server_API.app.core.Research.service import ResearchService

    service = ResearchService(
        research_db_path=tmp_path / "research.db",
        outputs_dir=tmp_path / "outputs",
        job_manager=None,
    )

    with pytest.raises(ValueError, match="artifact_not_allowed"):
        service.get_artifact(
            owner_user_id="1",
            session_id="rs_missing",
            artifact_name="not_allowed.bin",
        )
