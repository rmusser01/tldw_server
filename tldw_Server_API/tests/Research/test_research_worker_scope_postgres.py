"""Deep Research workers must bind the persisted owner for PostgreSQL content."""

import asyncio
from contextlib import nullcontext

import pytest

from tldw_Server_API.app.core.DB_Management.media_db.api import (
    create_media_database,
    get_media_by_id,
    get_media_repository,
)
from tldw_Server_API.app.core.DB_Management.ResearchSessionsDB import ResearchSessionsDB
from tldw_Server_API.app.core.DB_Management.scope_context import get_scope, scoped_context
from tldw_Server_API.app.core.Research import jobs_worker

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


def _session(tmp_path, owner):
    return ResearchSessionsDB(tmp_path / "research.db").create_session(
        owner_user_id=owner,
        query="Rowan Observatory",
        source_policy="local_only",
        autonomy_mode="checkpointed",
        limits_json={},
    )


@pytest.mark.parametrize("inherited_admin", [False, True])
async def test_research_worker_reads_only_persisted_owner_media(
    tmp_path,
    monkeypatch,
    pg_restricted_backend,
    inherited_admin,
):
    with scoped_context(user_id=1, is_admin=False):
        media = create_media_database(client_id="1", db_path=tmp_path / "media.db", backend=pg_restricted_backend)
    try:
        with scoped_context(user_id=1, is_admin=False):
            own = get_media_repository(media).add_media_with_keywords(
                url="research://rowan",
                title="Rowan",
                media_type="document",
                content="Rowan Observatory telescope is Selene.",
                keywords=[],
                overwrite=False,
            )[0]
        session = _session(tmp_path, "1")
        seen = []

        class OneJobSDK:
            def __init__(self, manager, config):
                pass

            async def run(self, *, handler, **kwargs):
                await handler(
                    {
                        "id": 41,
                        "owner_user_id": "1",
                        "payload": {"session_id": session.id, "phase": "collecting", "owner_user_id": "2"},
                    }
                )

        async def read_content(job, **kwargs):
            scope = get_scope()
            seen.append(None if scope is None else (scope.user_id, scope.is_admin))
            assert get_media_by_id(media, own)["content"] == "Rowan Observatory telescope is Selene."  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
            with scoped_context(user_id=2, is_admin=False):
                assert get_media_by_id(media, own) is None  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
            return {"owner_only": True}

        monkeypatch.setattr(jobs_worker, "WorkerSDK", OneJobSDK)
        monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object())
        monkeypatch.setattr(jobs_worker, "handle_research_phase_job", read_content)
        context = scoped_context(user_id=999, is_admin=True) if inherited_admin else nullcontext()
        with context:
            before = get_scope()
            try:
                await jobs_worker.run_research_jobs_worker(
                    research_db_path=tmp_path / "research.db",
                    outputs_dir=tmp_path / "outputs",
                )
            finally:
                assert get_scope() == before  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
        assert seen == [(1, False)]  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
    finally:
        media.close_connection()


@pytest.mark.parametrize(
    "owner",
    [
        pytest.param(None, id="missing"),
        pytest.param(True, id="boolean"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param("", id="empty"),
        pytest.param("../7", id="path"),
        pytest.param({}, id="mapping"),
        pytest.param(2**63, id="outside-pg-range"),
        pytest.param(10**5000, id="oversized-int"),
        pytest.param("9" * 5000, id="oversized-string"),
    ],
)
async def test_research_worker_refuses_invalid_persisted_owner(monkeypatch, tmp_path, owner):
    dispatched = []

    class OneJobSDK:
        def __init__(self, manager, config):
            pass

        async def run(self, *, handler, **kwargs):
            with pytest.raises(ValueError) as failure:
                await handler(
                    {
                        "id": 41,
                        "owner_user_id": owner,
                        "payload": {"owner_user_id": "7", "session_id": "rs_1", "phase": "collecting"},
                    }
                )
            assert failure.value.retryable is False  # nosec B101 -- Regression assertion checks worker authorization or cleanup.

    async def phase(job, **kwargs):
        dispatched.append(job)
        return {}

    monkeypatch.setattr(jobs_worker, "WorkerSDK", OneJobSDK)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object())
    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", phase)
    with scoped_context(user_id=999, is_admin=True):
        before = get_scope()
        await jobs_worker.run_research_jobs_worker(
            research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs"
        )
        assert get_scope() == before  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
    assert dispatched == []  # nosec B101 -- Regression assertion checks worker authorization or cleanup.


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("controlled phase failure"), asyncio.CancelledError("controlled cancellation")],
    ids=["exception", "cancellation"],
)
async def test_research_worker_restores_scope_after_phase_error(monkeypatch, tmp_path, failure):
    session = _session(tmp_path, "7")

    class OneJobSDK:
        def __init__(self, manager, config):
            pass

        async def run(self, *, handler, **kwargs):
            await handler({"id": 41, "owner_user_id": 7, "payload": {"owner_user_id": "999", "session_id": session.id}})

    async def phase(job, **kwargs):
        assert (get_scope().user_id, get_scope().is_admin) == (7, False)  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
        raise failure

    monkeypatch.setattr(jobs_worker, "WorkerSDK", OneJobSDK)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object())
    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", phase)
    with scoped_context(user_id=999, is_admin=True):
        before = get_scope()
        with pytest.raises(type(failure), match="controlled"):
            await jobs_worker.run_research_jobs_worker(
                research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs"
            )
        assert get_scope() == before  # nosec B101 -- Regression assertion checks worker authorization or cleanup.


@pytest.mark.parametrize("missing", [False, True])
async def test_research_worker_refuses_foreign_session_with_explicit_paths(monkeypatch, tmp_path, missing):
    session = _session(tmp_path, "2")
    before = ResearchSessionsDB(tmp_path / "research.db").get_session(session.id)
    dispatched = []

    class OneJobSDK:
        def __init__(self, manager, config):
            pass

        async def run(self, *, handler, **kwargs):
            with pytest.raises(ValueError) as failure:
                await handler(
                    {
                        "id": 41,
                        "owner_user_id": "1",
                        "payload": {"session_id": "rs_missing" if missing else session.id, "owner_user_id": "2"},
                    }
                )
            assert failure.value.retryable is False  # nosec B101 -- Regression assertion checks worker authorization or cleanup.

    async def phase(job, **kwargs):
        dispatched.append(job)
        return {}

    monkeypatch.setattr(jobs_worker, "WorkerSDK", OneJobSDK)
    monkeypatch.setattr(jobs_worker, "_jobs_manager", lambda: object())
    monkeypatch.setattr(jobs_worker, "handle_research_phase_job", phase)
    await jobs_worker.run_research_jobs_worker(
        research_db_path=tmp_path / "research.db", outputs_dir=tmp_path / "outputs"
    )
    assert dispatched == []  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
    assert ResearchSessionsDB(tmp_path / "research.db").get_session(session.id) == before  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
    assert not (tmp_path / "outputs").exists()  # nosec B101 -- Regression assertion checks worker authorization or cleanup.
