from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.jobs import (
    EXPLAINER_DOMAIN,
    EXPLAINER_JOB_TYPE,
    EXPLAINER_QUEUE,
    ExplainerGenerationNotConfiguredError,
    current_answer_revision,
    handle_explainer_node_expansion_job,
    make_configured_explainer_generator,
)
from tldw_Server_API.app.core.Explainer.jobs_worker import build_explainer_job_handler
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository
from tldw_Server_API.app.core.Explainer.retrieval import (
    ExplainerSourceAccessError,
    ExplainerSourceContext,
    ExplainerSourceExcerpt,
)
from tldw_Server_API.app.core.Explainer.service import ExplainerService, ExplainerValidationError

pytestmark = pytest.mark.unit


class FakeJobManager:
    def __init__(self) -> None:
        self.jobs: list[dict[str, Any]] = []
        self.next_id = 123

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        job = {
            "id": self.next_id,
            "uuid": f"job-{self.next_id}",
            "status": "queued",
            **kwargs,
        }
        self.jobs.append(job)
        self.next_id += 1
        return job


@pytest.fixture()
def explainer_repo(tmp_path) -> ExplainerRepository:
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    return ExplainerRepository(db)


@pytest.fixture()
def fake_job_manager() -> FakeJobManager:
    return FakeJobManager()


def _create_session(
    repo: ExplainerRepository,
    *,
    grounding: str = "open",
    selected_sources: list[dict[str, Any]] | None = None,
):
    return repo.create_session(
        owner_user_id="7",
        title="Learn attention",
        mode="goal",
        output_intent="both",
        grounding=grounding,
        depth_preset="standard",
        selected_sources=selected_sources or [],
        root_prompt="Explain transformer attention",
    )


def test_expand_marks_node_queued_and_creates_job(
    fake_job_manager: FakeJobManager,
    explainer_repo: ExplainerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPLAINER_GENERATOR_ENABLED", "1")
    monkeypatch.setenv("EXPLAINER_GENERATOR_PROVIDER", "openai")
    monkeypatch.setenv("EXPLAINER_GENERATOR_MODEL", "test-model")
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]
    service = ExplainerService(repo=explainer_repo, job_manager=fake_job_manager)

    accepted = service.enqueue_node_expansion(
        session_id=session.id,
        node_id=node_id,
        owner_user_id="7",
        intent="both",
    )

    assert accepted.status == "queued"
    assert accepted.job_id
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "queued"
    [job] = fake_job_manager.jobs
    assert job["domain"] == EXPLAINER_DOMAIN
    assert job["queue"] == EXPLAINER_QUEUE
    assert job["job_type"] == EXPLAINER_JOB_TYPE
    assert job["owner_user_id"] == "7"
    assert job["payload"]["session_id"] == session.id
    assert job["payload"]["node_id"] == node_id
    assert job["payload"]["intent"] == "both"
    assert "prompt" not in job["payload"]
    assert "source_context" not in job["payload"]
    assert job["idempotency_key"].startswith(f"explainer:{session.id}:{node_id}:both:")


def test_expand_rejects_unconfigured_open_generation(
    fake_job_manager: FakeJobManager,
    explainer_repo: ExplainerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EXPLAINER_GENERATOR_ENABLED", raising=False)
    monkeypatch.delenv("EXPLAINER_GENERATOR_PROVIDER", raising=False)
    monkeypatch.delenv("EXPLAINER_GENERATOR_MODEL", raising=False)
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]
    service = ExplainerService(repo=explainer_repo, job_manager=fake_job_manager)

    with pytest.raises(ExplainerValidationError):
        service.enqueue_node_expansion(
            session_id=session.id,
            node_id=node_id,
            owner_user_id="7",
            intent="both",
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "idle"
    assert fake_job_manager.jobs == []


def test_expand_does_not_mark_queued_for_terminal_idempotent_job(
    fake_job_manager: FakeJobManager,
    explainer_repo: ExplainerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPLAINER_GENERATOR_ENABLED", "1")
    monkeypatch.setenv("EXPLAINER_GENERATOR_PROVIDER", "openai")
    monkeypatch.setenv("EXPLAINER_GENERATOR_MODEL", "test-model")
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]

    def terminal_create_job(**kwargs: Any) -> dict[str, Any]:
        return {
            "id": 999,
            "uuid": "job-999",
            "status": "completed",
            **kwargs,
        }

    fake_job_manager.create_job = terminal_create_job
    service = ExplainerService(repo=explainer_repo, job_manager=fake_job_manager)

    with pytest.raises(ExplainerValidationError):
        service.enqueue_node_expansion(
            session_id=session.id,
            node_id=node_id,
            owner_user_id="7",
            intent="both",
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "idle"


@pytest.mark.asyncio()
async def test_fresh_queued_job_revision_survives_queue_status_update(
    fake_job_manager: FakeJobManager,
    explainer_repo: ExplainerRepository,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPLAINER_GENERATOR_ENABLED", "1")
    monkeypatch.setenv("EXPLAINER_GENERATOR_PROVIDER", "openai")
    monkeypatch.setenv("EXPLAINER_GENERATOR_MODEL", "test-model")
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]
    service = ExplainerService(repo=explainer_repo, job_manager=fake_job_manager)
    monkeypatch.setattr(
        explainer_repo.db,
        "utcnow_iso",
        lambda: "2099-01-01T00:00:00+00:00",
    )

    service.enqueue_node_expansion(
        session_id=session.id,
        node_id=node_id,
        owner_user_id="7",
        intent="explain",
    )
    [job] = fake_job_manager.jobs

    async def fake_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Fresh queued child",
                    "body": "A newly queued job should still generate.",
                    "kind": "explanation",
                    "intent": "explain",
                    "outside_knowledge_used": True,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    result = await handle_explainer_node_expansion_job(
        job,
        repo=explainer_repo,
        generator=fake_generator,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    assert loaded.nodes[child_id].title == "Fresh queued child"


@pytest.mark.asyncio()
async def test_job_handler_writes_child_nodes_and_generation_metadata(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]

    async def fake_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Attention scores",
                    "body": "Attention compares query and key vectors.",
                    "kind": "explanation",
                    "intent": "explain",
                    "citations": [],
                    "outside_knowledge_used": True,
                }
            ],
            "generation_metadata": {
                "provider": "fake",
                "model": "fake-model",
                "tokenUsage": {"totalTokens": 12},
            },
        }

    result = await handle_explainer_node_expansion_job(
        {
            "id": 101,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "both",
            },
        },
        repo=explainer_repo,
        generator=fake_generator,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.title == "Attention scores"
    assert child.status == "complete"
    assert child.evidence_state == "uncited"
    assert child.generation_metadata is not None
    assert child.generation_metadata["provider"] == "fake"
    assert child.generation_metadata["model"] == "fake-model"
    assert child.generation_metadata["promptTemplateVersion"] == "explainer_node_expansion_v1"
    assert child.generation_metadata["jobId"] == "101"
    assert loaded.nodes[node_id].status == "complete"


@pytest.mark.asyncio()
async def test_worker_handler_uses_configured_generator(
    tmp_path,
) -> None:
    db_path = tmp_path / "Explainer.db"
    db = ExplainerDatabase(db_path)
    repo = ExplainerRepository(db)
    session = _create_session(repo)
    node_id = session.root_node_ids[0]
    db.close_connection()

    async def fake_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Worker child",
                    "body": "Generated through worker handler.",
                    "kind": "explanation",
                    "intent": "explain",
                    "outside_knowledge_used": True,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "worker"},
        }

    handler = build_explainer_job_handler(
        db_path_resolver=lambda _owner_user_id: db_path,
        generator_factory=lambda: fake_generator,
    )

    result = await handler(
        {
            "id": 201,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": current_answer_revision(session.nodes[node_id]),
            },
        }
    )

    assert result["children_created"] == 1
    verify_db = ExplainerDatabase(db_path)
    loaded = ExplainerRepository(verify_db).get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    assert loaded.nodes[child_id].title == "Worker child"
    verify_db.close_connection()


@pytest.mark.asyncio()
async def test_job_handler_marks_target_node_error_on_provider_failure(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]

    async def failing_generator(_prompt):
        raise RuntimeError("provider unavailable")

    with pytest.raises(RuntimeError):
        await handle_explainer_node_expansion_job(
            {
                "id": 102,
                "job_type": EXPLAINER_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {
                    "session_id": session.id,
                    "node_id": node_id,
                    "intent": "explain",
                },
            },
            repo=explainer_repo,
            generator=failing_generator,
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "error"


@pytest.mark.asyncio()
async def test_source_only_insufficient_retrieval_creates_insufficient_child(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(
        explainer_repo,
        grounding="source_only",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
    )
    node_id = session.root_node_ids[0]

    async def generator_should_not_run(_prompt):
        raise AssertionError("generator should not run when source-only retrieval is insufficient")

    def insufficient_retriever(*, session, owner_user_id):
        return ExplainerSourceContext(
            excerpts=[],
            insufficient=True,
            retrieval_metadata={"reason": "no_selected_source_matches"},
        )

    result = await handle_explainer_node_expansion_job(
        {
            "id": 103,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
            },
        },
        repo=explainer_repo,
        generator=generator_should_not_run,
        retriever=insufficient_retriever,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.status == "complete"
    assert child.evidence_state == "insufficient"
    assert child.outside_knowledge_used is False
    assert loaded.nodes[node_id].status == "complete"


@pytest.mark.asyncio()
async def test_default_source_only_snapshot_metadata_is_insufficient_without_authoritative_context(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(
        explainer_repo,
        grounding="source_only",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "metadata": {
                    "excerpt": "This selected-source snapshot is not an authoritative retrieval result.",
                },
            }
        ],
    )
    node_id = session.root_node_ids[0]

    async def generator_should_not_run(_prompt):
        raise AssertionError("default source-only path should return insufficient without authoritative context")

    result = await handle_explainer_node_expansion_job(
        {
            "id": 104,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": current_answer_revision(session.nodes[node_id]),
            },
        },
        repo=explainer_repo,
        generator=generator_should_not_run,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.evidence_state == "insufficient"
    assert child.outside_knowledge_used is False


@pytest.mark.asyncio()
async def test_source_only_forged_selected_source_citation_becomes_insufficient(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(
        explainer_repo,
        grounding="source_only",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
    )
    node_id = session.root_node_ids[0]

    def authoritative_retriever(*, session, owner_user_id):
        return ExplainerSourceContext(
            excerpts=[
                ExplainerSourceExcerpt(
                    source_id="media-42",
                    source_type="media",
                    title="Attention paper notes",
                    excerpt="Attention weights are computed from query-key similarity.",
                    location_label="chunk 3",
                    snapshot_hash="sha256:real",
                )
            ],
            insufficient=False,
        )

    async def forged_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Forged citation",
                    "body": "Unsupported claim.",
                    "kind": "explanation",
                    "intent": "explain",
                    "citations": [
                        {
                            "source_id": "media-42",
                            "source_type": "media",
                            "title": "Attention paper notes",
                            "excerpt": "A fabricated excerpt that was not retrieved.",
                            "location_label": "chunk 3",
                            "snapshot_hash": "sha256:real",
                        }
                    ],
                    "outside_knowledge_used": False,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    result = await handle_explainer_node_expansion_job(
        {
            "id": 105,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": current_answer_revision(session.nodes[node_id]),
            },
        },
        repo=explainer_repo,
        generator=forged_generator,
        retriever=authoritative_retriever,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.evidence_state == "insufficient"
    assert child.outside_knowledge_used is False
    assert child.citations == []


@pytest.mark.asyncio()
async def test_source_led_uncited_claim_cannot_be_marked_supported(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo, grounding="source_led")
    node_id = session.root_node_ids[0]

    async def unsupported_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Unsupported claim",
                    "body": "This claim has no source evidence.",
                    "kind": "explanation",
                    "intent": "explain",
                    "evidence_state": "supported",
                    "outside_knowledge_used": False,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    result = await handle_explainer_node_expansion_job(
        {
            "id": 109,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": current_answer_revision(session.nodes[node_id]),
            },
        },
        repo=explainer_repo,
        generator=unsupported_generator,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.evidence_state == "uncited"
    assert child.citations == []


@pytest.mark.asyncio()
async def test_source_led_forged_citation_is_removed_and_downgraded(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(
        explainer_repo,
        grounding="source_led",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
    )
    node_id = session.root_node_ids[0]

    def authoritative_retriever(*, session, owner_user_id):
        return ExplainerSourceContext(
            excerpts=[
                ExplainerSourceExcerpt(
                    source_id="media-42",
                    source_type="media",
                    title="Attention paper notes",
                    excerpt="Attention weights are computed from query-key similarity.",
                    location_label="chunk 3",
                    snapshot_hash="sha256:real",
                )
            ],
            insufficient=False,
        )

    async def forged_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Forged source-led citation",
                    "body": "Unsupported claim.",
                    "kind": "explanation",
                    "intent": "explain",
                    "citations": [
                        {
                            "source_id": "media-42",
                            "source_type": "media",
                            "title": "Attention paper notes",
                            "excerpt": "A fabricated excerpt that was not retrieved.",
                            "location_label": "chunk 3",
                            "snapshot_hash": "sha256:real",
                        }
                    ],
                    "outside_knowledge_used": False,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    result = await handle_explainer_node_expansion_job(
        {
            "id": 110,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": current_answer_revision(session.nodes[node_id]),
            },
        },
        repo=explainer_repo,
        generator=forged_generator,
        retriever=authoritative_retriever,
    )

    assert result["children_created"] == 1
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    [child_id] = loaded.nodes[node_id].child_node_ids
    child = loaded.nodes[child_id]
    assert child.evidence_state == "uncited"
    assert child.citations == []


@pytest.mark.asyncio()
async def test_retriever_rejects_unselected_source_context(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(
        explainer_repo,
        grounding="source_only",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
            }
        ],
    )
    node_id = session.root_node_ids[0]

    def unselected_retriever(*, session, owner_user_id):
        return ExplainerSourceContext(
            excerpts=[
                ExplainerSourceExcerpt(
                    source_id="media-99",
                    source_type="media",
                    title="Unselected notes",
                    excerpt="This source was not selected for the session.",
                )
            ],
            insufficient=False,
        )

    async def generator_should_not_run(_prompt):
        raise AssertionError("unselected source context must be rejected before generation")

    with pytest.raises(ExplainerSourceAccessError):
        await handle_explainer_node_expansion_job(
            {
                "id": 106,
                "job_type": EXPLAINER_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {
                    "session_id": session.id,
                    "node_id": node_id,
                    "intent": "explain",
                    "answer_revision": current_answer_revision(session.nodes[node_id]),
                },
            },
            repo=explainer_repo,
            generator=generator_should_not_run,
            retriever=unselected_retriever,
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "error"
    assert loaded.nodes[node_id].child_node_ids == []


@pytest.mark.asyncio()
async def test_stale_answer_revision_skips_without_writing_children(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]
    stale_revision = current_answer_revision(session.nodes[node_id])
    explainer_repo.update_node(
        session.id,
        node_id,
        owner_user_id="7",
        selected_custom_answer="A newer answer",
    )

    async def generator_should_not_run(_prompt):
        raise AssertionError("stale jobs must not generate")

    result = await handle_explainer_node_expansion_job(
        {
            "id": 106,
            "job_type": EXPLAINER_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {
                "session_id": session.id,
                "node_id": node_id,
                "intent": "explain",
                "answer_revision": stale_revision,
            },
        },
        repo=explainer_repo,
        generator=generator_should_not_run,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "stale_answer_revision"
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].child_node_ids == []
    assert loaded.nodes[node_id].selected_custom_answer == "A newer answer"


@pytest.mark.asyncio()
async def test_retrying_same_expansion_job_does_not_duplicate_children(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]
    job = {
        "id": 107,
        "job_type": EXPLAINER_JOB_TYPE,
        "owner_user_id": "7",
        "payload": {
            "session_id": session.id,
            "node_id": node_id,
            "intent": "explain",
            "answer_revision": current_answer_revision(session.nodes[node_id]),
        },
    }

    async def fake_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Retry-safe child",
                    "body": "Generated once.",
                    "kind": "explanation",
                    "intent": "explain",
                    "outside_knowledge_used": True,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    first = await handle_explainer_node_expansion_job(job, repo=explainer_repo, generator=fake_generator)
    second = await handle_explainer_node_expansion_job(job, repo=explainer_repo, generator=fake_generator)

    assert first["children_created"] == 1
    assert second["children_created"] == 0
    assert second["status"] == "skipped"
    assert second["reason"] == "duplicate_expansion_batch"
    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert len(loaded.nodes[node_id].child_node_ids) == 1


@pytest.mark.asyncio()
async def test_mid_persist_failure_rolls_back_created_children(
    explainer_repo: ExplainerRepository,
) -> None:
    session = _create_session(explainer_repo)
    node_id = session.root_node_ids[0]

    class FailingMetadataRepository:
        def __init__(self, wrapped: ExplainerRepository) -> None:
            self.wrapped = wrapped

        def get_session(self, *args, **kwargs):
            return self.wrapped.get_session(*args, **kwargs)

        def create_node(self, *args, **kwargs):
            return self.wrapped.create_node(*args, **kwargs)

        def delete_node(self, *args, **kwargs):
            return self.wrapped.delete_node(*args, **kwargs)

        def update_node(self, session_id, node_id, *, owner_user_id, **updates):
            if "generation_metadata" in updates and node_id != session.root_node_ids[0]:
                raise RuntimeError("metadata write failed")
            return self.wrapped.update_node(
                session_id,
                node_id,
                owner_user_id=owner_user_id,
                **updates,
            )

    async def fake_generator(_prompt):
        return {
            "children": [
                {
                    "title": "Partial child",
                    "body": "This should be rolled back.",
                    "kind": "explanation",
                    "intent": "explain",
                    "outside_knowledge_used": True,
                }
            ],
            "generation_metadata": {"provider": "fake", "model": "fake"},
        }

    with pytest.raises(RuntimeError):
        await handle_explainer_node_expansion_job(
            {
                "id": 108,
                "job_type": EXPLAINER_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {
                    "session_id": session.id,
                    "node_id": node_id,
                    "intent": "explain",
                    "answer_revision": current_answer_revision(session.nodes[node_id]),
                },
            },
            repo=FailingMetadataRepository(explainer_repo),
            generator=fake_generator,
        )

    loaded = explainer_repo.get_session(session.id, owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].child_node_ids == []
    assert loaded.nodes[node_id].status == "error"


@pytest.mark.asyncio()
async def test_configured_generator_parses_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeAdapter:
        def chat(self, request, *, timeout=None):
            assert request["model"] == "fake-model"
            assert request["messages"]
            return {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"children":[{"title":"Configured child","body":"Adapter output",'
                                '"kind":"explanation","intent":"explain","outside_knowledge_used":true}],'
                                '"generation_metadata":{"provider":"fake","model":"fake-model"}}'
                            )
                        }
                    }
                ]
            }

    monkeypatch.setenv("EXPLAINER_GENERATOR_ENABLED", "1")
    monkeypatch.setenv("EXPLAINER_GENERATOR_PROVIDER", "fake-provider")
    monkeypatch.setenv("EXPLAINER_GENERATOR_MODEL", "fake-model")
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    monkeypatch.setattr(adapter_utils, "get_adapter_or_raise", lambda _provider: FakeAdapter())
    monkeypatch.setattr(adapter_utils, "resolve_provider_api_key_from_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter_utils, "ensure_app_config", lambda *_args, **_kwargs: {})

    generator = make_configured_explainer_generator()
    result = await generator(
        type(
            "Prompt",
            (),
            {
                "as_messages": lambda self: [{"role": "user", "content": "Expand"}],
            },
        )()
    )

    assert result["children"][0]["title"] == "Configured child"
    assert result["generation_metadata"]["provider"] == "fake"
