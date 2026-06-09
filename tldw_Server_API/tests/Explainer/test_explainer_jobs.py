from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.jobs import (
    EXPLAINER_DOMAIN,
    EXPLAINER_JOB_TYPE,
    EXPLAINER_QUEUE,
    handle_explainer_node_expansion_job,
)
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository
from tldw_Server_API.app.core.Explainer.retrieval import ExplainerSourceContext
from tldw_Server_API.app.core.Explainer.service import ExplainerService

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
) -> None:
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
