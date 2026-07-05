from __future__ import annotations

import importlib.util
import os
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Evaluations.embeddings_abtest_repository import (
    EmbeddingABTestRepository,
    RepositoryConfig,
)


pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_full_fixtures"]
pytestmark = [pytest.mark.integration, pytest.mark.postgres]


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def test_abtest_repository_crud_postgres(request: pytest.FixtureRequest) -> None:
    _client, _db_name = request.getfixturevalue("isolated_test_environment")  # noqa: F841
    db_url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    assert db_url and "postgres" in db_url
    has_psycopg = _module_available("psycopg")
    has_psycopg2 = _module_available("psycopg2")

    if not has_psycopg and not has_psycopg2:
        pytest.skip("No PostgreSQL DBAPI driver installed for SQLAlchemy (psycopg or psycopg2)")

    # SQLAlchemy defaults to psycopg2 for postgresql:// URLs; prefer psycopg3
    # explicitly when psycopg2 is unavailable in this environment.
    if has_psycopg and not has_psycopg2:
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
        elif db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql+psycopg://", 1)

    repo = EmbeddingABTestRepository.from_config(RepositoryConfig(db_url=db_url))
    test_id = f"abtest-{uuid4()}"
    arm_id = f"arm-{uuid4()}"
    query_id = f"query-{uuid4()}"
    result_id = f"result-{uuid4()}"

    repo.create_test(
        test_id=test_id,
        name="postgres-repo-test",
        created_by="tester",
        config={"arms": [{"provider": "openai", "model": "text-embedding-3-small"}]},
    )
    repo.add_arm(
        arm_id=arm_id,
        test_id=test_id,
        arm_index=0,
        provider="openai",
        model_id="text-embedding-3-small",
        status="ready",
    )
    repo.add_query(
        query_id=query_id,
        test_id=test_id,
        text="quick brown fox",
        ground_truth_ids=["doc-1"],
    )
    repo.record_result(
        result_id=result_id,
        test_id=test_id,
        arm_id=arm_id,
        query_id=query_id,
        ranked_ids=["doc-1", "doc-2"],
        scores=[0.9, 0.1],
        metrics={"recall@1": 1.0},
        latency_ms=42.5,
    )
    repo.update_test_status(test_id=test_id, status="completed", stats={"tests_completed": 1})

    snapshot = repo.get_test_with_children(test_id)
    assert snapshot is not None
    assert snapshot.status == "completed"
    assert snapshot.arms and snapshot.arms[0].arm_id == arm_id
    assert snapshot.queries and snapshot.queries[0].query_id == query_id
    assert snapshot.results and snapshot.results[0].result_id == result_id

    rows, total = repo.list_results(test_id, limit=10, offset=0)
    assert total == 1
    assert len(rows) == 1
    assert rows[0].result_id == result_id
