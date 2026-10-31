"""A skill tool call returns its PostgreSQL checkout before the skill runs (TASK-13861).

A forked skill makes an LLM call inside SkillExecutor.execute; no ChaCha checkout may
be held across it. Only the registry lookup before it touches ChaCha, and neither it nor
the service's setup checks out on the event loop (TASK-13896).
"""

import asyncio
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.operation_scope import chacha_operation
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Skills import context_integration
from tldw_Server_API.app.core.Skills.skills_service import SkillsService

pytestmark = pytest.mark.integration


@pytest.fixture
async def skills_owner(pg_database_config, tmp_path, monkeypatch):
    """Owner 1's handle, with one skill, on a pool that records which thread holds each checkout."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    pool = backend.get_pool()
    holders, taken, returned, on_loop = {}, Counter(), Counter(), []
    original_get, original_return = pool.get_connection, pool.return_connection

    def get_connection():
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        raw = original_get()
        taken[id(raw)] += 1
        holders[id(raw)] = threading.current_thread().name
        return raw

    def return_connection(raw):
        returned[id(raw)] += 1
        return original_return(raw)

    monkeypatch.setattr(pool, "get_connection", get_connection)
    monkeypatch.setattr(pool, "return_connection", return_connection)
    db = CharactersRAGDB(tmp_path / "skills.db", client_id="1", backend=backend)
    with chacha_operation(independent=True):
        await SkillsService(user_id=1, base_path=tmp_path, db=db).create_skill(
            "summarize", "---\ndescription: Summarize text\ncontext: fork\n---\nSummarize: $ARGUMENTS"
        )

    def outstanding():
        return sorted(holders[key] for key, count in (taken - returned).items() for _ in range(count))

    on_loop.clear()
    try:
        yield SimpleNamespace(db=db, base_path=tmp_path, outstanding=outstanding, on_loop=on_loop)
    finally:
        db.close_connection()
        pool.close_all()


async def test_no_checkout_is_held_while_the_skill_runs(skills_owner, monkeypatch):
    held_while_running = []

    class RecordingExecutor:
        """Stands in for SkillExecutor, whose forked skills make an LLM call."""

        async def execute(self, *, skill_data, arguments, context):  # noqa: ARG002
            held_while_running.append(skills_owner.outstanding())
            return SimpleNamespace(
                skill_name=skill_data["name"], rendered_prompt=arguments, allowed_tools=None,
                model_override=None, execution_mode="fork", fork_output="done",
            )

    monkeypatch.setattr(context_integration, "SkillExecutor", RecordingExecutor)

    result = await context_integration.handle_skill_tool_call(
        skill_name="summarize", args="text", user_id=1, base_path=skills_owner.base_path, db=skills_owner.db
    )

    assert result["success"] is True, result
    assert held_while_running == [[]]
    assert skills_owner.outstanding() == []


async def test_the_skill_setup_and_lookup_take_no_checkout_on_the_event_loop(skills_owner, monkeypatch):
    class InlineExecutor:
        async def execute(self, *, skill_data, arguments, context):  # noqa: ARG002
            return SimpleNamespace(
                skill_name=skill_data["name"], rendered_prompt=arguments, allowed_tools=None,
                model_override=None, execution_mode="inline", fork_output=None,
            )

    monkeypatch.setattr(context_integration, "SkillExecutor", InlineExecutor)

    result = await context_integration.handle_skill_tool_call(
        skill_name="summarize", args="text", user_id=1, base_path=skills_owner.base_path, db=skills_owner.db
    )

    assert (result["success"], len(skills_owner.on_loop) > 0, sum(skills_owner.on_loop)) == (True, True, 0)
