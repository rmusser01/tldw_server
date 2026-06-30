from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.services.admin_acp_sessions_service import ACPSessionStore


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_register_and_record_prompt_preserve_creation_config_and_bootstrap_ready_transcript(tmp_path):
    _db = ACPSessionsDB(db_path=str(tmp_path / "store_test.db"))
    store = ACPSessionStore(db=_db)

    await store.register_session(
        session_id="session-1",
        user_id=7,
        agent_type="codex",
        name="Seed Session",
        cwd="/tmp/project",
        tags=["alpha"],
        mcp_servers=[{"name": "filesystem", "command": "fs-server"}],
        persona_id="persona-1",
        workspace_id="workspace-1",
        workspace_group_id="group-1",
        scope_snapshot_id="scope-1",
    )

    usage = await store.record_prompt(
        "session-1",
        [{"role": "user", "content": "Hello"}],
        {"content": "World", "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
    )

    record = await store.get_session("session-1")
    assert record is not None
    assert usage is not None
    assert record.mcp_servers == [{"name": "filesystem", "command": "fs-server"}]
    assert record.bootstrap_ready is True
    assert record.needs_bootstrap is False
    assert [msg["content"] for msg in record.messages] == ["Hello", "World"]
    assert record.message_count == 2
    assert record.to_info_dict()["persona_id"] == "persona-1"


@pytest.mark.asyncio
async def test_session_store_preserves_sandbox_context_and_filters_by_workspace(tmp_path):
    _db = ACPSessionsDB(db_path=str(tmp_path / "store_test.db"))
    store = ACPSessionStore(db=_db)

    await store.register_session(
        session_id="session-workspace",
        user_id=7,
        agent_type="codex",
        name="Workspace Session",
        cwd="/tmp/project",
        workspace_id="workspace-1",
        sandbox_session_id="sandbox-session-1",
        sandbox_run_id="sandbox-run-1",
    )
    await store.register_session(
        session_id="session-other",
        user_id=7,
        agent_type="codex",
        name="Other Session",
        cwd="/tmp/project",
        workspace_id="workspace-2",
    )

    record = await store.get_session("session-workspace")
    assert record is not None
    assert record.sandbox_session_id == "sandbox-session-1"
    assert record.sandbox_run_id == "sandbox-run-1"
    assert record.to_info_dict()["sandbox_session_id"] == "sandbox-session-1"

    rows, total = await store.list_sessions(user_id=7, workspace_id="workspace-1")
    assert total == 1
    assert [row.session_id for row in rows] == ["session-workspace"]


@pytest.mark.asyncio
async def test_session_store_exposes_policy_snapshot_fields_in_info_and_detail_dict(tmp_path):
    _db = ACPSessionsDB(db_path=str(tmp_path / "store_test.db"))
    store = ACPSessionStore(db=_db)

    await store.register_session(
        session_id="session-policy",
        user_id=7,
        agent_type="codex",
        name="Policy Session",
        cwd="/tmp/project",
        policy_snapshot_version="v1",
        policy_snapshot_fingerprint="policy-fingerprint",
        policy_snapshot_refreshed_at="2026-03-14T12:00:00+00:00",
        policy_summary={"allowed_tools": 3},
        policy_provenance_summary={"sources": ["mcp_hub"]},
        policy_refresh_error="stale snapshot",
    )

    record = await store.get_session("session-policy")
    assert record is not None
    assert record.policy_snapshot_version == "v1"
    assert record.policy_snapshot_fingerprint == "policy-fingerprint"

    info = record.to_info_dict()
    detail = record.to_detail_dict()
    assert info["policy_snapshot_refreshed_at"] == "2026-03-14T12:00:00+00:00"
    assert info["policy_summary"] == {"allowed_tools": 3}
    assert info["policy_provenance_summary"] == {"sources": ["mcp_hub"]}
    assert detail["policy_refresh_error"] == "stale snapshot"


@pytest.mark.asyncio
async def test_record_prompt_marks_session_non_bootstrappable_when_assistant_text_cannot_be_normalized(tmp_path):
    _db = ACPSessionsDB(db_path=str(tmp_path / "store_test.db"))
    store = ACPSessionStore(db=_db)
    await store.register_session(
        session_id="session-2",
        user_id=7,
        agent_type="codex",
        name="Opaque Session",
        cwd="/tmp/project",
        mcp_servers=[{"name": "filesystem"}],
    )

    await store.record_prompt(
        "session-2",
        [{"role": "user", "content": "Hello"}],
        {"detail": {"structured": True}},
    )

    record = await store.get_session("session-2")
    assert record is not None
    assert record.bootstrap_ready is False
    assert record.messages[-1]["role"] == "assistant"
    assert record.messages[-1]["content"] is None
    assert record.messages[-1]["raw_result"] == {"detail": {"structured": True}}


@pytest.mark.asyncio
async def test_fork_session_copies_lineage_config_and_bootstrap_state(tmp_path):
    _db = ACPSessionsDB(db_path=str(tmp_path / "store_test.db"))
    store = ACPSessionStore(db=_db)
    await store.register_session(
        session_id="session-source",
        user_id=7,
        agent_type="codex",
        name="Seed Session",
        cwd="/tmp/project",
        tags=["alpha"],
        mcp_servers=[{"name": "filesystem"}],
        persona_id="persona-1",
        workspace_id="workspace-1",
        workspace_group_id="group-1",
        scope_snapshot_id="scope-1",
    )
    await store.record_prompt(
        "session-source",
        [{"role": "user", "content": "Hello"}],
        {"content": "World"},
    )

    forked = await store.fork_session(
        source_session_id="session-source",
        new_session_id="session-fork",
        message_index=1,
        user_id=7,
        name="Forked Session",
    )

    assert forked is not None
    assert forked.forked_from == "session-source"
    assert forked.mcp_servers == [{"name": "filesystem"}]
    assert forked.bootstrap_ready is True
    assert forked.needs_bootstrap is True
    assert forked.message_count == 2
