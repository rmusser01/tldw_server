from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat_Macros.acp_adapter import (
    resolve_acp_branch_capability,
    select_branch_strategy,
)
from tldw_Server_API.app.core.Chat_Macros.context_snapshot import MacroContextSnapshot

pytestmark = pytest.mark.unit


def test_resolve_acp_branch_capability_records_resumable_session_metadata():
    snapshot = MacroContextSnapshot(
        conversation_id="conv-1",
        workspace_id="workspace-1",
        acp_session_id="acp-1",
        messages=[],
        selected_message_ids=[],
        selected_source_ids={},
        model_selection={},
        output_profile="default",
        token_estimate=0,
        acp={"forkable": True, "resumable": True, "lineage_id": "root"},
    )

    capability = resolve_acp_branch_capability(snapshot)

    assert capability.available is True
    assert capability.resumable is True
    assert capability.session_id == "acp-1"
    assert capability.metadata["lineage_id"] == "root"


def test_auto_strategy_falls_back_to_chat_native_without_acp_session():
    snapshot = MacroContextSnapshot(
        conversation_id="conv-1",
        workspace_id=None,
        acp_session_id=None,
        messages=[],
        selected_message_ids=[],
        selected_source_ids={},
        model_selection={},
        output_profile="default",
        token_estimate=0,
        acp={},
    )
    capability = resolve_acp_branch_capability(snapshot)

    decision = select_branch_strategy(
        step_strategy=None,
        macro_strategy="auto",
        capability=capability,
    )

    assert decision.strategy == "chat_native"
    assert decision.fallback is True
    assert decision.required_failed is False
    assert decision.metadata["reason"] == "acp_unavailable"


def test_required_acp_fork_strategy_fails_when_unavailable():
    snapshot = MacroContextSnapshot(
        conversation_id="conv-1",
        workspace_id=None,
        acp_session_id=None,
        messages=[],
        selected_message_ids=[],
        selected_source_ids={},
        model_selection={},
        output_profile="default",
        token_estimate=0,
        acp={},
    )
    capability = resolve_acp_branch_capability(snapshot)

    decision = select_branch_strategy(
        step_strategy="acp_fork",
        macro_strategy="auto",
        capability=capability,
    )

    assert decision.strategy == "failed"
    assert decision.required_failed is True
    assert decision.error_code == "acp_unavailable"


def test_acp_capability_distinguishes_not_resumable_and_not_forkable() -> None:
    not_resumable = resolve_acp_branch_capability(
        MacroContextSnapshot(
            acp_session_id="session-1",
            acp={"forkable": True, "resumable": False},
        )
    )
    not_forkable = resolve_acp_branch_capability(
        MacroContextSnapshot(
            acp_session_id="session-2",
            acp={"forkable": False, "resumable": True},
        )
    )

    assert not_resumable.reason == "acp_not_resumable"
    assert not_forkable.reason == "acp_not_forkable"
    decision = select_branch_strategy(
        step_strategy="acp_fork",
        macro_strategy="auto",
        capability=not_forkable,
    )
    assert decision.required_failed is True
    assert decision.metadata["reason"] == "acp_not_forkable"
