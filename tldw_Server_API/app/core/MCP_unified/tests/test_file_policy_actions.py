"""Tests for shared MCP file-policy action metadata."""

from __future__ import annotations

import pytest


def test_file_policy_action_taxonomy_lists_existing_and_reserved_actions() -> None:
    from mcp_unified.interfaces.file_policy_actions import (
        FILE_POLICY_ACTIONS,
        FILE_POLICY_EXFILTRATION_ACTIONS,
        FILE_POLICY_EXISTING_TOOL_ACTIONS,
    )

    assert frozenset(
        {
            "read",
            "edit",
            "write",
            "delete",
            "rename",
            "move",
            "share",
            "export",
            "chmod",
            "admin",
            "lock",
        }
    ) == FILE_POLICY_ACTIONS
    assert frozenset({"read", "edit", "write", "lock"}) == FILE_POLICY_EXISTING_TOOL_ACTIONS
    assert frozenset({"share", "export"}) == FILE_POLICY_EXFILTRATION_ACTIONS


def test_file_policy_action_metadata_describes_reserved_actions_without_implementation() -> None:
    from mcp_unified.interfaces.file_policy_actions import get_file_policy_action_metadata

    metadata = get_file_policy_action_metadata("share")

    assert metadata.as_dict() == {
        "action": "share",
        "family": "exfiltration",
        "description": "Share or publish file content outside the active workspace boundary.",
        "implemented": False,
        "risk": "exfiltration",
    }


def test_file_policy_action_metadata_describes_lock_as_implemented() -> None:
    from mcp_unified.interfaces.file_policy_actions import get_file_policy_action_metadata

    metadata = get_file_policy_action_metadata("lock")

    assert metadata.as_dict() == {
        "action": "lock",
        "family": "lock",
        "description": "Acquire or release coordination locks for allowed workspace paths.",
        "implemented": True,
        "risk": "coordination",
    }


def test_file_policy_action_metadata_rejects_unknown_actions() -> None:
    from mcp_unified.interfaces.file_policy_actions import get_file_policy_action_metadata

    with pytest.raises(ValueError, match="unknown file policy action"):
        get_file_policy_action_metadata("destroy")


def test_file_policy_action_metadata_rejects_missing_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp_unified.interfaces import file_policy_actions

    monkeypatch.delitem(file_policy_actions._ACTION_METADATA, "lock")

    with pytest.raises(ValueError, match="metadata is missing"):
        file_policy_actions.get_file_policy_action_metadata("lock")
