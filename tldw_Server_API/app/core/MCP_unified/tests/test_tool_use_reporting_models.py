"""Tests for metadata-only MCP tool-use reporting models."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from mcp_unified.tool_use_reporting.models import ToolUseEvent
from mcp_unified.tool_use_reporting.sanitization import sanitize_safe_id


def test_tool_use_event_normalizes_created_at_to_utc_epoch_ordering() -> None:
    event = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 12, 0, tzinfo=timezone(timedelta(hours=-7))),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    assert event.created_at_utc.tzinfo == timezone.utc
    assert event.created_at_utc.isoformat() == "2026-06-06T19:00:00+00:00"
    assert event.created_at_epoch_us == 1_780_772_400_000_000


def test_tool_use_event_rejects_or_omits_sensitive_payload_fields() -> None:
    event = ToolUseEvent(
        runtime_surface="gateway",
        requested_tool_name="fs.read",
        status="error",
        reason_code="/Users/example/private.txt",
        raw_arguments={"path": "/Users/example/private.txt"},
    )

    dumped = event.model_dump(mode="json")
    assert "raw_arguments" not in dumped
    assert "/Users/example" not in str(dumped)
    assert event.reason_code == "unknown"


def test_tool_use_event_is_immutable() -> None:
    event = ToolUseEvent(
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    with pytest.raises((TypeError, ValueError)):
        event.status = "error"  # type: ignore[misc]


def test_sanitize_safe_id_allows_bounded_profile_model_mode_ids() -> None:
    assert sanitize_safe_id("Architect-01", field="profile_id") == "Architect-01"
    assert sanitize_safe_id("gpt-4.1-mini", field="model_id") == "gpt-4.1-mini"
    assert sanitize_safe_id("qa_mode.default", field="mode_id") == "qa_mode.default"


def test_sanitize_safe_id_drops_paths_emails_and_long_values() -> None:
    assert sanitize_safe_id("/Users/me/project", field="profile_id") is None
    assert sanitize_safe_id("person@example.com", field="profile_id") is None
    assert sanitize_safe_id("x" * 512, field="profile_id") is None


def test_sanitize_safe_id_drops_values_with_unsafe_characters() -> None:
    assert sanitize_safe_id("profile id with spaces", field="profile_id") is None
    assert sanitize_safe_id("../escape", field="tool_prompt_id") is None
