"""Tests for shared MCP tool observability metadata helpers."""

from __future__ import annotations

import inspect

import pytest

from tldw_Server_API.app.core.MCP_unified.tool_observability import (
    build_execution_eval_metadata,
    build_tool_eval_metadata,
)


def test_build_tool_eval_metadata_is_stable_and_non_empty() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id="mcp.git.status.v1",
        tool_prompt_version="2026.06.04",
        task_families=["code_review"],
        expected_result_kind="structured_git_state",
        success_signals=["avoided_mutation"],
    )

    assert metadata == {
        "eval": {
            "tool_prompt_id": "mcp.git.status.v1",
            "tool_prompt_version": "2026.06.04",
            "task_families": ["code_review"],
            "expected_result_kind": "structured_git_state",
            "success_signals": ["avoided_mutation"],
            "prompt_variant": "builtin",
        }
    }


def test_build_tool_eval_metadata_strips_values_and_removes_blank_list_items() -> None:
    metadata = build_tool_eval_metadata(
        tool_prompt_id=" mcp.git.diff.v1 ",
        tool_prompt_version=" 2026.06.04 ",
        task_families=[" code_review ", "", "  ", " repository_research"],
        expected_result_kind=" bounded_diff ",
        success_signals=[" avoided_mutation ", "", " selected_correct_scope "],
        prompt_variant="   ",
    )

    assert metadata == {
        "eval": {
            "tool_prompt_id": "mcp.git.diff.v1",
            "tool_prompt_version": "2026.06.04",
            "task_families": ["code_review", "repository_research"],
            "expected_result_kind": "bounded_diff",
            "success_signals": ["avoided_mutation", "selected_correct_scope"],
            "prompt_variant": "builtin",
        }
    }


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("tool_prompt_id", {"tool_prompt_id": " "}),
        ("tool_prompt_version", {"tool_prompt_version": ""}),
        ("expected_result_kind", {"expected_result_kind": "\t"}),
    ],
)
def test_build_tool_eval_metadata_rejects_blank_required_strings(
    field: str, kwargs: dict[str, str]
) -> None:
    values = {
        "tool_prompt_id": "mcp.git.status.v1",
        "tool_prompt_version": "2026.06.04",
        "task_families": ["code_review"],
        "expected_result_kind": "structured_git_state",
        "success_signals": ["avoided_mutation"],
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=field):
        build_tool_eval_metadata(**values)


def test_build_execution_eval_metadata_returns_only_safe_scalar_fields() -> None:
    metadata = build_execution_eval_metadata(
        tool_name=" git.status ",
        tool_prompt_id=" mcp.git.status.v1 ",
        tool_prompt_version=" 2026.06.04 ",
        action_family=" status ",
        result_kind=" structured_git_state ",
        profile_id=" code-reviewer ",
        path_filter_used=True,
        truncated=False,
        reason_code=" ok ",
        duration_ms=12.75,
    )

    assert metadata == {
        "tool_name": "git.status",
        "tool_prompt_id": "mcp.git.status.v1",
        "tool_prompt_version": "2026.06.04",
        "action_family": "status",
        "result_kind": "structured_git_state",
        "profile_id": "code-reviewer",
        "path_filter_used": True,
        "truncated": False,
        "reason_code": "ok",
        "duration_ms": 12.75,
    }
    assert all(
        isinstance(value, str | bool | int | float) for value in metadata.values()
    )


def test_build_execution_eval_metadata_omits_blank_optional_fields() -> None:
    metadata = build_execution_eval_metadata(
        tool_name="git.diff",
        tool_prompt_id="mcp.git.diff.v1",
        tool_prompt_version="2026.06.04",
        action_family="diff",
        result_kind="bounded_diff",
        profile_id=" ",
        path_filter_used=None,
        truncated=None,
        reason_code="",
        duration_ms=None,
    )

    assert metadata == {
        "tool_name": "git.diff",
        "tool_prompt_id": "mcp.git.diff.v1",
        "tool_prompt_version": "2026.06.04",
        "action_family": "diff",
        "result_kind": "bounded_diff",
    }


def test_build_execution_eval_metadata_does_not_accept_arbitrary_label_dicts() -> None:
    signature = inspect.signature(build_execution_eval_metadata)

    assert "labels" not in signature.parameters
    assert "metadata" not in signature.parameters
    assert "raw_payload" not in signature.parameters
    assert "absolute_path" not in signature.parameters
    assert "raw_diff" not in signature.parameters
    assert "file_contents" not in signature.parameters
    assert "author_email" not in signature.parameters

    with pytest.raises(TypeError):
        build_execution_eval_metadata(
            tool_name="git.diff",
            tool_prompt_id="mcp.git.diff.v1",
            tool_prompt_version="2026.06.04",
            action_family="diff",
            result_kind="bounded_diff",
            labels={
                "raw_payload": "diff --git a/secret.txt b/secret.txt",
                "absolute_path": "/Users/example/work/secret.txt",
                "author_email": "author@example.com",
            },
        )
