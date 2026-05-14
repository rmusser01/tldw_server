"""Regression tests for persona visual import commit eligibility guards."""

from __future__ import annotations

from tldw_Server_API.app.core.Persona.visual_portability.commit_eligibility import (
    import_preview_commit_blockers,
    import_preview_plan_from_stored_json,
    is_import_preview_plan_committable,
    is_import_preview_result_committable,
)


def test_import_preview_plan_rejects_non_mapping_plan() -> None:
    assert not is_import_preview_plan_committable([])
    assert import_preview_commit_blockers([]) == ["missing_or_invalid_plan"]


def test_import_preview_result_rejects_missing_plan() -> None:
    assert not is_import_preview_result_committable({"status": "completed"})
    assert not is_import_preview_result_committable({"status": "completed", "proposed_plan": []})


def test_import_preview_plan_from_stored_json_allows_empty_legacy_plan() -> None:
    """Treat missing legacy proposed-plan JSON as valid empty metadata."""
    plan, valid = import_preview_plan_from_stored_json("")

    assert valid is True
    assert plan == {}


def test_import_preview_plan_from_stored_json_accepts_bytes_json() -> None:
    """Parse byte-encoded proposed-plan JSON without coercing through repr text."""
    plan, valid = import_preview_plan_from_stored_json(b'{"target_mode": "create_new"}')

    assert valid is True
    assert plan == {"target_mode": "create_new"}


def test_import_preview_plan_from_stored_json_rejects_corrupted_or_non_object_plan() -> None:
    """Reject malformed or non-object proposed-plan JSON from stored previews."""
    for raw_value in ("{not-json", "[]"):
        plan, valid = import_preview_plan_from_stored_json(raw_value)

        assert valid is False
        assert plan == {}


def test_renderer_import_preview_without_can_commit_does_not_block_plan() -> None:
    plan = {
        "renderer_import_preview": {
            "status": "legacy_preview",
        },
    }

    assert is_import_preview_plan_committable(plan)


def test_import_preview_commit_blockers_ignore_null_entries() -> None:
    plan = {
        "commit_blockers": [
            None,
            "runtime_adapter_not_implemented",
            " ",
        ],
        "commit_eligible": False,
    }

    assert import_preview_commit_blockers(plan) == [
        "runtime_adapter_not_implemented",
        "commit_eligible_not_true",
    ]
