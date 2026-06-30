from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.VN_Scripts.playtest import build_script_playtest


def _valid() -> dict[str, Any]:
    return {"valid": True, "errors": [], "warnings": []}


def test_playtest_reports_choice_boundaries_and_endings() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {
            "start": [
                {
                    "op": "choice",
                    "id": "door",
                    "choices": [
                        {"id": "open", "text": "Open", "target": "open"},
                        {"id": "leave", "text": "Leave", "target": "leave"},
                    ],
                }
            ],
            "open": [{"op": "narrate", "text": "It opens."}, {"op": "end"}],
            "leave": [{"op": "end"}],
        },
    }

    result = build_script_playtest(program, source="stored_draft", validation_diagnostics=_valid())

    assert result["runtime_ready"] is True
    assert result["summary"]["choice_boundary_count"] == 1
    assert result["summary"]["ending_count"] == 2
    assert result["choice_boundaries"][0]["choices"] == [
        {"id": "open", "text": "Open", "target": "open"},
        {"id": "leave", "text": "Leave", "target": "leave"},
    ]
    assert {ending["label"] for ending in result["endings"]} == {"open", "leave"}


def test_playtest_reports_missing_choice_target_as_runtime_blocker() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {
            "start": [
                {
                    "op": "choice",
                    "id": "door",
                    "choices": [
                        {"id": "open", "text": "Open", "target": "open"},
                        {"id": "bad", "text": "Broken", "target": "missing"},
                    ],
                }
            ],
            "open": [{"op": "end"}],
        },
    }

    result = build_script_playtest(program, source="stored_draft", validation_diagnostics=_valid())

    error_codes = {error["code"] for error in result["diagnostics"]["errors"]}
    assert "playtest_choice_target_error" in error_codes
    assert result["runtime_ready"] is False
    assert result["summary"]["ending_count"] == 1
    assert {path["status"] for path in result["paths"]} == {"choice_target_error", "ended"}


def test_playtest_allows_cross_branch_convergence_without_loop_warning() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {
            "start": [
                {
                    "op": "choice",
                    "id": "route",
                    "choices": [
                        {"id": "left", "text": "Left", "target": "same_end"},
                        {"id": "right", "text": "Right", "target": "same_end"},
                    ],
                }
            ],
            "same_end": [{"op": "end"}],
        },
    }

    result = build_script_playtest(program, source="stored_draft", validation_diagnostics=_valid())

    warning_codes = {warning["code"] for warning in result["diagnostics"]["warnings"]}
    assert "playtest_loop_detected" not in warning_codes
    assert result["runtime_ready"] is True
    assert result["summary"]["ending_count"] == 2


def test_playtest_reports_generation_boundary_without_model_call() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {
            "start": [
                {
                    "op": "generate",
                    "id": "intro",
                    "prompt": "Write a safe intro.",
                    "profile_key": "default",
                    "output_schema": "narrative_dialogue",
                }
            ],
        },
    }

    result = build_script_playtest(program, source="stored_draft", validation_diagnostics=_valid())

    assert result["runtime_ready"] is True
    assert result["summary"]["generation_boundary_count"] == 1
    boundary = result["generation_boundaries"][0]
    assert boundary["generation_id"] == "intro"
    assert boundary["profile_key"] == "default"
    assert boundary["output_schema"] == "narrative_dialogue"
    assert "prompt_hash" in boundary


def test_playtest_reports_loop_truncation() -> None:
    program = {
        "schema_version": "vn_script_program.v1",
        "entry_label": "start",
        "labels": {"start": [{"op": "jump", "target": "start"}]},
    }

    result = build_script_playtest(program, source="stored_draft", validation_diagnostics=_valid(), max_steps=3)

    warning_codes = {warning["code"] for warning in result["diagnostics"]["warnings"]}
    assert "playtest_truncated" in warning_codes
    assert result["runtime_ready"] is False
    assert result["truncated"] is True


def test_playtest_carries_validation_blockers() -> None:
    result = build_script_playtest(
        {"schema_version": "vn_script_program.v1", "entry_label": "start", "labels": {"start": [{"op": "end"}]}},
        source="stored_draft",
        validation_diagnostics={
            "valid": False,
            "errors": [{"code": "visual_slot_key_missing", "path": "$.labels.start[0].slot_key"}],
            "warnings": [],
        },
    )

    assert result["valid"] is False
    assert result["runtime_ready"] is False
    assert result["validation_diagnostics"]["errors"][0]["code"] == "visual_slot_key_missing"
