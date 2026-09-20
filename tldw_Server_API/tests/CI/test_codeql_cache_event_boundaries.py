"""Guard the event-dependent checkout boundary behind reviewed CodeQL alerts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = json.loads((ROOT / "Docs/Evidence/PR2761-codeql-actions.json").read_text())
CASES = sorted({(p["workflow"], p["job"], p["event"]) for p in EVIDENCE["checkout_event_proofs"].values()})
ADMISSION_GUARD = (
    "vars.LICENSE_FIRST_CI_ENABLED == 'true' && "
    "github.event_name == 'workflow_run' && "
    "github.event.workflow_run.conclusion == 'success'"
)
PR_FIELDS = {
    "github.event.workflow_run.pull_requests[0].head.sha",
    "github.event.pull_request.head.sha",
}


def non_pr_checkout_ref(expression: str, jobs: dict[str, Any], job: dict[str, Any]) -> str:
    """Evaluate only the audited OR-chain form; reject unknown sources/operators.

    On schedule/workflow_dispatch the PR-specific event fields are absent. The
    reusable admission job can supply a SHA only on its guarded workflow_run.
    This deliberately does not interpret arbitrary GitHub expression syntax.
    """
    fields = expression.strip().removeprefix("${{").removesuffix("}}").strip().split("||")
    fields = [field.strip() for field in fields]
    assert fields[-1] == "github.sha"
    for field in fields:
        if field == "github.sha":
            return field
        if field == "needs.admission.outputs.head_sha":
            assert " ".join(jobs["admission"]["if"].split()) == ADMISSION_GUARD
            needs = job.get("needs", [])
            assert "admission" in ([needs] if isinstance(needs, str) else needs)
        else:
            assert field in PR_FIELDS, f"Unreviewed checkout source: {field}"
    raise AssertionError("No event-commit fallback")


@pytest.mark.parametrize(("workflow_path", "job_id", "event"), CASES)
def test_cache_writing_events_checkout_the_event_commit(workflow_path: str, job_id: str, event: str) -> None:
    workflow = yaml.safe_load((ROOT / workflow_path).read_text())
    assert event in {"schedule", "workflow_dispatch"}
    assert event in workflow.get("on", workflow.get(True))
    job = workflow["jobs"][job_id]
    checkouts = [step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")]
    assert checkouts
    for step in checkouts:
        assert step["with"]["persist-credentials"] is False
        assert non_pr_checkout_ref(step["with"]["ref"], workflow["jobs"], job) == "github.sha"


@pytest.mark.parametrize(
    "expression",
    [
        "${{ inputs.head_sha || github.sha }}",
        "${{ github.event.inputs.head_sha || github.sha }}",
        "${{ needs.unchecked.outputs.head_sha || github.sha }}",
        "${{ github.event.pull_request.head.sha }}",
    ],
)
def test_dispatch_input_or_missing_fallback_is_rejected(expression: str) -> None:
    with pytest.raises(AssertionError):
        non_pr_checkout_ref(expression, {}, {})


@pytest.mark.parametrize("guard", ["always()", "github.event_name != 'workflow_run'", "true"])
def test_admission_output_without_workflow_run_guard_is_rejected(guard: str) -> None:
    with pytest.raises(AssertionError):
        non_pr_checkout_ref(
            "${{ needs.admission.outputs.head_sha || github.sha }}",
            {"admission": {"if": guard}},
            {"needs": "admission"},
        )
