"""Shape of the merge-queue entry point, merge-queue.yml (spec sections 3 and 5)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "merge-queue.yml"
CHECKOUT_ACTION = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers() -> dict:
    wf = _wf()
    return wf.get("on", wf.get(True))  # `on` is a YAML 1.1 boolean


def test_triggers_are_dev_events_only():
    on = _triggers()
    assert set(on) == {"pull_request", "push"}
    assert on["pull_request"] == {"types": ["auto_merge_enabled", "auto_merge_disabled", "closed"], "branches": ["dev"]}
    assert on["push"] == {"branches": ["dev"]}


@pytest.mark.parametrize("trigger", ["pull_request_target", "schedule", "workflow_run", "check_suite", "check_run"])
def test_never_uses_a_trigger_that_reads_the_default_branch(trigger):
    """These run main's copy of the workflow, and nothing in the queue may depend on main."""
    assert trigger not in _triggers()
    assert f"{trigger}:" not in WORKFLOW.read_text(encoding="utf-8")


def test_write_permissions_are_job_level_only():
    wf = _wf()
    assert wf["permissions"] == {"contents": "read"}
    assert set(wf["jobs"]) == {"queue"}
    assert wf["jobs"]["queue"]["permissions"] == {
        "contents": "write", "pull-requests": "write", "actions": "write",
        "checks": "read", "statuses": "read",
    }


def test_runs_only_when_enabled_and_never_for_forks():
    job = _wf()["jobs"]["queue"]
    assert "vars.MERGE_QUEUE == 'dry' || vars.MERGE_QUEUE == 'on'" in job["if"]
    assert "github.event_name == 'push'" in job["if"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]


def test_checks_out_dev_and_runs_the_queue_module_with_the_builtin_token():
    """The code that runs with write permissions is dev's, never the PR's."""
    checkout, advance = _wf()["jobs"]["queue"]["steps"]
    assert checkout["uses"] == CHECKOUT_ACTION
    assert checkout["with"] == {"ref": "dev", "persist-credentials": False}
    assert advance["run"] == "python3 -m Helper_Scripts.ci.merge_queue"
    assert advance["env"] == {"GH_TOKEN": "${{ github.token }}", "MERGE_QUEUE": "${{ vars.MERGE_QUEUE }}"}
    assert (REPO_ROOT / "Helper_Scripts" / "ci" / "merge_queue.py").is_file()
    assert (REPO_ROOT / "Helper_Scripts" / "ci" / "__init__.py").is_file()
    assert "secrets." not in WORKFLOW.read_text(encoding="utf-8"), "no PAT, App key or deploy key (spec 3)"


def test_has_no_concurrency_group():
    """A job cancelled while pending shows as a red check on the PR; races are made safe in the
    script instead (spec section 7)."""
    wf = _wf()
    assert "concurrency" not in wf and "concurrency" not in wf["jobs"]["queue"]
