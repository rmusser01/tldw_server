"""Merge-queue contract of frontend-required.yml (spec section 4.8).

Spec: Docs/superpowers/specs/2026-10-04-merge-queue-design.md
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from Helper_Scripts.ci import merge_queue as mq

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "frontend-required.yml"
GATE = "frontend-required"


def _wf() -> dict:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _jobs() -> dict:
    return _wf()["jobs"]


# --- 4.8: the manual-dispatch name guard ----------------------------------------------------


def test_name_guard_exempts_only_the_queue_actor():
    """A dispatch publishes the protected name only when the queue's own token made it.

    Three rows: pull_request and workflow_run publish `frontend-required`; a dispatch by
    `github-actions[bot]` (GITHUB_TOKEN, how the queue starts the gate) publishes it too; every
    other dispatch publishes `frontend-required-diagnostic`. The rows are evaluated in
    test_admin_ui_vitest_ratchet_workflow.py; this pins the expression to the queue's actor.
    """
    assert mq.QUEUE_ACTOR == "github-actions[bot]"
    assert _jobs()[GATE]["name"] == (
        "${{ github.event_name == 'workflow_dispatch' && "
        f"github.actor != '{mq.QUEUE_ACTOR}' && "
        "'frontend-required-diagnostic' || 'frontend-required' }}"
    )


def test_nothing_else_in_the_workflow_depends_on_who_dispatched_it():
    """Apart from the job name, a queue dispatch takes the same path as any other dispatch,
    which for the same head and `base_sha` tests what the pull_request run tests."""
    rendered = json.dumps(_wf())  # parsed values only: comments are not behaviour
    assert rendered.count("github.actor") == 1
    assert rendered.count("frontend-required-diagnostic") == 1
    assert "triggering_actor" not in rendered
