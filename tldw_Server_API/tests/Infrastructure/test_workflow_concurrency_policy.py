"""Superseded runs of a PR's required checks must cancel each other.

Every required workflow puts the PR identity in its concurrency group so a new
push cancels the runs still going for the previous one. Get this wrong in a
single workflow and that workflow keeps burning runners on commits nobody is
waiting for, while the PR's other checks move on.

This pins the shape of that contract rather than one exact expression. The
expression has grown a fallback twice -- most recently ``workflow_run`` support,
in a0cce60c7b, so runs triggered by another workflow group with the PR that
caused them -- and both times the previous version of this test asserted a
string literal and simply went red, unnoticed, for weeks. What actually matters
is that the workflows agree with each other and key on the PR.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# The required checks: the set whose runs must collapse together.
TARGET_WORKFLOWS = (
    "ci.yml",
    "backend-required.yml",
    "coverage-required.yml",
    "frontend-required.yml",
    "security-required.yml",
    "e2e-required.yml",
    "frontend-ux-gates.yml",
    "pre-commit.yml",
    "pypi-package.yml",
    "sbom.yml",
)

# How each trigger names the pull request a run belongs to. These are GitHub's
# own event paths, not a style choice -- there is one way to reach the PR number
# from each event -- but which ones a workflow *needs* is read from its triggers
# rather than assumed, so dropping a trigger drops its requirement too.
#
# The distinction matters: on a workflow_run trigger there is no
# github.event.pull_request, so a group keyed only on that form degrades to the
# ref and stops collapsing anything.
#
# Every trigger whose payload carries a pull request is listed, not just the two
# in use today. A trigger that is absent from this mapping has no requirement at
# all, so an unlisted PR trigger would let a group with no PR identity through
# silently -- the guard failing open is worse than it failing loudly.
PR_IDENTITY_BY_TRIGGER = {
    "pull_request": "github.event.pull_request.number",
    "pull_request_target": "github.event.pull_request.number",
    "pull_request_review": "github.event.pull_request.number",
    "pull_request_review_comment": "github.event.pull_request.number",
    "workflow_run": "github.event.workflow_run.pull_requests[0].number",
}


def _workflow(name: str) -> dict:
    with (WORKFLOWS_DIR / name).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _triggers(name: str) -> set[str]:
    """Return the workflow's trigger names.

    ``on`` is a YAML 1.1 boolean, so PyYAML parses the key as ``True``.
    """
    workflow = _workflow(name)
    on = workflow.get("on", workflow.get(True))
    if isinstance(on, dict):
        return set(on)
    if isinstance(on, list):
        return set(on)
    return {on} if on else set()


def _concurrency(name: str) -> dict:
    concurrency = _workflow(name).get("concurrency")
    assert isinstance(concurrency, dict), f"{name} is missing a concurrency block"
    return concurrency


def _suffix(name: str, group: str) -> str:
    """Strip the per-workflow prefix, leaving the shared identity expression."""
    prefix = f"{name.removesuffix('.yml')}-"
    assert group.startswith(prefix), (
        f"{name} has concurrency group {group!r}, which does not start with "
        f"{prefix!r}. The prefix is what keeps one workflow's runs from "
        f"cancelling another's."
    )
    return group.removeprefix(prefix)


@pytest.mark.unit
@pytest.mark.parametrize("name", TARGET_WORKFLOWS)
def test_required_workflow_cancels_superseded_runs(name: str) -> None:
    """Without this, a superseded run keeps holding runners to completion."""
    assert _concurrency(name).get("cancel-in-progress") is True, (
        f"{name} does not cancel in-progress runs, so pushing to a PR leaves "
        f"the previous run going."
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", TARGET_WORKFLOWS)
def test_required_workflow_groups_on_the_pull_request(name: str) -> None:
    """A group that does not key on the PR never collapses anything.

    Checked per trigger the workflow actually declares, so a workflow that only
    ever runs on ``pull_request`` is not asked for the ``workflow_run`` form.
    """
    triggers = _triggers(name)
    group = _concurrency(name).get("group", "")

    missing = {
        trigger: token
        for trigger, token in PR_IDENTITY_BY_TRIGGER.items()
        if trigger in triggers and token not in group
    }
    assert not missing, (
        f"{name} triggers on {sorted(missing)} but its concurrency group "
        f"{group!r} has no way to identify the pull request on "
        f"{'that trigger' if len(missing) == 1 else 'those triggers'}:\n"
        + "\n".join(f"  {trigger}: expected {token}" for trigger, token in sorted(missing.items()))
        + "\nRuns arriving that way fall back to the ref and never collapse."
    )


@pytest.mark.unit
def test_required_workflows_agree_on_one_identity_expression() -> None:
    """One workflow drifting is the failure this guard exists to catch.

    They are prefixed per workflow and otherwise identical, so comparing the
    suffixes catches a single file being edited in isolation -- which is how
    this last broke.
    """
    suffixes = {name: _suffix(name, _concurrency(name).get("group", "")) for name in TARGET_WORKFLOWS}

    distinct = set(suffixes.values())
    assert len(distinct) == 1, (
        "required workflows disagree on how they identify a run, so their runs "
        "will not collapse together:\n"
        + "\n".join(f"  {name}: {suffix}" for name, suffix in sorted(suffixes.items()))
    )
