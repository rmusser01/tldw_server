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

# Whatever else the expression falls back to, the PR number is what makes runs
# for the same PR collapse. Without it they never group and nothing cancels.
REQUIRED_KEY = "github.event.pull_request.number"


def _concurrency(name: str) -> dict:
    with (WORKFLOWS_DIR / name).open("r", encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    concurrency = workflow.get("concurrency")
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
    """A group that does not key on the PR never collapses anything."""
    group = _concurrency(name).get("group", "")

    assert REQUIRED_KEY in group, (
        f"{name} has concurrency group {group!r}, which does not reference "
        f"{REQUIRED_KEY}. Runs for the same pull request will not group, so "
        f"none of them cancel."
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
