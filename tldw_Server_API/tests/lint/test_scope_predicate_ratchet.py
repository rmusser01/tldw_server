"""A tenant predicate must not be applied only when the scope is truthy.

    if user_id:
        conditions.append("user_id = ?")

A falsy scope drops the predicate and returns the whole table. Harmless on
SQLite, where each user has their own file; a full cross-tenant read on shared
PostgreSQL tables.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "scope_predicate_baseline.txt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.ci.scope_predicate_ratchet import (  # noqa: E402
    SCOPE_NAMES,
    _adds_predicate_on,
    _guarded_scope,
    diff_against_baseline,
    iter_findings,
    read_baseline,
)


def _parse_if(source: str) -> ast.If:
    """Return the single `if` statement in *source*."""
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.If)
    return node


@pytest.mark.unit
def test_no_new_conditional_tenant_predicates() -> None:
    """A newly added conditional scope predicate must fail CI by name."""
    added, stale = diff_against_baseline(sorted(iter_findings()), read_baseline())
    assert not added, "new conditional tenant predicates:\n  " + "\n  ".join(added)
    assert not stale, (
        "baseline entries no longer match the tree (line numbers move; "
        "regenerate with --write-baseline):\n  " + "\n  ".join(stale)
    )


@pytest.mark.unit
def test_baseline_is_sorted_and_unique() -> None:
    entries = [
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert entries == sorted(entries)
    assert len(entries) == len(set(entries))


@pytest.mark.unit
def test_bare_truthiness_on_a_scope_is_detected() -> None:
    node = _parse_if('if user_id:\n    conditions.append("user_id = ?")\n')
    assert _guarded_scope(node.test) == "user_id"
    assert _adds_predicate_on(node.body, "user_id")


@pytest.mark.unit
def test_explicit_none_check_is_not_the_footgun() -> None:
    """`is not None` is a deliberate check; 0 and "" still reach the predicate."""
    node = _parse_if('if user_id is not None:\n    conditions.append("user_id = ?")\n')
    assert _guarded_scope(node.test) is None


@pytest.mark.unit
def test_non_scope_filters_are_ignored() -> None:
    """`if status:` is ordinary optional filtering, not a tenant boundary."""
    node = _parse_if('if status:\n    conditions.append("status = ?")\n')
    assert _guarded_scope(node.test) is None


@pytest.mark.unit
def test_guard_that_builds_no_predicate_is_ignored() -> None:
    node = _parse_if('if user_id:\n    logger.info("saw a user")\n')
    assert _guarded_scope(node.test) == "user_id"
    assert not _adds_predicate_on(node.body, "user_id")


@pytest.mark.unit
def test_predicate_on_a_different_column_does_not_count() -> None:
    """Only a comparison on the guarded scope itself is the footgun."""
    node = _parse_if('if user_id:\n    conditions.append("status = ?")\n')
    assert not _adds_predicate_on(node.body, "user_id")


@pytest.mark.unit
def test_fstring_predicates_are_detected() -> None:
    """Postgres paths build placeholders with f-strings."""
    node = _parse_if(
        'if owner_user_id:\n    conditions.append(f"owner_user_id = {placeholder}")\n'
    )
    assert _adds_predicate_on(node.body, "owner_user_id")


@pytest.mark.unit
def test_scope_names_cover_the_tenant_columns_used_in_this_repo() -> None:
    for name in ("user_id", "owner_user_id", "client_id", "tenant_id"):
        assert name in SCOPE_NAMES
