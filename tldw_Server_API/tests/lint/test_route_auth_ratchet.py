"""Every route must carry an authentication dependency, or be a reviewed exception.

There is no global auth middleware, so a route that declares no auth dependency
is public.  The ratchet runs in a subprocess because it force-enables route
families that are disabled by policy by default -- including ``connectors``,
whose unauthenticated job read is what prompted this gate -- and those toggles
are read at import time.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RATCHET = REPO_ROOT / "Helper_Scripts" / "ci" / "route_auth_ratchet.py"
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "route_auth_baseline.txt"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.ci.route_auth_ratchet import (  # noqa: E402
    AUTHENTICATORS,
    AUTHORIZER_FACTORIES,
    NOT_AUTHENTICATION,
    diff_against_baseline,
    is_authenticated,
)


def _run_ratchet() -> subprocess.CompletedProcess[str]:
    """Run the ratchet in a clean interpreter and return the completed process."""
    return subprocess.run(
        [sys.executable, str(RATCHET)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _baseline_entries() -> list[str]:
    """Return the baseline's route entries, without comments or blank lines."""
    return [
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


class _FakeDependant:
    """Minimal stand-in for a FastAPI ``Dependant`` for traversal tests."""

    def __init__(self, call: object | None, dependencies: list["_FakeDependant"] | None = None):
        self.call = call
        self.dependencies = dependencies or []


def _named(qualname: str):
    """Return a callable whose ``__qualname__`` is *qualname*."""

    def _fn() -> None:  # pragma: no cover - never invoked
        return None

    _fn.__qualname__ = qualname
    return _fn


@pytest.mark.unit
def test_no_new_unauthenticated_routes() -> None:
    """A route added without an auth dependency must fail CI by name."""
    result = _run_ratchet()
    assert result.returncode == 0, (
        "Route authentication ratchet failed.\n" f"{result.stderr}"
    )


@pytest.mark.unit
def test_baseline_is_sorted() -> None:
    """Sorted order keeps the reviewable diff small when an entry changes."""
    entries = _baseline_entries()
    assert entries == sorted(entries)


@pytest.mark.unit
def test_baseline_has_no_duplicates() -> None:
    """A duplicated exception would survive one removal and stay in force."""
    entries = _baseline_entries()
    assert len(entries) == len(set(entries))


@pytest.mark.unit
def test_rate_limiters_are_not_authentication() -> None:
    """`rbac_rate_limit` reads like an RBAC gate and is only a rate limiter.

    Three `/sharing/admin/*` routes shipped guarded by nothing else.  Asserted
    through ``is_authenticated`` rather than set membership, so the guarantee
    holds however the classification is implemented.
    """
    for name in sorted(NOT_AUTHENTICATION):
        dependant = _FakeDependant(_named(f"{name}.<locals>._dep"))
        assert not is_authenticated(dependant), name


@pytest.mark.unit
def test_locality_guards_are_not_authentication() -> None:
    """`require_local_setup_access` authorizes by loopback, not by identity.

    An anonymous caller reaching loopback passes it, so routes relying on it are
    anonymous-capable and must stay visible in the baseline.
    """
    assert "require_local_setup_access" not in AUTHENTICATORS
    assert not is_authenticated(_FakeDependant(_named("require_local_setup_access")))


@pytest.mark.unit
def test_authenticator_is_found_through_nesting() -> None:
    """Auth applied on a parent router must count for the routes beneath it."""
    leaf = _FakeDependant(_named("get_request_user"))
    middle = _FakeDependant(_named("get_media_db_for_user"), [leaf])
    root = _FakeDependant(None, [middle])
    assert is_authenticated(root)


@pytest.mark.unit
def test_authorizer_factory_closure_counts() -> None:
    """`RequireRole("admin")` yields a closure named `_checker`; match the factory."""
    for factory in sorted(AUTHORIZER_FACTORIES):
        dependant = _FakeDependant(_named(f"{factory}.<locals>._checker"))
        assert is_authenticated(dependant), factory


@pytest.mark.unit
def test_cycles_do_not_hang_the_walk() -> None:
    """A self-referential dependency must terminate rather than recurse forever."""
    node = _FakeDependant(_named("something_unrelated"))
    node.dependencies = [node]
    assert not is_authenticated(node)


@pytest.mark.unit
def test_new_unauthenticated_route_is_reported() -> None:
    """A route missing from the baseline is reported as added."""
    added, stale = diff_against_baseline({"GET /leak"}, {"POST /login"})
    assert added == ["GET /leak"]
    assert stale == ["POST /login"]


@pytest.mark.unit
def test_stale_exception_cannot_be_reused() -> None:
    """A baseline entry whose route gained auth must fail, not pass as 'fewer'.

    Otherwise the exception stays listed and a later regression on the same
    method and path produces no added entry, so CI stays green while the route
    goes public.
    """
    baseline = {"PUT /api/v1/config/tokenizer", "POST /api/v1/auth/login"}
    current = {"POST /api/v1/auth/login"}  # tokenizer route gained auth

    added, stale = diff_against_baseline(current, baseline)
    assert added == []
    assert stale == ["PUT /api/v1/config/tokenizer"]

    # Regenerated baseline: clean. Regression re-opens it: caught as added.
    assert diff_against_baseline(current, current) == ([], [])
    reopened, _ = diff_against_baseline(baseline, current)
    assert reopened == ["PUT /api/v1/config/tokenizer"]
