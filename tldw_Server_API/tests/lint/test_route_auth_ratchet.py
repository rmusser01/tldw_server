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


def _run_ratchet() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RATCHET)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.unit
def test_no_new_unauthenticated_routes() -> None:
    result = _run_ratchet()
    assert result.returncode == 0, (
        "A route was added without an authentication dependency.\n"
        f"{result.stderr}"
    )


@pytest.mark.unit
def test_baseline_is_sorted_and_unique() -> None:
    entries = [
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert entries == sorted(entries), "baseline must stay sorted so diffs are readable"
    assert len(entries) == len(set(entries)), "baseline contains duplicate routes"


@pytest.mark.unit
def test_rate_limiters_do_not_count_as_authentication() -> None:
    """`rbac_rate_limit` reads like an RBAC gate and is only a rate limiter.

    Three `/sharing/admin/*` routes shipped guarded by nothing else.  If this
    ever starts counting as auth, the gate goes quietly blind.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from Helper_Scripts.ci.route_auth_ratchet import (
        AUTHENTICATORS,
        AUTHORIZER_FACTORIES,
    )

    for name in ("rbac_rate_limit", "check_rate_limit", "check_auth_rate_limit",
                 "get_rate_limiter_dep", "kanban_rate_limit",
                 "check_evaluation_rate_limit"):
        assert name not in AUTHENTICATORS
        assert name not in AUTHORIZER_FACTORIES
