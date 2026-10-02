"""route_map entries must be reachable and used; the real app is checked in a subprocess."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi import APIRouter, FastAPI

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Helper_Scripts.ci.rg_route_map_lint import lint  # noqa: E402

from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes  # noqa: E402

pytestmark = pytest.mark.unit


def _served():
    router = APIRouter(tags=["notes"])

    @router.get("/notes/{nid}")
    def note(nid: str) -> None:
        return None

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return list(iter_served_routes(app.routes))


def test_dead_pattern_is_reported():
    assert lint({"by_path": {"/api/v1/nowhere*": "p"}, "by_tag": {}}, _served(), set()) == ["by_path /api/v1/nowhere* matches no served route"]


def test_shadowed_pattern_is_reported():
    rm = {"by_path": {"/api/v1/*": "a", "/api/v1/notes*": "b"}, "by_tag": {}}
    assert lint(rm, _served(), set()) == ["by_path /api/v1/notes* is shadowed by earlier patterns"]


def test_unused_tag_is_reported():
    assert lint({"by_path": {}, "by_tag": {"ghost": "p", "notes": "q"}}, _served(), set()) == ["by_tag ghost is used by no served route"]


def test_tag_whose_routes_all_match_a_path_is_reported():
    # by_path wins over by_tag, so this tag mapping can never take effect.
    rm = {"by_path": {"/api/v1/notes*": "p"}, "by_tag": {"notes": "q"}}
    assert lint(rm, _served(), set()) == ["by_tag notes is shadowed by by_path entries"]


def test_allowlisted_problem_is_suppressed():
    problem = "by_tag ghost is used by no served route"
    assert lint({"by_path": {}, "by_tag": {"ghost": "p"}}, _served(), {problem}) == []


# Run subprocesses the way a shell does. PYTEST_CURRENT_TEST makes config.py defer
# reading config.txt, which hid TASK-13417's import-order bug from this suite.
_SHELL_ENV = {
    k: v for k, v in os.environ.items()
    if k not in {"MINIMAL_TEST_APP", "PYTEST_CURRENT_TEST", "TEST_MODE", "TLDW_TEST_MODE"}
}


def _printed_routes(script: str) -> set[str]:
    """Run ``script`` in a clean interpreter and return its ``ROUTE ...`` lines."""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n{script}"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900, env=_SHELL_ENV,
    )
    assert result.returncode == 0, result.stderr
    return {line for line in result.stdout.splitlines() if line.startswith("ROUTE ")}


def test_lint_sees_the_routes_a_fresh_app_serves():
    # TASK-13417: the lint imported the policy loader before load_app(), which read
    # config.txt early and hid every force-enabled router (benchmarks, connectors, ...).
    dump = "for r in served: print('ROUTE', r.path, sorted(r.methods))\n"
    fresh = _printed_routes(
        "from Helper_Scripts.ci.route_auth_ratchet import load_app\n"
        "app = load_app()\n"
        "from tldw_Server_API.app.core.Utils.fastapi_routes import iter_served_routes\n"
        "served = list(iter_served_routes(app.routes))\n" + dump
    )
    linted = _printed_routes(
        "from Helper_Scripts.ci.rg_route_map_lint import load_inputs\n"
        "_route_map, served = load_inputs()\n" + dump
    )
    assert any(" /api/v1/connectors" in line for line in fresh)
    assert linted == fresh, sorted(fresh ^ linted)[:40]


def test_shipped_route_map_is_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Helper_Scripts" / "ci" / "rg_route_map_lint.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900, env=_SHELL_ENV,
    )
    assert result.returncode == 0, result.stdout + result.stderr
