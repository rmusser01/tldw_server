"""route_map entries must be reachable and used; the real app is checked in a subprocess."""

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


def test_shipped_route_map_is_clean():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "Helper_Scripts" / "ci" / "rg_route_map_lint.py")],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=900,
    )
    assert result.returncode == 0, result.stdout + result.stderr
