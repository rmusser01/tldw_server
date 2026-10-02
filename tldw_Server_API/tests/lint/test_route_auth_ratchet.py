"""Every route must carry an authentication dependency, or be a reviewed exception.

There is no global auth middleware, so a route that declares no auth dependency
is public.  The ratchet runs in a subprocess because it force-enables route
families that are disabled by policy by default -- including ``connectors``,
whose unauthenticated job read is what prompted this gate -- and those toggles
are read at import time.
"""

from __future__ import annotations

import configparser
import os
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
    iter_routes,
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


def _load_app_route_paths(env_overrides: dict[str, str | None] | None = None) -> set[str]:
    """Build the ratchet's app out-of-process and return every served path.

    Runs in a subprocess for the same reason ``_run_ratchet`` does: ``load_app()``
    mutates ``os.environ`` (pops the pytest/test-mode markers, sets route
    policy and config dir), which must not leak into the rest of this test run.

    ``env_overrides`` is applied to the subprocess's starting environment
    before ``load_app()`` runs, so a test can simulate a caller environment
    that already sets e.g. ``TLDW_CONFIG_FILE``.
    """
    script = (
        f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "from Helper_Scripts.ci.route_auth_ratchet import load_app, iter_routes\n"
        "app = load_app()\n"
        "for path, _methods, _dependant in iter_routes(app):\n"
        "    print(path)\n"
    )
    env = os.environ.copy()
    for key, value in (env_overrides or {}).items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _baseline_entries() -> list[str]:
    """Return the baseline's route entries, without comments or blank lines."""
    return [
        line.strip()
        for line in BASELINE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


class _FakeDependant:
    """Minimal stand-in for a FastAPI ``Dependant`` for traversal tests."""

    def __init__(self, call: object | None, dependencies: list[_FakeDependant] | None = None):
        self.call = call
        self.dependencies = dependencies or []


@pytest.mark.unit
def test_routes_behind_nested_includes_are_inspected_with_include_time_auth() -> None:
    """A route two include_router calls deep is seen at its served path, auth included.

    On FastAPI >= 0.137 a nested include is an opaque branch inside the outer one;
    reading only the outer router's candidates hid every such route from the ratchet.
    """
    from fastapi import APIRouter, Depends, FastAPI

    def get_request_user() -> None:  # named like the real authenticator
        return None

    inner = APIRouter()

    @inner.get("/leaf")
    def leaf() -> None:
        return None

    middle = APIRouter()
    middle.include_router(inner, prefix="/inner")
    app = FastAPI()
    app.include_router(middle, prefix="/api", dependencies=[Depends(get_request_user)])

    routes = {path: dependant for path, _methods, dependant in iter_routes(app)}

    assert "/api/inner/leaf" in routes
    assert is_authenticated(routes["/api/inner/leaf"])


def _named(qualname: str):
    """Return a callable whose ``__qualname__`` is *qualname*."""

    def _fn() -> None:  # pragma: no cover - never invoked
        return None

    _fn.__qualname__ = qualname
    return _fn


@pytest.mark.unit
def test_force_enabled_routers_are_mounted() -> None:
    """benchmarks/connectors/personalization must be visible to the ratchet.

    ``ROUTE_POLICY_ENV`` force-enables these route keys, but
    ``config.py::_route_toggle_policy`` only honors the ``ROUTES_ENABLE`` env
    var under explicit pytest or server test-mode runtime -- both of which
    ``load_app()`` deliberately turns off so it measures production wiring.
    If the force-enable mechanism breaks, these routers (``default_stable =
    False``) silently drop out of the app the ratchet inspects, and an
    unauthenticated route added to one of them would pass CI unnoticed.
    """
    paths = _load_app_route_paths()
    assert any(p.startswith("/api/v1/benchmarks") for p in paths), sorted(paths)
    assert any(p.startswith("/api/v1/connectors") for p in paths), sorted(paths)
    assert any(p.startswith("/api/v1/personalization") for p in paths), sorted(paths)


@pytest.mark.unit
def test_force_enabled_routers_are_mounted_with_explicit_config_file() -> None:
    """A caller-set ``TLDW_CONFIG_FILE`` must not shadow the force-enable.

    ``config_paths._resolve_env_root()`` and ``resolve_config_file()`` both
    check ``TLDW_CONFIG_FILE`` before ``TLDW_CONFIG_PATH`` before
    ``TLDW_CONFIG_DIR``. A version of ``load_app()`` that only set
    ``TLDW_CONFIG_DIR`` went blind the moment a caller environment already
    set ``TLDW_CONFIG_FILE``: the app read the *original* config.txt --
    whose ``[API-Routes] enable`` list does not include these route keys --
    and the ratchet silently stopped inspecting benchmarks/connectors/
    personalization again.
    """
    real_config = REPO_ROOT / "tldw_Server_API" / "Config_Files" / "config.txt"
    assert real_config.exists(), "fixture assumption: repo config.txt exists"

    paths = _load_app_route_paths(env_overrides={"TLDW_CONFIG_FILE": str(real_config)})
    assert any(p.startswith("/api/v1/benchmarks") for p in paths), sorted(paths)
    assert any(p.startswith("/api/v1/connectors") for p in paths), sorted(paths)
    assert any(p.startswith("/api/v1/personalization") for p in paths), sorted(paths)


@pytest.fixture
def jobs_only_config(tmp_path: Path) -> Path:
    """Use a real production config with one explicitly enabled experimental route."""
    parser = configparser.ConfigParser()
    parser.read(REPO_ROOT / "tldw_Server_API" / "Config_Files" / "config.txt")
    parser.set("API-Routes", "stable_only", "true")
    parser.set("API-Routes", "enable", '["jobs"]')
    parser.set("API-Routes", "disable", "chat")
    path = tmp_path / "config.txt"
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return path


@pytest.mark.unit
def test_json_enabled_routes_remain_in_inspection(jobs_only_config: Path) -> None:
    """Copying a JSON list must not silently discard the operator's jobs routes."""
    paths = _load_app_route_paths({"TLDW_CONFIG_FILE": str(jobs_only_config)})
    assert "/api/v1/jobs/queue/status" in paths, sorted(paths)


@pytest.mark.unit
@pytest.mark.parametrize("config_key", ["TLDW_CONFIG_FILE", "TLDW_CONFIG_PATH", "TLDW_CONFIG_DIR"])
def test_dotenv_selected_routes_remain_in_inspection(
    jobs_only_config: Path, tmp_path: Path, config_key: str
) -> None:
    """Resolve dotenv-only config selection before pinning the inspection copy."""
    parser = configparser.ConfigParser()
    parser.read(jobs_only_config)
    parser.set("API-Routes", "enable", "jobs")
    with jobs_only_config.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    dotenv = tmp_path / ".env"
    selection = jobs_only_config.parent if config_key == "TLDW_CONFIG_DIR" else jobs_only_config
    dotenv.write_text(f'{config_key}="{selection}"\n', encoding="utf-8")
    paths = _load_app_route_paths(
        {
            "TLDW_CONFIG_FILE": None,
            "TLDW_CONFIG_PATH": None,
            "TLDW_CONFIG_DIR": None,
            "TLDW_ENV_FILE": str(dotenv),
            "TLDW_ENV_FILE_EXCLUSIVE": "1",
        }
    )
    assert "/api/v1/jobs/queue/status" in paths, sorted(paths)
    assert "/api/v1/chat/completions" not in paths


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
