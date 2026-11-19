"""One test per shared config key, pinning where it resolves today (TASK-13950.21, Task 7).

Row 1: ``AUTH_MODE`` (alias ``APP_MODE``). Every surface that reads the key is exercised through
its nearest callable, over the same input cases. The commit that moved the key changed only the
expected values the owner approved (D1 a-c).
"""

from __future__ import annotations

import configparser
from pathlib import Path

import pydantic
import pytest

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.AuthNZ import initialize as authnz_initialize
from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.Chat import command_router
from tldw_Server_API.app.core.config_sections.auth import (
    auth_mode_env,
    load_auth_config,
    resolve_auth_mode,
)
from tldw_Server_API.app.core.config_sections.precedence import Resolved
from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.core.Evaluations.config_validator import (
    EvaluationConfigValidator,
)
from tldw_Server_API.app.core.External_Sources import connectors_service
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Sync.v2 import security as sync_security
from tldw_Server_API.app.core.Web_Scraping import browser_transport
from tldw_Server_API.app.services import (
    claims_review_metrics_scheduler as claims_scheduler,
)

pytestmark = pytest.mark.unit

UNSET = None
REJECTED = "ValidationError"

# id -> (env AUTH_MODE, env APP_MODE, [AuthNZ] auth_mode)
CASES = {
    "env_set": ("multi_user", UNSET, UNSET),
    "env_blank": ("", UNSET, "multi_user"),
    "env_invalid": ("multi-user", UNSET, "multi_user"),
    "config_only": (UNSET, UNSET, "multi_user"),
    "neither": (UNSET, UNSET, UNSET),
    "app_mode_only": (UNSET, "multi", UNSET),
    "config_invalid_app_mode": (UNSET, "multi", "bogus"),
}


def _parser(auth_mode: str | None, *, full: bool = False) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    if full:
        real = _REAL_CONFIG()
        parser.read_dict({s: dict(real.items(s, raw=True)) for s in real.sections()})
    if not parser.has_section("AuthNZ"):
        parser.add_section("AuthNZ")
    parser.remove_option("AuthNZ", "auth_mode")
    if auth_mode is not None:
        parser.set("AuthNZ", "auth_mode", auth_mode)
    return parser


_REAL_CONFIG = config.load_comprehensive_config


@pytest.fixture
def apply_case(monkeypatch, tmp_path):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_mode, env_app_mode, cfg_mode = CASES[case]
        for key, value in (("AUTH_MODE", env_mode), ("APP_MODE", env_app_mode)):
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        for key in ("CONNECTORS_REQUIRE_TOKEN_ENCRYPTION", "API_BEARER", "SINGLE_USER_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        full = _parser(cfg_mode, full=True)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: full)
        monkeypatch.setattr(browser_transport, "load_comprehensive_config", lambda: full)
        minimal = _parser(cfg_mode)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: minimal)
        # the setup wizard reads its own copy of config.txt
        setup_file = Path(tmp_path) / "config.txt"
        with setup_file.open("w", encoding="utf-8") as handle:
            minimal.write(handle)
        monkeypatch.setattr(setup_manager, "get_config_file_path", lambda: setup_file)
        authnz_settings.reset_settings()
        return minimal

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


def _authnz_mode() -> str:
    authnz_settings.reset_settings()
    try:
        return authnz_settings.get_settings().AUTH_MODE
    except pydantic.ValidationError:
        return REJECTED


# config_sections and AuthNZ are unchanged by the move. Every other reader follows the effective mode
# (env, then [AuthNZ] auth_mode, then single_user; APP_MODE counts only when both are invalid) since D1(a)-(c): before, LazySettings
# read APP_MODE and the rest read the environment only.
EXPECTED = {
    #               config_sections  AuthNZ
    "env_set": ("multi_user", "multi_user"),
    "env_blank": ("multi_user", REJECTED),
    "env_invalid": ("multi_user", REJECTED),
    "config_only": ("multi_user", "multi_user"),
    "neither": ("single_user", "single_user"),
    "app_mode_only": ("single_user", "single_user"),
    "config_invalid_app_mode": ("multi_user", REJECTED),
}


@pytest.mark.parametrize("case", list(CASES))
def test_auth_mode_resolution_today(apply_case, case):
    minimal = apply_case(case)
    effective, expected_authnz = EXPECTED[case]
    single_user = effective == "single_user"

    assert load_auth_config(minimal).mode == effective
    assert _authnz_mode() == expected_authnz
    assert config.load_settings()["SINGLE_USER_MODE"] is single_user  # D1(a): follows the effective mode

    # D1(b)/(c): the setting readers follow the effective mode too
    assert command_router.is_single_user_mode() is single_user
    assert db_path_utils._get_auth_mode() == effective
    assert setup_endpoint._resolve_auth_mode({}) == effective
    assert connectors_service._connector_secret_encryption_required() is (not single_user)
    assert claims_scheduler.resolve_scheduler_config({}).single_user_mode is single_user
    assert sync_security.server_trusted_encryption_status_from_env().auth_mode == effective
    validator = EvaluationConfigValidator()
    validator._validate_authentication()
    assert any("single-user mode" in issue.message for issue in validator.issues) is single_user


@pytest.mark.parametrize(
    "environment,expected",
    [
        ({"AUTH_MODE": "multi_user"}, "multi_user"),
        ({"AUTH_MODE": " Single_User "}, "single_user"),
        ({"AUTH_MODE": "multi-user"}, ""),
        ({"AUTH_MODE": ""}, ""),
        ({}, "single_user"),
    ],
)
def test_browser_transport_effective_auth_mode_today(apply_case, environment, expected):
    apply_case("neither")
    assert browser_transport._effective_auth_mode(environment) == expected


@pytest.mark.parametrize(
    "case,expected",
    [("env_set", "multi_user"), ("env_blank", "from_dotenv"), ("config_only", "from_dotenv"), ("neither", "from_dotenv")],
)
def test_setup_wizard_env_check_today(apply_case, case, expected):
    apply_case(case)
    assert authnz_initialize._effective_env_value("AUTH_MODE", {"AUTH_MODE": "from_dotenv"}) == expected


# resolve_auth_mode is the AuthNZ overlay's rule: exact-case env, then [AuthNZ] auth_mode, then single_user.
RESOLVED = {
    "env_set": Resolved("multi_user", "env", "AUTH_MODE"),
    "env_blank": Resolved("", "env", "AUTH_MODE"),
    "env_invalid": Resolved("multi-user", "env", "AUTH_MODE"),
    "config_only": Resolved("multi_user", "config", "AuthNZ.auth_mode"),
    "neither": Resolved("single_user", "default", None),
    "app_mode_only": Resolved("single_user", "default", None),
    "config_invalid_app_mode": Resolved("bogus", "config", "AuthNZ.auth_mode"),
}


@pytest.mark.parametrize("case", list(CASES))
def test_auth_mode_resolves_through_config_sections(apply_case, case):
    minimal = apply_case(case)
    assert resolve_auth_mode(minimal) == RESOLVED[case]
    assert auth_mode_env() == CASES[case][0]


def test_resolve_auth_mode_passes_a_blank_through():
    blank_config = _parser("")
    assert resolve_auth_mode(_parser("multi_user"), {"AUTH_MODE": ""}) == Resolved("", "env", "AUTH_MODE")
    assert resolve_auth_mode(blank_config, {}) == Resolved("", "config", "AuthNZ.auth_mode")
    assert load_auth_config(_parser("multi_user"), env={"AUTH_MODE": ""}).mode == "multi_user"
    assert resolve_auth_mode(None, {}) == Resolved("single_user", "default", None)


def test_auth_mode_env_is_exact_case_and_live():
    assert auth_mode_env({"AUTH_MODE": " Multi "}) == " Multi "
    assert auth_mode_env({"auth_mode": "multi_user"}) is None
    assert auth_mode_env({}) is None


def test_setup_wizard_auth_mode_reads_the_config_file_fresh(monkeypatch, tmp_path):
    """The wizard rewrites config.txt; first-run metadata must see the new mode without a restart."""
    monkeypatch.delenv("AUTH_MODE", raising=False)
    monkeypatch.delenv("APP_MODE", raising=False)
    setup_file = tmp_path / "config.txt"
    monkeypatch.setattr(setup_manager, "get_config_file_path", lambda: setup_file)
    for mode in ("multi_user", "single_user"):
        setup_file.write_text(f"[AuthNZ]\nauth_mode = {mode}\n", encoding="utf-8")
        assert setup_endpoint._resolve_auth_mode({}) == mode
