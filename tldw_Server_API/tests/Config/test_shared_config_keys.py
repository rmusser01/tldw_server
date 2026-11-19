"""One test per shared config key, pinning where it resolves today (TASK-13950.21, Task 7).

Row 1: ``AUTH_MODE`` (alias ``APP_MODE``). Every surface that reads the key is exercised through
its nearest callable, over the same five input cases, so a move to ``config_sections`` shows up
as a reviewed change in an expected value and nowhere else.
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
from tldw_Server_API.app.core.config_sections.auth import load_auth_config
from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.core.Evaluations.config_validator import EvaluationConfigValidator
from tldw_Server_API.app.core.External_Sources import connectors_service
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Sync.v2 import security as sync_security
from tldw_Server_API.app.core.Web_Scraping import browser_transport
from tldw_Server_API.app.services import claims_review_metrics_scheduler as claims_scheduler

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


# Today every reader but AuthNZ and config_sections looks at the environment only.
TODAY = {
    #               config_sections  AuthNZ         SINGLE_USER_MODE (APP_MODE)
    "env_set": ("multi_user", "multi_user", True),
    "env_blank": ("multi_user", REJECTED, True),
    "env_invalid": ("multi_user", REJECTED, True),
    "config_only": ("multi_user", "multi_user", True),
    "neither": ("single_user", "single_user", True),
    "app_mode_only": ("single_user", "single_user", False),
    "config_invalid_app_mode": ("multi_user", REJECTED, False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_auth_mode_resolution_today(apply_case, case):
    minimal = apply_case(case)
    env_only = CASES[case][0]
    single_user_default = "single_user"
    expected_cs, expected_authnz, expected_single_user_mode = TODAY[case]

    assert load_auth_config(minimal).mode == expected_cs
    assert _authnz_mode() == expected_authnz
    assert config.load_settings()["SINGLE_USER_MODE"] is expected_single_user_mode

    # env-only readers: the effective mode is invisible to them today
    stripped = (env_only or "").strip().lower()
    assert command_router.is_single_user_mode() is (stripped == "single_user")
    assert db_path_utils._get_auth_mode() == (stripped or "single_user")
    assert setup_endpoint._resolve_auth_mode({}) == ((env_only or "").strip() or CASES[case][2] or "single_user")
    assert connectors_service._connector_secret_encryption_required() is (stripped == "multi_user")
    assert claims_scheduler.resolve_scheduler_config({}).single_user_mode is ((single_user_default if env_only is None else env_only).lower() == "single_user")
    assert sync_security.server_trusted_encryption_status_from_env().auth_mode == ((env_only or "").strip() or None)
    validator = EvaluationConfigValidator()
    validator._validate_authentication()
    assert any("single-user mode" in issue.message for issue in validator.issues) is ((single_user_default if env_only is None else env_only) == "single_user")


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
