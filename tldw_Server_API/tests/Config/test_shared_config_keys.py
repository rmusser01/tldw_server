"""One test per shared config key, pinning where it resolves today (TASK-13950.21, Task 7).

Row 1: ``AUTH_MODE`` (alias ``APP_MODE``). Row 2: ``SINGLE_USER_FIXED_ID``. Row 3: ``ACCESS_TOKEN_EXPIRE_MINUTES``. Row 5: ``REDIS_URL``. Row 6: ``JWT_SECRET_KEY``. Every surface that reads the key is exercised through
its nearest callable, over the same input cases. The commit that moved the key changed only the
expected values the owner approved (D1 a-c).
"""

from __future__ import annotations

import configparser
import contextlib
from pathlib import Path

from types import SimpleNamespace

import pydantic
import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint
from tldw_Server_API.app.core import config, startup_preflight
from tldw_Server_API.app.core.AuthNZ import initialize as authnz_initialize
from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings
from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig
from tldw_Server_API.app.core.Chat import command_router
from tldw_Server_API.app.core.config_sections import auth as auth_section
from tldw_Server_API.app.core.config_sections import redis as redis_section
from tldw_Server_API.app.core.config_sections.auth import (
    auth_mode_env,
    load_auth_config,
    resolve_access_token_expire_minutes,
    resolve_auth_mode,
    resolve_database_url,
    resolve_redis_url,
    resolve_single_user_fixed_id,
    single_user_fixed_id_env,
)
from tldw_Server_API.app.core.config_sections.database import database_url_env
from tldw_Server_API.app.core.config_sections.precedence import Resolved
from tldw_Server_API.app.core.DB_Management import db_path_utils
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.Evaluations.config_validator import (
    EvaluationConfigValidator,
)
from tldw_Server_API.app.core.External_Sources import connectors_service
from tldw_Server_API.app.core.Setup import setup_manager
from tldw_Server_API.app.core.Sync.v2 import security as sync_security
from tldw_Server_API.app.core.Web_Scraping import browser_transport
from tldw_Server_API.app.core.Watchlists import websub
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


# ---- Row 2: SINGLE_USER_FIXED_ID ----

# id -> (env SINGLE_USER_FIXED_ID, [AuthNZ] single_user_fixed_id)
ID_CASES = {
    "env_set": ("7", "9"),
    "env_blank": ("", "9"),
    "env_invalid": ("abc", "9"),
    "config_only": (UNSET, "9"),
    "neither": (UNSET, UNSET),
}

ERR = "raises"


def _id_parser(value: str | None) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    if value is not None:
        parser.set("AuthNZ", "single_user_fixed_id", value)
    return parser


@pytest.fixture
def apply_id_case(monkeypatch, tmp_path):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_id, cfg_id = ID_CASES[case]
        if env_id is None:
            monkeypatch.delenv("SINGLE_USER_FIXED_ID", raising=False)
        else:
            monkeypatch.setenv("SINGLE_USER_FIXED_ID", env_id)
        monkeypatch.delenv("AUTH_MODE", raising=False)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        parser = _id_parser(cfg_id)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
        # websub's directory scan calls DatabasePaths.get_base_directory, which does not exist, so it
        # always takes the single-user fallback (a latent dead scan, left alone here).
        authnz_settings.reset_settings()
        return parser

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


def _outcome(fn):
    try:
        return fn()
    except (ValueError, pydantic.ValidationError):
        return ERR


# D1(g): config_sections no longer reads [AuthNZ] single_user_fixed_id (before, a blank env value or
# config only gave 9); AuthNZ, LazySettings and websub never did.
ID_EXPECTED = {
    #                 config_sections AuthNZ  LazySettings websub
    "env_set": (7, 7, 7, 7),
    "env_blank": (1, ERR, ERR, 1),
    "env_invalid": (1, ERR, ERR, ERR),
    "config_only": (1, 1, 1, 1),
    "neither": (1, 1, 1, 1),
}


@pytest.mark.parametrize("case", list(ID_CASES))
def test_single_user_fixed_id_resolution_today(apply_id_case, case):
    parser = apply_id_case(case)
    cfg_sections, authnz, lazy, watch = ID_EXPECTED[case]

    assert load_auth_config(parser).single_user_fixed_id == cfg_sections
    assert _outcome(lambda: authnz_settings.get_settings().SINGLE_USER_FIXED_ID) == authnz
    assert _outcome(lambda: config.load_settings()["SINGLE_USER_FIXED_ID"]) == lazy
    assert _outcome(lambda: websub._iter_user_ids_with_websub()) == ([watch] if watch != ERR else ERR)


# D1(g): config.txt no longer sets the id. The raw env text is returned unvalidated (blank kept).
ID_RESOLVED = {
    "env_set": Resolved("7", "env", "SINGLE_USER_FIXED_ID"),
    "env_blank": Resolved("", "env", "SINGLE_USER_FIXED_ID"),
    "env_invalid": Resolved("abc", "env", "SINGLE_USER_FIXED_ID"),
    "config_only": Resolved("1", "default", None),
    "neither": Resolved("1", "default", None),
}


@pytest.mark.parametrize("case", list(ID_CASES))
def test_single_user_fixed_id_resolves_through_config_sections(apply_id_case, case):
    parser = apply_id_case(case)
    assert resolve_single_user_fixed_id(parser) == ID_RESOLVED[case]
    assert resolve_single_user_fixed_id(None) == ID_RESOLVED[case]
    assert single_user_fixed_id_env() == ID_CASES[case][0]


def test_single_user_fixed_id_env_is_exact_case_and_live():
    assert single_user_fixed_id_env({"SINGLE_USER_FIXED_ID": " 3 "}) == " 3 "
    assert single_user_fixed_id_env({"single_user_fixed_id": "3"}) is None
    assert single_user_fixed_id_env({}) is None


def test_config_only_fixed_id_warns_once_per_process():
    messages: list[str] = []
    sink = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    auth_section._warn_fixed_id_ignored_once.cache_clear()
    try:
        for _ in range(3):
            assert load_auth_config(_id_parser("9"), env={}).single_user_fixed_id == 1
        assert load_auth_config(_id_parser(None), env={}).single_user_fixed_id == 1
    finally:
        logger.remove(sink)
    hits = [m for m in messages if "single_user_fixed_id" in m and "SINGLE_USER_FIXED_ID" in m]
    assert len(hits) == 1


# ---- Row 3: ACCESS_TOKEN_EXPIRE_MINUTES ----

# id -> (env ACCESS_TOKEN_EXPIRE_MINUTES, [AuthNZ] access_token_expire_minutes)
TTL_CASES = {
    "env_set": ("7", "45"),
    "env_blank": ("", "45"),
    "env_invalid": ("abc", "45"),
    "config_only": (UNSET, "45"),
    "neither": (UNSET, UNSET),
}


def _ttl_parser(value: str | None) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    if value is not None:
        parser.set("AuthNZ", "access_token_expire_minutes", value)
    return parser


@pytest.fixture
def apply_ttl_case(monkeypatch):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_ttl, cfg_ttl = TTL_CASES[case]
        if env_ttl is None:
            monkeypatch.delenv("ACCESS_TOKEN_EXPIRE_MINUTES", raising=False)
        else:
            monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", env_ttl)
        monkeypatch.delenv("AUTH_MODE", raising=False)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        parser = _ttl_parser(cfg_ttl)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
        authnz_settings.reset_settings()
        return parser

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


# config_sections has no function for this key today. The LazySettings entry is deleted (no reader used it), so the key is absent.
ABSENT = "absent"
TTL_EXPECTED = {
    #                 AuthNZ LazySettings
    "env_set": (7, ABSENT),
    "env_blank": (ERR, ABSENT),
    "env_invalid": (ERR, ABSENT),
    "config_only": (45, ABSENT),
    "neither": (30, ABSENT),
}


@pytest.mark.parametrize("case", list(TTL_CASES))
def test_access_token_expire_minutes_resolution_today(apply_ttl_case, case):
    apply_ttl_case(case)
    authnz, lazy = TTL_EXPECTED[case]

    assert _outcome(lambda: authnz_settings.get_settings().ACCESS_TOKEN_EXPIRE_MINUTES) == authnz
    assert _outcome(lambda: config.load_settings().get("ACCESS_TOKEN_EXPIRE_MINUTES", ABSENT)) == lazy


# The overlay's rule: env (blank kept), then [AuthNZ] access_token_expire_minutes (blank kept), then nothing.
TTL_RESOLVED = {
    "env_set": Resolved("7", "env", "ACCESS_TOKEN_EXPIRE_MINUTES"),
    "env_blank": Resolved("", "env", "ACCESS_TOKEN_EXPIRE_MINUTES"),
    "env_invalid": Resolved("abc", "env", "ACCESS_TOKEN_EXPIRE_MINUTES"),
    "config_only": Resolved("45", "config", "AuthNZ.access_token_expire_minutes"),
    "neither": Resolved(None, "default", None),
}


@pytest.mark.parametrize("case", list(TTL_CASES))
def test_access_token_expire_minutes_resolves_through_config_sections(apply_ttl_case, case):
    parser = apply_ttl_case(case)
    assert resolve_access_token_expire_minutes(parser) == TTL_RESOLVED[case]


def test_access_token_expire_minutes_is_not_in_lazy_settings(apply_ttl_case):
    apply_ttl_case("env_set")
    assert "ACCESS_TOKEN_EXPIRE_MINUTES" not in config.load_settings()


# ---- Row 4: DATABASE_URL (split: AuthNZ users DB vs the content-backend override) ----

# id -> (env DATABASE_URL, [AuthNZ] database_url, [AuthNZ] db_type, pg_password)
DB_CASES = {
    "env_set": ("sqlite:///./env.db", "sqlite:///./cfg.db", UNSET, UNSET),
    "env_blank": ("", "sqlite:///./cfg.db", UNSET, UNSET),
    "config_only": (UNSET, "sqlite:///./cfg.db", UNSET, UNSET),
    "config_blank": (UNSET, "", "sqlite", UNSET),
    "synth_sqlite": (UNSET, UNSET, "sqlite", UNSET),
    "synth_postgres": (UNSET, UNSET, "postgres", "pw"),
    "synth_postgres_no_password": (UNSET, UNSET, "postgres", UNSET),
    "neither": (UNSET, UNSET, UNSET, UNSET),
}


def _db_parser(cfg_url, db_type, pg_password) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None)
    parser.add_section("AuthNZ")
    for option, value in (("database_url", cfg_url), ("db_type", db_type), ("pg_password", pg_password)):
        if value is not None:
            parser.set("AuthNZ", option, value)
    if db_type == "sqlite":
        parser.set("AuthNZ", "sqlite_path", "./synth.db")
    return parser


@pytest.fixture
def apply_db_case(monkeypatch):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_url, cfg_url, db_type, pg_password = DB_CASES[case]
        if env_url is None:
            monkeypatch.delenv("DATABASE_URL", raising=False)
        else:
            monkeypatch.setenv("DATABASE_URL", env_url)
        monkeypatch.delenv("AUTH_MODE", raising=False)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        parser = _db_parser(cfg_url, db_type, pg_password)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
        authnz_settings.reset_settings()
        return parser

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


AUTHNZ_DEFAULT_URL = "sqlite:///./Databases/users.db"
DB_AUTHNZ_EXPECTED = {
    "env_set": "sqlite:///./env.db",
    "env_blank": "",
    "config_only": "sqlite:///./cfg.db",
    "config_blank": "",
    "synth_sqlite": "sqlite:///./synth.db",
    "synth_postgres": "postgresql://tldw_user:pw@localhost:5432/tldw_users?sslmode=prefer",
    "synth_postgres_no_password": AUTHNZ_DEFAULT_URL,
    "neither": AUTHNZ_DEFAULT_URL,
}


@pytest.mark.parametrize("case", list(DB_CASES))
def test_database_url_resolution_today(apply_db_case, monkeypatch, case):
    apply_db_case(case)
    env_url = DB_CASES[case][0]

    assert _outcome(lambda: authnz_settings.get_settings().DATABASE_URL) == DB_AUTHNZ_EXPECTED[case]

    # LazySettings, the generic content backend and preflight read the environment alone (blank kept).
    lazy = config.load_settings()["DATABASE_URL"]
    assert (lazy == env_url) if env_url is not None else lazy.startswith("sqlite:///")

    backend = DatabaseConfig.from_env()
    if env_url:
        assert backend.connection_string == env_url
    else:
        assert backend.connection_string is None or backend.connection_string != "sqlite:///./cfg.db"

    assert startup_preflight.check_database_connectivity()["status"] == "ok"  # no case sets a postgres URL


def test_auth_database_url_reader_prefers_the_environment_today(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@envhost:5432/envdb")
    reader = AuthDatabaseConfig.__new__(AuthDatabaseConfig)
    reader.settings = SimpleNamespace(DATABASE_URL="postgresql://u:p@settingshost:5432/settingsdb")
    assert reader._get_postgresql_config().connection_string == "postgresql://u:p@envhost:5432/envdb"
    monkeypatch.setenv("DATABASE_URL", "")
    assert reader._get_postgresql_config().connection_string.endswith("@localhost:5432/tldw_users")
    monkeypatch.delenv("DATABASE_URL")
    assert reader._get_postgresql_config().connection_string == "postgresql://u:p@settingshost:5432/settingsdb"


def test_preflight_reads_the_live_database_url_today(monkeypatch):
    def refuse(*_a, **_k):
        raise OSError("blocked")

    monkeypatch.setattr("socket.create_connection", refuse)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@dbhost:6543/x")
    assert startup_preflight.check_database_connectivity()["message"].startswith("Cannot reach Postgres at dbhost:6543")
    monkeypatch.setenv("DATABASE_URL", "")
    assert startup_preflight.check_database_connectivity()["status"] == "ok"


DB_RESOLVED = {
    "env_set": Resolved("sqlite:///./env.db", "env", "DATABASE_URL"),
    "env_blank": Resolved("", "env", "DATABASE_URL"),
    "config_only": Resolved("sqlite:///./cfg.db", "config", "AuthNZ.database_url"),
    "config_blank": Resolved("", "config", "AuthNZ.database_url"),
    "synth_sqlite": Resolved("sqlite:///./synth.db", "config", "AuthNZ.db_type"),
    "synth_postgres": Resolved(DB_AUTHNZ_EXPECTED["synth_postgres"], "config", "AuthNZ.db_type"),
    "synth_postgres_no_password": Resolved(None, "default", None),
    "neither": Resolved(None, "default", None),
}


@pytest.mark.parametrize("case", list(DB_CASES))
def test_database_url_resolves_through_config_sections(apply_db_case, case):
    parser = apply_db_case(case)
    assert resolve_database_url(parser) == DB_RESOLVED[case]


def test_database_url_env_is_env_only_blank_kept_and_live(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database_url_env() is None
    monkeypatch.setenv("DATABASE_URL", "")
    assert database_url_env() == ""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///x.db")
    assert database_url_env() == "sqlite:///x.db"
    assert database_url_env({"DATABASE_URL": "y"}) == "y"
    assert database_url_env({}) is None


def test_resolve_database_url_logs_the_password_warning_without_a_secret():
    seen: list[str] = []
    sink = logger.add(lambda message: seen.append(str(message)), level="WARNING")
    try:
        no_password = _db_parser(UNSET, "postgres", UNSET)
        no_password.set("AuthNZ", "pg_user", "someone")
        resolve_database_url(no_password, env={})
        resolve_database_url(_db_parser(UNSET, "postgres", "pw"), env={})
    finally:
        logger.remove(sink)
    text = "".join(seen)
    assert "pg_password not set in config.txt; not synthesizing DATABASE_URL." in text
    assert "someone" not in text and "pw@" not in text


# ---- Row 5: REDIS_URL (split: AuthNZ overlay, LazySettings, env-layer readers) ----

AZ_URL, CFG_URL, ENV_URL = "redis://az:6379/2", "redis://cfg:6380/3", "redis://env:6379/1"
# id -> (env REDIS_URL, [AuthNZ] redis_url, [Redis] redis_url)
REDIS_CASES = {
    "env_set": (ENV_URL, AZ_URL, CFG_URL),
    "env_blank": ("", AZ_URL, CFG_URL),
    "env_spaces": ("  ", AZ_URL, CFG_URL),
    "env_invalid": ("http://bad", AZ_URL, CFG_URL),
    "config_only": (UNSET, AZ_URL, CFG_URL),
    "authnz_only": (UNSET, AZ_URL, UNSET),
    "redis_section_only": (UNSET, UNSET, CFG_URL),
    "config_blank": (UNSET, "", ""),
    "neither": (UNSET, UNSET, UNSET),
}
SYNTH = "redis://localhost:6379/0"


@pytest.fixture
def apply_redis_case(monkeypatch):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_url, authnz_url, redis_url = REDIS_CASES[case]
        for key in ("REDIS_HOST", "REDIS_PORT", "REDIS_DB", "AUTH_MODE"):
            monkeypatch.delenv(key, raising=False)
        if env_url is None:
            monkeypatch.delenv("REDIS_URL", raising=False)
        else:
            monkeypatch.setenv("REDIS_URL", env_url)
        parser = configparser.ConfigParser(interpolation=None)
        for section, value in (("AuthNZ", authnz_url), ("Redis", redis_url)):
            parser.add_section(section)
            if value is not None:
                parser.set(section, "redis_url", value)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
        authnz_settings.reset_settings()
        return parser

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


# AuthNZ's field; LazySettings' (url, host, port, db): the URL's parts fill the host, port and db
# only when it came from config and REDIS_HOST/PORT/DB are unset.
REDIS_AUTHNZ_EXPECTED = {
    "env_set": ENV_URL,
    "env_blank": "",
    "env_spaces": ERR,
    "env_invalid": ERR,
    "config_only": AZ_URL,
    "authnz_only": AZ_URL,
    "redis_section_only": UNSET,
    "config_blank": "",
    "neither": UNSET,
}
REDIS_LAZY_EXPECTED = {
    "env_set": (ENV_URL, "localhost", 6379, 0),
    "env_blank": (CFG_URL, "cfg", 6380, 3),
    "env_spaces": ("  ", "localhost", 6379, 0),
    "env_invalid": ("http://bad", "localhost", 6379, 0),
    "config_only": (CFG_URL, "cfg", 6380, 3),
    "authnz_only": (SYNTH, "localhost", 6379, 0),
    "redis_section_only": (CFG_URL, "cfg", 6380, 3),
    "config_blank": (SYNTH, "localhost", 6379, 0),
    "neither": (SYNTH, "localhost", 6379, 0),
}


@pytest.mark.parametrize("case", list(REDIS_CASES))
def test_redis_url_resolution_today(apply_redis_case, case):
    apply_redis_case(case)

    assert _outcome(lambda: authnz_settings.get_settings().REDIS_URL) == REDIS_AUTHNZ_EXPECTED[case]

    lazy = config.load_settings()
    assert (lazy["REDIS_URL"], lazy["REDIS_HOST"], lazy["REDIS_PORT"], lazy["REDIS_DB"]) == REDIS_LAZY_EXPECTED[case]


REDIS_ENV_STATES = {"unset": UNSET, "blank": "", "set": "redis://h:1/0"}


@pytest.fixture
def redis_env(monkeypatch):
    def apply(state: str) -> str | None:
        value = REDIS_ENV_STATES[state]
        if value is None:
            monkeypatch.delenv("REDIS_URL", raising=False)
        else:
            monkeypatch.setenv("REDIS_URL", value)
        return value

    return apply


@pytest.mark.parametrize("state", list(REDIS_ENV_STATES))
def test_migration_lock_readers_pass_the_live_redis_url_today(monkeypatch, tmp_path, redis_env, state):
    from contextlib import contextmanager

    from tldw_Server_API.app.core.AuthNZ import migrations as authnz_migrations
    from tldw_Server_API.app.core.DB_Management import db_migration

    seen: list[object] = []

    @contextmanager
    def fake_lock(**kwargs):
        seen.append(kwargs["redis_url"])
        yield object()

    expected = redis_env(state)
    monkeypatch.setattr(authnz_migrations, "acquire_migration_lock", fake_lock)
    monkeypatch.setattr(authnz_migrations, "_apply_authnz_migrations_locked", lambda *_a, **_k: None)
    authnz_migrations.apply_authnz_migrations(tmp_path / "users.db")

    migrator = db_migration.DatabaseMigrator(str(tmp_path / "x.db"), str(tmp_path / "migrations"))
    monkeypatch.setattr(db_migration, "acquire_migration_lock", fake_lock)
    with contextlib.suppress(Exception):  # only the lock call matters; the rest of the migration may fail
        migrator.migrate_to_version(create_backup=False)

    assert len(seen) == 2  # one lock per reader
    assert seen == [expected, expected]


@pytest.mark.parametrize("state, expected", [("unset", "redis://localhost:6379"), ("blank", ""), ("set", "redis://h:1/0")])
def test_embeddings_worker_redis_default_today(redis_env, state, expected):
    from tldw_Server_API.app.core.Embeddings.worker_config import RedisConfig

    redis_env(state)
    assert RedisConfig().url == expected


@pytest.mark.parametrize("state, expected", [("unset", None), ("blank", None), ("set", "redis://h:1/0")])
def test_governor_factory_probes_the_live_redis_url_today(monkeypatch, redis_env, state, expected):
    import redis as redis_lib

    from tldw_Server_API.app.core.Resource_Governance.governor_factory import create_governor

    probed: list[str] = []

    def refuse(url, **_kwargs):
        probed.append(url)
        raise OSError("blocked")

    redis_env(state)
    monkeypatch.setattr(redis_lib, "from_url", refuse)
    create_governor(policies={})
    assert (probed[0] if probed else None) == expected


@pytest.mark.parametrize(
    "state, embeddings_url, expected",
    [("unset", None, False), ("blank", None, False), ("set", None, True), ("blank", "redis://e:1/0", True)],
)
def test_embeddings_backpressure_redis_switch_today(monkeypatch, redis_env, state, embeddings_url, expected):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as embeddings_endpoint

    redis_env(state)
    monkeypatch.delenv("REDIS_ENABLED", raising=False)
    monkeypatch.setattr(config, "settings", {})
    if embeddings_url is None:
        monkeypatch.delenv("EMBEDDINGS_REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("EMBEDDINGS_REDIS_URL", embeddings_url)
    assert embeddings_endpoint._is_embeddings_backpressure_redis_enabled() is expected


@pytest.mark.parametrize(
    "state, sandbox_url, expected",
    [
        ("unset", None, "redis://127.0.0.1:6379/0"),
        ("blank", None, "redis://127.0.0.1:6379/0"),
        ("set", None, "redis://h:1/0"),
        ("set", "redis://sandbox:1/0", "redis://sandbox:1/0"),
    ],
)
def test_sandbox_streams_fanout_url_today(monkeypatch, redis_env, state, sandbox_url, expected):
    from tldw_Server_API.app.core.Sandbox import streams

    redis_env(state)
    for key in ("REDIS_HOST", "REDIS_PORT", "REDIS_DB"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SANDBOX_WS_REDIS_FANOUT", "1")
    if sandbox_url is None:
        monkeypatch.delenv("SANDBOX_REDIS_URL", raising=False)
    else:
        monkeypatch.setenv("SANDBOX_REDIS_URL", sandbox_url)
    seen: list[str] = []

    def refuse(*, preferred_url, **_kwargs):
        seen.append(preferred_url)
        raise OSError("blocked")

    monkeypatch.setattr(streams, "create_sync_redis_client", refuse)
    streams.RunStreamHub()
    assert len(seen) == 1
    assert seen[0] == expected


REDIS_AUTHNZ_RESOLVED = {
    "env_set": Resolved(ENV_URL, "env", "REDIS_URL"),
    "env_blank": Resolved("", "env", "REDIS_URL"),
    "env_spaces": Resolved("  ", "env", "REDIS_URL"),
    "env_invalid": Resolved("http://bad", "env", "REDIS_URL"),
    "config_only": Resolved(AZ_URL, "config", "AuthNZ.redis_url"),
    "authnz_only": Resolved(AZ_URL, "config", "AuthNZ.redis_url"),
    "redis_section_only": Resolved(None, "default", None),
    "config_blank": Resolved("", "config", "AuthNZ.redis_url"),
    "neither": Resolved(None, "default", None),
}
REDIS_LAZY_RESOLVED = {
    "env_set": Resolved(ENV_URL, "env", "REDIS_URL"),
    "env_blank": Resolved(CFG_URL, "config", "Redis.redis_url"),
    "env_spaces": Resolved("  ", "env", "REDIS_URL"),
    "env_invalid": Resolved("http://bad", "env", "REDIS_URL"),
    "config_only": Resolved(CFG_URL, "config", "Redis.redis_url"),
    "authnz_only": Resolved(None, "default", None),
    "redis_section_only": Resolved(CFG_URL, "config", "Redis.redis_url"),
    "config_blank": Resolved(None, "default", None),
    "neither": Resolved(None, "default", None),
}


@pytest.mark.parametrize("case", list(REDIS_CASES))
def test_redis_url_resolves_through_config_sections(apply_redis_case, case):
    parser = apply_redis_case(case)
    assert resolve_redis_url(parser) == REDIS_AUTHNZ_RESOLVED[case]
    assert redis_section.resolve_redis_url(parser) == REDIS_LAZY_RESOLVED[case]


def test_redis_url_env_is_env_only_blank_kept_and_live(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert redis_section.redis_url_env() is None
    monkeypatch.setenv("REDIS_URL", "")
    assert redis_section.redis_url_env() == ""
    monkeypatch.setenv("REDIS_URL", "redis://x:1/0")
    assert redis_section.redis_url_env() == "redis://x:1/0"
    assert redis_section.redis_url_env({"REDIS_URL": "y"}) == "y"
    assert redis_section.redis_url_env({}) is None


def test_resolve_redis_url_without_a_parser_reads_the_environment_alone():
    assert redis_section.resolve_redis_url(None, env={"REDIS_URL": "redis://x:1/0"}) == Resolved(
        "redis://x:1/0", "env", "REDIS_URL"
    )
    assert redis_section.resolve_redis_url(None, env={}) == Resolved(None, "default", None)


# ---- Row 6: JWT_SECRET_KEY (AuthNZ overlay, LazySettings, setting readers, env-layer readers) ----

JWT_ENV, JWT_CFG, JWT_DEFAULT = "e" * 32, "c" * 32, "a_very_insecure_default_secret_key_for_dev_only"
# id -> (env JWT_SECRET_KEY, [AuthNZ] jwt_secret_key)
JWT_CASES = {
    "env_set": (JWT_ENV, JWT_CFG),
    "env_blank": ("", JWT_CFG),
    "env_short": ("short", JWT_CFG),
    "config_only": (UNSET, JWT_CFG),
    "config_short": (UNSET, "short"),
    "config_blank": (UNSET, ""),
    "neither": (UNSET, UNSET),
}


@pytest.fixture
def apply_jwt_case(monkeypatch):
    def apply(case: str) -> configparser.ConfigParser:
        config.clear_config_cache()
        env_secret, cfg_secret = JWT_CASES[case]
        for key in ("AUTH_MODE", "KNOWLEDGE_QA_SHARE_LINK_SECRET"):
            monkeypatch.delenv(key, raising=False)
        if env_secret is None:
            monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
        else:
            monkeypatch.setenv("JWT_SECRET_KEY", env_secret)
        parser = configparser.ConfigParser(interpolation=None)
        parser.add_section("AuthNZ")
        if cfg_secret is not None:
            parser.set("AuthNZ", "jwt_secret_key", cfg_secret)
        monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
        monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
        monkeypatch.setattr(authnz_settings, "load_comprehensive_config", lambda: parser)
        authnz_settings.reset_settings()
        return parser

    yield apply
    monkeypatch.undo()
    config.clear_config_cache()
    authnz_settings.reset_settings()


# AuthNZ's field (single-user mode, so no secret is required); LazySettings' entry; the validator's
# "no JWT secret" error; the notes cursor secret; the share-link fallback after a failed derivation.
JWT_EXPECTED = {
    #                 AuthNZ    LazySettings  validator_error  share fallback
    "env_set": (JWT_ENV, JWT_ENV, False, JWT_ENV),
    "env_blank": ("", "", True, None),
    "env_short": ("short", "short", False, "short"),
    "config_only": (JWT_CFG, JWT_DEFAULT, True, None),
    "config_short": ("short", JWT_DEFAULT, True, None),
    "config_blank": ("", JWT_DEFAULT, True, None),
    "neither": (UNSET, JWT_DEFAULT, True, None),
}


def _share_fallback_secret(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chat as chat_mod

    def boom(*_args, **_kwargs):
        raise ValueError("no usable HMAC key material")

    monkeypatch.setattr("tldw_Server_API.app.core.AuthNZ.crypto_utils.derive_hmac_key", boom)
    chat_mod._get_knowledge_qa_share_signing_key.cache_clear()
    try:
        return chat_mod._get_knowledge_qa_share_signing_key().decode()
    except RuntimeError:
        return None
    finally:
        chat_mod._get_knowledge_qa_share_signing_key.cache_clear()


@pytest.mark.parametrize("case", list(JWT_CASES))
def test_jwt_secret_key_resolution_today(apply_jwt_case, monkeypatch, case):
    apply_jwt_case(case)
    authnz_expected, lazy_expected, validator_error, share_expected = JWT_EXPECTED[case]

    assert _outcome(lambda: authnz_settings.get_settings().JWT_SECRET_KEY) == authnz_expected
    assert config.load_settings()["JWT_SECRET_KEY"] == lazy_expected

    monkeypatch.setenv("AUTH_MODE", "multi_user")
    validator = EvaluationConfigValidator()
    validator._validate_authentication()
    assert any("No JWT secret key" in issue.message for issue in validator.issues) is validator_error

    assert _share_fallback_secret(monkeypatch) == share_expected


@pytest.mark.parametrize("case", ["env_set", "env_blank", "config_only", "neither"])
def test_notes_cursor_secret_follows_lazy_settings_today(apply_jwt_case, case):
    from fastapi import HTTPException

    from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint

    apply_jwt_case(case)
    try:
        secret = notes_endpoint._attachment_cursor_secret().decode()
    except HTTPException:
        secret = None
    assert secret == {"env_set": JWT_ENV, "env_blank": None, "config_only": JWT_DEFAULT, "neither": JWT_DEFAULT}[case]


JWT_RING_STATES = {"unset": UNSET, "blank": "", "spaces": "  ", "set": "ring-secret"}


@pytest.mark.parametrize("state", list(JWT_RING_STATES))
def test_webhook_key_ring_lists_the_jwt_secret_today(monkeypatch, state):
    from tldw_Server_API.app.core.AuthNZ.admin_webhook_secrets import _iter_admin_webhook_secret_keys

    for key in ("BYOK_ENCRYPTION_KEY", "BYOK_SECONDARY_ENCRYPTION_KEY", "SESSION_ENCRYPTION_KEY", "SINGLE_USER_API_KEY", "API_KEY"):
        monkeypatch.delenv(key, raising=False)
    if JWT_RING_STATES[state] is None:
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("JWT_SECRET_KEY", JWT_RING_STATES[state])
    assert _iter_admin_webhook_secret_keys() == ([("jwt_secret", "ring-secret")] if state == "set" else [])


@pytest.mark.parametrize("state", list(JWT_RING_STATES))
def test_prototype_signing_secret_readers_use_the_live_env_jwt_secret_today(monkeypatch, state):
    from tldw_Server_API.app.core.Prototype_Workspaces import access, preview_broker

    monkeypatch.setattr(access, "get_settings", lambda: SimpleNamespace(JWT_SECRET_KEY=None, SINGLE_USER_API_KEY=None))
    monkeypatch.setattr(preview_broker, "get_settings", lambda: SimpleNamespace(JWT_SECRET_KEY=None, SINGLE_USER_API_KEY=None))
    for key in ("SINGLE_USER_API_KEY", "PROTOTYPE_PREVIEW_SIGNING_SECRET"):
        monkeypatch.delenv(key, raising=False)
    if JWT_RING_STATES[state] is None:
        monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    else:
        monkeypatch.setenv("JWT_SECRET_KEY", JWT_RING_STATES[state])
    for module in (access, preview_broker):
        if state == "set":
            assert module._resolve_stable_signing_secret(None) == "ring-secret"
        else:
            with pytest.raises(RuntimeError):
                module._resolve_stable_signing_secret(None)


@pytest.mark.parametrize(
    "state, expected",
    [("unset", True), ("blank", True), ("set", False)],
)
def test_setup_wizard_jwt_check_today(monkeypatch, state, expected):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    if state != "unset":
        monkeypatch.setenv("JWT_SECRET_KEY", "" if state == "blank" else "j" * 40)
    _missing, issues = authnz_initialize._detect_env_issues("multi_user", {"JWT_SECRET_KEY": ""})
    assert any("JWT_SECRET_KEY must be set" in issue for issue in issues) is expected
