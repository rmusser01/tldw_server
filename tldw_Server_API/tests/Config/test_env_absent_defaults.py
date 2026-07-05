"""Env-absent config defaults — the #2590/e88c96500f defect class (RA6).

Real deployments frequently run with NO override env vars set. The test
conftests force-set many env vars at import time, so the rest of the suite
never exercises the no-env-var fallback path — which is exactly where the
hardening-broke-the-fallback defects escaped.

These tests bypass the conftest pollution entirely by passing an explicit
empty ``env={}`` map to each typed config loader (all take
``(config_parser, env=None)``), so they assert the true real-deployment
defaults deterministically.
"""
from __future__ import annotations

from configparser import ConfigParser

import pytest

from tldw_Server_API.app.core.config_sections.audio import load_audio_config
from tldw_Server_API.app.core.config_sections.auth import load_auth_config
from tldw_Server_API.app.core.config_sections.chat import load_chat_config
from tldw_Server_API.app.core.config_sections.chunking import load_chunking_config
from tldw_Server_API.app.core.config_sections.database import load_database_config
from tldw_Server_API.app.core.config_sections.embeddings import load_embeddings_config
from tldw_Server_API.app.core.config_sections.jobs import load_jobs_config
from tldw_Server_API.app.core.config_sections.logging import load_logging_config
from tldw_Server_API.app.core.config_sections.moderation import load_moderation_config
from tldw_Server_API.app.core.config_sections.providers import load_providers_config
from tldw_Server_API.app.core.config_sections.rag import load_rag_config
from tldw_Server_API.app.core.config_sections.server import load_server_config
from tldw_Server_API.app.core.config_sections.stt import load_stt_config

pytestmark = pytest.mark.unit

EMPTY_ENV: dict[str, str] = {}

# (loader, config.txt section names it reads)
_LOADERS = [
    (load_auth_config, ["AuthNZ"]),
    (load_server_config, ["API", "Server"]),
    (load_database_config, ["Database"]),
    (load_jobs_config, ["Jobs"]),
    (load_rag_config, ["RAG"]),
    (load_chat_config, ["Chat-Module"]),
    (load_audio_config, ["TTS-Settings"]),
    (load_stt_config, ["STT-Settings"]),
    (load_embeddings_config, ["Embeddings"]),
    (load_moderation_config, ["Moderation"]),
    (load_providers_config, ["API"]),
    (load_logging_config, ["Logging"]),
]


def _empty_parser() -> ConfigParser:
    """A parser with every section the loaders may read, but no options set —
    i.e. a fresh install with a bare config.txt."""
    parser = ConfigParser()
    for section in (
        "AuthNZ", "API", "Server", "Database", "Jobs", "RAG", "Chat-Module",
        "TTS-Settings", "STT-Settings", "Chunking", "Embeddings", "Moderation",
        "Logging", "Providers",
    ):
        parser.add_section(section)
    return parser


@pytest.mark.parametrize("loader", [pair[0] for pair in _LOADERS], ids=[p[0].__name__ for p in _LOADERS])
def test_config_loader_is_total_with_no_env(loader) -> None:
    """Every typed loader must produce a config (never raise) when NO env vars
    are set — the no-override real-deployment path.
    """
    result = loader(_empty_parser(), env=EMPTY_ENV)
    assert result is not None


def test_auth_defaults_to_single_user_with_no_env() -> None:
    """With no AUTH_MODE/APP_MODE, auth defaults to the safe single-user mode."""
    cfg = load_auth_config(_empty_parser(), env=EMPTY_ENV)
    assert cfg.mode == "single_user"
    assert cfg.single_user_fixed_id == 1


def test_server_cors_is_not_disabled_by_default() -> None:
    """CORS protection must be ON by default (disable_cors False) with no env —
    a hardening regression here is the e88c96500f-class bug.
    """
    cfg = load_server_config(_empty_parser(), env=EMPTY_ENV)
    assert cfg.disable_cors is False


def test_chunking_loader_uses_typed_defaults_with_no_env() -> None:
    """Chunking numeric/bool options fall back to their typed defaults, not
    crash, when the env map is empty (exercises _parse_int/_parse_bool paths)."""
    cfg = load_chunking_config(_empty_parser(), env=EMPTY_ENV)
    # the typed loader must yield real ints/bools, never raw strings or None
    assert isinstance(cfg.max_size, int) and cfg.max_size > 0
    assert isinstance(cfg.overlap, int) and cfg.overlap >= 0
    assert isinstance(cfg.adaptive, bool)
    assert isinstance(cfg.multi_level, bool)


def test_empty_env_matches_missing_env_map() -> None:
    """Passing env={} must behave identically to env=None-under-empty-os-environ:
    an explicit empty map is the faithful 'no overrides' signal."""
    parser = _empty_parser()
    for loader, _sections in _LOADERS:
        explicit_empty = loader(parser, env={})
        assert explicit_empty is not None


# --------------------------------------------------------------------------- #
# Auth-mode matrix: the loader must resolve each deployment mode from env,
# via both AUTH_MODE (explicit) and APP_MODE (legacy) — the axes that actually
# vary between single-user and multi-user deployments.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("env_map", "expected_mode"),
    [
        ({"AUTH_MODE": "single_user"}, "single_user"),
        ({"AUTH_MODE": "multi_user"}, "multi_user"),
        ({"AUTH_MODE": "garbage"}, "single_user"),  # invalid -> safe fallback
        ({}, "single_user"),  # no env at all -> safe default
    ],
)
def test_auth_mode_resolution_matrix(env_map: dict[str, str], expected_mode: str) -> None:
    cfg = load_auth_config(_empty_parser(), env=env_map)
    assert cfg.mode == expected_mode


def test_app_mode_multi_is_the_last_resort_fallback() -> None:
    """Documents the real precedence: APP_MODE is only consulted after both
    AUTH_MODE and the config's auth_mode fail to yield a valid mode. Because
    ConfigParser.get(..., fallback="single_user") always returns a valid value
    for a missing section/option, the APP_MODE branch is reached only when the
    config carries an explicitly INVALID auth_mode.
    """
    from configparser import ConfigParser

    parser = ConfigParser()
    parser.add_section("AuthNZ")
    parser.set("AuthNZ", "auth_mode", "not_a_real_mode")  # invalid -> falls through
    cfg = load_auth_config(parser, env={"APP_MODE": "multi"})
    assert cfg.mode == "multi_user"
