from __future__ import annotations

import configparser
import os
from dataclasses import dataclass
from functools import cache
from typing import Mapping

from loguru import logger

from .precedence import Resolved, resolve
from .types import ConfigParserLike


@dataclass(frozen=True)
class AuthConfig:
    mode: str
    single_user_fixed_id: int


def resolve_auth_mode(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> Resolved:
    """The AuthNZ overlay's rule: exact-case env, then ``[AuthNZ] auth_mode``, then ``single_user``.

    A blank value counts as set and is returned as is; the raw text is never validated here, so a
    bad mode reaches pydantic's ``Literal`` and stops AuthNZ (fail closed).
    """
    return resolve(
        ("AUTH_MODE",), (("AuthNZ", "auth_mode"),), config_parser=config_parser, env=env,
        default="single_user", blank_env=True, blank_config=True,
    )


def auth_mode_env(env: Mapping[str, str] | None = None) -> str | None:
    """The environment layer alone: the live ``AUTH_MODE`` text, blank kept, ``None`` when unset."""
    return resolve(("AUTH_MODE",), env=env, blank_env=True).value


def resolve_single_user_fixed_id(
    config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None
) -> Resolved:
    """AuthNZ's and LazySettings' rule: env ``SINGLE_USER_FIXED_ID`` (blank kept), then ``1``.

    ``[AuthNZ] single_user_fixed_id`` is not read (D1g); the raw text is never validated here.
    """
    return resolve(("SINGLE_USER_FIXED_ID",), config_parser=config_parser, env=env, default="1", blank_env=True)


def single_user_fixed_id_env(env: Mapping[str, str] | None = None) -> str | None:
    """The environment layer alone: the live ``SINGLE_USER_FIXED_ID`` text, blank kept, ``None`` when unset."""
    return resolve(("SINGLE_USER_FIXED_ID",), env=env, blank_env=True).value


@cache
def _warn_fixed_id_ignored_once() -> None:
    """Runs its body once per process (``cache``): ``[AuthNZ] single_user_fixed_id`` is ignored (D1g)."""
    logger.warning(
        "[AuthNZ] single_user_fixed_id in config.txt is ignored; set the SINGLE_USER_FIXED_ID "
        "environment variable instead."
    )


def _warn_ignored_fixed_id(config_parser: ConfigParserLike) -> None:
    if config_parser.get("AuthNZ", "single_user_fixed_id", fallback=None) is not None:
        _warn_fixed_id_ignored_once()


def load_auth_config(
    config_parser: ConfigParserLike | None = None,
    env: Mapping[str, str] | None = None,
) -> AuthConfig:
    """Load the tolerant auth view; ``config_parser=None`` loads ``config.txt`` now (never raises)."""
    if config_parser is None:
        from tldw_Server_API.app.core.config import load_comprehensive_config

        try:
            config_parser = load_comprehensive_config() or configparser.ConfigParser()
        except (OSError, configparser.Error, ValueError):
            config_parser = configparser.ConfigParser()
    env_map: Mapping[str, str] = env if env is not None else os.environ

    mode = (auth_mode_env(env_map) or "").strip().lower()
    if mode not in {"single_user", "multi_user"}:
        mode = str(config_parser.get("AuthNZ", "auth_mode", fallback="single_user")).strip().lower()
    if mode not in {"single_user", "multi_user"}:
        app_mode = str(resolve(("APP_MODE",), env=env_map, default="single", blank_env=True).value).strip().lower()
        mode = "multi_user" if app_mode == "multi" else "single_user"

    _warn_ignored_fixed_id(config_parser)
    single_user_raw = single_user_fixed_id_env(env_map) or "1"
    try:
        single_user_fixed_id = int(str(single_user_raw))
    except (TypeError, ValueError):
        single_user_fixed_id = 1

    return AuthConfig(mode=mode, single_user_fixed_id=single_user_fixed_id)
