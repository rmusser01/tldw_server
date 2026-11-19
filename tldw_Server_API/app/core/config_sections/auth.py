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


def resolve_access_token_expire_minutes(
    config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None
) -> Resolved:
    """The AuthNZ overlay's rule: env ``ACCESS_TOKEN_EXPIRE_MINUTES``, then ``[AuthNZ] access_token_expire_minutes``.

    Blank values count as set; the raw text is never validated here (pydantic's ``int`` field does that).
    """
    return resolve(
        ("ACCESS_TOKEN_EXPIRE_MINUTES",), (("AuthNZ", "access_token_expire_minutes"),),
        config_parser=config_parser, env=env, blank_env=True, blank_config=True,
    )


def resolve_redis_url(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> Resolved:
    """The AuthNZ overlay's rule: env ``REDIS_URL``, then ``[AuthNZ] redis_url``; blank values count as set.

    The raw text is never validated here (the ``REDIS_URL`` field's validator does that).
    """
    return resolve(
        ("REDIS_URL",), (("AuthNZ", "redis_url"),), config_parser=config_parser, env=env,
        blank_env=True, blank_config=True,
    )


def resolve_database_url(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> Resolved:
    """AuthNZ's users DB rule: env ``DATABASE_URL``, then ``[AuthNZ] database_url``, then ``db_type``.

    Blank values count as set. A URL built from ``[AuthNZ] db_type`` reports ``source="config"`` with
    origin ``AuthNZ.db_type``. With ``pg_password`` unset it logs a warning (never the URL or password)
    and reports ``source="default"``, so nothing is overridden. The raw text is never validated here.
    """
    resolved = resolve(
        ("DATABASE_URL",), (("AuthNZ", "database_url"),), config_parser=config_parser, env=env,
        blank_env=True, blank_config=True,
    )
    if resolved.source != "default" or config_parser is None:
        return resolved

    def option(name: str, fallback: str) -> str:
        return str(config_parser.get("AuthNZ", name, fallback=fallback)).strip()

    db_type = config_parser.get("AuthNZ", "db_type", fallback=None)
    kind = str(db_type).strip().lower() if db_type is not None else ""
    if kind == "sqlite":
        return Resolved(f"sqlite:///{option('sqlite_path', './Databases/users.db')}", "config", "AuthNZ.db_type")
    if kind in {"postgres", "postgresql"}:
        password = option("pg_password", "")
        if not password:
            logger.warning("AuthNZ: pg_password not set in config.txt; not synthesizing DATABASE_URL.")
            return resolved
        host, port = option("pg_host", "localhost"), option("pg_port", "5432")
        db, user = option("pg_db", "tldw_users"), option("pg_user", "tldw_user")
        sslmode = option("pg_sslmode", "prefer")
        return Resolved(f"postgresql://{user}:{password}@{host}:{port}/{db}?sslmode={sslmode}", "config", "AuthNZ.db_type")
    return resolved


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
