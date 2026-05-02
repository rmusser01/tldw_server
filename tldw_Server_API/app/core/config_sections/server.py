from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from tldw_Server_API.app.core.testing import is_truthy

from .types import ConfigParserLike

_FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class ServerConfig:
    disable_cors: bool
    cors_allow_credentials: bool


def _get_raw(
    config_parser: ConfigParserLike,
    env_map: Mapping[str, str],
    env_key: str,
    option: str,
    default: str,
) -> str:
    """Return an environment override, parser value, or default for a server option."""
    raw = env_map.get(env_key)
    if raw is None or str(raw).strip() == "":
        raw = config_parser.get("Server", option, fallback=default)
    text = str(raw).strip()
    return text or default


def _parse_bool(raw: object, default: bool) -> bool:
    """Parse a boolean value with project-standard truthy tokens."""
    text = str(raw).strip().lower()
    if not text:
        return default
    if is_truthy(text):
        return True
    if text in _FALSE_VALUES:
        return False
    return default


def load_server_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ServerConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    return ServerConfig(
        disable_cors=_parse_bool(
            _get_raw(config_parser, env_map, "DISABLE_CORS", "disable_cors", "false"),
            False,
        ),
        cors_allow_credentials=_parse_bool(
            _get_raw(
                config_parser,
                env_map,
                "CORS_ALLOW_CREDENTIALS",
                "cors_allow_credentials",
                "false",
            ),
            False,
        ),
    )
