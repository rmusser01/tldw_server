from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .types import ConfigParserLike


@dataclass(frozen=True)
class ServerConfig:
    disable_cors: bool
    cors_allow_credentials: bool


def load_server_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ServerConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    disable_cors = str(
        env_map.get("DISABLE_CORS", "")
        or config_parser.get("Server", "disable_cors", fallback="false")
    ).strip().lower() in ("true", "1", "yes")

    cors_allow_credentials = str(
        env_map.get("CORS_ALLOW_CREDENTIALS", "")
        or config_parser.get("Server", "cors_allow_credentials", fallback="false")
    ).strip().lower() in ("true", "1", "yes")

    return ServerConfig(
        disable_cors=disable_cors,
        cors_allow_credentials=cors_allow_credentials,
    )
