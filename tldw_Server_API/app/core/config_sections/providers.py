from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .precedence import resolve
from .types import ConfigParserLike


@dataclass(frozen=True)
class ProvidersConfig:
    default_api: str
    default_provider: str


def load_providers_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> ProvidersConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    default_api = str(
        resolve(("DEFAULT_API",), env=env_map, blank_env=True).value
        or config_parser.get("API", "default_api", fallback="openai")
    ).strip() or "openai"

    default_provider = str(
        resolve(("DEFAULT_PROVIDER",), env=env_map, blank_env=True).value
        or config_parser.get("API", "default_provider", fallback=default_api)
    ).strip() or default_api

    return ProvidersConfig(default_api=default_api, default_provider=default_provider)
