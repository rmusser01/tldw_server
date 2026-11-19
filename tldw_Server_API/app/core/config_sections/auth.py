from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from .precedence import resolve
from .types import ConfigParserLike


@dataclass(frozen=True)
class AuthConfig:
    mode: str
    single_user_fixed_id: int


def load_auth_config(
    config_parser: ConfigParserLike,
    env: Mapping[str, str] | None = None,
) -> AuthConfig:
    env_map: Mapping[str, str] = env if env is not None else os.environ

    mode = (resolve(("AUTH_MODE",), env=env_map, blank_env=True).value or "").strip().lower()
    if mode not in {"single_user", "multi_user"}:
        mode = str(config_parser.get("AuthNZ", "auth_mode", fallback="single_user")).strip().lower()
    if mode not in {"single_user", "multi_user"}:
        app_mode = str(resolve(("APP_MODE",), env=env_map, default="single", blank_env=True).value).strip().lower()
        mode = "multi_user" if app_mode == "multi" else "single_user"

    single_user_raw = (
        resolve(("SINGLE_USER_FIXED_ID",), env=env_map, blank_env=True).value
        or config_parser.get("AuthNZ", "single_user_fixed_id", fallback="1")
    )
    try:
        single_user_fixed_id = int(str(single_user_raw))
    except (TypeError, ValueError):
        single_user_fixed_id = 1

    return AuthConfig(mode=mode, single_user_fixed_id=single_user_fixed_id)
