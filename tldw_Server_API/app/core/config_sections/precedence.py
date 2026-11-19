"""The one env-over-config resolver (TASK-13950.21).

``resolve`` returns the first environment key that is set, then the first ``config.txt`` location
that is set, then the default. "Set" means not None and, unless ``blank_env``/``blank_config``,
not blank after ``strip()``. Values come back unstripped and unvalidated; callers parse them.
A missing section or option is absent (``fallback=None``); every other parser error, such as
``InterpolationSyntaxError``, propagates. Stdlib only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Literal, NamedTuple

from .types import ConfigParserLike

Source = Literal["env", "config", "default"]


class Resolved(NamedTuple):
    value: str | None
    source: Source
    origin: str | None  # the env key, or "<Section>.<option>"


def _is_set(value: Any, allow_blank: bool) -> bool:
    return value is not None and (allow_blank or str(value).strip() != "")


def resolve(
    env_keys: Sequence[str] = (),
    locations: Sequence[tuple[str, str]] = (),
    *,
    config_parser: ConfigParserLike | None = None,
    env: Mapping[str, str] | None = None,
    default: str | None = None,
    blank_env: bool = False,
    blank_config: bool = False,
) -> Resolved:
    """Resolve one setting: env keys in order, then config locations in order, then ``default``.

    ``env=None`` reads ``os.environ`` at call time; ``config_parser=None`` skips the config layer.
    """
    env_map = os.environ if env is None else env
    for key in env_keys:
        value = env_map.get(key)
        if _is_set(value, blank_env):
            return Resolved(value, "env", key)
    if config_parser is not None:
        for section, option in locations:
            value = config_parser.get(section, option, fallback=None)
            if _is_set(value, blank_config):
                return Resolved(value, "config", f"{section}.{option}")
    return Resolved(default, "default", None)


__all__ = ["Resolved", "Source", "resolve"]
