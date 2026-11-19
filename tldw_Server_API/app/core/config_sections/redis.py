"""Redis settings shared by LazySettings and the features that read ``REDIS_URL`` (TASK-13950.21)."""

from __future__ import annotations

from collections.abc import Mapping

from .precedence import Resolved, resolve
from .types import ConfigParserLike


def redis_url_env(env: Mapping[str, str] | None = None) -> str | None:
    """The environment layer alone: the live ``REDIS_URL`` text, blank kept, ``None`` when unset."""
    return resolve(("REDIS_URL",), env=env, blank_env=True).value


def resolve_redis_url(config_parser: ConfigParserLike | None, env: Mapping[str, str] | None = None) -> Resolved:
    """LazySettings' rule: a non-empty env ``REDIS_URL``, then a non-empty ``[Redis] redis_url``.

    "Non-empty" is plain truthiness, as before: a whitespace-only value counts as set, a blank one is
    skipped. A blank env value falls through to config. Nothing is validated; the caller builds the
    host/port URL when the source is ``default``.
    """
    from_env = redis_url_env(env)
    if from_env:
        return Resolved(from_env, "env", "REDIS_URL")
    from_config = resolve((), (("Redis", "redis_url"),), config_parser=config_parser, blank_config=True)
    return from_config if from_config.value else Resolved(None, "default", None)
