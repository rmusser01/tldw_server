"""Anthropic model-ID spelling shared by pricing, model-availability checks and the request builders."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any

_DOTTED_VERSION = re.compile(r"^(claude-(?:opus|sonnet|haiku|fable|mythos)-\d+)\.(\d+)", re.IGNORECASE)


def canonical_anthropic_model_id(model: str | None) -> str:
    """Return the dashed alias of a dotted Claude name, and any other name unchanged.

    No current Anthropic API ID has a dot (``claude-sonnet-4-5``, not ``claude-sonnet-4.5``), but the WebUI model
    picker, saved conversations and config defaults still hold dotted names. This is the one place that maps them.
    """
    value = model or ""
    return _DOTTED_VERSION.sub(r"\1-\2", value)


def allowlist_has_model(allowed_models: Collection[Any], model: Any) -> bool:
    """Whether ``model`` is in a model allowlist, a dotted Claude name and its dashed ID counting as one model.

    Only strings are mapped; any other value is compared as written, so a malformed entry or request body never
    raises (the budget middleware skips enforcement on a TypeError) and never widens the list.
    """
    if model in allowed_models:
        return True
    if not isinstance(model, str):
        return False
    wanted = canonical_anthropic_model_id(model)
    return any(isinstance(item, str) and canonical_anthropic_model_id(item) == wanted for item in allowed_models)
