"""Give providers test API keys the way the gateway reads them; replaces patching API_KEYS on chat_request_schemas or endpoints.chat (TASK-13950.13)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tldw_Server_API.app.core import config_llm_providers


def api_keys_source(keys: Mapping[str, str | None]) -> Callable[[], dict[str, str | None]]:
    """A ``get_api_keys`` replacement: the real keys, with ``keys`` layered on top.

    The real source is read now, before the caller installs the patch, so the lambda never calls itself.
    """
    real = config_llm_providers.get_api_keys
    return lambda: {**real(), **keys}
