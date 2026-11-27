"""The one provider call Prompt Studio makes, through the LLM gateway (TASK-13950.27 PR H)."""

from __future__ import annotations

from typing import Any

from ....core.LLM_Calls.adapter_utils import PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY
from ....core.LLM_Calls.gateway import perform_chat_api_call
from ....core.LLM_Calls.provider_results import raise_if_content_filtered

# Keys a request mapping must never carry into the gateway: the surface and the provider are ours to set.
_RESERVED_KEYS = frozenset({"llm_surface", "api_provider", "api_endpoint", "provider"})


def call_prompt_studio_llm(provider: str, request: dict[str, Any], *, timeout: float | None = None) -> Any:
    """Call ``provider`` through the gateway with an adapter-style ``request`` (sync; run it in a worker).

    R10: ``request`` may come from a project, prompt or test config, so ``llm_surface`` (and the provider
    keys) are dropped from it and the surface is passed explicitly.
    """
    kwargs = {key: value for key, value in request.items() if key not in _RESERVED_KEYS}
    if kwargs.get("credentials_resolved") is True and PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY not in kwargs:
        # The adapter took a resolved key and config as given; the gateway wants a runtime handle for that
        # flag. Without a handle the explicit key and app_config stand on their own.
        del kwargs["credentials_resolved"]
    if timeout is not None:
        kwargs["timeout"] = timeout
    response = perform_chat_api_call(api_provider=provider, **kwargs, llm_surface="prompt_studio")
    raise_if_content_filtered(response)
    return response
