"""The call path and provider/model resolution live in the LLM gateway; chat_service only re-exports (TASK-13950.13)."""

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.LLM_Calls import gateway

pytestmark = pytest.mark.unit

REEXPORTED = (
    "perform_chat_api_call", "perform_chat_api_call_async", "resolve_provider_and_model",
    "normalize_request_provider_and_model", "infer_provider_from_model_catalog",
    "invalidate_model_alias_caches", "resolve_provider_api_key",
)
# chat_service's staying known-models functions still call these two; they are imported, not re-exported.
STAYING_CODE_IMPORTS = ("_provider_inventory_endpoint", "_raise_inventory_failure")


@pytest.mark.parametrize("name", REEXPORTED)
def test_chat_service_re_exports_the_gateway_object(name: str) -> None:
    assert getattr(chat_service, name) is getattr(gateway, name)


@pytest.mark.provider_inventory  # keeps the gateway's real model check instead of the conftest stub
def test_chat_service_holds_no_other_gateway_function() -> None:
    """A patch or call on chat_service of any other moved name must fail loudly, not miss the gateway."""
    on_chat_service = {
        name
        for name, value in vars(gateway).items()
        if callable(value) and getattr(value, "__module__", None) == gateway.__name__ and hasattr(chat_service, name)
    }

    assert on_chat_service == set(REEXPORTED) | set(STAYING_CODE_IMPORTS)


class _Adapter:
    def chat(self, request, *, timeout=None):
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class _Registry:
    def get_adapter(self, name):
        return _Adapter()


def test_the_call_path_looks_up_its_registry_in_the_gateway(monkeypatch) -> None:
    monkeypatch.setattr(gateway, "_get_llm_registry", lambda: _Registry())

    result = chat_service.perform_chat_api_call(
        api_endpoint="e2e-fake", model="m", api_key="k", messages_payload=[{"role": "user", "content": "hi"}]
    )

    assert result["choices"][0]["message"]["content"] == "ok"
