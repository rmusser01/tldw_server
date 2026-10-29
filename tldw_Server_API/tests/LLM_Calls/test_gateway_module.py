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
NOT_REEXPORTED = (
    "_get_llm_registry", "_load_models_with_case_cached", "_provider_has_model_cached",
    "_find_catalog_providers_for_model_cached", "_load_alias_overrides_cached", "_privacy_safe_provider_errors_enabled",
    "_build_adapter_request_from_chat_args", "_map_provider_egress_error", "_resolve_chat_provider_name",
    "parse_provider_model_for_metrics", "_validate_provider_model_selection",
)


@pytest.mark.parametrize("name", REEXPORTED)
def test_chat_service_re_exports_the_gateway_object(name: str) -> None:
    assert getattr(chat_service, name) is getattr(gateway, name)


@pytest.mark.parametrize("name", NOT_REEXPORTED)
def test_moved_internals_are_not_re_exported(name: str) -> None:
    """A patch or call on chat_service would no longer reach the gateway, so it must fail loudly instead."""
    assert hasattr(gateway, name)
    assert not hasattr(chat_service, name)


def test_the_offline_inventory_default_stubs_the_gateway_check() -> None:
    """The root conftest finds the model check by module name; a stale name would let real discovery run."""
    assert gateway._validate_provider_model_selection.__module__ != gateway.__name__


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
