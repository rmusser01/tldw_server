import importlib

from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    LLMProviderOverride,
    apply_llm_provider_overrides_to_listing,
    get_override_model_priority,
    set_llm_provider_overrides_cache_for_tests,
    validate_provider_override,
)


def test_apply_overrides_filters_models_and_status() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=False,
                allowed_models=["gpt-4o"],
                api_key_hint="abcd",
            )
        }
    )

    payload = {
        "providers": [
            {
                "name": "openai",
                "enabled": True,
                "models": ["gpt-4o", "gpt-3.5-turbo"],
                "models_info": [
                    {"name": "gpt-4o", "notes": "ok"},
                    {"name": "gpt-3.5-turbo", "notes": "legacy"},
                ],
            }
        ]
    }

    updated = apply_llm_provider_overrides_to_listing(payload)
    provider = updated["providers"][0]
    assert provider["enabled"] is False
    assert provider["models"] == ["gpt-4o"]
    assert provider["models_info"] == [{"name": "gpt-4o", "notes": "ok"}]

    set_llm_provider_overrides_cache_for_tests({})


def _capture_provider_override_warnings(module):
    messages: list[str] = []
    sink_id = module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    return messages, sink_id


def test_parse_override_row_sanitizes_secret_decrypt_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    def fail_decrypt(_payload):
        raise RuntimeError("decrypt failed at /private/provider-secret.key")

    monkeypatch.setattr(module, "loads_envelope", lambda _blob: {"ciphertext": "blob"})
    monkeypatch.setattr(module, "decrypt_byok_payload", fail_decrypt)
    opaque_payload = "opaque-" + "provider-payload"
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        override = module._parse_override_row(
            {"provider": "OpenAI", "secret_blob": opaque_payload}
        )
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert override.api_key is None
    assert "Provider override decrypt failed" in joined
    assert "decrypt failed at" not in joined
    assert "/private/provider-secret.key" not in joined


async def test_refresh_provider_overrides_sanitizes_load_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    async def fail_get_pool():
        raise RuntimeError("provider override DB failed at /private/provider-overrides.db")

    monkeypatch.setattr(module, "get_db_pool", fail_get_pool)
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        overrides = await module.refresh_llm_provider_overrides()
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert overrides == {}
    assert "Failed to load provider overrides" in joined
    assert "provider override DB failed" not in joined
    assert "/private/provider-overrides.db" not in joined


async def test_refresh_provider_overrides_sanitizes_row_parse_warning(monkeypatch) -> None:
    module = importlib.import_module("tldw_Server_API.app.core.AuthNZ.llm_provider_overrides")

    class FakeRepo:
        def __init__(self, _pool):
            pass

        async def ensure_tables(self):
            return None

        async def list_overrides(self):
            return [{"provider": "openai"}]

    def fail_parse(_row):
        raise RuntimeError("provider override row failed at /private/provider-row.json")

    monkeypatch.setattr(module, "AuthnzLLMProviderOverridesRepo", FakeRepo)
    monkeypatch.setattr(module, "_parse_override_row", fail_parse)
    messages, sink_id = _capture_provider_override_warnings(module)

    try:
        overrides = await module.refresh_llm_provider_overrides(pool=object())
    finally:
        module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert overrides == {}
    assert "Failed to parse provider override row" in joined
    assert "provider override row failed" not in joined
    assert "/private/provider-row.json" not in joined


def test_validate_provider_override_blocks_disallowed_model() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                is_enabled=True,
                allowed_models=["gpt-4o"],
            )
        }
    )

    blocked = validate_provider_override("openai", "gpt-3.5-turbo")
    assert blocked is not None
    assert blocked["error_code"] == "model_not_allowed"

    allowed = validate_provider_override("openai", "gpt-4o")
    assert allowed is None

    set_llm_provider_overrides_cache_for_tests({})


def test_get_override_model_priority_reads_routing_rankings() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={
                    "routing": {
                        "model_rankings": {
                            "highest_quality": ["gpt-4.1", "gpt-4.1-mini"],
                        }
                    }
                },
            )
        }
    )

    assert get_override_model_priority("openai", "highest_quality") == [
        "gpt-4.1",
        "gpt-4.1-mini",
    ]

    updated = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models": ["gpt-4.1-mini", "gpt-4.1"],
                    "models_info": [
                        {"name": "gpt-4.1-mini"},
                        {"name": "gpt-4.1"},
                    ],
                }
            ]
        }
    )
    assert updated["providers"][0]["models"] == ["gpt-4.1", "gpt-4.1-mini"]
    assert [
        model["name"] for model in updated["providers"][0]["models_info"]
    ] == ["gpt-4.1", "gpt-4.1-mini"]

    set_llm_provider_overrides_cache_for_tests({})


def test_apply_overrides_sorts_models_info_without_crashing_on_non_dict_entries() -> None:
    set_llm_provider_overrides_cache_for_tests(
        {
            "openai": LLMProviderOverride(
                provider="openai",
                config={
                    "routing": {
                        "model_rankings": {
                            "highest_quality": ["gpt-4.1", "gpt-4.1-mini"],
                        }
                    }
                },
            )
        }
    )

    updated = apply_llm_provider_overrides_to_listing(
        {
            "providers": [
                {
                    "name": "openai",
                    "models_info": [
                        None,
                        {"name": "gpt-4.1-mini"},
                        "broken",
                        {"name": "gpt-4.1"},
                    ],
                }
            ]
        }
    )

    assert [
        model["name"] for model in updated["providers"][0]["models_info"]
    ] == ["gpt-4.1", "gpt-4.1-mini"]

    set_llm_provider_overrides_cache_for_tests({})
