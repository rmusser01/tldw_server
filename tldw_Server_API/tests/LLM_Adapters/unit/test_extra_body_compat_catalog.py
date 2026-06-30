from tldw_Server_API.app.core.LLM_Calls.extra_body_compat_catalog import (
    get_model_extra_body_compat,
    get_provider_extra_body_compat,
)


def test_known_provider_returns_supported_shape():
    data = get_provider_extra_body_compat("openai")
    assert isinstance(data["known_params"], list)
    assert "logit_bias" in data["known_params"]
    assert "min_p" in data["known_params"]
    assert "penalize_nl" in data["known_params"]
    assert "source" in data


def test_model_level_shape_present():
    data = get_model_extra_body_compat("openai", "gpt-4o-mini")
    assert isinstance(data["known_params"], list)
    assert "effective_reason" in data


def test_runtime_strict_context_disables_nonstandard_support():
    data = get_model_extra_body_compat(
        "openai",
        "gpt-4o-mini",
        runtime_context={"strict_openai_compat": True},
    )
    assert data["supported"] is False


def test_unknown_provider_returns_safe_fallback():
    data = get_provider_extra_body_compat("definitely-unknown-provider")
    assert data["supported"] is False
    assert data["known_params"] == []
    assert data["example"] == {"extra_body": {}}


def test_provider_aliases_resolve_to_catalog_entries():
    custom_openai = get_provider_extra_body_compat("custom-openai-api")
    custom_openai2 = get_provider_extra_body_compat("custom_openai_api_2")
    custom_openai99 = get_provider_extra_body_compat("custom-openai-api-99")
    llama_cpp = get_provider_extra_body_compat("llama.cpp")
    openai = get_provider_extra_body_compat("openai")

    assert custom_openai["supported"] is True
    assert custom_openai2["supported"] is True
    assert custom_openai99["supported"] is True
    assert llama_cpp["supported"] is True
    assert "logit_bias" in llama_cpp["known_params"]
    assert "post_sampling_probs" in llama_cpp["known_params"]
    assert "post_sampling_probs" not in openai["known_params"]


def test_model_lookup_works_with_provider_aliases():
    data = get_model_extra_body_compat("custom-openai-api", "any-model")
    assert data["supported"] is True
    assert "known_params" in data
