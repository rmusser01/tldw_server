import numpy as np
import pytest

from tldw_Server_API.app.core.Embeddings.embedding_policy import (
    adjust_dimensions,
    decide_and_apply_l2,
    enforce_embedding_policy,
    map_model_for_provider,
    resolve_fallback_chain,
    supports_openai_dimensions,
    validate_dimensions_request,
)
from tldw_Server_API.app.core.Embeddings.provider_resolution import resolve_provider_model
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingPolicyError,
    EmbeddingRequestContext,
    ProviderModelIntent,
)


def _intent(
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    *,
    explicit: bool = False,
) -> ProviderModelIntent:
    return ProviderModelIntent(
        provider=provider,
        model=model,
        requested_provider=provider if explicit else None,
        requested_model=model,
        provider_was_explicit=explicit,
        model_was_provider_qualified=False,
    )


def _context(
    *,
    provider_header: str | None = None,
    dimensions: int | None = None,
    encoding_format: str | None = "float",
) -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="u1",
        model_field="text-embedding-3-small",
        provider_header=provider_header,
        dimensions=dimensions,
        encoding_format=encoding_format,
    )


def test_openai_dimensions_are_limited_to_small_and_large_models():
    assert supports_openai_dimensions("text-embedding-3-small") is True
    assert supports_openai_dimensions("openai:text-embedding-3-large") is True
    assert supports_openai_dimensions("text-embedding-3-experimental") is False


def test_validate_dimensions_rejects_invalid_requests():
    assert validate_dimensions_request("openai", "text-embedding-3-small", 1536) == 1536
    assert validate_dimensions_request("huggingface", "sentence-transformers/all-MiniLM-L6-v2", 4096) == 4096

    with pytest.raises(EmbeddingPolicyError, match="only supported"):
        validate_dimensions_request("openai", "text-embedding-ada-002", 256)
    with pytest.raises(EmbeddingPolicyError, match="positive"):
        validate_dimensions_request("huggingface", "m", 0)
    with pytest.raises(EmbeddingPolicyError, match="<= 4096"):
        validate_dimensions_request("huggingface", "m", 4097)


def test_adjust_dimensions_uses_policy_and_metrics_callback():
    metrics: list[tuple[str, str, str]] = []

    reduced = adjust_dimensions(
        [[1.0, 2.0, 3.0]],
        2,
        "huggingface",
        "m",
        dimension_policy="reduce",
        record_adjustment=lambda provider, model, method: metrics.append((provider, model, method)),
    )
    assert reduced == [[1.0, 2.0]]
    assert metrics == [("huggingface", "m", "reduce")]

    padded = adjust_dimensions([[1.0, 2.0]], 4, "huggingface", "m", dimension_policy="pad")
    assert padded == [[1.0, 2.0, 0.0, 0.0]]

    ignored = adjust_dimensions([[1.0, 2.0, 3.0]], 2, "huggingface", "m", dimension_policy="ignore")
    assert ignored == [[1.0, 2.0, 3.0]]


def test_l2_policy_preserves_base64_and_adapter_defaults():
    base64_arr, base64_l2 = decide_and_apply_l2([3.0, 4.0], "base64", False)
    assert base64_l2 is False
    assert pytest.approx(np.linalg.norm(base64_arr), rel=0.0, abs=1e-6) == 5.0

    numeric_arr, numeric_l2 = decide_and_apply_l2([3.0, 4.0], "float", False)
    assert numeric_l2 is True
    assert pytest.approx(np.linalg.norm(numeric_arr), rel=0.0, abs=1e-6) == 1.0

    adapter_arr, adapter_l2 = decide_and_apply_l2([3.0, 4.0], "float", True)
    assert adapter_l2 is False
    assert pytest.approx(np.linalg.norm(adapter_arr), rel=0.0, abs=1e-6) == 5.0

    forced_arr, forced_l2 = decide_and_apply_l2([3.0, 4.0], "float", True, normalize_requested=True)
    assert forced_l2 is True
    assert pytest.approx(np.linalg.norm(forced_arr), rel=0.0, abs=1e-6) == 1.0


def test_fallback_chain_defaults_and_configured_chain():
    assert resolve_fallback_chain("openai") == ["openai", "huggingface", "onnx", "local_api"]
    assert resolve_fallback_chain("custom") == ["custom"]
    assert resolve_fallback_chain(
        "openai",
        settings_fallback_chain={"openai": ["onnx", "huggingface"]},
    ) == ["openai", "onnx", "huggingface"]
    assert resolve_fallback_chain(
        "openai",
        settings_fallback_chain={"openai": ["openai", "onnx", "huggingface", "onnx"]},
    ) == ["openai", "onnx", "huggingface"]


def test_resolve_provider_model_reports_missing_model_as_validation_error():
    with pytest.raises(EmbeddingPolicyError) as exc_info:
        resolve_provider_model(
            "",
            None,
            settings_config={},
            require_model=True,
        )

    assert exc_info.value.code == "model_required"
    assert exc_info.value.message == "Model is required"


def test_fallback_model_mapping_defaults_and_configured_map():
    assert (
        map_model_for_provider("openai", "huggingface", "text-embedding-3-small")
        == "sentence-transformers/all-MiniLM-L6-v2"
    )
    assert (
        map_model_for_provider(
            "openai",
            "onnx",
            "custom-model",
            settings_fallback_model_map={"openai:custom-model": {"onnx": "mapped-model"}},
        )
        == "mapped-model"
    )


def test_enforce_policy_suppresses_fallback_for_explicit_provider_header():
    decision = enforce_embedding_policy(
        _intent(explicit=True),
        _context(provider_header="openai"),
        allowed_providers=None,
        allowed_models=None,
        implemented_providers={"openai", "huggingface", "onnx", "local_api"},
        enforce_policy=True,
        allow_fallback_with_header=False,
        settings_fallback_chain=None,
        settings_fallback_model_map=None,
    )

    assert decision.provider == "openai"
    assert decision.model == "text-embedding-3-small"
    assert decision.fallback_allowed is False
    assert decision.fallback_chain == ["openai"]


def test_enforce_policy_allows_configured_fallback_when_header_override_is_enabled():
    decision = enforce_embedding_policy(
        _intent(explicit=True),
        _context(provider_header="openai"),
        allowed_providers={"openai", "huggingface"},
        allowed_models={"text-embedding-3-*"},
        implemented_providers={"openai", "huggingface", "onnx", "local_api"},
        enforce_policy=True,
        allow_fallback_with_header=True,
        settings_fallback_chain={"openai": ["huggingface", "onnx"]},
        settings_fallback_model_map=None,
    )

    assert decision.fallback_allowed is True
    assert decision.fallback_chain == ["openai", "huggingface"]


def test_enforce_policy_raises_stable_allowlist_denials():
    with pytest.raises(EmbeddingPolicyError) as provider_exc:
        enforce_embedding_policy(
            _intent(provider="openai"),
            _context(),
            allowed_providers={"huggingface"},
            allowed_models=None,
            implemented_providers={"openai", "huggingface"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )
    assert provider_exc.value.code == "provider_denied"
    assert "Provider 'openai' is not allowed" == provider_exc.value.message

    with pytest.raises(EmbeddingPolicyError) as model_exc:
        enforce_embedding_policy(
            _intent(model="text-embedding-3-large"),
            _context(),
            allowed_providers={"openai"},
            allowed_models={"text-embedding-3-small"},
            implemented_providers={"openai"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )
    assert model_exc.value.code == "model_denied"
    assert "Model 'text-embedding-3-large' is not allowed" == model_exc.value.message


def test_enforce_policy_reports_recognized_unimplemented_provider():
    with pytest.raises(EmbeddingPolicyError) as exc_info:
        enforce_embedding_policy(
            _intent(provider="voyage", model="voyage-3"),
            _context(),
            allowed_providers=None,
            allowed_models=None,
            implemented_providers={"openai", "huggingface"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )

    assert exc_info.value.code == "provider_unsupported"
    assert exc_info.value.message == "Provider 'voyage' not implemented"


def test_enforce_policy_reports_unknown_provider_before_execution_planning():
    with pytest.raises(EmbeddingPolicyError) as exc_info:
        enforce_embedding_policy(
            _intent(provider="unknown", model="custom-model"),
            _context(),
            allowed_providers=None,
            allowed_models=None,
            implemented_providers={"openai", "huggingface"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )

    assert exc_info.value.code == "unknown_provider"
    assert exc_info.value.message == "Unknown provider: unknown"


def test_enforce_policy_classifies_provider_before_dimensions():
    with pytest.raises(EmbeddingPolicyError) as unknown_exc:
        enforce_embedding_policy(
            _intent(provider="unknown", model="custom-model"),
            _context(dimensions=5000),
            allowed_providers=None,
            allowed_models=None,
            implemented_providers={"openai", "huggingface"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )
    assert unknown_exc.value.code == "unknown_provider"

    with pytest.raises(EmbeddingPolicyError) as unsupported_exc:
        enforce_embedding_policy(
            _intent(provider="voyage", model="voyage-3"),
            _context(dimensions=5000),
            allowed_providers=None,
            allowed_models=None,
            implemented_providers={"openai", "huggingface"},
            enforce_policy=True,
            allow_fallback_with_header=False,
            settings_fallback_chain=None,
            settings_fallback_model_map=None,
        )
    assert unsupported_exc.value.code == "provider_unsupported"
