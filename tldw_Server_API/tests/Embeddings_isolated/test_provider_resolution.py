from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.provider_resolution import (
    resolve_provider_model,
    split_provider_model,
)
from tldw_Server_API.app.core.Embeddings.request_types import EmbeddingPolicyError


def _fallback_guess(model: str, explicit_provider: str | None = None) -> str:
    del model
    return explicit_provider.lower() if explicit_provider else "fallback-provider"


@pytest.mark.unit
def test_split_provider_model_strips_provider_prefix_and_whitespace():
    assert split_provider_model(" OpenAI : text-embedding-3-small ") == (
        "openai",
        "text-embedding-3-small",
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("text-embedding-3-small", (None, "text-embedding-3-small")),
        (123, (None, "123")),
    ],
)
def test_split_provider_model_returns_no_prefix_for_unqualified_and_non_string(model, expected):
    assert split_provider_model(model) == expected


@pytest.mark.unit
def test_explicit_provider_lowercases_and_wins():
    intent = resolve_provider_model(
        "text-embedding-3-small",
        "OpenAI",
        settings_config={},
        require_model=True,
        guess_provider=_fallback_guess,
    )

    assert intent.provider == "openai"
    assert intent.model == "text-embedding-3-small"
    assert intent.requested_provider == "OpenAI"
    assert intent.requested_model == "text-embedding-3-small"
    assert intent.provider_was_explicit is True
    assert intent.model_was_provider_qualified is False


@pytest.mark.unit
def test_provider_qualified_model_strips_prefix():
    intent = resolve_provider_model(
        "openai:text-embedding-3-small",
        None,
        settings_config={},
        require_model=True,
        guess_provider=_fallback_guess,
    )

    assert intent.provider == "openai"
    assert intent.model == "text-embedding-3-small"
    assert intent.model_was_provider_qualified is True


@pytest.mark.unit
def test_prefix_mismatch_raises_policy_error_with_stable_code_and_message():
    with pytest.raises(EmbeddingPolicyError) as exc_info:
        resolve_provider_model(
            "openai:text-embedding-3-small",
            "huggingface",
            settings_config={},
            require_model=True,
            guess_provider=_fallback_guess,
        )

    assert exc_info.value.code == "provider_model_mismatch"
    assert (
        exc_info.value.message
        == "Model provider prefix 'openai' does not match provider 'huggingface'"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-small-en-v1.5",
        "thenlper/gte-base",
        "intfloat/e5-small-v2",
        "hkunlp/instructor-base",
        "Qwen/Qwen3-Embedding-0.6B",
        "microsoft/mpnet-base",
        "google-bert/bert-base-uncased",
        "facebook/contriever",
        "bert-base-uncased",
        "roberta-base",
        "xlm-roberta-base",
        "distilbert-base-uncased",
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
    ],
)
def test_huggingface_heuristic_patterns(model):
    intent = resolve_provider_model(
        model,
        None,
        settings_config={},
        require_model=True,
        guess_provider=_fallback_guess,
    )

    assert intent.provider == "huggingface"
    assert intent.model == model


@pytest.mark.unit
def test_slash_containing_non_openai_model_resolves_huggingface():
    intent = resolve_provider_model(
        "custom-org/custom-embedding-model",
        None,
        settings_config={},
        require_model=True,
        guess_provider=_fallback_guess,
    )

    assert intent.provider == "huggingface"


@pytest.mark.unit
@pytest.mark.parametrize(
    "model",
    [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
)
def test_openai_model_names_use_injected_fallback_guesser(model):
    intent = resolve_provider_model(
        model,
        None,
        settings_config={},
        require_model=True,
        guess_provider=_fallback_guess,
    )

    assert intent.provider == "fallback-provider"


@pytest.mark.unit
def test_openai_model_names_default_to_openai_without_injected_guesser():
    intent = resolve_provider_model(
        "text-embedding-3-small",
        None,
        settings_config={},
        require_model=True,
    )

    assert intent.provider == "openai"
    assert intent.model == "text-embedding-3-small"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("settings_config", "expected_model", "expected_provider"),
    [
        (
            {"embedding_model": "openai:text-embedding-3-small"},
            "text-embedding-3-small",
            "openai",
        ),
        (
            {"default_model_id": "openai:text-embedding-ada-002"},
            "text-embedding-ada-002",
            "openai",
        ),
        ({}, "sentence-transformers/all-MiniLM-L6-v2", "huggingface"),
    ],
)
def test_compatibility_default_when_model_absent_and_model_not_required(
    settings_config,
    expected_model,
    expected_provider,
):
    intent = resolve_provider_model(
        None,
        None,
        settings_config=settings_config,
        require_model=False,
        guess_provider=_fallback_guess,
    )

    assert intent.model == expected_model
    assert intent.provider == expected_provider


@pytest.mark.unit
@pytest.mark.parametrize("model", [None, "", "   "])
def test_require_model_true_with_missing_or_blank_model_raises_stable_policy_error(model):
    with pytest.raises(EmbeddingPolicyError) as exc_info:
        resolve_provider_model(
            model,
            None,
            settings_config={"embedding_model": "openai:text-embedding-3-small"},
            require_model=True,
            guess_provider=_fallback_guess,
        )

    assert exc_info.value.code == "model_denied"
    assert exc_info.value.message == "Model is required"
