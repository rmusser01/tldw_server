"""Every mapper of an admin provider-override refusal knows model_denied (TASK-15582)."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import (
    character_chat_sessions,
    chat,
    chunking,
    embeddings_v5_production_enhanced,
    messages,
)
from tldw_Server_API.app.api.v1.endpoints.audio import audio
from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import (
    prompt_studio_evaluations,
    prompt_studio_prompts,
    prompt_studio_test_cases,
)
from tldw_Server_API.app.core.exceptions import ChatConfigurationError, ProviderCredentialTerminalError
from tldw_Server_API.app.core.LLM_Calls.provider_results import normalize_provider_stream_error
from tldw_Server_API.app.core.Provider_Credentials.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.RAG.rag_service.streaming_executor import classify_rag_provider_error
from tldw_Server_API.app.core.TTS.tts_jobs_worker import _bounded_tts_credential_failure

pytestmark = pytest.mark.unit


def _denied() -> ByokResolutionError:
    """A denied model at the credential boundary, shaped as ProviderOverridePolicyError carries it."""
    error = ByokResolutionError("invalid_provider_credentials", "openai")
    error.policy_code = "model_denied"
    return error


def _http(exception: Any) -> tuple[int, str]:
    """The status and code of a mapper's HTTPException."""
    return exception.status_code, exception.detail["error_code"]


def _rag(error: BaseException) -> tuple[int, str]:
    """RAG's (code, status, message) as (status, code)."""
    code, status, _message = classify_rag_provider_error(error)
    return status, code


def _stream() -> tuple[int, str]:
    """A refusal raised mid-call, normalized for a stream."""
    normalized = normalize_provider_stream_error(ChatConfigurationError("refused", error_code="model_denied"))
    return normalized.status_code, normalized.code


_MAPPERS = {
    "chat": lambda: _http(chat._provider_credential_http_exception(_denied())),
    "messages": lambda: _http(messages._messages_credential_http_exception(_denied())),
    "character-chat": lambda: _http(character_chat_sessions._character_credential_http_exception(_denied())),
    "chunking": lambda: _http(chunking._chunking_credential_http_exception(_denied())),
    "embeddings": lambda: _http(embeddings_v5_production_enhanced._embeddings_credential_http_exception(_denied())),
    "tts": lambda: _http(audio._tts_credential_http_exception(_denied())),
    "prompt-studio-evaluations": lambda: _http(prompt_studio_evaluations._prompt_studio_credential_http_exception(_denied())),
    "prompt-studio-prompts": lambda: _http(prompt_studio_prompts._credential_http_exception(_denied())),
    "prompt-studio-test-cases": lambda: _http(prompt_studio_test_cases._credential_http_exception(_denied())),
    "rag-resolution": lambda: _rag(_denied()),
    "rag-adapter": lambda: _rag(ChatConfigurationError("refused", error_code="model_denied")),
    "evaluations": lambda: _http(
        evaluations_unified._eval_provider_http_exception(ChatConfigurationError("refused", error_code="model_denied"))
    ),
    "stream": _stream,
}


@pytest.mark.parametrize("mapper", sorted(_MAPPERS))
def test_every_mapper_answers_a_denied_model_with_403(mapper: str) -> None:
    assert _MAPPERS[mapper]() == (403, "model_denied")


def test_a_denied_model_keeps_its_code_through_every_error_type() -> None:
    from tldw_Server_API.app.core.LLM_Calls.provider_results import ADMIN_POLICY_CODES, ADMIN_REFUSAL_STATUS
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import ProviderOverridePolicyError

    assert (
        ProviderOverridePolicyError("model_denied", "openai").policy_code,
        ChatConfigurationError("refused", error_code="model_denied").error_code,
        ProviderCredentialTerminalError("model_denied").code,
        _bounded_tts_credential_failure(_denied()).failure_code,
        sorted(ADMIN_POLICY_CODES),
        ADMIN_REFUSAL_STATUS["model_denied"],
    ) == (
        "model_denied", "model_denied", "model_denied", "model_denied",
        ["model_denied", "model_not_allowed", "provider_disabled"], 403,
    )


def test_the_allowlist_refusal_says_the_users_own_key_may_serve_the_model() -> None:
    """TASK-15582: a model off allowed_models is only not offered on the server's key."""
    from tldw_Server_API.app.core.LLM_Calls.provider_results import PROVIDER_STREAM_ERROR_MESSAGES

    assert (PROVIDER_STREAM_ERROR_MESSAGES["model_not_allowed"], PROVIDER_STREAM_ERROR_MESSAGES["model_denied"]) == (
        "The selected model is not offered on the server's key for this provider. Your own provider key may serve it.",
        "The selected model is disabled by administrator policy.",
    )
