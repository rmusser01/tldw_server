"""
HyDE (Hypothetical Document Embeddings) utilities.

This module provides helpers to generate a hypothetical answer for a query
using a lightweight LLM and to compute its embedding for use in retrieval.
"""
import asyncio
import math
from collections.abc import Awaitable
from functools import partial
from numbers import Real
from typing import Any, Callable, Optional

from loguru import logger


def _hyde_instruction_prompt() -> str:
    """Build retrieval-oriented HyDE instructions without duplicating input."""
    return (
        "You are helping with retrieval. Write a concise, factual, neutral "
        "paragraph (2-5 sentences) that likely answers this question. Avoid hedging, "
        "cite plausible entities, metrics, and terminology."
    )


def _hyde_prompt(query: str) -> str:
    """Build the legacy prompt that includes the query inline."""
    return f"{_hyde_instruction_prompt()}\n\nQuestion: {query}\n"


def _heuristic_hypothetical_answer(query: str) -> str:
    """Return the deterministic fallback used by optional HyDE generation."""
    return f"Summary: An explanation of '{query}' including key facts, definitions, examples, and typical metrics."


def _hyde_response_text(response: Any) -> str:
    """Coerce supported provider response shapes into HyDE text."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        value = response.get("text") or response.get("content")
        return value if isinstance(value, str) else ""
    return ""


async def _mark_runtime_used_for_hyde(
    response: Any,
    *,
    credential_runtime: Any,
    handle: Any,
) -> None:
    """Mark credentials only after the provider returns usable, non-error text."""
    text = _hyde_response_text(response).strip()
    if not text or text.startswith("Error:"):
        raise RuntimeError("HyDE provider request failed")
    await credential_runtime.mark_used(handle)


def _generate_with_llm(prompt: str, provider: Optional[str], model: Optional[str]) -> Optional[str]:
    """Call the existing LLM utility to generate text.

    Uses Summarization_General_Lib.analyze() if available. Returns None on failure.
    """
    try:
        # Lazy import to avoid startup overhead
        import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

        class _LLMClient:
            def __init__(self, provider: Optional[str], model: Optional[str]):
                self.provider = (provider or "openai").strip()
                self.model = (model or "gpt-4o-mini").strip()

            def generate(self, prompt_text: str) -> str:
                try:
                    resp = sgl.analyze(
                        api_name=self.provider,
                        input_data="",
                        custom_prompt_arg=prompt_text,
                        api_key=None,
                        system_message=None,
                        temp=None,
                        model_override=self.model,
                    )
                    # sgl.analyze returns string content
                    if isinstance(resp, str):
                        return resp
                    # If dict-like, try common fields
                    if isinstance(resp, dict):
                        return resp.get("text") or resp.get("content") or str(resp)
                    return str(resp)
                except Exception:  # pragma: no cover - defensive
                    logger.warning("HyDE LLM generation failed")
                    return ""

        client = _LLMClient(provider, model)
        out = client.generate(prompt)
        return out.strip() if isinstance(out, str) else None
    except Exception:
        logger.debug("HyDE LLM utility unavailable")
        return None


def generate_hypothetical_answer(query: str, provider: Optional[str] = None, model: Optional[str] = None) -> str:
    """Generate a concise hypothetical answer for the query.

    Falls back to a heuristic template if LLM is unavailable.
    """
    text = _generate_with_llm(_hyde_prompt(query), provider, model)
    if text and len(text.split()) >= 5:
        return text
    return _heuristic_hypothetical_answer(query)


async def generate_hypothetical_answer_async(
    query: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    *,
    credential_runtime: Any,
    stage_metadata: dict[str, Any] | None = None,
) -> str:
    """Generate HyDE text with request-scoped credentials and fail closed."""
    effective_provider = (provider or "openai").strip().lower()
    effective_model = model.strip() if isinstance(model, str) and model.strip() else None
    try:
        import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

        handle = await credential_runtime.resolve(effective_provider)
        response = await _run_sync_embedding_call(
            partial(
                sgl.analyze,
                api_name=str(getattr(handle, "provider", effective_provider) or effective_provider),
                input_data=query,
                custom_prompt_arg=_hyde_instruction_prompt(),
                api_key=handle.api_key,
                system_message=None,
                temp=None,
                model_override=effective_model,
                app_config=handle.app_config,
                credentials_resolved=True,
                raise_on_error=True,
            ),
            on_success=partial(
                _mark_runtime_used_for_hyde,
                credential_runtime=credential_runtime,
                handle=handle,
            ),
        )
        text = _hyde_response_text(response)
        if len(text.split()) >= 5:
            return text.strip()
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - optional stage degrades to heuristic
        _record_embedding_degraded(stage_metadata, exc)
        logger.warning("Runtime HyDE generation failed")
    return _heuristic_hypothetical_answer(query)


def _embedding_provider_from_config(
    user_app_config: dict[str, Any],
    model_id_override: str | None = None,
) -> str | None:
    """Return the provider for the exact embedding model selected by a call."""
    raw_config = user_app_config.get("embedding_config") or user_app_config.get("EMBEDDING_CONFIG") or {}
    selected = str(model_id_override or raw_config.get("default_model_id") or "").strip()
    model_spec = _embedding_model_spec_from_config(user_app_config, model_id_override)
    provider = (
        model_spec.get("provider")
        if isinstance(model_spec, dict)
        else getattr(model_spec, "provider", None)
    )
    if provider:
        return str(provider).strip().lower()
    if ":" in selected:
        return selected.split(":", 1)[0].strip().lower() or None
    return None


def _embedding_model_spec_from_config(
    user_app_config: dict[str, Any],
    model_id_override: str | None = None,
) -> Any:
    """Return the exact embedding model configuration selected by a call."""
    raw_config = user_app_config.get("embedding_config") or user_app_config.get("EMBEDDING_CONFIG") or {}
    models = raw_config.get("models") or {}
    selected = str(model_id_override or raw_config.get("default_model_id") or "").strip()
    if not selected:
        return None

    model_spec = models.get(selected)
    if model_spec is None and ":" in selected:
        model_spec = models.get(selected.split(":", 1)[1])
    if model_spec is None:
        bare = selected.split(":", 1)[-1]
        guessed_providers = (["huggingface"] if "/" in bare else []) + [
            "openai",
            "local_api",
        ]
        for guessed_provider in guessed_providers:
            model_spec = models.get(f"{guessed_provider}:{bare}")
            if model_spec is not None:
                break
        if model_spec is None:
            matches = [
                spec for key, spec in models.items() if str(key).endswith(f":{bare}")
            ]
            if len(matches) == 1:
                model_spec = matches[0]

    return model_spec


def _runtime_local_embedding_call_kwargs(
    user_app_config: dict[str, Any],
    provider: str | None,
    model_id_override: str | None = None,
) -> dict[str, Any]:
    """Build a key-free execution boundary for runtime-bound local embeddings."""
    normalized = str(provider or "").strip().lower()
    if normalized not in {"local", "local_api"}:
        return {}
    call_kwargs: dict[str, Any] = {
        "api_key_override": None,
        "credentials_resolved": True,
    }
    if normalized == "local_api":
        from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingEndpointError

        model_spec = _embedding_model_spec_from_config(user_app_config, model_id_override)
        endpoint = (
            model_spec.get("api_url")
            if isinstance(model_spec, dict)
            else getattr(model_spec, "api_url", None)
        )
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise EmbeddingEndpointError(normalized)
        call_kwargs["base_url_override"] = endpoint.strip()
    return call_kwargs


def _credential_base_url(handle: Any) -> str | None:
    """Read the authorized endpoint from a redacted credential handle."""
    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import resolve_provider_section

    app_config = getattr(handle, "app_config", None) or {}
    provider = str(getattr(handle, "provider", "") or "")
    provider_config = app_config.get(resolve_provider_section(provider)) or {}
    for key in ("base_url", "api_base_url", "api_url", "endpoint"):
        value = provider_config.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


async def _resolve_runtime_embedding_call(
    credential_runtime: Any,
    provider: str,
) -> tuple[Any, dict[str, Any]]:
    """Resolve one hosted embedding call into explicit, fail-closed overrides."""
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingCredentialError

    handle = await credential_runtime.resolve(provider)
    api_key = getattr(handle, "api_key", None)
    if not isinstance(api_key, str) or not api_key.strip():
        raise EmbeddingCredentialError(provider)
    return handle, {
        "api_key_override": api_key,
        "base_url_override": _credential_base_url(handle),
        "credentials_resolved": True,
    }


def _record_embedding_degraded(stage_metadata: dict[str, Any] | None, exc: BaseException) -> None:
    """Record only bounded optional-stage failure metadata."""
    if stage_metadata is None:
        return
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Embeddings.async_embeddings import (
        EmbeddingCredentialError,
        EmbeddingEndpointError,
    )
    code = str(getattr(exc, "code", "") or getattr(exc, "error_code", "") or "")
    code = {
        "missing_credentials": "missing_provider_credentials",
        "configuration": "provider_configuration_invalid",
        "authentication": "invalid_provider_credentials",
    }.get(code, code)
    status_code = getattr(exc, "status_code", None)
    if isinstance(exc, EmbeddingCredentialError):
        code = "missing_provider_credentials"
    elif isinstance(exc, EmbeddingEndpointError):
        code = "provider_configuration_invalid"
    elif isinstance(exc, ChatConfigurationError):
        code = str(getattr(exc, "error_code", "") or "provider_configuration_invalid")
    elif code == "authentication" or status_code in {401, 403}:
        code = "invalid_provider_credentials"
    if code not in {
        "invalid_provider_credentials",
        "missing_provider_credentials",
        "provider_configuration_invalid",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }:
        code = "provider_unavailable"
    stage_metadata.update(
        embedding_coverage="degraded",
        failure_code=code,
    )


async def _drain_embedding_task(
    task: asyncio.Task[Any],
) -> tuple[bool, Any]:
    """Drain a shielded task despite repeated cancellation requests."""
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - caller cancellation remains authoritative
            break
    if task.cancelled():
        return False, None
    try:
        return True, task.result()
    except Exception:  # noqa: BLE001 - caller cancellation remains authoritative
        return False, None


async def _run_sync_embedding_call(
    call: Callable[[], Any],
    *,
    on_success: Callable[[Any], Awaitable[None]] | None = None,
) -> Any:
    """Drain provider work and success bookkeeping before cancellation escapes."""
    work_task = asyncio.create_task(asyncio.to_thread(call))
    try:
        result = await asyncio.shield(work_task)
    except asyncio.CancelledError:
        completed, result = await _drain_embedding_task(work_task)
        if completed and on_success is not None:
            mark_task = asyncio.create_task(on_success(result))
            await _drain_embedding_task(mark_task)
        raise
    if on_success is not None:
        mark_task = asyncio.create_task(on_success(result))
        try:
            await asyncio.shield(mark_task)
        except asyncio.CancelledError:
            await _drain_embedding_task(mark_task)
            raise
    return result


async def _mark_runtime_used_for_embeddings(
    embeddings: Any,
    *,
    credential_runtime: Any,
    handle: Any,
) -> None:
    """Mark runtime credentials used only after a usable vector is returned."""
    if handle is None:
        return
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingProviderError

    try:
        vector = embeddings[0]
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, (list, tuple)) or not vector:
            raise ValueError
        if not all(
            not isinstance(value, bool)
            and isinstance(value, Real)
            and math.isfinite(float(value))
            for value in vector
        ):
            raise ValueError
    except (AttributeError, IndexError, OverflowError, TypeError, ValueError):
        raise EmbeddingProviderError(
            str(getattr(handle, "provider", "") or "unknown"),
            code="provider_failure",
        ) from None
    await credential_runtime.mark_used(handle)


async def embed_text(
    text: str,
    *,
    credential_runtime: Any = None,
    stage_metadata: dict[str, Any] | None = None,
) -> Optional[list]:
    """Create an embedding vector for text using the existing embeddings service.

    Returns a Python list (not numpy) for portability.
    """
    try:
        from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
            create_embeddings_batch,
            get_embedding_config,
        )

        cfg = get_embedding_config()
        provider = _embedding_provider_from_config(cfg)
        handle = None
        call_kwargs: dict[str, Any] = {}
        # Hugging Face in this synchronous service is an in-process provider.
        if credential_runtime is not None:
            if provider == "openai":
                handle, call_kwargs = await _resolve_runtime_embedding_call(
                    credential_runtime,
                    provider,
                )
            else:
                call_kwargs = _runtime_local_embedding_call_kwargs(cfg, provider)
        embeddings = await _run_sync_embedding_call(
            partial(
                create_embeddings_batch,
                [text],
                cfg,
                None,
                **call_kwargs,
            ),
            on_success=(
                partial(
                    _mark_runtime_used_for_embeddings,
                    credential_runtime=credential_runtime,
                    handle=handle,
                )
                if handle is not None
                else None
            ),
        )
        if embeddings and embeddings[0] is not None:
            vec = embeddings[0]
            if hasattr(vec, "tolist"):
                maybe_list = vec.tolist()
                if isinstance(maybe_list, list):
                    return maybe_list
                if isinstance(maybe_list, tuple):
                    return list(maybe_list)
                return None
            if isinstance(vec, (list, tuple)):
                return list(vec)
        return None
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_embedding_degraded(stage_metadata, exc)
        logger.warning("HyDE embedding failed")
        return None
