"""
HyDE (Hypothetical Document Embeddings) utilities.

This module provides helpers to generate a hypothetical answer for a query
using a lightweight LLM and to compute its embedding for use in retrieval.
"""
import asyncio
from collections.abc import Awaitable
from functools import partial
from typing import Any, Callable, Optional

from loguru import logger


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
    prompt = (
        "You are helping with retrieval. Write a concise, factual, neutral "
        "paragraph (2-5 sentences) that likely answers this question. Avoid hedging, "
        "cite plausible entities, metrics, and terminology.\n\n"
        f"Question: {query}\n"
    )
    text = _generate_with_llm(prompt, provider, model)
    if text and len(text.split()) >= 5:
        return text
    # Fallback heuristic
    return f"Summary: An explanation of '{query}' including key facts, definitions, examples, and typical metrics."


def _embedding_provider_from_config(
    user_app_config: dict[str, Any],
    model_id_override: str | None = None,
) -> str | None:
    """Return the provider for the exact embedding model selected by a call."""
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
    handle = await credential_runtime.resolve(provider)
    return handle, {
        "api_key_override": getattr(handle, "api_key", None),
        "base_url_override": _credential_base_url(handle),
        "credentials_resolved": True,
    }


def _record_embedding_degraded(stage_metadata: dict[str, Any] | None, exc: BaseException) -> None:
    """Record only bounded optional-stage failure metadata."""
    if stage_metadata is None:
        return
    code = str(getattr(exc, "code", "") or "")
    if code not in {
        "invalid_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }:
        code = "provider_unavailable"
    stage_metadata.update(
        embedding_coverage="degraded",
        failure_code=code,
    )


async def _run_sync_embedding_call(
    call: Callable[[], Any],
    *,
    on_success: Callable[[Any], Awaitable[None]] | None = None,
) -> Any:
    """Drain a sync embedding thread before propagating caller cancellation."""
    work_task = asyncio.create_task(asyncio.to_thread(call))
    try:
        result = await asyncio.shield(work_task)
    except asyncio.CancelledError:
        while not work_task.done():
            try:
                await asyncio.shield(work_task)
            except asyncio.CancelledError:
                continue
            except Exception:  # noqa: BLE001 - discard failure after cancellation
                break
        if on_success is not None and not work_task.cancelled():
            try:
                result = work_task.result()
            except Exception:  # noqa: BLE001 - cancellation remains authoritative
                result = None
            else:
                mark_task = asyncio.create_task(on_success(result))
                while not mark_task.done():
                    try:
                        await asyncio.shield(mark_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:  # noqa: BLE001 - cancellation remains authoritative
                        break
        raise
    if on_success is not None:
        await on_success(result)
    return result


async def _mark_runtime_used_for_embeddings(
    embeddings: Any,
    *,
    credential_runtime: Any,
    handle: Any,
) -> None:
    """Mark runtime credentials used only after a usable vector is returned."""
    if handle is not None and embeddings and embeddings[0] is not None:
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
        if credential_runtime is not None and provider == "openai":
            handle, call_kwargs = await _resolve_runtime_embedding_call(
                credential_runtime,
                provider,
            )
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
