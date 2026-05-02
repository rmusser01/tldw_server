"""
HyDE (Hypothetical Document Embeddings) utilities.

This module provides helpers to generate a hypothetical answer for a query
using a lightweight LLM and to compute its embedding for use in retrieval.
"""
from typing import Optional

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


async def embed_text(text: str) -> Optional[list]:
    """Create an embedding vector for text using the existing embeddings service.

    Returns a Python list (not numpy) for portability.
    """
    try:
        import asyncio

        from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
            create_embeddings_batch,
            get_embedding_config,
        )

        cfg = get_embedding_config()
        embeddings = await asyncio.get_event_loop().run_in_executor(
            None,
            create_embeddings_batch,
            [text],
            cfg,
            None,
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
    except Exception:
        logger.warning("HyDE embedding failed")
        return None
