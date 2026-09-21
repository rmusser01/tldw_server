"""Resolve QA provider/model defaults without mixing models across providers."""

import os
from typing import Any

from tldw_Server_API.app.core.LLM_Calls.adapter_utils import resolve_provider_model
from tldw_Server_API.app.core.LLM_Calls.provider_identity import canonical_provider_name


def resolve_generation_defaults(
    config: dict[str, Any], provider: str | None = None, model: str | None = None
) -> tuple[str, str | None]:
    """Prefer explicit QA choices, otherwise inherit the configured Chat pair."""
    chat_settings = config.get("llm_api_settings") or config.get("API") or {}
    chat_provider = chat_settings.get("default_api") or config.get("default_api") or "openai"
    configured_rag_provider = str(config.get("RAG_DEFAULT_LLM_PROVIDER") or "").strip()
    configured_provider = canonical_provider_name(configured_rag_provider or chat_provider)
    rag_provider = str(os.getenv("RAG_DEFAULT_LLM_PROVIDER", configured_rag_provider)).strip()
    default_provider = canonical_provider_name(rag_provider or chat_provider)
    # A provider-only environment override cannot inherit another provider's model.
    configured_model = config.get("RAG_DEFAULT_LLM_MODEL") if configured_provider == default_provider else ""
    rag_model = str(os.getenv("RAG_DEFAULT_LLM_MODEL", configured_model or "")).strip()
    selected_provider = canonical_provider_name((provider or "").strip() or default_provider)
    selected_model = (model or "").strip() or (
        rag_model if selected_provider == default_provider else None
    ) or resolve_provider_model(selected_provider, config)
    if not selected_model and selected_provider == "openai":
        selected_model = "gpt-4o-mini"
    return selected_provider, selected_model
