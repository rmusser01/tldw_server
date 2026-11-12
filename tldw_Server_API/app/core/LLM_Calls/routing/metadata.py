"""Shared routing metadata helpers for model/provider catalogs."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.LLM_Calls.anthropic_model_ids import canonical_anthropic_model_id

ROUTING_MODEL_RANKS: dict[str, dict[str, dict[str, int]]] = {
    "openai": {
        "gpt-4o": {"quality_rank": 20, "latency_rank": 40, "cost_rank": 60},
        "gpt-4o-mini": {"quality_rank": 60, "latency_rank": 10, "cost_rank": 10},
    },
    "anthropic": {
        "claude-opus-4-1": {"quality_rank": 10, "latency_rank": 70, "cost_rank": 90},
        "claude-sonnet-4-5": {"quality_rank": 20, "latency_rank": 30, "cost_rank": 50},
        "claude-haiku-4-5": {"quality_rank": 55, "latency_rank": 10, "cost_rank": 20},
    },
    "google": {
        "gemini-3-pro-preview": {"quality_rank": 25, "latency_rank": 35, "cost_rank": 45},
        "gemini-3-flash-preview": {"quality_rank": 50, "latency_rank": 15, "cost_rank": 15},
    },
}


def merge_routing_metadata(
    metadata: dict[str, Any],
    *,
    provider: str,
    model: str,
    routing_model_ranks: dict[str, dict[str, dict[str, int]]] | None = None,
) -> dict[str, Any]:
    """Return routing metadata with derived capability flags and ranking keys."""

    normalized_provider = (provider or "").strip().lower()
    normalized_model = (model or "").strip()
    if normalized_provider == "anthropic":
        # The ranks are keyed by API IDs; a saved dotted name (claude-sonnet-4.5) finds its dashed entry.
        normalized_model = canonical_anthropic_model_id(normalized_model)
    merged = dict(metadata)
    capabilities = merged.get("capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    ranks = (routing_model_ranks or ROUTING_MODEL_RANKS).get(normalized_provider, {}).get(
        normalized_model,
        {},
    )
    merged.update(ranks)
    merged["tool_support"] = bool(
        merged.get("tool_support")
        or capabilities.get("tool_use")
        or capabilities.get("function_calling")
    )
    merged["vision_support"] = bool(merged.get("vision_support") or capabilities.get("vision"))
    merged["json_mode_support"] = bool(
        merged.get("json_mode_support") or capabilities.get("json_mode")
    )
    merged["reasoning_support"] = bool(
        merged.get("reasoning_support") or capabilities.get("thinking")
    )
    merged.setdefault("quality_rank", None)
    merged.setdefault("latency_rank", None)
    merged.setdefault("cost_rank", None)
    return merged
