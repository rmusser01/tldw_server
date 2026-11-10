import pytest

from tldw_Server_API.app.core.LLM_Calls.model_metadata import get_model_metadata
from tldw_Server_API.app.core.LLM_Calls.routing.metadata import (
    ROUTING_MODEL_RANKS,
    merge_routing_metadata,
)


@pytest.mark.unit
def test_merge_routing_metadata_applies_rank_overrides_and_capability_flags():
    merged = merge_routing_metadata(
        {
            "capabilities": {
                "tool_use": True,
                "vision": True,
                "json_mode": True,
                "thinking": True,
            },
            "tool_support": False,
            "vision_support": False,
            "json_mode_support": False,
            "reasoning_support": False,
        },
        provider="openai",
        model="gpt-4o",
    )

    assert merged["tool_support"] is True
    assert merged["vision_support"] is True
    assert merged["json_mode_support"] is True
    assert merged["reasoning_support"] is True
    assert merged["quality_rank"] == ROUTING_MODEL_RANKS["openai"]["gpt-4o"]["quality_rank"]
    assert merged["latency_rank"] == ROUTING_MODEL_RANKS["openai"]["gpt-4o"]["latency_rank"]
    assert merged["cost_rank"] == ROUTING_MODEL_RANKS["openai"]["gpt-4o"]["cost_rank"]


@pytest.mark.unit
def test_get_model_metadata_merges_core_routing_metadata():
    metadata = get_model_metadata("openai", "gpt-4o")

    assert metadata["tool_support"] is True
    assert metadata["vision_support"] is True
    assert metadata["json_mode_support"] is True
    assert metadata["quality_rank"] == ROUTING_MODEL_RANKS["openai"]["gpt-4o"]["quality_rank"]
