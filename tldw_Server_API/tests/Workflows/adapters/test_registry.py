"""Tests for the adapter registry system."""

import pytest

pytestmark = pytest.mark.unit


def test_all_adapters_registered():
    """Verify all 120 adapters are registered."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    adapters = registry.list_adapters()
    assert len(adapters) >= 120, f"Expected at least 120 adapters, got {len(adapters)}"


def test_all_adapters_have_config_models():
    """Verify all adapters have Pydantic config models."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    missing = []
    for name in registry.list_adapters():
        spec = registry.get_spec(name)
        if spec.config_model is None:
            missing.append(name)

    assert not missing, f"Adapters missing config_model: {missing}"


def test_registry_get_adapter():
    """Test get_adapter returns callable for known adapters."""
    from tldw_Server_API.app.core.Workflows.adapters import get_adapter

    adapter = get_adapter("llm")
    assert adapter is not None
    assert callable(adapter)


def test_json_validate_catalog_preserves_schema_input_key():
    """Catalog consumers receive the public schema key after the Python rename."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    entry = next(item for item in registry.get_catalog()["text"] if item["name"] == "json_validate")
    metadata = entry["config_schema"]
    assert set(metadata["properties"]) == {"timeout_seconds", "save_artifact", "data", "schema", "strict"}
    assert set(metadata["required"]) == {"data", "schema"}


def test_registry_get_adapter_unknown():
    """Test get_adapter returns None for unknown adapter."""
    from tldw_Server_API.app.core.Workflows.adapters import get_adapter

    adapter = get_adapter("nonexistent_adapter_xyz")
    assert adapter is None


def test_get_parallelizable():
    """Test get_parallelizable returns a set of adapter names."""
    from tldw_Server_API.app.core.Workflows.adapters import get_parallelizable

    parallel = get_parallelizable()
    assert isinstance(parallel, (set, frozenset))
    # LLM should be parallelizable
    assert "llm" in parallel
    # Control flow adapters should not be parallelizable
    assert "branch" not in parallel


def test_registry_categories():
    """Test that adapters are grouped by categories."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    expected_categories = {
        "audio",
        "video",
        "media",
        "rag",
        "knowledge",
        "content",
        "text",
        "integration",
        "evaluation",
        "research",
        "utility",
        "control",
        "llm",
    }
    actual_categories = set(registry.get_categories())
    assert expected_categories <= actual_categories, f"Missing categories: {expected_categories - actual_categories}"


def test_adapter_spec_has_required_fields():
    """Test that each adapter spec has required fields."""
    from tldw_Server_API.app.core.Workflows.adapters import registry

    for name in registry.list_adapters():
        spec = registry.get_spec(name)
        assert spec.name == name
        assert spec.category, f"{name} missing category"
        assert spec.description, f"{name} missing description"
        assert spec.func is not None, f"{name} missing func"
        assert callable(spec.func), f"{name} func not callable"


def test_adapter_functions_are_async():
    """Test that all adapter functions are async."""
    import asyncio
    from tldw_Server_API.app.core.Workflows.adapters import registry

    for name in registry.list_adapters():
        spec = registry.get_spec(name)
        assert asyncio.iscoroutinefunction(spec.func), f"{name} is not an async function"


def test_backward_compatible_imports():
    """Test that backward-compatible imports work."""
    from tldw_Server_API.app.core.Workflows.adapters import (
        run_llm_adapter,
        run_tts_adapter,
        run_stt_transcribe_adapter,
        run_summarize_adapter,
        run_webhook_adapter,
        run_branch_adapter,
        run_map_adapter,
        run_rag_search_adapter,
    )

    # All should be callable
    assert callable(run_llm_adapter)
    assert callable(run_tts_adapter)
    assert callable(run_stt_transcribe_adapter)
    assert callable(run_summarize_adapter)
    assert callable(run_webhook_adapter)
    assert callable(run_branch_adapter)
    assert callable(run_map_adapter)
    assert callable(run_rag_search_adapter)
