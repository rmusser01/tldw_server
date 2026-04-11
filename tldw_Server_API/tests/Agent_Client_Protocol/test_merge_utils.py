"""Tests for the shared merge_config utility."""

from tldw_Server_API.app.core.Agent_Client_Protocol.merge_utils import merge_config


def test_scalar_override():
    assert merge_config({"model": "old"}, {"model": "new"})["model"] == "new"


def test_scalar_preserved_when_overlay_missing():
    assert merge_config({"model": "old", "budget": 100}, {})["budget"] == 100


def test_dict_merge():
    result = merge_config(
        {"tool_tier_overrides": {"Bash(git:*)": "auto"}},
        {"tool_tier_overrides": {"Bash(rm:*)": "individual"}},
    )
    assert result["tool_tier_overrides"] == {
        "Bash(git:*)": "auto",
        "Bash(rm:*)": "individual",
    }


def test_dict_overlay_overrides_same_key():
    result = merge_config(
        {"tool_tier_overrides": {"Bash(git:*)": "auto"}},
        {"tool_tier_overrides": {"Bash(git:*)": "batch"}},
    )
    assert result["tool_tier_overrides"]["Bash(git:*)"] == "batch"


def test_list_append_dedup():
    result = merge_config(
        {"denied_tools": ["Bash(rm:*)"]},
        {"denied_tools": ["Bash(rm:*)", "Bash(dd:*)"]},
    )
    assert sorted(result["denied_tools"]) == ["Bash(dd:*)", "Bash(rm:*)"]


def test_nested_dict_merge():
    result = merge_config(
        {"nested": {"a": 1, "b": {"c": 2}}},
        {"nested": {"b": {"d": 3}}},
    )
    assert result["nested"]["a"] == 1
    assert result["nested"]["b"] == {"c": 2, "d": 3}


def test_none_overlay_values_skipped():
    result = merge_config({"model": "keep"}, {"model": None})
    assert result["model"] == "keep"


def test_empty_overlay():
    base = {"a": 1, "b": 2}
    assert merge_config(base, {}) == {"a": 1, "b": 2}


def test_non_union_list_key_overrides():
    """List keys not in _UNION_LIST_KEYS are treated as scalars (replaced)."""
    result = merge_config({"tags": ["old"]}, {"tags": ["new"]})
    assert result["tags"] == ["new"]


def test_base_not_mutated():
    """merge_config must not modify the base dict."""
    base = {"model": "gpt-4", "denied_tools": ["Bash(rm:*)"]}
    overlay = {"model": "claude", "denied_tools": ["Bash(dd:*)"]}
    merge_config(base, overlay)
    assert base == {"model": "gpt-4", "denied_tools": ["Bash(rm:*)"]}


def test_union_list_keys_all_supported():
    """All documented union-list keys should merge via append+dedup."""
    for key in (
        "allowed_tools",
        "denied_tools",
        "tool_names",
        "tool_patterns",
        "capabilities",
        "tool_modules",
        "module_ids",
        "allowed_models",
        "denied_models",
    ):
        result = merge_config({key: ["a"]}, {key: ["a", "b"]})
        assert sorted(result[key]) == ["a", "b"], f"Failed for key={key}"
