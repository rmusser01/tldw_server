"""Tests for the shared merge_utils module.

Covers: scalar override, dict merge, union-list append+dedup, nested merge,
None skip, and non-union list replacement.
"""
from __future__ import annotations

import copy

import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.merge_utils import (
    UNION_LIST_KEYS,
    merge_config,
)


# ---------------------------------------------------------------------------
# 1. Scalar override
# ---------------------------------------------------------------------------


class TestScalarOverride:
    def test_string_override(self) -> None:
        base = {"approval_mode": "require"}
        overlay = {"approval_mode": "auto"}
        result = merge_config(base, overlay)
        assert result["approval_mode"] == "auto"

    def test_int_override(self) -> None:
        base = {"max_retries": 3}
        overlay = {"max_retries": 5}
        result = merge_config(base, overlay)
        assert result["max_retries"] == 5

    def test_bool_override(self) -> None:
        base = {"enabled": False}
        overlay = {"enabled": True}
        result = merge_config(base, overlay)
        assert result["enabled"] is True

    def test_new_key_added(self) -> None:
        base = {"a": 1}
        overlay = {"b": 2}
        result = merge_config(base, overlay)
        assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# 2. Dict merge (recursive)
# ---------------------------------------------------------------------------


class TestDictMerge:
    def test_nested_dicts_merge(self) -> None:
        base = {"tool_tier_overrides": {"Read(*)": "auto", "Write(*)": "batch"}}
        overlay = {"tool_tier_overrides": {"Write(*)": "individual", "Bash(*)": "auto"}}
        result = merge_config(base, overlay)
        assert result["tool_tier_overrides"] == {
            "Read(*)": "auto",
            "Write(*)": "individual",
            "Bash(*)": "auto",
        }

    def test_deeply_nested_merge(self) -> None:
        base = {"level1": {"level2": {"a": 1, "b": 2}}}
        overlay = {"level1": {"level2": {"b": 3, "c": 4}}}
        result = merge_config(base, overlay)
        assert result["level1"]["level2"] == {"a": 1, "b": 3, "c": 4}

    def test_overlay_dict_replaces_scalar(self) -> None:
        """When base has a scalar and overlay has a dict, overlay wins."""
        base = {"x": "old"}
        overlay = {"x": {"nested": True}}
        result = merge_config(base, overlay)
        assert result["x"] == {"nested": True}

    def test_overlay_scalar_replaces_dict(self) -> None:
        """When base has a dict and overlay has a non-dict, overlay wins."""
        base = {"x": {"nested": True}}
        overlay = {"x": "flat"}
        result = merge_config(base, overlay)
        assert result["x"] == "flat"


# ---------------------------------------------------------------------------
# 3. Union-list append + dedup
# ---------------------------------------------------------------------------


class TestUnionListAppendDedup:
    @pytest.mark.parametrize("key", sorted(UNION_LIST_KEYS))
    def test_all_union_keys_append(self, key: str) -> None:
        base = {key: ["a", "b"]}
        overlay = {key: ["b", "c"]}
        result = merge_config(base, overlay)
        assert result[key] == ["a", "b", "c"]

    def test_allowed_tools_dedup(self) -> None:
        base = {"allowed_tools": ["web.search", "file.read"]}
        overlay = {"allowed_tools": ["file.read", "db.query"]}
        result = merge_config(base, overlay)
        assert result["allowed_tools"] == ["web.search", "file.read", "db.query"]

    def test_union_list_from_string(self) -> None:
        base = {"capabilities": "read"}
        overlay = {"capabilities": "write"}
        result = merge_config(base, overlay)
        assert result["capabilities"] == ["read", "write"]

    def test_union_list_empty_base(self) -> None:
        base: dict = {}
        overlay = {"denied_tools": ["rm", "kill"]}
        result = merge_config(base, overlay)
        assert result["denied_tools"] == ["rm", "kill"]


# ---------------------------------------------------------------------------
# 4. Nested merge (combines dict + union-list inside nested)
# ---------------------------------------------------------------------------


class TestNestedMerge:
    def test_nested_with_union_list(self) -> None:
        base = {
            "policy": {
                "allowed_tools": ["a"],
                "mode": "strict",
            },
        }
        overlay = {
            "policy": {
                "allowed_tools": ["b"],
                "timeout": 30,
            },
        }
        result = merge_config(base, overlay)
        assert result["policy"]["allowed_tools"] == ["a", "b"]
        assert result["policy"]["mode"] == "strict"
        assert result["policy"]["timeout"] == 30


# ---------------------------------------------------------------------------
# 5. None values in overlay are skipped
# ---------------------------------------------------------------------------


class TestNoneSkip:
    def test_none_value_preserves_base(self) -> None:
        base = {"approval_mode": "require", "max_retries": 3}
        overlay = {"approval_mode": None, "max_retries": 5}
        result = merge_config(base, overlay)
        assert result["approval_mode"] == "require"
        assert result["max_retries"] == 5

    def test_none_for_missing_key_does_not_add(self) -> None:
        base = {"a": 1}
        overlay = {"b": None}
        result = merge_config(base, overlay)
        assert "b" not in result

    def test_none_union_list_preserves_base(self) -> None:
        base = {"allowed_tools": ["x"]}
        overlay = {"allowed_tools": None}
        result = merge_config(base, overlay)
        assert result["allowed_tools"] == ["x"]


# ---------------------------------------------------------------------------
# 6. Non-union list is replaced (not appended)
# ---------------------------------------------------------------------------


class TestNonUnionListReplaced:
    def test_regular_list_replaced(self) -> None:
        base = {"tags": ["old1", "old2"]}
        overlay = {"tags": ["new1"]}
        result = merge_config(base, overlay)
        assert result["tags"] == ["new1"]


# ---------------------------------------------------------------------------
# 7. Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_base_not_mutated(self) -> None:
        base = {"allowed_tools": ["a"], "nested": {"x": 1}}
        original = copy.deepcopy(base)
        merge_config(base, {"allowed_tools": ["b"], "nested": {"y": 2}})
        assert base == original

    def test_overlay_not_mutated(self) -> None:
        overlay = {"nested": {"x": 1}}
        original = copy.deepcopy(overlay)
        merge_config({"nested": {"y": 2}}, overlay)
        assert overlay == original
