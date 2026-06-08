"""Tests for authored MCP path-grant policy compilation."""

from __future__ import annotations


def test_compile_hierarchical_path_grants_returns_flat_runtime_grants() -> None:
    from mcp_unified.profiles.path_grants import compile_hierarchical_path_grants

    result = compile_hierarchical_path_grants(
        {
            "org": [
                {"prefix": ".", "actions": ["read"]},
            ],
            "workspace": [
                {"prefix": "documents", "actions": ["read", "edit", "write"]},
            ],
            "folders": [
                {"path": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
                {"path": "documents/private", "actions": ["write"], "effect": "deny"},
            ],
            "files": [
                {"path": "downloads/report.md", "actions": "read"},
            ],
        }
    )

    assert result.has_errors is False
    assert result.path_grants == [
        {"prefix": ".", "actions": ["read"], "effect": "allow"},
        {"prefix": "documents", "actions": ["edit", "read", "write"], "effect": "allow"},
        {"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
        {"prefix": "downloads/report.md", "actions": ["read"], "effect": "allow"},
    ]
    assert result.preview[2] == {
        "prefix": "documents/private",
        "actions": ["edit", "write"],
        "effect": "deny",
        "source": "folders[0]",
        "level": "folder",
    }
    assert result.diagnostics == []


def test_compile_hierarchical_path_grants_reports_invalid_rules() -> None:
    from mcp_unified.profiles.path_grants import compile_hierarchical_path_grants

    result = compile_hierarchical_path_grants(
        {
            "workspace": [
                {"prefix": "/etc", "actions": ["read"]},
                {"prefix": "docs/../secrets", "actions": ["read"]},
                {"prefix": "C:secrets", "actions": ["read"]},
                {"prefix": "docs", "actions": ["share"]},
                {"prefix": "docs", "actions": ["read"], "effect": "prompt"},
            ]
        }
    )

    assert result.path_grants == []
    assert result.has_errors is True
    assert [item["code"] for item in result.diagnostics] == [
        "invalid_prefix",
        "invalid_prefix",
        "invalid_prefix",
        "invalid_actions",
        "invalid_effect",
    ]


def test_compile_policy_path_grants_prefers_explicit_flat_grants() -> None:
    from mcp_unified.profiles.path_grants import compile_policy_path_grants

    result = compile_policy_path_grants(
        {
            "path_grants": [
                {"prefix": "flat", "actions": ["read"]},
            ],
            "path_grant_authoring": {
                "workspace": [
                    {"prefix": "authored", "actions": ["write"]},
                ]
            },
        }
    )

    assert result.path_grants == [{"prefix": "flat", "actions": ["read"], "effect": "allow"}]
    assert result.preview[0]["source"] == "path_grants[0]"
