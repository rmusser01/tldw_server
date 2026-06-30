"""Tests for the root PyPI artifact content guard."""

from Helper_Scripts.Packaging import check_pypi_artifacts as guard


def test_normalize_name_converts_windows_and_absolute_paths() -> None:
    """Archive member names normalize to relative POSIX-style paths."""
    assert (
        guard._normalize_name(r".\apps\tldw-frontend\package.json")
        == "apps/tldw-frontend/package.json"
    )
    assert (
        guard._normalize_name(r"/apps\tldw-frontend\.next\BUILD_ID")
        == "apps/tldw-frontend/.next/BUILD_ID"
    )


def test_blocked_paths_uses_exact_components() -> None:
    """The guard blocks real Node/WebUI paths without substring false positives."""
    blocked = guard._blocked_paths(
        [
            "apps/tldw-frontend/package.json",
            "tldw_Server_API/node_modules/file.js",
            "docs/node_modules_connector.py",
            "apps/tldw-frontend-dev/README.md",
        ]
    )

    assert blocked == [
        "apps/tldw-frontend/package.json",
        "tldw_Server_API/node_modules/file.js",
    ]


def test_missing_required_roots_accepts_sdist_src_layout() -> None:
    """Required roots may appear below the sdist source-layout prefix."""
    assert (
        guard._missing_required_roots(
            [
                "apps/mcp-unified/src/mcp_unified/__init__.py",
                "tldw_Server_API/__init__.py",
            ]
        )
        == []
    )


def test_missing_required_roots_rejects_substring_only_matches() -> None:
    """Package-root checks use exact path components, not substrings."""
    assert guard._missing_required_roots(
        [
            "apps/mcp-unified/src/not_mcp_unified/__init__.py",
            "tldw_Server_API_extra/__init__.py",
        ]
    ) == ["tldw_Server_API", "mcp_unified"]
