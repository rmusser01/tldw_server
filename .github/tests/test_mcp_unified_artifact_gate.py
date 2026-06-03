"""Standalone MCP Unified artifact gate without host pytest package imports.

The real artifact assertions live beside the host MCP Unified tests. This shim
loads them by file path so the CI package gate can run with only standalone
``mcp_unified[dev]`` dependencies installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_TESTS_MODULE = "_mcp_unified_runtime_package_boundary"
BOUNDARY_TESTS_PATH = (
    REPO_ROOT
    / "tldw_Server_API"
    / "app"
    / "core"
    / "MCP_unified"
    / "tests"
    / "test_runtime_package_boundary.py"
)


def _load_boundary_tests() -> ModuleType:
    """Load boundary tests by file path so pytest does not import host packages."""

    spec = importlib.util.spec_from_file_location(
        BOUNDARY_TESTS_MODULE,
        BOUNDARY_TESTS_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load {BOUNDARY_TESTS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[BOUNDARY_TESTS_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(BOUNDARY_TESTS_MODULE, None)
        raise
    return module


_boundary_tests = _load_boundary_tests()

standalone_distributions = _boundary_tests.standalone_distributions
test_mcp_unified_standalone_distribution_metadata_matches_extras = (
    _boundary_tests.test_mcp_unified_standalone_distribution_metadata_matches_extras
)
test_mcp_unified_standalone_sdist_contains_only_package_boundary = (
    _boundary_tests.test_mcp_unified_standalone_sdist_contains_only_package_boundary
)
test_mcp_unified_standalone_artifacts_include_typed_marker = (
    _boundary_tests.test_mcp_unified_standalone_artifacts_include_typed_marker
)
test_mcp_unified_standalone_artifacts_include_package_docs = (
    _boundary_tests.test_mcp_unified_standalone_artifacts_include_package_docs
)
