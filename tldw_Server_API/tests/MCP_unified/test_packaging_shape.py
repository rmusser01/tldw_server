from importlib import import_module
from pathlib import Path


def test_mcp_unified_is_packaged_in_tree():
    module = import_module("tldw_Server_API.app.core.MCP_unified")

    assert module.__name__ == "tldw_Server_API.app.core.MCP_unified"
    assert getattr(module, "__version__", None)


def test_active_branch_does_not_depend_on_root_mcp_unified_package():
    repo_root = Path(__file__).resolve().parents[3]

    assert not (repo_root / "mcp_unified").exists()
