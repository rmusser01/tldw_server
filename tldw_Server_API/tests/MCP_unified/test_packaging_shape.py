from importlib import import_module
from pathlib import Path
from unittest import TestCase


class TestMCPUnifiedPackagingShape(TestCase):
    def test_mcp_unified_is_packaged_in_tree(self):
        module = import_module("tldw_Server_API.app.core.MCP_unified")

        self.assertEqual(module.__name__, "tldw_Server_API.app.core.MCP_unified")
        self.assertTrue(getattr(module, "__version__", None))

    def test_active_branch_does_not_depend_on_root_mcp_unified_package(self):
        repo_root = Path(__file__).resolve().parents[3]

        self.assertFalse((repo_root / "mcp_unified").exists())
