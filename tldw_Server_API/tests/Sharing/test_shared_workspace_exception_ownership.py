"""Architecture contract for shared-workspace domain exception ownership."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_EXCEPTIONS = REPO_ROOT / "tldw_Server_API/app/core/exceptions.py"
FEATURE_MODULES = (
    REPO_ROOT
    / "tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py",
    REPO_ROOT
    / "tldw_Server_API/app/core/Sharing/shared_workspace_chat_service.py",
)
SHARED_WORKSPACE_EXCEPTIONS = {
    "SharedWorkspaceAccessError",
    "SharedWorkspaceNotFound",
    "SharedWorkspaceUnavailable",
    "SharedWorkspaceChatServiceError",
    "SharedWorkspaceSourceScopeInvalid",
    "SharedWorkspaceSourceSubsetRequired",
    "SharedWorkspaceSourceChanged",
    "SharedWorkspaceRetrievalUnavailable",
    "SharedWorkspaceNoRelevantEvidence",
    "SharedWorkspaceChatContextTooLarge",
    "SharedWorkspaceNoProviderConfigured",
    "SharedWorkspaceGenerationFailed",
    "_SharedWorkspaceDataUnavailable",
    "_NonQueryableSource",
}


def _defined_classes(path: Path) -> set[str]:
    """Return top-level class names defined by a Python module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }


def test_shared_workspace_exceptions_are_owned_by_core_exceptions() -> None:
    """Feature services should import, rather than define, domain exceptions."""
    assert _defined_classes(CORE_EXCEPTIONS) >= SHARED_WORKSPACE_EXCEPTIONS
    for feature_module in FEATURE_MODULES:
        assert SHARED_WORKSPACE_EXCEPTIONS.isdisjoint(
            _defined_classes(feature_module)
        )
