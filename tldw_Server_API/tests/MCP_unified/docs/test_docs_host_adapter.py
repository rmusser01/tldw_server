from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.MCP_unified.adapters.docs.config import (
    docs_scope_from_context,
    docs_settings_from_module_config,
)
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

pytestmark = pytest.mark.unit


def test_docs_settings_from_module_config_keeps_locked_down_defaults(tmp_path: Path) -> None:
    config = ModuleConfig(
        name="docs",
        settings={"db_path": str(tmp_path / "docs.db"), "trusted_roots": [str(tmp_path)]},
    )

    settings = docs_settings_from_module_config(config)

    assert settings.db_path == tmp_path / "docs.db"  # nosec B101
    assert settings.trusted_roots == (tmp_path.resolve(),)  # nosec B101
    assert settings.enable_web_acquisition is False  # nosec B101
    assert settings.web_source_profile == "locked_down"  # nosec B101


def test_docs_scope_from_request_context_maps_user_and_profile() -> None:
    context = RequestContext(
        request_id="docs-scope",
        user_id="user-1",
        client_id="unit",
        metadata={"profile_scope": "profile-1"},
    )

    scope = docs_scope_from_context(context)

    assert scope.owner_scope == "user-1"  # nosec B101
    assert scope.profile_scope == "profile-1"  # nosec B101


def test_docs_scope_from_missing_context_uses_public_scope() -> None:
    scope = docs_scope_from_context(None)

    assert scope.owner_scope is None  # nosec B101
    assert scope.profile_scope is None  # nosec B101
