"""TASK-13294 AC#6: filesystem's verbatim-argument exemptions must reach production.

FilesystemModule declares that some arguments are file content and must be byte-exact:
fs.edit's old_string/new_string, fs.write's content, notebook.edit_cell's source. But
tool_execution/security.harden_and_sanitize_tool_arguments ran module.sanitize_input on
the whole argument dict first, with no tool name, so those values arrived already
stripped of control characters (a form feed in C source, an ESC in a terminal fixture)
and the module's own table had nothing left to protect. fs.edit could then never match
an old_string containing such a byte.

The upstream pass now takes the tool name and asks the module which top-level keys are
verbatim. Only str values are passed through; anything else still gets the full
sanitising pass, depth guard included, and then fails schema validation.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
    FilesystemModule,
)
from tldw_Server_API.app.core.MCP_unified.tool_execution.security import ToolExecutionSecurity

pytestmark = pytest.mark.unit

FORM_FEED_SOURCE = "int a;\x0c\nint b;\n"


def _security() -> ToolExecutionSecurity:
    """The upstream pass needs only its noncritical-exception tuple."""
    security = ToolExecutionSecurity.__new__(ToolExecutionSecurity)
    security._noncritical_exceptions = (ValueError, TypeError)
    return security


def _filesystem() -> FilesystemModule:
    """sanitize_input and the verbatim table are pure; skip __init__ side effects."""
    return FilesystemModule.__new__(FilesystemModule)


class _Plain(BaseModule):
    """A module that declares no verbatim arguments."""

    async def on_initialize(self) -> None:  # pragma: no cover - not exercised
        return None

    async def on_shutdown(self) -> None:  # pragma: no cover
        return None

    async def check_health(self) -> dict[str, bool]:  # pragma: no cover
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:  # pragma: no cover
        return []

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any = None) -> None:  # pragma: no cover
        return None


@pytest.mark.parametrize(
    ("tool_name", "key"),
    [("fs.write", "content"), ("fs.edit", "old_string"), ("fs.edit", "new_string"), ("notebook.edit_cell", "source")],
)
def test_declared_verbatim_content_survives_the_upstream_pass(tool_name: str, key: str) -> None:
    args = {"path": "src/a.c", key: FORM_FEED_SOURCE}

    hardened = _security().harden_and_sanitize_tool_arguments(_filesystem(), args, tool_name=tool_name)

    assert hardened[key] == FORM_FEED_SOURCE


def test_non_verbatim_keys_are_still_sanitised() -> None:
    """The exemption is per key: the path beside the content is still stripped."""
    args = {"path": "src/\x0ba.c", "content": FORM_FEED_SOURCE}

    hardened = _security().harden_and_sanitize_tool_arguments(_filesystem(), args, tool_name="fs.write")

    assert hardened["path"] == "src/a.c"
    assert hardened["content"] == FORM_FEED_SOURCE


def test_a_non_string_value_under_a_verbatim_key_is_still_sanitised() -> None:
    """Only str is trusted verbatim; nested data keeps the depth guard and the strip."""
    args = {"path": "a", "content": {"nested": "x\x0cy"}}

    hardened = _security().harden_and_sanitize_tool_arguments(_filesystem(), args, tool_name="fs.write")

    assert hardened["content"] == {"nested": "xy"}


def test_other_tools_on_the_same_module_are_fully_sanitised() -> None:
    """fs.list declares nothing verbatim, so every value is stripped as before."""
    args = {"path": "src/\x0ca"}

    hardened = _security().harden_and_sanitize_tool_arguments(_filesystem(), args, tool_name="fs.list")

    assert hardened == {"path": "src/a"}


def test_modules_without_declarations_are_unchanged() -> None:
    """The base hook returns nothing verbatim, so the other 21 modules behave as before."""
    module = _Plain(ModuleConfig(name="plain"))
    args = {"content": FORM_FEED_SOURCE}

    hardened = _security().harden_and_sanitize_tool_arguments(module, args, tool_name="anything")

    assert hardened == {"content": "int a;\nint b;\n"}


def test_forbidden_ownership_overrides_are_still_removed() -> None:
    """Verbatim handling must not bypass the ownership-override strip."""
    args = {"content": FORM_FEED_SOURCE, "user_id": "someone-else", "db_path": "/tmp/x.db"}

    hardened = _security().harden_and_sanitize_tool_arguments(_filesystem(), args, tool_name="fs.write")

    assert "user_id" not in hardened
    assert "db_path" not in hardened
    assert hardened["content"] == FORM_FEED_SOURCE
