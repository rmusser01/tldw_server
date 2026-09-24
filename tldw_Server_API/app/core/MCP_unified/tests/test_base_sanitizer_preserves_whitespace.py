"""Regression guard for TASK-13294 (whitespace half).

`BaseModule.sanitize_input` stripped control characters with

    "".join(ch for ch in s if ch >= " " or ch == "\\n")

keeping `\\n` but dropping `\\t` and `\\r`. It is applied to every tool call via
`tool_execution/security.py:harden_and_sanitize_tool_arguments`, and 22 modules
inherit it unchanged, so:

- `fs.write` of a tab-significant file (a Makefile, a TSV) silently lost its tabs and
  reported success. The `expected_sha256` receipt hashes the on-disk pre-image, so the
  integrity guard structurally could not catch it.
- `fs.edit` does exact string replacement, so an `old_string` containing a tab could
  never match a tab-indented file -- the tool was permanently unusable on such files.

The correct whitespace class already existed 700 lines away in the same package:
`filesystem_module.py:_sanitize_patch_diff` preserves `\\n`, `\\r` and `\\t`, with a
docstring saying exactly that.

Scope note: the `dangerous_patterns` denylist in the same function (which rejects
`--`, `/*`, `xp_`, and so ordinary Markdown rules, `src/*.py` pathspecs and
`exp_` filenames) is the *other* half of TASK-13294 and is deliberately NOT changed
here. Removing SQL-looking guards requires confirming that all 22 inheriting modules
bind their arguments to parameterised queries, which is an owner decision rather than
a drive-by edit.
"""

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule

# Suite marker: these are fast, isolated regression guards.
pytestmark = pytest.mark.unit


class _Module(BaseModule):
    """Concrete BaseModule so the inherited sanitizer can be exercised."""

    def __init__(self) -> None:  # bypass BaseModule.__init__ wiring
        pass

    async def on_initialize(self) -> None:
        """No-op: the sanitizer under test needs no module wiring."""

    async def on_shutdown(self) -> None:
        """No-op: nothing is acquired, so nothing is released."""

    async def check_health(self) -> None:
        """No-op: health is irrelevant to input sanitization."""

    async def execute_tool(self, *_args: Any, **_kwargs: Any) -> None:
        """No-op: no tool is invoked; only sanitize_input is exercised."""

    def get_tools(self) -> list[Any]:
        """Return no tools; the suite calls sanitize_input directly."""
        return []


@pytest.fixture
def mod() -> _Module:
    """A concrete BaseModule whose inherited sanitize_input can be called directly."""
    return _Module()


def test_tab_survives_sanitization(mod: _Module) -> None:
    """A Makefile's tab is syntactically required; losing it corrupts the file."""
    makefile = "all:\n\tgcc -o x x.c\n"

    assert mod.sanitize_input(makefile) == makefile, (
        "the tab was stripped -- fs.write would corrupt this file and report success, "
        "and fs.edit could never match it again"
    )


def test_carriage_return_survives_sanitization(mod: _Module) -> None:
    """CRLF content must round-trip; \\r was stripped alongside \\t."""
    crlf = "line one\r\nline two\r\n"
    assert mod.sanitize_input(crlf) == crlf


def test_tsv_content_survives(mod: _Module) -> None:
    """A TSV loses its column structure entirely if tabs are removed."""
    tsv = "name\tvalue\nalpha\t1\nbeta\t2\n"
    assert mod.sanitize_input(tsv) == tsv


def test_real_control_characters_are_still_stripped(mod: _Module) -> None:
    """Controls other than \\n, \\r, \\t must still go, including NUL."""
    assert mod.sanitize_input("a\x00b") == "ab"
    assert mod.sanitize_input("a\x01b\x1fc") == "abc"
    # \x0b (vertical tab) and \x0c (form feed) are not whitelisted
    assert mod.sanitize_input("a\x0bb\x0cc") == "abc"


def test_whitespace_preserved_through_nested_structures(mod: _Module) -> None:
    """Tool arguments arrive as nested dicts/lists, so recursion must preserve it too."""
    payload = {"files": [{"path": "Makefile", "content": "all:\n\tgcc x.c\n"}]}
    assert mod.sanitize_input(payload) == payload


def test_depth_guard_still_raises(mod: _Module) -> None:
    """Control: the abuse guard must be unaffected."""
    nested: object = "leaf"
    for _ in range(25):
        nested = [nested]
    with pytest.raises(ValueError, match="too deeply nested"):
        mod.sanitize_input(nested)


def test_matches_the_filesystem_patch_sanitizer(mod: _Module) -> None:
    """The base class and the diff sanitizer must agree on the whitespace class."""
    from tldw_Server_API.app.core.MCP_unified.modules.implementations import (
        filesystem_module,
    )

    sample = "a\tb\r\nc\x01d"
    patch_sanitizer = filesystem_module.FilesystemModule._sanitize_patch_diff
    assert mod.sanitize_input(sample) == patch_sanitizer(sample)


def test_filesystem_override_preserves_whitespace_too() -> None:
    """FilesystemModule overrides sanitize_input, so the base fix alone is not enough.

    Production argument hardening dispatches through the override, so this is the
    path `fs.write` and `fs.edit` actually take. Fixing only BaseModule left the
    defect live on exactly the tools it was about.
    """
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
        FilesystemModule,
    )

    fs = FilesystemModule.__new__(FilesystemModule)
    makefile = "all:\n\tgcc -o x x.c\n"

    assert fs.sanitize_input(makefile) == makefile, (
        "the FilesystemModule override still strips tabs -- fs.write corrupts "
        "tab-significant files and fs.edit cannot match them"
    )
    assert fs.sanitize_input("a\tb\r\nc") == "a\tb\r\nc"
    # real control characters still removed
    assert fs.sanitize_input("a\x00b\x01c") == "abc"
    # and it agrees with its own patch-diff sanitizer
    sample = "x\ty\r\nz\x01"
    assert fs.sanitize_input(sample) == FilesystemModule._sanitize_patch_diff(sample)
