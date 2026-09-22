"""The base sanitizer must not corrupt or reject ordinary content.

BaseModule.sanitize_input runs on EVERY tool call, via
tool_execution/security.harden_and_sanitize_tool_arguments, and 22 modules inherit it
unchanged. Two defects made it hostile to normal input:

* it stripped every character below U+0020 except "\\n", so tabs and carriage returns
  were silently removed. fs.write then wrote a Makefile whose recipe had lost its
  required tab and reported success, and fs.edit could never match a tab-indented
  old_string, making it permanently unusable on such files.
* a SQL-injection denylist rejected "--", "/*", "*/", "xp_" and "sp_" anywhere in any
  string, so ordinary content was refused with a security-flavoured error: a Markdown
  "---" rule, "SELECT 1 -- note", a git pathspec "-- src/app.py", the glob "src/*.py",
  and the filename "exp_data.csv".

The existing base-sanitizer test used os.urandom(4).hex() as its "safe" fixture, which
can never contain a denied substring, so nothing asserted that legitimate content
survives. That is what these cases add.
"""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig


class _Probe(BaseModule):
    async def on_initialize(self) -> None:  # pragma: no cover - not exercised
        return None

    async def on_shutdown(self) -> None:  # pragma: no cover
        return None

    async def check_health(self) -> dict[str, bool]:  # pragma: no cover
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:  # pragma: no cover
        return []

    async def execute_tool(self, tool_name, arguments, context=None):  # pragma: no cover
        return None


@pytest.fixture
def sanitize():
    return _Probe(ModuleConfig(name="probe")).sanitize_input


# --- whitespace preservation ------------------------------------------------------

def test_tabs_survive(sanitize) -> None:
    """A Makefile recipe without its tab is a broken Makefile."""
    makefile = "all:\n\tgcc -o x x.c\n"
    assert sanitize(makefile) == makefile


def test_carriage_returns_survive(sanitize) -> None:
    assert sanitize("a\r\nb") == "a\r\nb"


def test_tab_indented_python_round_trips(sanitize) -> None:
    src = "def f():\n\treturn 1\n"
    assert sanitize(src) == src


def test_real_nul_and_other_control_chars_are_still_stripped(sanitize) -> None:
    assert sanitize("a\x00b") == "ab"
    assert sanitize("a\x07b") == "ab"      # BEL
    assert sanitize("a\x1bb") == "ab"      # ESC


# --- content that must not be rejected --------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "Heading\n---\nbody",              # Markdown horizontal rule
        "SELECT 1 -- note",                # a comment in sample SQL
        "wait--what",
        "-- src/app.py",                   # git pathspec
        "run tests in src/*.py",           # a glob
        "/* keep this */",                 # a C comment
        "exp_data.csv",                    # contains "xp_"
        "resp_body.json",                  # contains "sp_"
        "a';b",                            # an apostrophe followed by a semicolon
    ],
)
def test_ordinary_content_is_not_rejected(sanitize, text: str) -> None:
    assert sanitize(text) == text


# --- structure and guards unchanged -----------------------------------------------

def test_nested_structures_still_recurse(sanitize) -> None:
    out = sanitize({"a": ["x\x00y", {"b": "c\td"}]})
    assert out == {"a": ["xy", {"b": "c\td"}]}


def test_depth_guard_still_raises(sanitize) -> None:
    deep: Any = "leaf"
    for _ in range(25):
        deep = {"k": deep}
    with pytest.raises(ValueError, match="deeply nested"):
        sanitize(deep)


def test_non_strings_pass_through(sanitize) -> None:
    assert sanitize(42) == 42
    assert sanitize(None) is None
    assert sanitize(True) is True


# --- the four ex-overrides now share one implementation ----------------------------
#
# filesystem, run_command, sandbox and web_tool_base each carried a near-identical
# sanitize_input whose only purpose was to escape the SQL denylist, and each had
# drifted to a different whitespace class. filesystem and sandbox kept only "\n", so
# fs.write still ate the Makefile tab even after the base itself was fixed -- the
# override shadowed the fix. They are gone; these cases keep them gone.

def _ex_override_modules():
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
        FilesystemModule,
    )
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module import (
        RunCommandModule,
    )
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.sandbox_module import (
        SandboxModule,
    )
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_tool_base import (
        WebToolBase,
    )

    return [FilesystemModule, RunCommandModule, SandboxModule, WebToolBase]


def test_no_module_shadows_the_base_sanitizer() -> None:
    for cls in _ex_override_modules():
        assert cls.sanitize_input is BaseModule.sanitize_input, (
            f"{cls.__name__} re-declares sanitize_input; the base is the one "
            "implementation, and a private copy will drift from it again"
        )


def test_filesystem_module_preserves_a_makefile_recipe_tab() -> None:
    """The reported corruption path, asserted end to end through the real module."""
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
        FilesystemModule,
    )

    # sanitize_input is pure; skip the module's __init__ side effects.
    module = FilesystemModule.__new__(FilesystemModule)
    makefile = "all:\n\tgcc -o x x.c\n"
    assert module.sanitize_input({"content": makefile})["content"] == makefile
