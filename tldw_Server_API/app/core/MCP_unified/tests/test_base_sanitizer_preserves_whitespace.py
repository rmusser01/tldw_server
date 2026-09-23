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


# ---------------------------------------------------------------------------
# TASK-13294, second defect in the same function: the SQL-injection denylist.
# ---------------------------------------------------------------------------


def _base_sanitizer():
    """A concrete BaseModule just for sanitize_input.

    Calling it unbound as `BaseModule.sanitize_input(None, x)` works for a plain
    string but blows up the moment it recurses, because the recursive call goes
    through `self`. Dicts, lists and the depth guard all need a real instance.
    """
    from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule

    class _Probe(BaseModule):
        async def on_initialize(self):  # pragma: no cover - never called
            return None

        async def on_shutdown(self):  # pragma: no cover - never called
            return None

        async def check_health(self):  # pragma: no cover - never called
            return None

        def get_tools(self):  # pragma: no cover - never called
            return []

        async def execute_tool(self, *args, **kwargs):  # pragma: no cover
            return None

    return _Probe.__new__(_Probe)


_ORDINARY_CONTENT = [
    "exp_data.csv",             # contains "xp_"
    "sp_reports.txt",           # contains "sp_"
    "src/*.py",                 # contains "/*"
    "a/*glob*/b",               # contains "/*" and "*/"
    "--- a markdown rule",      # contains "--"
    "git log -- path/to/file",  # git pathspec separator
    "pip install --no-cache-dir",
    "https://xn--bcher-kva.example",  # punycode, contains "--"
    "SELECT 1 -- note",
]


@pytest.mark.parametrize("value", _ORDINARY_CONTENT)
def test_ordinary_content_is_not_rejected(value: str) -> None:
    """The denylist buys nothing and refuses routine input.

    It matched `\';`, `";`, `--`, `/*`, `*/`, `xp_` and `sp_` anywhere in any string,
    on data bound to parameterised queries -- an audit of MCP_unified finds zero
    f-string or %-formatted SQL, so there is no concatenation for these substrings to
    escape into. What it did instead was reject a filename containing "xp_", every
    glob, every markdown rule and every git pathspec, with a protocol InvalidParams
    error on a tool call that was never dangerous.

    `web_tool_base.sanitize_input` already carries this fix for web tools only, and its
    docstring diagnoses the problem in the same terms.
    """
    assert _base_sanitizer().sanitize_input(value) == value, (
        f"{value!r} was rejected or altered by the base sanitizer"
    )


def test_the_base_and_web_sanitizers_agree() -> None:
    """The web override existed only to escape the denylist; pin that they match now."""
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.web_tool_base import (
        CONTROL_CHARS_RE,
    )

    sanitizer = _base_sanitizer()
    for value in [*_ORDINARY_CONTENT, "a\tb\r\nc", "x\x00y\x01z", "plain"]:
        assert sanitizer.sanitize_input(value) == CONTROL_CHARS_RE.sub("", value), (
            f"base and web sanitizers disagree on {value!r}"
        )


def test_control_characters_are_still_stripped() -> None:
    """Control: dropping the denylist must not drop the sanitising."""
    sanitizer = _base_sanitizer()

    assert sanitizer.sanitize_input("a\x00b\x01c\x7f") == "abc"
    assert sanitizer.sanitize_input("keep\ttabs\r\nand newlines") == (
        "keep\ttabs\r\nand newlines"
    )


def test_depth_guard_survives() -> None:
    """Control: the recursion limit is the other thing this function does."""
    nested: object = "leaf"
    for _ in range(25):
        nested = {"k": nested}
    with pytest.raises(ValueError, match="too deeply nested"):
        _base_sanitizer().sanitize_input(nested)


def test_nested_structures_are_still_sanitised() -> None:
    """Control: recursion into dicts and lists must keep working."""
    payload = {"path": "src/*.py", "items": ["a\x01b", {"deep": "--flag"}]}
    assert _base_sanitizer().sanitize_input(payload) == {
        "path": "src/*.py",
        "items": ["ab", {"deep": "--flag"}],
    }


# ---------------------------------------------------------------------------
# TASK-13294, third defect: four subclass overrides that outlived the denylist.
#
# web_tool_base, filesystem_module, sandbox_module and run_command_module each
# carried a sanitize_input override whose only reason to exist was escaping the
# base denylist -- their docstrings say so ("allowing portable glob syntax",
# "allowing CLI flags like `--help`", "allows CLI-style args and comment tokens").
# With the denylist gone the base does that for free, and each override was by
# then a strictly worse copy of it: every one let DEL through, sandbox_module kept
# only "\n" (so it stripped tabs and carriage returns from every payload), and
# run_command_module stripped carriage returns. Deleting them removes the drift
# that let the base be fixed while fs.write stayed broken.
# ---------------------------------------------------------------------------

_OVERRIDE_MODULES = [
    (
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.web_tool_base",
        "WebToolBase",
    ),
    (
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module",
        "FilesystemModule",
    ),
    (
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.sandbox_module",
        "SandboxModule",
    ),
    (
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.run_command_module",
        "RunCommandModule",
    ),
]


def _module_class(module_path: str, class_name: str) -> type:
    import importlib

    return getattr(importlib.import_module(module_path), class_name)


def _sanitizer_for(cls: type):
    """An instance of `cls` good enough to call sanitize_input on.

    WebToolBase is itself abstract, so it needs a trivial concrete subclass; the
    others can be allocated directly. Either way __init__ is skipped -- sanitize_input
    touches no instance state.
    """
    if getattr(cls, "__abstractmethods__", frozenset()):
        concrete = type(
            f"_Concrete{cls.__name__}",
            (cls,),
            {name: (lambda self, *a, **k: None) for name in cls.__abstractmethods__},
        )
        return concrete.__new__(concrete)
    return cls.__new__(cls)


@pytest.mark.parametrize(("module_path", "class_name"), _OVERRIDE_MODULES)
def test_no_module_shadows_the_base_sanitizer(module_path: str, class_name: str) -> None:
    """One definition, so a fix to it cannot be shadowed by a stale copy."""
    cls = _module_class(module_path, class_name)
    assert "sanitize_input" not in vars(cls), (
        f"{class_name} defines its own sanitize_input again. The base already strips "
        "control characters while preserving \\t, \\n and \\r, and carries no denylist "
        "to escape; a second copy only drifts. If a module genuinely needs a different "
        "rule, parameterise the base -- see TASK-13294."
    )


@pytest.mark.parametrize(("module_path", "class_name"), _OVERRIDE_MODULES)
def test_every_module_preserves_load_bearing_whitespace(module_path: str, class_name: str) -> None:
    """Tabs and carriage returns survive on every module, not just the base.

    sandbox_module's override kept only "\\n", so a Makefile or TSV passed inline to
    sandbox.exec lost every tab and the tool reported success. run_command_module's
    dropped "\\r", corrupting CRLF payloads the same way.
    """
    sanitizer = _sanitizer_for(_module_class(module_path, class_name))

    payload = "target:\n\tgcc -o x x.c\r\n"
    assert sanitizer.sanitize_input(payload) == payload, (
        f"{class_name} mangled tab/CR-significant content"
    )


@pytest.mark.parametrize(("module_path", "class_name"), _OVERRIDE_MODULES)
def test_every_module_strips_delete(module_path: str, class_name: str) -> None:
    """DEL (\\x7f) is a control character, and `ch >= " "` let all four keep it."""
    sanitizer = _sanitizer_for(_module_class(module_path, class_name))

    assert sanitizer.sanitize_input("a\x7fb") == "ab", f"{class_name} kept DEL"


def test_patch_diff_sanitizer_matches_the_base() -> None:
    """_sanitize_patch_diff is the one remaining second copy; keep it derived.

    Its own comment said it had to match BaseModule.sanitize_input, which is exactly
    the constraint a hand-copied predicate cannot hold -- it kept DEL after the base
    stopped.
    """
    from tldw_Server_API.app.core.MCP_unified.modules.base import CONTROL_CHARS_RE
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.filesystem_module import (
        FilesystemModule,
    )

    for value in ["--- a/x\n+++ b/x\n@@\n-\tone\n+\ttwo\r\n", "a\x7fb", "x\x00y", "plain"]:
        assert FilesystemModule._sanitize_patch_diff(value) == CONTROL_CHARS_RE.sub("", value), (
            f"_sanitize_patch_diff disagrees with the base on {value!r}"
        )
