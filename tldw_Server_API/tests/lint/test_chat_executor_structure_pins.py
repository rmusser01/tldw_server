"""Structure pins for the TASK-13950.25 split of the chat_service executors.

* ``PATCHED_CHAT_SERVICE_NAMES``: every name a test patches on ``chat_service`` (AST scan, R4/R20).
* ``moved_chat_code_violations``: code that a ``c25*`` manifest moves or lifts must read a patched name
  through ``cs.<name>``, never through an import of its own, or the test's patch goes inert.
* A fresh interpreter can import each chat module first (R5's late-import block keeps that true).

Nothing here imports chat code except the subprocess pin.
"""

from __future__ import annotations

import ast
import functools
import importlib.util
import os
import re
import subprocess  # nosec B404  # fixed argv, no shell
import sys
import textwrap
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT

pytestmark = pytest.mark.unit

CHAT_DIR = APP_ROOT / "core" / "Chat"
MANIFEST_DIR = REPO_ROOT / "Helper_Scripts" / "rag_split" / "manifests"
SCAN_ROOTS = (
    REPO_ROOT / "tldw_Server_API" / "tests",
    APP_ROOT / "core" / "MCP_unified" / "tests",
)
MODULE = "chat_service"
STRING_TARGET = re.compile(r"(?:^|\.)chat_service\.(\w+)")
PATCHERS = {"patch", "setattr", "delattr"}
KNOWN_TARGETS = {"perform_chat_api_call", "perform_chat_api_call_async", "log_llm_usage", "get_request_queue", "time", "logger"}
DESTINATIONS = (
    "completion_pipeline",
    "moderation_pipeline",
    "persistence_service",
    "response_processor",
    "streaming_pipeline",
    "tool_execution_service",
    "history_context",
)


# ---- the patch scan ---------------------------------------------------------------------------


def _files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
        elif root.is_dir():
            yield from sorted(root.rglob("*.py"))


def _alias_exprs(tree: ast.AST) -> set[str]:
    """Expressions that name ``chat_service`` here: an import of it, an ``x = <alias>`` binding, or ``*._chat_service``."""
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            out.update(a.asname or a.name for a in n.names if a.name == MODULE)
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.split(".")[-1] == MODULE:
                    out.add(a.asname or a.name)
    assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    grew = True
    while grew:
        grew = False
        for n in assigns:
            if ast.unparse(n.value) in out:
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id not in out:
                        out.add(t.id)
                        grew = True
    return out


def _is_alias(expr: ast.expr, aliases: set[str]) -> bool:
    text = ast.unparse(expr)
    return text in aliases or text.split(".")[-1] == "_chat_service"


def _call_name(func: ast.expr) -> str | None:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None


def _names_in(tree: ast.AST) -> set[str]:
    aliases = _alias_exprs(tree)
    found: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and (_call_name(n.func) in PATCHERS or _call_name(n.func) == "object") and n.args:
            first = n.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                m = STRING_TARGET.search(first.value)
                if m:
                    found.add(m.group(1))
                elif first.value.endswith(MODULE) and len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                    found.add(str(n.args[1].value))
            elif _is_alias(first, aliases) and len(n.args) > 1 and isinstance(n.args[1], ast.Constant):
                if isinstance(n.args[1].value, str):
                    found.add(n.args[1].value)
        elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                if isinstance(t, ast.Attribute) and _is_alias(t.value, aliases):
                    found.add(t.attr)
    return found


def patched_chat_service_names(test_roots: Iterable[Path]) -> frozenset[str]:
    """Names tests patch on ``chat_service``, in every form (prefiltered by the text ``chat_service``)."""
    names: set[str] = set()
    for path in _files(test_roots):
        text = path.read_text(encoding="utf-8", errors="replace")
        if MODULE in text:
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError as exc:
                pytest.fail(f"cannot scan {path}: {exc}", pytrace=False)
            names |= _names_in(tree)
    return frozenset(names)


@functools.cache
def _patched_names() -> frozenset[str]:
    return patched_chat_service_names(SCAN_ROOTS)


def __getattr__(name: str) -> frozenset[str]:
    """``PATCHED_CHAT_SERVICE_NAMES``, computed on first use so an unparsable test file cannot break collection."""
    if name == "PATCHED_CHAT_SERVICE_NAMES":
        return _patched_names()
    raise AttributeError(name)


SEEDED_SCAN = '''
from unittest.mock import patch
import pkg.core.Chat.chat_service as cs_mod
from pkg.core.Chat import chat_service


def test_forms(monkeypatch, mocker):
    alias = chat_service
    again = alias
    with patch("pkg.core.Chat.chat_service.NameA"):
        pass
    mocker.patch("pkg.core.Chat.chat_service.NameB")
    monkeypatch.setattr("pkg.core.Chat.chat_service.NameC", 1)
    monkeypatch.setattr(cs_mod, "NameD", 1)
    setattr(alias, "NameE", 1)
    patch.object(again, "NameF")
    cs_mod.NameG = 1
    monkeypatch.setattr("pkg.core.Chat.chat_service", "NameH", 1)
    monkeypatch.setattr(self._chat_service, "NameI", 1)
    patch("pkg.other_module.NotCounted")
'''


def test_patch_scan_finds_the_known_targets() -> None:
    assert _patched_names() >= KNOWN_TARGETS


def test_patch_scan_sees_each_form(tmp_path: Path) -> None:
    (tmp_path / "test_seeded.py").write_text(SEEDED_SCAN)
    (tmp_path / "test_unrelated.py").write_text("patch('pkg.other.X')\n")
    assert patched_chat_service_names([tmp_path]) == {f"Name{c}" for c in "ABCDEFGHI"}


# ---- moved code reads patched names through cs ------------------------------------------------


def c25_manifest_functions() -> dict[str, set[str]]:
    """``{module: {function names}}`` that the ``c25*`` manifests move, lift, relocate or give an entry."""
    out: dict[str, set[str]] = defaultdict(set)
    for path in sorted(MANIFEST_DIR.glob("c25*.py")):
        manifest: dict = {}
        for n in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MANIFEST" for t in n.targets):
                manifest = ast.literal_eval(n.value)
        for mv in manifest.get("moves", []):
            out[mv["module"]].add(mv["name"])
        for name, module in manifest.get("lift", {}).items():
            out[module].add(name)
        for item in [manifest["entry"]] if manifest.get("entry") else []:
            out[item["module"]].add(item["name"])
        for item in manifest.get("relocate", []):
            out[item["module"]].add(item["name"])
    return dict(out)


def _own_scope_nodes(fn: ast.AST) -> Iterable[ast.AST]:
    """Nodes of ``fn``'s own scope: nested def/lambda/class bodies are skipped (their names are theirs)."""
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            stack.extend(ast.iter_child_nodes(n))


def _local_names(fn: ast.AST) -> set[str]:
    out: set[str] = set()
    for a in ast.walk(fn.args):  # type: ignore[attr-defined]
        if isinstance(a, ast.arg):
            out.add(a.arg)
    for n in _own_scope_nodes(fn):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.alias):
            out.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
    return out


def _bare_reads(scope: ast.AST, names: set[str], outer: set[str]) -> list[ast.Name]:
    """Loads of ``names`` that no enclosing function scope binds, descending into nested scopes."""
    bound = outer | _local_names(scope)
    found: list[ast.Name] = []
    for n in _own_scope_nodes(scope):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in names and n.id not in bound:
            found.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            found += _bare_reads(n, names, bound)
    return found


def moved_chat_code_violations(chat_dir: Path, functions: dict[str, set[str]], patched: Iterable[str]) -> list[str]:
    """Moved functions that read a patched name bare, where the name is bound by an import of the module.

    ``cs.<name>`` is an attribute read, so it never matches. A name the function binds itself is its own.
    """
    patched = set(patched)
    out: list[str] = []
    for module, names in sorted(functions.items()):
        path = chat_dir / f"{module}.py"
        if not path.is_file():
            out.append(f"{module}.py: the manifest names functions in a module that does not exist")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            (a.asname or a.name).split(".")[0]
            for top in tree.body
            for n in ast.walk(top)
            if isinstance(n, (ast.Import, ast.ImportFrom)) and not isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef))
            for a in n.names
        }
        for fn in (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names):
            out += [
                f"{path.name}:{n.lineno}: {fn.name} reads {n.id} bare; read it as cs.{n.id}"
                for n in _bare_reads(fn, patched & imported, set())
            ]
    return out


def test_moved_chat_code_reads_patched_names_through_chat_service() -> None:
    """Vacuous until PR E: no ``c25*`` manifest exists before then, so ``c25_manifest_functions()`` is empty.

    PR E (Task 5) adds the assertion that it is non-empty. The seeded test below proves the check itself.
    """
    assert moved_chat_code_violations(CHAT_DIR, c25_manifest_functions(), _patched_names()) == []


def test_a_bare_patched_name_in_a_moved_def_is_reported(tmp_path: Path) -> None:
    (tmp_path / "mini.py").write_text(
        textwrap.dedent(
            """\
            from pkg.llm import perform_chat_api_call
            from pkg.llm import unpatched_helper
            from pkg.Chat import chat_service as cs


            async def lifted(cst):
                unpatched_helper()
                return await perform_chat_api_call(cst)


            async def lifted_ok(cst):
                return await cs.perform_chat_api_call(cst)


            async def lifted_local(cst, perform_chat_api_call):
                return perform_chat_api_call(cst)


            async def not_moved(cst):
                return perform_chat_api_call(cst)
            """
        )
    )
    patched = {"perform_chat_api_call"}
    got = moved_chat_code_violations(tmp_path, {"mini": {"lifted", "lifted_ok", "lifted_local"}}, patched)
    assert len(got) == 1 and "lifted reads perform_chat_api_call bare" in got[0]
    (tmp_path / "nested.py").write_text(
        "from pkg.llm import perform_chat_api_call\n\n\nasync def outer(cst):\n"
        "    def inner(perform_chat_api_call):\n        perform_chat_api_call = 1\n        return perform_chat_api_call\n"
        "    return await perform_chat_api_call(inner)\n"
    )
    nested = moved_chat_code_violations(tmp_path, {"nested": {"outer"}}, patched)
    assert len(nested) == 1 and "outer reads perform_chat_api_call bare" in nested[0]
    assert moved_chat_code_violations(tmp_path, {"missing": {"f"}}, patched) != []
    assert moved_chat_code_violations(tmp_path, {}, patched) == []


def test_c25_manifest_functions_reads_every_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "c25x.py").write_text(
        "MANIFEST = {'base': 'x', 'moves': [{'name': 'm1', 'module': 'a', 'span': [1, 2]}], 'lift': {'l1': 'b'},\n"
        " 'entry': {'name': 'e1', 'module': 'a', 'doc': 'd'}, 'relocate': [{'name': 'r1', 'module': 'c'}]}\n"
    )
    (tmp_path / "t03_config.py").write_text("MANIFEST = {'moves': [{'name': 'rag', 'module': 'z', 'span': [1, 2]}]}\n")
    monkeypatch.setitem(globals(), "MANIFEST_DIR", tmp_path)
    assert c25_manifest_functions() == {"a": {"m1", "e1"}, "b": {"l1"}, "c": {"r1"}}


# ---- every chat module imports first in a fresh interpreter (R5) ------------------------------

IMPORT_FIRST = """
import importlib, sys
mod = importlib.import_module(sys.argv[1])
cs = importlib.import_module("tldw_Server_API.app.core.Chat.chat_service")
if "cs" in vars(mod):
    assert mod.cs is cs, f"{sys.argv[1]}.cs is not chat_service"
"""


def _fresh_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}}
    env.update(
        PYTHONPATH=str(REPO_ROOT),
        TEST_MODE="true",
        DISABLE_HEAVY_STARTUP="1",
        HTTP_PROXY="http://127.0.0.1:9",
        HTTPS_PROXY="http://127.0.0.1:9",
        NO_PROXY="localhost,127.0.0.1",
    )
    return env


CHAT_PKG = "tldw_Server_API.app.core.Chat"
IMPORT_FIRST_MODULES = (
    *(f"{CHAT_PKG}.{m}" for m in DESTINATIONS),
    f"{CHAT_PKG}.chat_service",
    "tldw_Server_API.app.api.v1.endpoints.chat",
)


@pytest.mark.parametrize("module", IMPORT_FIRST_MODULES)
def test_each_chat_module_imports_first_in_a_fresh_interpreter(module: str) -> None:
    """Import ``module`` first, then ``chat_service``; a module that binds ``cs`` must bind ``chat_service`` itself.

    Trivial until PR E: no destination binds ``cs`` yet, so only the clean first import is checked.
    """
    assert importlib.util.find_spec(module) is not None
    r = subprocess.run(  # noqa: S603  # nosec B603  # sys.executable and a fixed script
        [sys.executable, "-c", IMPORT_FIRST, module],
        cwd=REPO_ROOT,
        env=_fresh_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert r.returncode == 0, r.stderr[-2000:]


def test_an_unparsable_test_file_is_a_named_failure_not_a_collection_error(tmp_path: Path) -> None:
    (tmp_path / "test_bad.py").write_text("import chat_service\ndef (:\n")
    with pytest.raises(pytest.fail.Exception, match=r"cannot scan .*test_bad\.py"):
        patched_chat_service_names([tmp_path])
