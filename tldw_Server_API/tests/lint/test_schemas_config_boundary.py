"""Guard: api/v1/schemas must not read config at import time (TASK-13950.14).

Two measurements, each frozen by an allowlist that may only shrink:

* ``schema_config_calls``: calls made at import time in ``api/v1/schemas`` that reach
  ``core.config``. "Import time" is everything outside ``def``, ``async def`` and
  ``lambda`` bodies: module and class bodies, decorators, argument defaults, and
  module-level ``if``/``try``. A call is flagged when its callee is reached through a
  name bound from a config module (``tldw_Server_API.app.core.config``, or any module
  named ``...core.config.*`` or ``...core.config_*``), or when the callee is a
  module-level function of the same file that makes such a call, directly or
  transitively (fixpoint). Bindings are collected file-wide, function-level imports
  included, so the matcher fails closed. Forms: ``from <config> import x [as y]``,
  ``from tldw_Server_API.app.core import config [as c]``, ``import <config> as c`` and
  un-aliased ``import <config>`` (dotted prefix). Relative imports are resolved.
  Attribute calls count through their root name (``c.settings.get(...)``).
* ``core_chat_request_schemas_importers``: core modules (not ``tests`` dirs) that import
  ``api.v1.schemas.chat_request_schemas`` at any level.

Allowlist semantics: a finding not in the allowlist fails (growth); an allowlisted key
that is no longer found fails as stale (delete it); every reason must cite
``TASK-<n>`` or ``#<n>`` and carry no unfilled ``<...>``.

Known limits: a config load reached through an imported non-config helper is not seen
(importing ``core.LLM_Calls.routing.models`` loads config today); ``importlib``,
``getattr`` and module rebinding are not seen either. Only calls count: a bare read
such as ``X = settings.FOO`` or ``settings["a"]`` at import time is not flagged (no
schema does that today), and method calls on call results (``M().g()``) or a shadowed
name are not followed.

``python -m tldw_Server_API.tests.lint.test_schemas_config_boundary`` prints both
measurements; it is the committed counter.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

_PKG = "tldw_Server_API.app"
_CONFIG = f"{_PKG}.core.config"
_TARGET = f"{_PKG}.api.v1.schemas.chat_request_schemas"
_SCHEMAS = f"{_PKG}.api.v1.schemas"

IMPORT_TIME_CONFIG_ALLOWLIST: dict[str, str] = {
    "api/v1/schemas/chat_validators.py": (
        "MAX_REQUEST_SIZE reads [Chat-Module] max_request_size_chars once at import, and tests patch and import the "
        "constant; config reads move to call time under TASK-13950.21"
    ),
}
_SCHEMA_MODELS = "imports schema models; TASK-13950.15 moves them to core"
CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS: dict[str, str] = {
    "core.Chat.chat_helpers": _SCHEMA_MODELS,
    "core.Chat.chat_service": _SCHEMA_MODELS,
    "core.Prompt_Management.prompt_improvement_dispatch": _SCHEMA_MODELS,
}


def _is_config(name: str) -> bool:
    return name == _CONFIG or name.startswith((_CONFIG + ".", _CONFIG + "_"))


def _resolve(node: ast.ImportFrom, module: str) -> str:
    if not node.level:
        return node.module or ""
    base = module.split(".")[: -node.level]
    return ".".join(base + ([node.module] if node.module else []))


def _dotted(func: ast.expr) -> tuple[str | None, str]:
    """(root name, dotted source) of a callee, walking through attributes, calls, subscripts."""
    parts: list[str] = []
    node = func
    while True:
        if isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        elif isinstance(node, (ast.Call, ast.Subscript)):
            parts.clear()
            node = node.func if isinstance(node, ast.Call) else node.value
        else:
            break
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return node.id, ".".join(reversed(parts))
    return None, ast.unparse(func)


def _import_time_nodes(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield every node outside function/lambda bodies (decorators and defaults included)."""
    stack = list(ast.iter_child_nodes(tree))
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            stack.extend(node.decorator_list)
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
        elif isinstance(node, ast.Lambda):
            stack.extend(node.args.defaults)
            stack.extend(d for d in node.args.kw_defaults if d is not None)
        else:
            stack.extend(ast.iter_child_nodes(node))


def import_time_config_calls(tree: ast.Module, module: str) -> list[tuple[int, str]]:
    """``(line, callee source)`` for each import-time call that reaches a config module."""
    names: set[str] = set()
    prefixes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _resolve(node, module)
            for alias in node.names:
                if _is_config(base) or _is_config(f"{base}.{alias.name}"):
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if _is_config(alias.name):
                    if alias.asname:
                        names.add(alias.asname)
                    else:
                        prefixes.add(alias.name)

    def reaches_config(call: ast.Call) -> bool:
        root, dotted = _dotted(call.func)
        return root in names or any(dotted == p or dotted.startswith(p + ".") for p in prefixes)

    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    tainted: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, fn in funcs.items():
            if name in tainted:
                continue
            for call in (n for n in ast.walk(fn) if isinstance(n, ast.Call)):
                root, dotted = _dotted(call.func)
                if reaches_config(call) or dotted in tainted:
                    tainted.add(name)
                    changed = True
                    break

    found = []
    for node in _import_time_nodes(tree):
        if isinstance(node, ast.Call):
            _, dotted = _dotted(node.func)
            if reaches_config(node) or dotted in tainted:
                found.append((node.lineno, ast.unparse(node.func)))
    return sorted(found)


def _module_name(path: Path, root: Path) -> str:
    return ".".join([_PKG, *path.relative_to(root).with_suffix("").parts])


def schema_config_calls(root: Path = APP_ROOT) -> dict[str, list[tuple[int, str]]]:
    out: dict[str, list[tuple[int, str]]] = {}
    schemas = root.resolve() / "api" / "v1" / "schemas"
    for path, tree in parse_tree(root).items():
        if not path.is_relative_to(schemas):
            continue
        calls = import_time_config_calls(tree, _module_name(path, root.resolve()))
        if calls:
            out[path.relative_to(root.resolve()).as_posix()] = calls
    return out


def core_chat_request_schemas_importers(root: Path = APP_ROOT) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    core = root.resolve() / "core"
    for path, tree in parse_tree(root).items():
        if not path.is_relative_to(core) or "tests" in path.relative_to(core).parts:
            continue
        module = _module_name(path, root.resolve())
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                hit = any(a.name == _TARGET or a.name.startswith(_TARGET + ".") for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = _resolve(node, module)
                hit = base == _TARGET or base.startswith(_TARGET + ".") or (
                    base == _SCHEMAS and any(a.name == "chat_request_schemas" for a in node.names)
                )
            else:
                continue
            if hit:
                lines.append(node.lineno)
        if lines:
            key = module.removeprefix(_PKG + ".").removesuffix(".__init__")
            out[key] = sorted(lines)
    return out


_REASON_REF = re.compile(r"(TASK-\d+|#\d+)")


def problems(found: Mapping[str, object], allowlist: Mapping[str, str]) -> list[str]:
    out = [f"{k}: found but not allowlisted" for k in sorted(found) if k not in allowlist]
    out += [f"{k}: stale allowlist entry, no longer found; delete it" for k in sorted(allowlist) if k not in found]
    for k, reason in sorted(allowlist.items()):
        if not _REASON_REF.search(reason) or re.search(r"<[^>]*>", reason):
            out.append(f"{k}: reason needs a TASK-<n> or #<n> reference and no unfilled <...>")
    return out


def _seed(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "app"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


@pytest.mark.parametrize("source", [
    "from tldw_Server_API.app.core.config import load_and_log_configs\nCFG = load_and_log_configs()\n",
    "from tldw_Server_API.app.core import config as c\nCFG = c.load_and_log_configs()\n",
    "import tldw_Server_API.app.core.config as c\nX = c.settings.get('k')\n",
    "import tldw_Server_API.app.core.config\nX = tldw_Server_API.app.core.config.load_comprehensive_config()\n",
    "from tldw_Server_API.app.core.config_llm_providers import get_api_keys\nKEYS = get_api_keys()\n",
    "def _k():\n    from tldw_Server_API.app.core.config import load_and_log_configs\n    return load_and_log_configs()\n\nK = _k()\n",
    "from tldw_Server_API.app.core.config import settings\nclass M:\n    n: int = int(settings.get('n', 1))\n",
    "from tldw_Server_API.app.core.config import load_and_log_configs\ndef f(cfg=load_and_log_configs()):\n    return cfg\n",
])
def test_every_import_time_config_call_form_is_seen(tmp_path, source):
    assert list(schema_config_calls(_seed(tmp_path, {"api/v1/schemas/x.py": source}))) == ["api/v1/schemas/x.py"]


def test_call_time_reads_and_pure_helpers_are_not_flagged(tmp_path):
    source = (
        "from tldw_Server_API.app.core.config import load_and_log_configs\n"
        "from tldw_Server_API.app.core.custom_openai_providers import iter_custom_openai_provider_names\n"
        "NAMES = list(iter_custom_openai_provider_names())\n"
        "def f():\n    return load_and_log_configs()\n"
        "g = lambda: load_and_log_configs()\n"
        "class M:\n    def m(self):\n        return load_and_log_configs()\n"
    )
    assert schema_config_calls(_seed(tmp_path, {"api/v1/schemas/x.py": source})) == {}


@pytest.mark.parametrize("source", [
    "from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
    "def f():\n    from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
    "from tldw_Server_API.app.api.v1.schemas import chat_request_schemas\n",
    "import tldw_Server_API.app.api.v1.schemas.chat_request_schemas as s\n",
    "from ...api.v1.schemas.chat_request_schemas import ChatCompletionRequest\n",
])
def test_every_core_import_form_of_chat_request_schemas_is_seen(tmp_path, source):
    assert list(core_chat_request_schemas_importers(_seed(tmp_path, {"core/X/y.py": source}))) == ["core.X.y"]


def test_unlisted_stale_and_unreasoned_entries_fail():
    assert [p.split(":")[0] for p in problems({"a": [1]}, {})] == ["a"]
    assert "stale" in problems({}, {"b": "why (TASK-1)"})[0]
    assert "reason" in problems({"a": [1]}, {"a": "because"})[0]
    assert "reason" in problems({"a": [1]}, {"a": "see TASK-1 <fill me>"})[0]
    assert problems({"a": [1]}, {"a": "why (TASK-1)"}) == []


def test_schemas_modules_do_not_load_config_at_import():
    assert problems(schema_config_calls(), IMPORT_TIME_CONFIG_ALLOWLIST) == []


def test_core_imports_of_chat_request_schemas_only_shrink():
    assert problems(core_chat_request_schemas_importers(), CHAT_REQUEST_SCHEMAS_CORE_IMPORTERS) == []


if __name__ == "__main__":
    print("Schema import-time config calls:")
    for key, calls in schema_config_calls().items():
        print(f"  {key}: {calls}")
    print("Core importers of chat_request_schemas:")
    for key, lines in core_chat_request_schemas_importers().items():
        print(f"  {key}: {lines}")
