"""Ratchet: test patches of endpoint modules do not grow (TASK-13950.20).

A test that patches a name on an endpoint module pins the endpoint's import list: moving the
collaborator, or the code that calls it, breaks the test. This freezes those patches per endpoint
module in endpoint_monkeypatch_baseline.json, each a ceiling
``{"monkeypatch": int, "other": int, "reason": ...}``.

Counting. Every ``*.py`` under tldw_Server_API with a ``tests`` directory component (in-app tests
packages included; conftest and helper modules too) is parsed, file by file, if its bytes contain
``endpoints``. Each patch site (``patch_sites``) counts against the endpoint module its receiver
names (``endpoint_module_key``: the longest prefix that is a module or package file under
``app/api/<version>/endpoints/``, the package ``__init__`` included):
  * monkeypatch: ``<x>.setattr(...)`` and ``<x>.delattr(...)``, ``<x>`` not the name ``object`` or
    ``builtins``;
  * other: ``patch(...)``, ``<x>.patch(...)``, ``patch.object(...)``, ``<x>.patch.object(...)``, and
    attribute targets of an assignment, augmented assignment or ``del`` (``chat_ep.x = 1``).
String form: the first positional argument is a str literal with a dot, ``"<receiver>.<attr>"``;
it counts for ``setattr`` with exactly 2 positional arguments, ``delattr`` with exactly 1, and
``patch`` with any number. Object form: any other first positional argument, for ``setattr``,
``delattr`` and ``patch.object``; its receiver is that argument's dotted module path.

Scope rule (``ScopeResolver``). A name resolves in the innermost function or lambda holding the
reference, then outward to the module (class bodies are transparent; decorators, defaults and
annotations belong to the enclosing scope). Bindings are imports, assignments (``del`` and loop,
``with`` and ``except`` targets included) and parameters. In the first scope that binds the name,
the binding with the greatest line number at or before the reference is used (else the scope's
first). Only module bindings give a path: ``import a.b`` (``a``), ``import a.b as n``, an absolute
``from a import b [as n]``, ``n = [<x>.]import_module("<literal>")`` and
``n = <x>.modules["<literal>"]``. Any other binding (a parameter, a relative import, a call's result,
tuple unpacking) is unresolved, and so is anything it shadows.

Exit metric. Only ``monkeypatch`` counts toward the exit (``other`` is a ceiling too, but no target).
Phase 2 of TASK-13950.20 drains it from SEED, the total on dev when this ratchet landed (5809 on
343e7f304c), to EXIT_TARGET, 55% of it, or below.

Known limits. These escape the count: the builtin ``setattr(ep, ...)``; ``patch.dict``,
``patch.multiple`` and ``monkeypatch.setitem(<endpoint>.X, ...)``; a module object supplied by a
fixture, a helper's return value, tuple unpacking or a parameter; attribute chains through another
module's import (``runner_client.sandbox_ep._service``); control-flow-dependent bindings (the rule is
lexical); targets built with ``getattr`` or an f-string. Comprehension variables and
``global``/``nonlocal`` names bind as if in the enclosing function.

Growth fails: patch the collaborator where the endpoint looks it up (PATCHES.fix). Each drain PR
locks its gain in with ``--tighten`` and pastes ``--summary`` into its body:
``python -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet --tighten``
(``--write-baseline`` records every current measurement; ``--summary`` prints the exit metric).
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections import defaultdict
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT

pytestmark = pytest.mark.unit

PATCHES = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("endpoint_monkeypatch_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_endpoint_monkeypatch_ratchet",
    fix=(
        "Patch the collaborator where the endpoint looks it up: its home module as a string target, "
        "app.dependency_overrides for a dependency, or monkeypatch.setattr(app_services(), 'db_pool', fake) "
        "and the like for a container service (TASK-13950.20). Patches of endpoint modules only fall."
    ),
    stale="No test patches this endpoint module any more.",
)
SEED: dict[str, int] = {"monkeypatch": 5809, "other": 937}  # the dev tip 343e7f304c, after #3433
EXIT_TARGET = SEED["monkeypatch"] * 55 // 100


# --- Name resolution ----------------------------------------------------------------------------

_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


class _Scope:
    __slots__ = ("parent", "bindings")

    def __init__(self, parent: _Scope | None) -> None:
        self.parent = parent
        self.bindings: dict[str, list[tuple[int, int, str | None]]] = defaultdict(list)  # (line, col, path)


def _module_literal(value: ast.AST | None) -> str | None:
    """``"x"`` for ``[<m>.]import_module("x")`` or ``<m>.modules["x"]``, else None."""
    if isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Constant):
        func = value.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name == "import_module" and isinstance(value.args[0].value, str):
            return value.args[0].value
    if (
        isinstance(value, ast.Subscript)
        and isinstance(value.value, ast.Attribute)
        and value.value.attr == "modules"
        and isinstance(value.slice, ast.Constant)
        and isinstance(value.slice.value, str)
    ):
        return value.slice.value
    return None


class ScopeResolver:
    """Resolves a name at a node to the dotted module path it is bound to; see the module docstring."""

    def __init__(self, tree: ast.Module) -> None:
        self._tree = tree  # keeps the nodes, and so the ids _scope_of is keyed by, alive
        self._module = _Scope(None)
        self._scope_of: dict[int, _Scope] = {}
        scopes = [self._module]
        literals: dict[int, str] = {}  # id(Name target) -> module literal of its assignment
        stack: list[tuple[ast.AST, _Scope]] = [(tree, self._module)]  # iterative: long BinOp chains nest deep
        while stack:
            node, scope = stack.pop()
            self._scope_of[id(node)] = scope
            if isinstance(node, _FUNCTIONS):
                scopes.append(inner := _Scope(scope))
                stack.extend(self._enter_function(node, scope, inner))
                continue
            line = getattr(node, "lineno", 0)
            col = getattr(node, "col_offset", 0)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.partition(".")[0]
                    self._bind(scope, bound, line, col, alias.name if alias.asname else bound)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        path = f"{node.module}.{alias.name}" if not node.level else None
                        self._bind(scope, alias.asname or alias.name, line, col, path)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and (literal := _module_literal(node.value)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                literals.update((id(target), literal) for target in targets if isinstance(target, ast.Name))
            elif isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                self._bind(scope, node.id, line, col, literals.get(id(node)))
            elif isinstance(node, ast.ExceptHandler) and node.name:
                self._bind(scope, node.name, line, col, None)
            stack.extend((child, scope) for child in ast.iter_child_nodes(node))
        for scope in scopes:
            for bindings in scope.bindings.values():
                bindings.sort(key=lambda binding: binding[:2])

    def _enter_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda, scope: _Scope, inner: _Scope
    ) -> Iterator[tuple[ast.AST, _Scope]]:
        """Bind the parameters in ``inner``; decorators, defaults and annotations stay in ``scope``."""
        args = node.args
        params = [*args.posonlyargs, *args.args, *args.kwonlyargs, *filter(None, (args.vararg, args.kwarg))]
        for param in params:
            self._bind(inner, param.arg, param.lineno, param.col_offset, None)
            self._scope_of[id(param)] = inner
            if param.annotation is not None:
                yield param.annotation, scope
        yield from ((default, scope) for default in [*args.defaults, *args.kw_defaults] if default is not None)
        if isinstance(node, ast.Lambda):
            yield node.body, inner
            return
        yield from ((decorator, scope) for decorator in node.decorator_list)
        if node.returns is not None:
            yield node.returns, scope
        yield from ((stmt, inner) for stmt in node.body)

    @staticmethod
    def _bind(scope: _Scope, name: str, line: int, col: int, path: str | None) -> None:
        scope.bindings[name].append((line, col, path))

    def resolve(self, name: str, at: ast.AST) -> str | None:
        """The dotted module path ``name`` is bound to where ``at`` stands, or None."""
        scope: _Scope | None = self._scope_of.get(id(at), self._module)
        line = getattr(at, "lineno", 0)
        while scope is not None:
            bindings = scope.bindings.get(name)
            if bindings:
                preceding = [binding for binding in bindings if binding[0] <= line]
                return (preceding[-1] if preceding else bindings[0])[2]
            scope = scope.parent
        return None

    def dotted(self, expr: ast.AST) -> str | None:
        """``a.b.c`` for an ``ast.Name`` or an ``ast.Attribute`` chain on one whose name resolves."""
        attrs = []
        while isinstance(expr, ast.Attribute):
            attrs.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name):
            return None
        base = self.resolve(expr.id, expr)
        return ".".join([base, *reversed(attrs)]) if base is not None else None


# --- The counter --------------------------------------------------------------------------------


class PatchSite(NamedTuple):
    node: ast.AST  # the Call, or the Attribute target of an assignment or del
    field: str  # "monkeypatch" | "other"
    verb: str  # "setattr" | "delattr" | "patch" | "patch.object" | "assign"
    form: str  # "str" | "obj"
    receiver: str | None  # dotted path of what is patched, without the attribute
    attr: str | None  # the attribute, when it is a str literal


def _verb(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return "patch" if func.id == "patch" else None
    if not isinstance(func, ast.Attribute):
        return None
    if func.attr in ("setattr", "delattr"):
        receiver = func.value
        return None if isinstance(receiver, ast.Name) and receiver.id in ("object", "builtins") else func.attr
    if func.attr == "patch":
        return "patch"
    inner = func.value
    if func.attr == "object" and (
        (isinstance(inner, ast.Name) and inner.id == "patch") or (isinstance(inner, ast.Attribute) and inner.attr == "patch")
    ):
        return "patch.object"
    return None


_STRING_ARITY = {"setattr": 2, "delattr": 1}


def _call_site(call: ast.Call, resolver: ScopeResolver) -> PatchSite | None:
    verb = _verb(call.func)
    if verb is None or not call.args:
        return None
    field = "monkeypatch" if verb in _STRING_ARITY else "other"
    arg0 = call.args[0]
    if (
        verb != "patch.object"
        and isinstance(arg0, ast.Constant)
        and isinstance(arg0.value, str)
        and "." in arg0.value
        and _STRING_ARITY.get(verb, len(call.args)) == len(call.args)
    ):
        receiver, name = arg0.value.rsplit(".", 1)
        return PatchSite(call, field, verb, "str", receiver, name)
    if verb == "patch":
        return None
    arg1 = call.args[1] if len(call.args) > 1 else None
    attr = arg1.value if isinstance(arg1, ast.Constant) and isinstance(arg1.value, str) else None
    return PatchSite(call, field, verb, "obj", resolver.dotted(arg0), attr)


def patch_sites(tree: ast.Module) -> Iterator[PatchSite]:
    """Every counted patch shape in ``tree``, wherever its receiver points; see the module docstring."""
    resolver = ScopeResolver(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (site := _call_site(node, resolver)) is not None:
                yield site
            continue
        if isinstance(node, (ast.Assign, ast.Delete)):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Attribute):
                yield PatchSite(target, "other", "assign", "obj", resolver.dotted(target.value), target.attr)


def iter_test_modules(repo_root: Path = REPO_ROOT) -> Iterator[Path]:
    """Every ``*.py`` under ``repo_root/tldw_Server_API`` with a ``tests`` directory component, sorted."""
    package = repo_root / "tldw_Server_API"
    for path in sorted(package.rglob("*.py")):
        if "tests" in path.relative_to(package).parts[:-1]:
            yield path


@cache
def endpoint_module_key(dotted: str, repo_root: Path = REPO_ROOT) -> str | None:
    """The app-relative path of the longest prefix of ``dotted`` that is an endpoint module, or None."""
    parts = dotted.split(".")
    if parts[:3] != ["tldw_Server_API", "app", "api"] or len(parts) < 5 or parts[4] != "endpoints":
        return None
    app = repo_root / "tldw_Server_API" / "app"
    for end in range(len(parts), 4, -1):
        base = app.joinpath(*parts[2:end])
        module = [base.with_name(base.name + ".py")] if end > 5 else []  # api/<version>/endpoints.py is not under it
        for candidate in (*module, base / "__init__.py"):
            if candidate.is_file():
                return candidate.relative_to(app).as_posix()
    return None


def measure_endpoint_patches(repo_root: Path = REPO_ROOT) -> dict[str, dict[str, int]]:
    """``{endpoint module: {"monkeypatch": n, "other": m}}`` over the test modules; see the module docstring."""
    found: dict[str, dict[str, int]] = defaultdict(lambda: {"monkeypatch": 0, "other": 0})
    for path in iter_test_modules(repo_root):
        source = path.read_bytes()
        if b"endpoints" not in source:
            continue
        for site in patch_sites(ast.parse(source, filename=str(path))):
            if site.receiver and (key := endpoint_module_key(site.receiver, repo_root)):
                found[key][site.field] += 1
    return dict(sorted(found.items()))


def summary(measured: Mapping[str, Mapping[str, int]]) -> str:
    """The exit metric on one line."""
    monkeypatch = sum(entry["monkeypatch"] for entry in measured.values())
    other = sum(entry["other"] for entry in measured.values())
    seed = SEED["monkeypatch"]
    return (
        f"monkeypatch {monkeypatch} ({100 * monkeypatch / seed:.1f}% of seed {seed}; "
        f"Phase 2 exit at or below {EXIT_TARGET}), other {other}"
    )


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


SEED_TREE = {
    "tldw_Server_API/app/api/v1/endpoints/chat.py": "class Thing:\n    y = 1\n",
    "tldw_Server_API/app/api/v1/endpoints/audio/__init__.py": "",
    "tldw_Server_API/app/api/v1/endpoints/audio/tts.py": "",
    "tldw_Server_API/app/core/Chat/svc.py": "",
    "tldw_Server_API/tests/Chat/test_shapes.py": (
        "import importlib, sys\n"
        "from unittest import mock\n"
        "from unittest.mock import patch\n"
        "import tldw_Server_API.app.api.v1.endpoints.chat as chat_ep\n"
        "from tldw_Server_API.app.core.Chat import svc\n"
        "tts = importlib.import_module('tldw_Server_API.app.api.v1.endpoints.audio.tts')\n"
        "pkg = sys.modules['tldw_Server_API.app.api.v1.endpoints.audio']\n"
        "def test_shapes(monkeypatch, mocker, name):\n"
        "    monkeypatch.setattr(chat_ep, 'a', 1)\n"
        "    monkeypatch.setattr('tldw_Server_API.app.api.v1.endpoints.chat._b', 1)\n"
        "    monkeypatch.setattr(chat_ep.Thing, 'y', 2)\n"
        "    monkeypatch.setattr(chat_ep, name, 3)\n"
        "    monkeypatch.setattr(tts, 'c', 1)\n"
        "    monkeypatch.delattr(pkg, 'd')\n"
        "    from tldw_Server_API.app.api.v1.endpoints import audio\n"
        "    monkeypatch.setattr(audio.tts, 'e', 1)\n"
        "    patch('tldw_Server_API.app.api.v1.endpoints.chat.f')\n"
        "    patch.object(chat_ep, 'g')\n"
        "    mock.patch('tldw_Server_API.app.api.v1.endpoints.chat.h')\n"
        "    mocker.patch.object(chat_ep, 'i')\n"
        "    chat_ep.j = 1\n"
        "    del chat_ep.k\n"
        "    monkeypatch.setattr(svc, 'x', 1)\n"
        "    monkeypatch.setattr('tldw_Server_API.app.core.Chat.svc.x', 1)\n"
        "    monkeypatch.setenv('X', '1')\n"
        "    object.__setattr__(chat_ep, 'z', 1)\n"
        "def test_shadowed(chat_ep, monkeypatch):\n"
        "    monkeypatch.setattr(chat_ep, 'p', 1)\n"
        "def test_rebound(monkeypatch):\n"
        "    chat_ep = object()\n"
        "    monkeypatch.setattr(chat_ep, 'q', 1)\n"
        "def test_local(monkeypatch):\n"
        "    from tldw_Server_API.app.api.v1.endpoints.audio import tts as chat_ep\n"
        "    monkeypatch.setattr(chat_ep, 'r', 1)\n"
    ),
    "tldw_Server_API/app/core/MCP_unified/tests/test_in_app.py": (
        "from tldw_Server_API.app.api.v1.endpoints import chat\n"
        "def test_in_app(monkeypatch):\n    monkeypatch.setattr(chat, 'o', 1)\n"
    ),
    "tldw_Server_API/app/core/Chat/not_a_test.py": (
        "import tldw_Server_API.app.api.v1.endpoints.chat as c\n"
        "def f(monkeypatch):\n    monkeypatch.setattr(c, 'n', 1)\n"
    ),
}


def test_the_counter_counts_each_patch_shape(tmp_path: Path) -> None:
    measured = measure_endpoint_patches(_seed(tmp_path, SEED_TREE))
    assert measured == {
        # a, _b, Thing.y, a computed name, o (an in-app tests package); f, g, h, i, j, k.
        # Not p (a parameter shadows the import), not q (rebound locally), not n (not a test module).
        "api/v1/endpoints/chat.py": {"monkeypatch": 5, "other": 6},
        "api/v1/endpoints/audio/tts.py": {"monkeypatch": 3, "other": 0},  # c (import_module), e (package chain), r (a local alias)
        "api/v1/endpoints/audio/__init__.py": {"monkeypatch": 1, "other": 0},  # delattr through sys.modules
    }
    seed, exit_target = SEED["monkeypatch"], EXIT_TARGET
    assert summary({"a.py": {"monkeypatch": seed, "other": 2}, "b.py": {"monkeypatch": 0, "other": 1}}) == (
        f"monkeypatch {seed} (100.0% of seed {seed}; Phase 2 exit at or below {exit_target}), other 3"
    )


def test_growth_a_new_module_and_a_stale_entry_fail_and_shrinking_passes(tmp_path: Path) -> None:
    measured = measure_endpoint_patches(_seed(tmp_path, SEED_TREE))
    key = "api/v1/endpoints/chat.py"
    grown, gone, new = _ceilings.violations(
        PATCHES,
        {**measured, "api/v1/endpoints/new.py": {"monkeypatch": 1, "other": 0}},
        {
            key: {"monkeypatch": 4, "other": 6, "reason": None},
            "api/v1/endpoints/audio/tts.py": {"monkeypatch": 3, "other": 0, "reason": None},
            "api/v1/endpoints/audio/__init__.py": {"monkeypatch": 1, "other": 0, "reason": None},
            "api/v1/endpoints/gone.py": {"monkeypatch": 1, "other": 0, "reason": None},
        },
    )
    assert grown.startswith(f"{key}: monkeypatch 5 > ceiling 4")
    assert gone.startswith("api/v1/endpoints/gone.py: stale entry")
    assert new.startswith("api/v1/endpoints/new.py: monkeypatch 1, other 0, and no ceiling")
    roomy = {k: {**v, "monkeypatch": v["monkeypatch"] + 1, "reason": None} for k, v in measured.items()}
    assert _ceilings.violations(PATCHES, measured, roomy) == []


# --- The ratchet --------------------------------------------------------------------------------


def test_endpoint_patches_do_not_grow() -> None:
    measured, baseline = measure_endpoint_patches(), _ceilings.load(PATCHES.baseline)

    problems = _ceilings.violations(PATCHES, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(PATCHES, measured, baseline)


def test_raised_endpoint_patch_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(PATCHES, measure_endpoint_patches())
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        print(summary(measure_endpoint_patches()))
        raise SystemExit(0)
    raise SystemExit(_ceilings.main(PATCHES, measure_endpoint_patches))
