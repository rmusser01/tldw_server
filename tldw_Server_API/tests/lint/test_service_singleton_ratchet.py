"""Ratchet: module singletons do not grow (TASK-13950.18).

Three counts, each frozen per module in its own baseline JSON as a ceiling
``{"<field>": int, "reason": ...}``:

  * globals (global_statements_baseline.json): the ``ast.Global`` nodes in each module under
    tldw_Server_API/app. Modules in an in-app ``tests`` package directory are skipped.
  * getters (endpoint_getter_calls_baseline.json): in modules under ``api/<version>/endpoints/``
    (any depth), the distinct calls inside a function body (at any nesting) whose callee is a bare
    name matching ``get_\\w+`` and that take no arguments. Module-level calls, decorators and
    defaults do not count, nor do attribute calls (``mod.get_x()``).
  * state (api_deps_module_state_baseline.json): in the direct children of
    ``api/<version>/API_Deps/``, the module-level ``Assign``/``AnnAssign`` statements whose value is
    a call to a STATE_CONSTRUCTORS name (``Lock()`` or ``threading.Lock()``). Module-level means
    top-level, or nested in module-level ``if``/``try``/``with``/``for``/``while``; never in a
    ``def`` or ``class``.

Known limits. These escape the counts: getters called through an alias
(``pool = get_db_pool; pool()``), state built by a module-level helper call, and ``global`` writes
in comprehensions.

Each slice of TASK-13950.18 tightens this. Growth fails: hold the state on an object its owner
owns instead. If growth is unavoidable, raise the count with a "reason" (see _ceilings.py).
Lock a gain in with
``python -m tldw_Server_API.tests.lint.test_service_singleton_ratchet --tighten``
(``--write-baseline`` records every current measurement).
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree

pytestmark = pytest.mark.unit

STATE_CONSTRUCTORS = frozenset({
    "Lock", "RLock", "Semaphore", "BoundedSemaphore", "Condition", "Event",
    "ThreadPoolExecutor", "ProcessPoolExecutor", "LRUCache", "TTLCache",
    "OrderedDict", "WeakValueDictionary", "WeakKeyDictionary",
})
_COMMAND = "python -m tldw_Server_API.tests.lint.test_service_singleton_ratchet"
_GETTER = re.compile(r"get_\w+")
_ENDPOINTS = re.compile(r"api/[^/]+/endpoints/")
_API_DEPS = re.compile(r"api/[^/]+/API_Deps/[^/]+\.py")


def _ratchet(name: str, fix: str, stale: str) -> _ceilings.Ratchet:
    return _ceilings.Ratchet(
        baseline=Path(__file__).resolve().with_name(name), command=_COMMAND, fix=fix, stale=stale
    )


GLOBALS = _ratchet(
    "global_statements_baseline.json",
    "Keep the state on an object its owner holds (a slot in app_services() or a constructor argument), not in a module global.",
    "The module has no `global` statement any more.",
)
GETTERS = _ratchet(
    "endpoint_getter_calls_baseline.json",
    "Take the service as a parameter: Depends(<provider>) in a route, an argument in a helper. "
    "Bare zero-argument getters in endpoint modules only fall.",
    "The endpoint module makes no bare getter call any more.",
)
STATE = _ratchet(
    "api_deps_module_state_baseline.json",
    "Move the lock, cache or executor into a core runtime object held by app_services() "
    "(as ChaChaRuntimeManager does); API_Deps modules stay adapters.",
    "The module holds no module-level lock, cache or executor any more.",
)


# --- The counters -------------------------------------------------------------------------------


def _modules(root: Path):
    """``(app-relative posix key, tree)`` for every module outside an in-app tests package."""
    root = root.resolve()
    for path, tree in parse_tree(root).items():
        relative = path.relative_to(root)
        if "tests" not in relative.parts[:-1]:
            yield relative.as_posix(), tree


def measure_globals(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{module: {"globals": n}}`` for modules with ``global`` statements."""
    found = {}
    for key, tree in _modules(root):
        if count := sum(isinstance(node, ast.Global) for node in ast.walk(tree)):
            found[key] = {"globals": count}
    return found


def _is_bare_getter(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and _GETTER.fullmatch(node.func.id) is not None
        and not node.args
        and not node.keywords
    )


def measure_getter_calls(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{endpoint module: {"calls": n}}`` for modules with bare getter calls in function bodies."""
    found = {}
    for key, tree in _modules(root):
        if not _ENDPOINTS.match(key):
            continue
        calls: set[int] = set()  # ast.walk visits a nested function's calls once per enclosing function
        for func in ast.walk(tree):
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for stmt in func.body:
                    calls.update(id(node) for node in ast.walk(stmt) if _is_bare_getter(node))
        if calls:
            found[key] = {"calls": len(calls)}
    return found


def _module_level(body: list[ast.stmt]):
    """Statements at module level: top-level and nested in if/try/with/for/while, never def/class."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            yield from _module_level(stmt.body + stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            yield from _module_level(stmt.body)
        elif isinstance(stmt, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            handlers = [s for h in stmt.handlers for s in h.body]
            yield from _module_level(stmt.body + handlers + stmt.orelse + stmt.finalbody)


def _is_state_call(node: ast.AST | None) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
    return name in STATE_CONSTRUCTORS


def measure_module_state(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    """``{API_Deps module: {"state": n}}`` for modules holding module-level locks, caches or executors."""
    found = {}
    for key, tree in _modules(root):
        if not _API_DEPS.fullmatch(key):
            continue
        count = sum(
            isinstance(stmt, (ast.Assign, ast.AnnAssign)) and _is_state_call(stmt.value)
            for stmt in _module_level(tree.body)
        )
        if count:
            found[key] = {"state": count}
    return found


RATCHETS = (
    (GLOBALS, measure_globals),
    (GETTERS, measure_getter_calls),
    (STATE, measure_module_state),
)


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def test_each_counter_counts_its_shape(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": "_x = None\ndef f():\n    global _x\n    _x = 1\nclass C:\n    def g(self):\n        global _x, _y\n",
        "core/X/tests/test_a.py": "def f():\n    global _z\n",
        "api/v1/endpoints/e.py": (
            "X = get_y()\n"
            "async def route(dep=get_default()):\n"
            "    a = await get_db_pool()\n"
            "    b = get_settings(1)\n"
            "    c = mod.get_z()\n"
            "    def inner():\n        return get_w()\n"
            "    return get_q()\n"
        ),
        "api/v1/endpoints/sub/f.py": "def h():\n    return get_mcp_server()\n",
        "api/v1/API_Deps/d.py": (
            "import threading\nfrom cachetools import LRUCache\n"
            "_lock = threading.Lock()\n"
            "try:\n    _cache: LRUCache = LRUCache(maxsize=2)\nexcept ImportError:\n    _cache = {}\n"
            "_plain = {}\n"
            "def f():\n    lock = threading.Lock()\n"
            "class C:\n    lock = threading.RLock()\n"
        ),
    })
    assert measure_globals(root) == {"core/X/a.py": {"globals": 2}}
    assert measure_getter_calls(root) == {
        "api/v1/endpoints/e.py": {"calls": 3},  # get_db_pool, get_w (nested), get_q
        "api/v1/endpoints/sub/f.py": {"calls": 1},
    }
    assert measure_module_state(root) == {"api/v1/API_Deps/d.py": {"state": 2}}


@pytest.mark.parametrize(("ratchet", "measure", "field"), [(GLOBALS, measure_globals, "globals"), (GETTERS, measure_getter_calls, "calls"), (STATE, measure_module_state, "state")])
def test_growth_new_and_stale_entries_fail_and_shrinking_passes(tmp_path, ratchet, measure, field) -> None:
    root = _seed(tmp_path / "app", {
        "core/X/a.py": "def f():\n    global a\ndef g():\n    global b\n",
        "api/v1/endpoints/e.py": "def r():\n    get_a()\n    get_b()\n",
        "api/v1/API_Deps/d.py": "import threading\nA = threading.Lock()\nB = threading.RLock()\n",
    })
    (key,) = measure(root)
    grown, gone, new = _ceilings.violations(ratchet, {**measure(root), "z.py": {field: 1}}, {key: {field: 1, "reason": None}, "gone.py": {field: 1, "reason": None}})
    assert grown.startswith(f"{key}: {field} 2 > ceiling 1")
    assert gone.startswith("gone.py: stale entry")
    assert new.startswith(f"z.py: {field} 1, and no ceiling")
    assert _ceilings.violations(ratchet, measure(root), {key: {field: 3, "reason": None}}) == []


# --- The ratchets -------------------------------------------------------------------------------

_REAL = pytest.mark.parametrize(("ratchet", "measure"), RATCHETS, ids=["globals", "getters", "state"])


@_REAL
def test_singleton_counts_do_not_grow(ratchet, measure) -> None:
    measured, baseline = measure(), _ceilings.load(ratchet.baseline)

    problems = _ceilings.violations(ratchet, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(ratchet, measured, baseline)


@_REAL
def test_raised_singleton_ceilings_carry_a_new_reason(ratchet, measure) -> None:
    problems = _ceilings.reasonless_bumps(ratchet, measure())
    assert problems == [], "\n".join(problems)


if __name__ == "__main__":
    raise SystemExit(max(_ceilings.main(ratchet, measure) for ratchet, measure in RATCHETS))
