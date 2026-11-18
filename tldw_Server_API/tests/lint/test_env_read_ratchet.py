"""Ratchet: environment reads do not grow, and the shared keys stay in one place (TASK-13950.21).

Configuration is read from the environment in thousands of places. This freezes the count per
module in env_read_baseline.json, each a ceiling ``{"reads": int, "import_time": int,
"reason": ...}``, so a new read has to go through the typed section in core/config_sections
instead. Every later .21 PR retargets readers and tightens the ceilings.

Counted, per module under tldw_Server_API/app (modules in an in-app ``tests`` package directory
are skipped, and so is every module in EXEMPT):
  * direct reads: ``os.getenv(k)``, ``os.getenvb(k)``, ``getenv(k)`` imported from ``os``,
    ``<env>.get(k)``, ``<env>[k]`` in Load context, and ``k in <env>`` / ``not in``. ``<env>`` is
    ``os.environ`` under any alias, ``environ`` imported from ``os``, or a name the module binds
    (``=`` or annotated ``=``) to ``os.environ`` or to a conditional expression with an
    ``os.environ`` branch;
  * helper calls: a function whose parameter is the key of a direct read (or is iterated and the
    loop variable is the key) is a helper, and so is one that forwards a parameter into a helper's
    key position. A call to a helper with a literal key (or a tuple or list of literals) is a read;
    a call with a non-literal key is not, because the helper's own body read is counted;
  * resolver reads: a call to config_sections.precedence.resolve, or to a function that forwards
    its key into it, with a literal key. Counted outside the config modules only, because inside
    them each resolver read is the implementation.
Import time: outside every ``def``/``lambda`` body (decorators and defaults included), skipping
``if __name__ == "__main__":`` blocks, plus the bodies of module-level functions that import-time
code calls by bare name, transitively.

Not counted:
  * test-mode flags (TEST_MODE, TESTING, TLDW_TEST_MODE, PYTEST_CURRENT_TEST, TLDW_TEST_*):
    TASK-13950.19 owns them. Calls with no key, such as ``is_test_mode()``, are not env reads;
  * core/testing.py, which is the flag layer itself (its helpers still count at their callers);
  * writes (``os.environ[k] = v``, ``setdefault``), whole-environ copies and iteration;
  * a key built at run time that is not a helper parameter, and a mapping passed in as a parameter.

Growth fails: read the setting through its typed section in core/config_sections (or the
settings object that owns it). If growth is unavoidable, raise the count with a "reason"
(reviewed in the PR; see _ceilings.py). A count below its ceiling passes; lock the gain in with
``python -m tldw_Server_API.tests.lint.test_env_read_ratchet --tighten``.
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections.abc import Iterator, Mapping
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple

import pytest

from tldw_Server_API.tests.lint import _ceilings
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree
from tldw_Server_API.tests.lint.test_test_mode_branch_ratchet import is_signal_constant

pytestmark = pytest.mark.unit

CONFIG_MODULES: frozenset[str] = frozenset({"core/config.py", "core/AuthNZ/settings.py", "core/MCP_unified/config.py"})
EXEMPT: frozenset[str] = frozenset({"core/testing.py"})
RESOLVER: tuple[str, str] = ("tldw_Server_API.app.core.config_sections.precedence", "resolve")
SHARED_KEYS: frozenset[str] = frozenset({
    "AUTH_MODE", "APP_MODE", "SINGLE_USER_API_KEY", "API_KEY", "JWT_SECRET_KEY", "DATABASE_URL", "REDIS_URL",
    "SINGLE_USER_FIXED_ID", "ACCESS_TOKEN_EXPIRE_MINUTES", "RAG_DEFAULT_LLM_PROVIDER", "RAG_DEFAULT_LLM_MODEL",
})
MOVED_SHARED_KEYS: frozenset[str] = frozenset()

RATCHET = _ceilings.Ratchet(
    baseline=Path(__file__).resolve().with_name("env_read_baseline.json"),
    command="python -m tldw_Server_API.tests.lint.test_env_read_ratchet",
    fix=(
        "Read the setting through its typed section in core/config_sections (or the settings object "
        "that owns it) instead of a new env read (TASK-13950.21)."
    ),
    stale="The module no longer reads the environment.",
)

_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
Helper = tuple[int, str, Literal["direct", "resolver"]]
_RESOLVER_HELPER: Helper = (0, "env_keys", "resolver")


def is_config_module(key: str) -> bool:
    return key in CONFIG_MODULES or key.startswith("core/config_sections/")


class Read(NamedTuple):
    line: int
    names: tuple[str, ...]  # empty for a non-literal key
    kind: Literal["direct", "helper", "resolver"]
    import_time: bool


def _dotted(key: str) -> tuple[str, bool]:
    parts = key[:-3].split("/")
    is_pkg = parts[-1] == "__init__"
    if is_pkg:
        parts = parts[:-1]
    return ".".join(["tldw_Server_API.app", *parts]), is_pkg


class _Scope:
    """One module's import bindings and ``os.environ`` aliases, built once."""

    def __init__(self, tree: ast.Module, key: str) -> None:
        self.module, is_pkg = _dotted(key)
        self.os: set[str] = set()
        self.environ: set[str] = set()
        self.getenv: set[str] = set()
        self.imports: dict[str, tuple[str, str]] = {}  # bound name -> (module, name) from ``from m import n``
        self.modules: dict[str, str] = {}  # bound name -> dotted module it may refer to
        self.env_aliases: set[str] = set()
        package = self.module if is_pkg else self.module.rpartition(".")[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "os":
                        self.os.add(alias.asname or "os")
                    if alias.asname:
                        self.modules[alias.asname] = alias.name
            elif isinstance(node, ast.ImportFrom):
                source = node.module or ""
                if node.level:
                    base = package
                    for _ in range(node.level - 1):
                        base = base.rpartition(".")[0]
                    source = f"{base}.{source}" if source else base
                for alias in node.names:
                    bound = alias.asname or alias.name
                    if source == "os" and alias.name == "environ":
                        self.environ.add(bound)
                    elif source == "os" and alias.name in ("getenv", "getenvb"):
                        self.getenv.add(bound)
                    else:
                        self.imports[bound] = (source, alias.name)
                        self.modules.setdefault(bound, f"{source}.{alias.name}")
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if self._environ_like(value):
                self.env_aliases.update(t.id for t in targets if isinstance(t, ast.Name))

    def _is_os_environ(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Attribute):
            return node.attr == "environ" and isinstance(node.value, ast.Name) and node.value.id in self.os
        return isinstance(node, ast.Name) and node.id in self.environ

    def _environ_like(self, node: ast.AST) -> bool:
        return self._is_os_environ(node) or (
            isinstance(node, ast.IfExp) and (self._is_os_environ(node.body) or self._is_os_environ(node.orelse))
        )

    def _is_env(self, node: ast.AST) -> bool:
        return self._is_os_environ(node) or (isinstance(node, ast.Name) and node.id in self.env_aliases)

    def direct_key(self, node: ast.AST) -> ast.AST | None | Literal[False]:
        """The key expression of a direct read (None when it has no argument), or False."""
        if isinstance(node, ast.Call):
            func = node.func
            is_getenv = (
                isinstance(func, ast.Attribute)
                and func.attr in ("getenv", "getenvb")
                and isinstance(func.value, ast.Name)
                and func.value.id in self.os
            ) or (isinstance(func, ast.Name) and func.id in self.getenv)
            if is_getenv or (isinstance(func, ast.Attribute) and func.attr == "get" and self._is_env(func.value)):
                if node.args:
                    return node.args[0]
                return next((kw.value for kw in node.keywords if kw.arg == "key"), None)
        elif isinstance(node, ast.Subscript):
            if isinstance(node.ctx, ast.Load) and self._is_env(node.value):
                return node.slice
        elif isinstance(node, ast.Compare):
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and self._is_env(right):
                    return left
                left = right
        return False

    def target(self, func: ast.AST) -> tuple[str, str] | None:
        """The (module, name) a call resolves to, by the three binding forms the spec lists."""
        if isinstance(func, ast.Name):
            return self.imports.get(func.id, (self.module, func.id))
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            if func.value.id == "self":
                return (self.module, func.attr)
            if func.value.id in self.modules:
                return (self.modules[func.value.id], func.attr)
        return None


def _literal_names(node: ast.AST | None) -> tuple[str, ...] | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, (ast.Tuple, ast.List)) and node.elts:
        if all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.elts):
            return tuple(e.value for e in node.elts)  # type: ignore[attr-defined]
    return None


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = fn.args
    names = [a.arg for a in args.posonlyargs + args.args]
    if names and names[0] in ("self", "cls"):
        names = names[1:]
    return names + [a.arg for a in args.kwonlyargs]


def _key_arg(call: ast.Call, helper: Helper) -> ast.AST | None:
    index, name, _ = helper
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if index < len(call.args) and not any(isinstance(a, ast.Starred) for a in call.args[: index + 1]):
        return call.args[index]
    return None


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterator[ast.AST]:
    """The nodes of ``fn``'s body, not descending into nested defs, lambdas or classes."""
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        stack.extend(
            c for c in ast.iter_child_nodes(node) if not isinstance(c, (*_DEFS, ast.Lambda, ast.ClassDef))
        )


def _helper_of(target: tuple[str, str] | None, helpers: Mapping[tuple[str, str], Helper]) -> Helper | None:
    """RESOLVER is classified by identity first, so ``resolve``'s own direct backing never hides it."""
    if target == RESOLVER:
        return _RESOLVER_HELPER
    return helpers.get(target) if target else None


def env_helpers(trees: Mapping[str, ast.Module]) -> dict[tuple[str, str], Helper]:
    """Functions that read an env var named by a parameter, over every non-``tests`` module (exempt ones too)."""
    helpers: dict[tuple[str, str], Helper] = {}
    forwarders = []  # (key, params, loop-variable -> param, [(call, target)])
    for rel, tree in trees.items():
        scope = _Scope(tree, rel)
        for fn in (n for n in ast.walk(tree) if isinstance(n, _DEFS)):
            params = _params(fn)
            if not params:
                continue
            key = (scope.module, fn.name)
            derived: dict[str, str] = {}
            calls = []
            for node in _own_nodes(fn):
                if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
                    if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Name) and node.iter.id in params:
                        derived[node.target.id] = node.iter.id
            for node in _own_nodes(fn):
                direct = scope.direct_key(node)
                if isinstance(direct, ast.Name) and (param := direct.id if direct.id in params else derived.get(direct.id)):
                    helpers[key] = (params.index(param), param, "direct")
                if isinstance(node, ast.Call) and (target := scope.target(node.func)):
                    calls.append((node, target))
            if calls:
                forwarders.append((key, params, derived, calls))
    changed = True
    while changed:
        changed = False
        for key, params, derived, calls in forwarders:
            for call, target in calls:
                helper = _helper_of(target, helpers)
                arg = _key_arg(call, helper) if helper else None
                if helper is None or arg is None:
                    continue
                for elt in arg.elts if isinstance(arg, (ast.Tuple, ast.List)) else [arg]:
                    param = None
                    if isinstance(elt, ast.Name):
                        param = elt.id if elt.id in params else derived.get(elt.id)
                    old = helpers.get(key)
                    if param and (old is None or (old[2] == "resolver" and helper[2] == "direct")):
                        helpers[key] = (params.index(param), param, helper[2])
                        changed = True
    return helpers


def _import_time_nodes(tree: ast.Module) -> set[int]:
    """ids of the nodes that run when the module is imported."""
    found: set[int] = set()
    top_funcs = {n.name: n for n in tree.body if isinstance(n, _DEFS)}

    def visit(node: ast.AST, at_import: bool) -> None:
        if at_import:
            found.add(id(node))
        if isinstance(node, (*_DEFS, ast.Lambda)):
            args = node.args
            decorators = node.decorator_list if isinstance(node, _DEFS) else []
            for expr in [*decorators, *args.defaults, *(d for d in args.kw_defaults if d is not None)]:
                visit(expr, at_import)
            for stmt in node.body if isinstance(node, _DEFS) else [node.body]:
                visit(stmt, False)
            return
        for child in ast.iter_child_nodes(node):
            visit(child, at_import)

    for stmt in tree.body:
        is_main = (
            isinstance(stmt, ast.If)
            and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name)
            and stmt.test.left.id == "__name__"
        )
        if is_main:
            for child in ast.iter_child_nodes(stmt):
                visit(child, False)
        else:
            visit(stmt, True)
    seen: set[str] = set()
    queue = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and id(n) in found]
    while queue:
        call = queue.pop()
        if isinstance(call.func, ast.Name) and call.func.id in top_funcs and call.func.id not in seen:
            seen.add(call.func.id)
            for node in _own_nodes(top_funcs[call.func.id]):
                found.add(id(node))
                if isinstance(node, ast.Call):
                    queue.append(node)
    return found


def module_reads(tree: ast.Module, module: str, helpers: Mapping[tuple[str, str], Helper]) -> list[Read]:
    """``module`` is the root-relative path (``core/X/a.py``), which names the dotted module for call targets."""
    scope = _Scope(tree, module)
    at_import = _import_time_nodes(tree)
    reads = []
    for node in ast.walk(tree):
        key = scope.direct_key(node)
        if key is not False:
            names = _literal_names(key) if isinstance(key, ast.Constant) else None
            reads.append(Read(node.lineno, names or (), "direct", id(node) in at_import))
        elif isinstance(node, ast.Call):
            helper = _helper_of(scope.target(node.func), helpers)
            names = _literal_names(_key_arg(node, helper)) if helper else None
            if names:
                kind = "resolver" if helper[2] == "resolver" else "helper"  # type: ignore[index]
                reads.append(Read(node.lineno, names, kind, id(node) in at_import))
    return reads


def _is_app_module(key: str) -> bool:
    return "tests" not in key.split("/")[:-1]


def _trees(root: Path) -> dict[str, ast.Module]:
    root = root.resolve()
    return {p.relative_to(root).as_posix(): t for p, t in parse_tree(root).items() if _is_app_module(p.relative_to(root).as_posix())}


def measure_reads(root: Path = APP_ROOT) -> dict[str, list[Read]]:
    trees = _trees(root)
    helpers = env_helpers(trees)
    found = {k: module_reads(t, k, helpers) for k, t in trees.items() if k not in EXEMPT}
    return {k: reads for k, reads in found.items() if reads}


def _counts(key: str, read: Read) -> bool:
    if read.names and all(is_signal_constant(n) for n in read.names):
        return False
    return not (read.kind == "resolver" and is_config_module(key))


def _measure_reads(reads: Mapping[str, list[Read]]) -> dict[str, dict[str, int]]:
    found = {}
    for key, module in reads.items():
        counted = [r for r in module if _counts(key, r)]
        if counted:
            found[key] = {"reads": len(counted), "import_time": sum(r.import_time for r in counted)}
    return found


def measure(root: Path = APP_ROOT) -> dict[str, dict[str, int]]:
    return _measure_reads(measure_reads(root))


def summary(measured: Mapping[str, Mapping[str, int]]) -> str:
    lines = []
    for label, inside in (("outside config modules", False), ("config modules", True)):
        rows = [v for k, v in measured.items() if is_config_module(k) == inside]
        lines.append(
            f"{label}: {sum(v['reads'] for v in rows)} reads in {len(rows)} modules, "
            f"{sum(v['import_time'] for v in rows)} at import time"
        )
    return "\n".join(lines)


def settings_env_names(tree: ast.Module) -> frozenset[str]:
    """Env names of the ``BaseSettings`` fields: the ``validation_alias`` constant, else an upper-case field name."""
    names = set()
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        bases = [b.id if isinstance(b, ast.Name) else b.attr if isinstance(b, ast.Attribute) else "" for b in cls.bases]
        if not any(b.endswith("BaseSettings") for b in bases):
            continue
        for stmt in cls.body:
            if not (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)):
                continue
            alias = None
            if isinstance(stmt.value, ast.Call):
                for kw in stmt.value.keywords:
                    if kw.arg == "validation_alias" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                        alias = kw.value.value
            if alias:
                names.add(alias)
            elif stmt.target.id.isupper():
                names.add(stmt.target.id)
    return frozenset(names)


def moved_key_problems(reads: Mapping[str, list[Read]], moved: frozenset[str]) -> list[str]:
    return sorted(
        f"{key}: reads {name}, which moved to core/config_sections (TASK-13950.21); read it through the typed section"
        for key, module in reads.items()
        if not key.startswith("core/config_sections/")
        for name in sorted({n for r in module for n in r.names} & moved)
    )


@cache
def app_reads() -> dict[str, list[Read]]:
    """One counter pass over the app tree (about 25 s), shared by the tests in this module and Task 2's."""
    return measure_reads()


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


SEED = {
    "core/X/direct.py": 'import os\nA = os.getenv("A1")\nB = os.environ.get("B1")\nC = os.environ["C1"]\nD = "D1" in os.environ\n',
    "core/X/aliased.py": 'import os as _os\nfrom os import environ as E, getenv as g\n\n\ndef f(env=None):\n'
                         '    env = _os.environ if env is None else env\n    return env.get("A2"), E["B2"], g("C2")\n',
    "core/X/annotated.py": 'import os\nfrom typing import Mapping\n\n\ndef f(env=None):\n'
                           '    env_map: Mapping[str, str] = env if env is not None else os.environ\n    return env_map.get("A3")\n',
    "core/X/helpers.py": 'import os\n\n\ndef _env_int(name, default=0):\n    return int(os.getenv(name) or default)\n\n\n'
                         'def _first(keys):\n    for key in keys:\n        if os.environ.get(key):\n            return key\n\n\n'
                         'def _outer(k):\n    return _env_int(k)\n\n\ndef _raw(cp, *, env_key):\n    return os.environ.get(env_key) or cp\n\n\n'
                         'X = _env_int("H1")\n\n\ndef g():\n'
                         '    return _env_int("H2"), _first(("H3", "H4")), _outer("H6"), _raw(None, env_key="H7")\n',
    "core/X/imported.py": 'from tldw_Server_API.app.core.X.helpers import _env_int\n\n\ndef h():\n    return _env_int("H5")\n',
    "core/X/flags.py": 'from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode\n\n'
                       'A = env_flag_enabled("FEATURE_X")\nB = env_flag_enabled("TESTING")\nC = is_test_mode()\n',
    "core/X/import_time.py": 'import os\n\n\ndef _limit():\n    return os.getenv("I1")\n\n\nLIMIT = _limit()\n\n\n'
                             'def later():\n    return os.getenv("I2")\n\n\nif __name__ == "__main__":\n    print(os.getenv("I3"))\n',
    "core/X/not_reads.py": 'import os\nos.environ["W1"] = "1"\nos.environ.setdefault("W2", "1")\nENV = dict(os.environ)\n'
                           'KEYS = [k for k in os.environ.keys() if k.startswith("TLDW_")]\n',
    "core/X/test_flags.py": 'import os\nA = os.getenv("TEST_MODE")\nB = os.getenv("TLDW_TEST_DB")\n',
    "core/X/resolver_user.py": 'from tldw_Server_API.app.core.config_sections.precedence import resolve\n\n\n'
                               'def _via(k):\n    return resolve((k,), config_parser=None)\n\n\n'
                               'def f(cp):\n    return resolve(("R1",), (("S", "o"),), config_parser=cp), _via("R3")\n',
    "core/config_sections/precedence.py": 'import os\n\n\ndef resolve(env_keys=(), locations=(), *, config_parser=None, env=None, default=None):\n'
                                          '    env = os.environ if env is None else env\n    for key in env_keys:\n'
                                          '        if env.get(key):\n            return key\n    return default\n',
    "core/config_sections/x.py": 'from .precedence import resolve\n\n\ndef _get_raw(cp, *, env_keys):\n'
                                 '    return resolve(env_keys, config_parser=cp)\n\n\ndef f(cp):\n'
                                 '    return resolve(("R2",), config_parser=cp), _get_raw(cp, env_keys=("K1",))\n',
    "core/config_sections/y.py": 'import os\n\n\ndef _get_raw(env_key):\n    return os.environ.get(env_key)\n\n\nA = _get_raw("K2")\n',
    "core/testing.py": 'import os\nX = os.getenv("Z1")\n\n\ndef env_flag_enabled(name):\n    return bool(os.getenv(name))\n',
    "core/X/tests/test_x.py": 'import os\nX = os.getenv("Z2")\n',
}


def test_the_counter_counts_each_read_shape(tmp_path: Path) -> None:
    assert measure(_seed(tmp_path / "app", SEED)) == {
        "core/X/aliased.py": {"reads": 3, "import_time": 0},
        "core/X/annotated.py": {"reads": 1, "import_time": 0},
        "core/X/direct.py": {"reads": 4, "import_time": 4},
        "core/X/flags.py": {"reads": 1, "import_time": 1},  # FEATURE_X; TESTING excluded; is_test_mode() has no key
        "core/X/helpers.py": {"reads": 8, "import_time": 2},  # 3 helper bodies + H1, H2, (H3, H4), H6 via _outer, H7 by keyword
        "core/X/import_time.py": {"reads": 3, "import_time": 1},  # I1 via LIMIT = _limit(); not I2, not the __main__ block
        "core/X/imported.py": {"reads": 1, "import_time": 0},
        "core/X/resolver_user.py": {"reads": 2, "import_time": 0},  # R1 and R3 (resolver-backed _via) count outside config
        "core/config_sections/precedence.py": {"reads": 1, "import_time": 0},  # resolve's own body; its callers are resolver reads
        "core/config_sections/y.py": {"reads": 2, "import_time": 2},  # a direct-backed helper counts even in a config module
    }


def test_reads_carry_their_literal_names(tmp_path: Path) -> None:
    reads = measure_reads(_seed(tmp_path / "app", SEED))
    assert sorted(r.names for r in reads["core/X/helpers.py"]) == [(), (), (), ("H1",), ("H2",), ("H3", "H4"), ("H6",), ("H7",)]
    assert {(r.kind, r.names) for r in reads["core/config_sections/x.py"]} == {("resolver", ("R2",)), ("resolver", ("K1",))}
    assert [r.names for r in reads["core/X/test_flags.py"]] == [("TEST_MODE",), ("TLDW_TEST_DB",)]  # names kept, reads not counted
    assert "core/testing.py" not in reads and "core/X/tests/test_x.py" not in reads


def test_growth_a_new_module_and_a_stale_entry_fail(tmp_path: Path) -> None:
    measured = measure(_seed(tmp_path / "app", {"core/X/a.py": 'import os\nA = os.getenv("A")\nB = os.getenv("B")\n',
                                                "core/X/new.py": 'import os\nA = os.getenv("A")\n'}))
    baseline = {"core/X/a.py": {"reads": 1, "import_time": 2, "reason": None},
                "core/X/gone.py": {"reads": 1, "import_time": 0, "reason": None}}
    problems = _ceilings.violations(RATCHET, measured, baseline)
    assert [p.split(":")[0] for p in problems] == ["core/X/a.py", "core/X/gone.py", "core/X/new.py"]
    assert "reads 2 > ceiling 1" in problems[0]
    assert "stale entry" in problems[1]


def test_a_moved_key_read_outside_config_sections_fails(tmp_path: Path) -> None:
    reads = measure_reads(_seed(tmp_path / "app", {
        "core/X/a.py": 'import os\nA = os.getenv("AUTH_MODE")\n',
        "core/config_sections/auth.py": 'import os\nA = os.environ.get("AUTH_MODE")\n',
    }))
    assert moved_key_problems(reads, frozenset({"AUTH_MODE"})) == [
        "core/X/a.py: reads AUTH_MODE, which moved to core/config_sections (TASK-13950.21); read it through the typed section"
    ]


# --- The ratchet, against the real tree ---------------------------------------------------------


def _app_measurement() -> dict[str, dict[str, int]]:
    return _measure_reads(app_reads())


def test_env_reads_do_not_grow() -> None:
    measured, baseline = _app_measurement(), _ceilings.load(RATCHET.baseline)

    problems = _ceilings.violations(RATCHET, measured, baseline)

    assert problems == [], "\n".join(problems)
    _ceilings.warn_if_tightenable(RATCHET, measured, baseline)


def test_raised_env_read_ceilings_carry_a_new_reason() -> None:
    problems = _ceilings.reasonless_bumps(RATCHET, _app_measurement())
    assert problems == [], "\n".join(problems)


def test_moved_shared_keys_are_read_only_in_config_sections() -> None:
    assert MOVED_SHARED_KEYS <= SHARED_KEYS
    assert moved_key_problems(app_reads(), MOVED_SHARED_KEYS) == []


def test_mcp_config_resolves_no_shared_key() -> None:
    names = settings_env_names(parse_tree(APP_ROOT)[APP_ROOT / "core/MCP_unified/config.py"])
    assert len(names) >= 70  # the pin is not vacuous: dev has 76
    assert names & SHARED_KEYS == frozenset()


if __name__ == "__main__":
    if sys.argv[1:] == ["--summary"]:
        print(summary(measure()))
        raise SystemExit(0)
    raise SystemExit(_ceilings.main(RATCHET, measure))
