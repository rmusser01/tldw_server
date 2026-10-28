"""Ratchet: the backend's layering contract (TASK-13950.9).

About 100 backend packages sit in one import cycle, and only core -> api was ratcheted
(file-keyed, in no required gate). This freezes every import that breaks the layer order,
keyed by (importer package, target package), so the count of each edge may only fall.
Phase 1 of the 2026-10-25 architecture roadmap cuts the cycle by deleting these lines.

Layers, lowest first (LAYERS holds the package patterns; first match wins):

    foundation  core.config*, core.exception* (exceptions, exception_types), core.Utils,
                core.Logging, core.testing, core.http_client, core.feature_flags,
                core.custom_openai_providers
    platform    core.Metrics, core.Infrastructure, core.Security, core.Audit
    persistence core.DB_Management
    domain      every other core package (and the empty ``core`` package itself)
    services    services
    api         api.*
    main        main, and ``app`` (tldw_Server_API/app/__init__.py)

Packages. A module's package is a prefix of its name under tldw_Server_API/app:
``core.<X>`` (a top-level core package or module: ``core.Chat``, ``core.config``),
``api.<version>.<X>`` (``api.v1.schemas``, ``api.v1.API_Deps``, ``api.v1.utils``), except
that endpoints are split by family, ``api.<version>.endpoints.<family>`` (``.media``,
``.chat``, ``._pagination_utils``), and ``services``/``main`` are one package each. The
importer is the package of the importing module; moving an import between modules of one
package (a shimmed move) leaves the baseline alone. Failures print the importing modules.

Counting. One statement counts once for each package it imports from, other than its own:
  * ``import a.b.c``                 -> the package of ``a.b.c``;
  * ``from a.b import c``            -> the package of ``a.b.c`` if that is a module or
                                        package in the tree, else the package of ``a.b``;
  * relative imports are resolved against the importing module first.
Imports at module level, in class bodies and inside functions all count (a function-level
import still closes the cycle, and counting it stops a gate from being met by moving
imports into functions). The body of ``if TYPE_CHECKING:`` is skipped only when the name
is bound from typing: ``from typing import TYPE_CHECKING [as X]``, or ``typing.TYPE_CHECKING``
after ``import typing [as t]``. A module-local ``TYPE_CHECKING = True`` hides nothing. The
``else`` branch runs, so it counts. Imports in the body of a ``try`` that catches ImportError
or ModuleNotFoundError count; the counter reports how many. Modules in a ``tests`` package
directory inside the app (core/MCP_unified/tests, core/Scheduler/tests) are test code the app
never imports, so they are not scanned; a module named ``tests.py`` is.

Known limits:
  * ``importlib.import_module``/``__import__`` calls are not counted. At the time of writing
    3 of them, named by a module-level string constant, reach pairs that static imports
    already baseline: core.Evaluations -> api.v1.endpoints.embeddings_v5_production_enhanced
    (Evaluations/recipes/embeddings_recipe_hints.py) and core.MCP_unified -> services, twice
    (MCP_unified/adapters/tldw_policy.py). Turning a pair's last static import into an
    importlib call is not a cut, so reviewers must not accept a deleted line on that basis.
  * Only direct edges are ruled, so a same-layer rule can be laundered through an unruled
    package of the same layer: core.LLM_Calls -> core.X -> core.Chat passes the sibling
    bans. An upward edge cannot be laundered: the second hop is still upward.

Rules. A (importer, target) pair is ratcheted when it
  * goes up the layers ("upward (<layer> -> <layer>)"): this covers core -> api,
    services -> api and DB_Management -> domain;
  * joins two endpoint families ("endpoint -> endpoint");
  * imports an endpoint family from any other api package ("api -> endpoint"), except the
    router assembly (ROUTER_ASSEMBLY), so endpoint code cannot be re-exported through
    api/v1/utils, schemas or API_Deps. Shared code moves, without a re-export, into
    api/v1/utils or a core/service module;
  * imports the router assembly from anywhere but ASSEMBLERS (main and the router
    registry) ("router assembly"), so the exempt assembly cannot launder endpoints either;
  * is a sibling ban ("sibling ban"): core.LLM_Calls -> core.Chat and -> core.AuthNZ.
tests/lint/test_core_to_api_import_boundary.py is a view of the core -> api pairs.

Baseline: layering_baseline.txt, one ``<importer> -> <target>: <statements>`` line per
ratcheted pair. A new pair or a higher count fails. Unlike the size ratchets, whose
numbers are ceilings that may shrink without an edit, a count here must equal the code:
an edge that is gone or fell is stale and fails until its line is deleted or lowered
(``--tighten`` does it). The
baseline is Phase 1's progress gate, so every cut is locked in by the PR that makes it,
and an edge cut in one PR cannot quietly come back under its old count in the next.
On a baseline conflict or a stale failure after merging dev, take dev's side, run
``--tighten``, and re-add any ``# bump:`` lines of your own.

Bump with reason. A new pair, or a raised count, needs a reason on its line,
``<importer> -> <target>: <n>  # bump: <reason> (TASK-<n> or #<PR>)``: non-empty, citing a
task or PR, with no unfilled ``<...>`` placeholder, and different from the line's reason
at the merge base of $BACKLOG_TASK_FORMAT_BASE (backend-required sets it to the PR base;
a local run defaults to origin/dev) and HEAD. The bump is then reviewed in the PR. As the
count must equal the code, a raise cannot reserve headroom. The same rule as _ceilings.py,
for a text baseline.

Since TASK-13950.12, foundation and core.Metrics have no baseline entries, and
test_foundation_and_metrics_import_only_foundation and
test_foundation_and_metrics_load_only_foundation_at_import_time pin them at zero,
statically and at import time, with no ``# bump:`` exception.

Commands:
    python -m tldw_Server_API.tests.lint.test_layering_contract                    # report
    python -m tldw_Server_API.tests.lint.test_layering_contract --tighten          # cuts only
    python -m tldw_Server_API.tests.lint.test_layering_contract --write-baseline   # everything
The report gives the pair, statement and module counts and both package SCC sizes
(module-level, and with function-level imports) for the roadmap's phase notes.
``--tighten`` only lowers fallen counts and deletes gone pairs; it never records growth.
``--write-baseline`` records every measurement and prints each raised or new pair, which
then needs a bump reason. Both keep each surviving line's reason.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from fnmatch import fnmatchcase
from functools import cache
from pathlib import Path
from typing import NamedTuple

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT, parse_tree
from tldw_Server_API.tests.lint._ceilings import BASE_ENV

pytestmark = pytest.mark.unit

APP = "tldw_Server_API.app"
BASELINE = Path(__file__).resolve().with_name("layering_baseline.txt")
COMMAND = "python -m tldw_Server_API.tests.lint.test_layering_contract"
REASON_PLACEHOLDER = "<why the growth is needed> (TASK-<n> or #<PR>)"

LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "foundation",
        (
            "core.config*",
            "core.exception*",
            "core.Utils",
            "core.Logging",
            "core.testing",
            "core.http_client",
            "core.feature_flags",
            "core.custom_openai_providers",
        ),
    ),
    ("platform", ("core.Metrics", "core.Infrastructure", "core.Security", "core.Audit")),
    ("persistence", ("core.DB_Management",)),
    ("domain", ("core", "core.*")),
    ("services", ("services",)),
    ("api", ("api", "api.*")),
    ("main", ("main", "app")),
)
SIBLING_BANS = frozenset({("core.LLM_Calls", "core.Chat"), ("core.LLM_Calls", "core.AuthNZ")})
FOUNDATION_ONLY_IMPORTERS = frozenset({"core.Metrics"})  # platform, but pinned like foundation (TASK-13950.12)
ROUTER_ASSEMBLY = frozenset({"api.v1.router_groups", "api.v1.router_registry", "api.v2.router"})
ASSEMBLERS = frozenset({"main", "api.v1.router_registry"})  # the only importers of ROUTER_ASSEMBLY
_SHARE = "Move the code (don't re-export it) into api/v1/utils or a core/service module."
FIXES = {
    "upward": "Invert it: move the shared code down to the importer's layer, or inject it from above.",
    "endpoint -> endpoint": _SHARE,
    "api -> endpoint": f"Endpoint modules are not a library. {_SHARE}",
    "router assembly": f"Only main and api/v1/router_registry import the router assembly. {_SHARE}",
    "sibling ban": "core.LLM_Calls must not depend on core.Chat or core.AuthNZ: have the caller pass it in.",
}

Pair = tuple[str, str]
Baseline = dict[Pair, tuple[int, str]]  # pair -> (statements, bump reason or "")
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})
_SCOPES = ("body", "orelse", "finalbody", "handlers", "cases")
_LINE = re.compile(r"(\S+) -> (\S+): (\d+)(?:\s+# bump:(.*))?")
_REFERENCE = re.compile(r"TASK-\d|#\d")
_UNFILLED = re.compile(r"<[^<>]*>")


# --- The counter --------------------------------------------------------------------------------


class Import(NamedTuple):
    module: str  # the importing module, relative to the app ("core.Chat.chat_service")
    target: str  # the imported package
    function_level: bool
    guarded: bool  # in the body of a try that catches ImportError or ModuleNotFoundError


def package_of(module: str) -> str:
    """The layer package of an app-relative module name; see the module docstring."""
    parts = module.split(".") if module else []
    if not parts:
        return "app"
    depth = {"core": 2, "api": 4 if parts[2:3] == ["endpoints"] else 3}.get(parts[0], 1)
    return ".".join(parts[:depth])


@cache
def layer_of(package: str) -> int:
    for index, (_, patterns) in enumerate(LAYERS):
        if any(fnmatchcase(package, pattern) for pattern in patterns):
            return index
    raise LookupError(f"{package} is in no layer: add it to LAYERS in {__file__}")


def _is_endpoint(package: str) -> bool:
    return package.split(".")[0] == "api" and package.split(".")[2:3] == ["endpoints"]


@cache
def rule(importer: str, target: str) -> str | None:
    """The rule an import from ``importer`` into ``target`` breaks, or None."""
    if (importer, target) in SIBLING_BANS:
        return "sibling ban"
    if _is_endpoint(importer) and _is_endpoint(target):
        return "endpoint -> endpoint"
    if target in ROUTER_ASSEMBLY and importer not in ASSEMBLERS:
        return "router assembly"
    if _is_endpoint(target) and importer.split(".")[0] == "api" and importer not in ROUTER_ASSEMBLY:
        return "api -> endpoint"
    low, high = layer_of(importer), layer_of(target)
    return f"upward ({LAYERS[low][0]} -> {LAYERS[high][0]})" if low < high else None


def _module_name(relative: Path) -> str:
    parts = relative.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _app_relative(name: str) -> str | None:
    if name == APP:
        return ""
    return name[len(APP) + 1 :] if name.startswith(APP + ".") else None


def _dotted(test: ast.expr) -> str:
    if isinstance(test, ast.Attribute) and isinstance(test.value, ast.Name):
        return f"{test.value.id}.{test.attr}"
    return test.id if isinstance(test, ast.Name) else ""


def _bind_type_checking(node: ast.stmt, names: set[str]) -> None:
    """Track the names bound to typing.TYPE_CHECKING, in source order; a rebinding drops one."""
    if isinstance(node, ast.ImportFrom) and node.module == "typing" and not node.level:
        names.update(alias.asname or alias.name for alias in node.names if alias.name == "TYPE_CHECKING")
    elif isinstance(node, ast.Import):
        names.update(f"{alias.asname or alias.name}.TYPE_CHECKING" for alias in node.names if alias.name == "typing")
    elif isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names.difference_update(target.id for target in targets if isinstance(target, ast.Name))


def _catches_import_error(node: ast.Try | ast.TryStar) -> bool:
    for handler in node.handlers:
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        if any(isinstance(kind, ast.Name) and kind.id in _IMPORT_ERRORS for kind in types):
            return True
    return False


def _walk(
    statements: Sequence[ast.AST], function_level: bool, guarded: bool, type_checking: set[str]
) -> Iterator[tuple[ast.Import | ast.ImportFrom, bool, bool]]:
    """Every import statement, in source order, outside ``if TYPE_CHECKING:`` bodies."""
    for node in statements:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            _bind_type_checking(node, type_checking)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, function_level, guarded
        elif isinstance(node, ast.If) and _dotted(node.test) in type_checking:
            yield from _walk(node.orelse, function_level, guarded, type_checking)
        else:
            inner = function_level or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            catches = isinstance(node, (ast.Try, ast.TryStar)) and _catches_import_error(node)
            for scope in _SCOPES:
                body = getattr(node, scope, ())
                yield from _walk(body, inner, guarded or (catches and scope == "body"), type_checking)


def _targets(node: ast.Import | ast.ImportFrom, module: str, is_package: bool, known: frozenset[str]) -> set[str]:
    """The app packages one import statement reads from."""
    if isinstance(node, ast.Import):
        names = [_app_relative(alias.name) for alias in node.names]
        return {package_of(name) for name in names if name is not None}
    if node.level:
        parts = module.split(".") if module else []
        parts = parts if is_package else parts[:-1]
        if node.level - 1 > len(parts):
            return set()
        base = ".".join([APP, *parts[: len(parts) - node.level + 1], *([node.module] if node.module else [])])
    else:
        base = node.module or ""
    targets = set()
    for alias in node.names:
        name = _app_relative(f"{base}.{alias.name}")
        name = name if name in known else _app_relative(base)
        if name is not None:
            targets.add(package_of(name))
    return targets


def scan(root: Path = APP_ROOT) -> tuple[Import, ...]:
    """One record per (import statement, app package it reads from), self-imports excluded."""
    return _scan(root.resolve())


@cache
def _scan(root: Path) -> tuple[Import, ...]:
    trees = parse_tree(root)
    names = {path: _module_name(path.relative_to(root)) for path in trees}
    known = frozenset(".".join(n.split(".")[:i]) for n in names.values() for i in range(n.count(".") + 2))
    records = []
    for path, tree in trees.items():
        module = names[path]
        if "tests" in path.relative_to(root).parts[:-1]:
            continue  # an in-app test package (core/MCP_unified/tests, core/Scheduler/tests), not app code
        importer = package_of(module)
        for node, function_level, guarded in _walk(tree.body, False, False, set()):
            for target in sorted(_targets(node, module, path.name == "__init__.py", known) - {importer}):
                records.append(Import(module, target, function_level, guarded))
    return tuple(records)


def measure(root: Path = APP_ROOT) -> dict[Pair, Counter[str]]:
    """Every ratcheted (importer package, target package) pair -> statements per importing module."""
    measured: defaultdict[Pair, Counter[str]] = defaultdict(Counter)
    for record in scan(root):
        pair = (package_of(record.module), record.target)
        if rule(*pair):
            measured[pair][record.module] += 1
    return dict(measured)


def scan_upward_edges(root: Path = APP_ROOT) -> set[Pair]:
    """Every (importer package, target package) pair that goes up the layers."""
    pairs = {(package_of(record.module), record.target) for record in scan(root)}
    return {pair for pair in pairs if layer_of(pair[0]) < layer_of(pair[1])}


def _largest_scc(edges: set[Pair]) -> int:
    """Tarjan's algorithm; the package graph is a few hundred nodes, so recursion is fine."""
    graph: defaultdict[str, set[str]] = defaultdict(set)
    for importer, target in edges:
        graph[importer].add(target)
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    largest = 0

    def visit(node: str) -> None:
        nonlocal largest
        index[node] = low[node] = len(index)
        stack.append(node)
        for nxt in graph[node]:
            if nxt not in index:
                visit(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            size = len(stack) - stack.index(node)
            del stack[-size:]
            largest = max(largest, size)

    for node in list(graph):
        if node not in index:
            visit(node)
    return largest


def package_scc_sizes(root: Path = APP_ROOT) -> dict[str, int]:
    """The largest package import cycle, counting module-level imports only, and all imports."""
    records = scan(root)
    module_level = {(package_of(r.module), r.target) for r in records if not r.function_level}
    every = {(package_of(r.module), r.target) for r in records}
    return {"module_level": _largest_scc(module_level), "with_function_level": _largest_scc(every)}


def foundation_breaches(records: Sequence[Import]) -> list[str]:
    """``<importer> -> <target> (<module>)`` for each import from foundation or core.Metrics into a higher layer."""
    breaches: set[str] = set()
    for record in records:
        importer = package_of(record.module)
        from_pinned = layer_of(importer) == 0 or importer in FOUNDATION_ONLY_IMPORTERS
        if from_pinned and layer_of(record.target) > 0:
            breaches.add(f"{importer} -> {record.target} ({record.module})")
    return sorted(breaches)


def foundation_modules(root: Path = APP_ROOT) -> list[str]:
    """Every foundation module, app-relative and sorted; in-app ``tests`` packages are skipped, as in _scan."""
    names = (
        _module_name(path.relative_to(root))
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts[:-1]
    )
    return sorted(name for name in names if layer_of(package_of(name)) == 0)


def upper_layer_loads(modules: Sequence[str]) -> dict[str, list[str]]:
    """Import ``modules`` in a fresh interpreter: the imports that failed, and the packages above foundation loaded."""
    script = textwrap.dedent(
        f"""
        import importlib, json, sys
        failed = []
        for module in {list(modules)!r}:
            try:
                importlib.import_module({APP!r} + "." + module)
            except Exception as exc:
                failed.append(f"{{module}}: {{type(exc).__name__}}: {{exc}}")
        print(json.dumps({{"failed": failed, "loaded": sorted(sys.modules)}}))
        """
    )
    result = subprocess.run(  # fixed argv, no shell
        [sys.executable, "-c", script], cwd=REPO_ROOT, check=True, capture_output=True, text=True, timeout=300
    )
    report = json.loads(result.stdout.splitlines()[-1])
    loaded = {package_of(name) for name in map(_app_relative, report["loaded"]) if name is not None}
    return {"failed": report["failed"], "upper": sorted(p for p in loaded - {"app", "core"} if layer_of(p) > 0)}


# --- The baseline -------------------------------------------------------------------------------


def _shown(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def parse(text: str, source: str = "layering baseline") -> Baseline:
    entries: Baseline = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        match = _LINE.fullmatch(line.rstrip())
        if not match or (match[1], match[2]) in entries:
            raise ValueError(
                f"{source} line {number}: expected a unique '<importer> -> <target>: <count>' "
                f"with an optional '  # bump: <reason>', got {line!r}"
            )
        entries[(match[1], match[2])] = (int(match[3]), (match[4] or "").strip())
    return entries


def load(path: Path) -> Baseline:
    return parse(path.read_text(encoding="utf-8"), _shown(path))


def dump(path: Path, entries: Mapping[Pair, tuple[int, str]]) -> None:
    header = f"# <importer package> -> <target package>: <statements>. Counts only fall; see {Path(__file__).name}.\n"
    lines = [
        f"{a} -> {b}: {count}" + (f"  # bump: {reason}" if reason else "") + "\n"
        for (a, b), (count, reason) in sorted(entries.items())
    ]
    path.write_text(header + "".join(lines), encoding="utf-8")


def rewritten(
    baseline: Mapping[Pair, tuple[int, str]], measured: Mapping[Pair, Counter[str]], *, tighten: bool
) -> tuple[Baseline, list[str]]:
    """(The baseline ``--tighten`` or ``--write-baseline`` writes, the raised or new pairs it records).

    ``tighten`` only lowers counts and drops gone pairs, so it never records growth; both keep reasons.
    """
    counts = {pair: sum(modules.values()) for pair, modules in measured.items()}
    if tighten:
        return {pair: (min(n, counts[pair]), why) for pair, (n, why) in baseline.items() if pair in counts}, []
    raised = [
        f"{a} -> {b}: {n} " + (f"(was {baseline[(a, b)][0]})" if (a, b) in baseline else "(new)")
        for (a, b), n in sorted(counts.items())
        if n > baseline.get((a, b), (0, ""))[0]
    ]
    return {pair: (n, baseline.get(pair, (0, ""))[1]) for pair, n in counts.items()}, raised


def _statements(count: int) -> str:
    return f"{count} statement{'s' if count != 1 else ''}"


def violations(measured: Mapping[Pair, Counter[str]], baseline: Mapping[Pair, tuple[int, str]]) -> list[str]:
    """One message per pair that is new, grew, or is stale (fell or is gone)."""
    problems = []
    for pair in sorted(measured.keys() | baseline.keys()):
        modules = measured.get(pair, Counter())
        now, cap = sum(modules.values()), baseline.get(pair, (0, ""))[0]
        name = f"{pair[0]} -> {pair[1]}"
        if now > cap:
            label = rule(*pair) or ""
            grown = f" > baseline {cap}" if pair in baseline else ", not in the baseline"
            importing = ", ".join(f"{module} ({count})" for module, count in sorted(modules.items()))
            problems.append(
                f"{name}: {_statements(now)}{grown} [{label}]. Importing modules: {importing}. "
                f"{FIXES.get(label.split(' (')[0], '')} If it is unavoidable, set this line in {_shown(BASELINE)}; "
                f"the raise and its reason are reviewed in the PR:\n    {name}: {now}  # bump: {REASON_PLACEHOLDER}"
            )
        elif now < cap:
            gone = (
                "the edge is gone. Delete the line"
                if not now
                else f"baseline {cap} but {now} measured. Lower it to {now}"
            )
            problems.append(
                f"{name}: stale, {gone} in {_shown(BASELINE)} to lock the cut in, or run: {COMMAND} --tighten. "
                "After merging dev, take dev's side of a baseline conflict, run --tighten, and re-add any "
                "'# bump:' lines of your own."
            )
    return problems


def unreasoned_bumps(
    baseline: Mapping[Pair, tuple[int, str]], base: Mapping[Pair, tuple[int, str]] | None
) -> list[str]:
    """Pairs added or raised since ``base`` without a new reason; ``base`` None means no check."""
    if base is None:
        return []
    problems = []
    for pair, (count, reason) in sorted(baseline.items()):
        old_count, old_reason = base.get(pair, (0, ""))
        if pair in base and count <= old_count:
            continue
        if reason and reason != old_reason and _REFERENCE.search(reason) and not _UNFILLED.search(reason):
            continue
        problems.append(
            f"{pair[0]} -> {pair[1]}: raised to {count} without a new reason in {_shown(BASELINE)}. Give one that "
            f"cites the task or PR; the raise and its reason are reviewed in the PR:\n"
            f"    {pair[0]} -> {pair[1]}: {count}  # bump: {REASON_PLACEHOLDER}"
        )
    return problems


def _git_out(repo_root: Path, *args: str) -> str:
    return subprocess.run(  # fixed git argv, no shell
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout


def base_version(path: Path, base: str, repo_root: Path = REPO_ROOT) -> Baseline | None:
    """The baseline at the merge base of ``base`` and HEAD, or None if it did not exist there."""
    merge_base = _git_out(repo_root, "merge-base", base, "HEAD").strip()
    relative = path.relative_to(repo_root).as_posix()
    if not _git_out(repo_root, "ls-tree", "--name-only", merge_base, "--", relative).strip():
        return None
    return parse(_git_out(repo_root, "show", f"{merge_base}:{relative}"), f"{relative} at {merge_base[:10]}")


# --- Self-tests: each seeds a tree under a fresh tmp root and runs the real scan ---------------


def _seed(root: Path, files: dict[str, str]) -> Path:
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    return root


def _counts(measured: Mapping[Pair, Counter[str]]) -> dict[Pair, int]:
    return {pair: sum(modules.values()) for pair, modules in measured.items()}


@pytest.mark.parametrize(
    "source",
    [
        f"from {APP}.core.Chat import chat_service\n",
        f"def build():\n    import {APP}.core.Chat.chat_service\n",
    ],
    ids=["module_level", "function_level"],
)
def test_a_seeded_upward_import_fails(tmp_path: Path, source: str) -> None:
    root = _seed(tmp_path / "app", {"core/config_x.py": source})

    (problem,) = violations(measure(root), {})

    assert problem.startswith("core.config_x -> core.Chat: 1 statement, not in the baseline"), problem
    assert "upward (foundation -> domain)" in problem and "core.config_x (1)" in problem
    assert "core.config_x -> core.Chat: 1  # bump: <why" in problem  # the line to add, reviewed in the PR


def test_type_checking_imports_are_not_counted_but_their_else_branch_is(tmp_path: Path) -> None:
    source = f"""
        import typing as t
        from typing import TYPE_CHECKING
        from typing import TYPE_CHECKING as TC
        if TYPE_CHECKING:
            from {APP}.core.Chat import a
        if t.TYPE_CHECKING:
            from {APP}.core.Chat import b
        def f():
            if TC:
                from {APP}.core.Chat import c
            else:
                from {APP}.services import d
    """
    root = _seed(tmp_path / "app", {"core/config_x.py": source})

    assert _counts(measure(root)) == {("core.config_x", "services"): 1}


def test_only_type_checking_bound_from_typing_hides_imports(tmp_path: Path) -> None:
    files = {
        "core/config_a.py": f"TYPE_CHECKING = True\nif TYPE_CHECKING:\n    from {APP}.core.Chat import a\n",
        "core/config_b.py": f"""
            from typing import TYPE_CHECKING
            TYPE_CHECKING = True
            if TYPE_CHECKING:
                from {APP}.core.Chat import b
        """,
        "core/config_c.py": f"""
            import typing
            from mylib import TYPE_CHECKING as OTHER
            if typing.TYPE_CHECKING:
                from {APP}.core.Chat import c
            if OTHER:
                from {APP}.core.RAG import d
        """,
    }

    assert _counts(measure(_seed(tmp_path / "app", files))) == {
        ("core.config_a", "core.Chat"): 1,
        ("core.config_b", "core.Chat"): 1,
        ("core.config_c", "core.RAG"): 1,
    }


def test_moving_an_import_inside_the_importer_package_keeps_the_baseline_and_growth_fails(tmp_path: Path) -> None:
    baseline = {("core.Utils", "core.Chat"): (1, "")}

    def problems(name: str, files: dict[str, str]) -> list[str]:
        return violations(measure(_seed(tmp_path / name / "app", files)), baseline)

    assert problems("before", {"core/Utils/a.py": f"from {APP}.core.Chat import x\n", "core/Utils/b.py": ""}) == []
    assert problems("renamed", {"core/Utils/b.py": f"from {APP}.core.Chat import x\n"}) == []
    (grown,) = problems(
        "grown",
        {
            "core/Utils/b.py": f"from {APP}.core.Chat import x\n",
            "core/Utils/c.py": "def f():\n    from ..Chat import y\n",
        },
    )
    assert grown.startswith("core.Utils -> core.Chat: 2 statements > baseline 1"), grown
    assert "core.Utils.b (1), core.Utils.c (1)" in grown  # the importing modules are printed
    (new_edge,) = problems(
        "new_edge", {"core/Utils/b.py": f"from {APP}.core.Chat import x\nfrom {APP}.core.RAG import y\n"}
    )
    assert new_edge.startswith("core.Utils -> core.RAG: 1 statement, not in the baseline"), new_edge


def test_counts_may_only_fall_so_a_lower_count_or_a_gone_edge_is_stale(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {"core/Utils/a.py": f"from {APP}.core.Chat import x\n"})
    baseline = {("core.Utils", "core.Chat"): (3, ""), ("core.Utils", "core.RAG"): (1, "")}

    lowered, gone = violations(measure(root), baseline)

    assert lowered.startswith("core.Utils -> core.Chat: stale, baseline 3 but 1 measured. Lower it to 1"), lowered
    assert gone.startswith("core.Utils -> core.RAG: stale, the edge is gone. Delete the line"), gone


def test_import_forms_resolve_to_their_target_package(tmp_path: Path) -> None:
    source = f"""
        import os, {APP}.core.Chat.a, {APP}.services.b
        from {APP}.core import Chat, NAME_IN_CORE_INIT
        from {APP}.api.v1.endpoints import media
        from .. import Chat
        from ..Chat.sub import thing
        from . import sibling
        from .pkg import helper
        try:
            from {APP}.main import app
        except (ModuleNotFoundError, ValueError):
            app = None
    """
    files = {
        "core/__init__.py": "NAME_IN_CORE_INIT = 1\n",
        "core/Chat/__init__.py": "",
        "core/Utils/mod.py": source,
        "core/Utils/pkg/__init__.py": "from ... import Chat\nfrom ...Chat.sub import thing\nfrom .. import mod\n",
        "api/v1/endpoints/media/__init__.py": "",
        "core/Utils/tests/test_mod.py": f"from {APP}.main import app\n",  # in-app test code: not scanned
        "core/Logging/tests.py": f"from {APP}.main import app\n",  # a module named tests is app code
    }
    root = _seed(tmp_path / "app", files)

    assert _counts(measure(root)) == {
        ("core.Utils", "core.Chat"): 6,  # import a.b, from core import Chat, two relative forms, twice in pkg
        ("core.Utils", "services"): 1,  # one ``import`` statement naming two packages counts once for each
        ("core.Utils", "core"): 1,  # a name defined in core/__init__.py is the ``core`` package
        ("core.Utils", "api.v1.endpoints.media"): 1,
        ("core.Utils", "main"): 1,
        ("core.Logging", "main"): 1,
    }
    assert sum(record.guarded for record in scan(root)) == 1  # under try/except ImportError, still counted


def test_endpoint_sibling_services_and_db_management_rules(tmp_path: Path) -> None:
    files = {
        "api/v1/endpoints/chat.py": f"""
            from {APP}.api.v1.endpoints.media import upload
            from {APP}.api.v1.utils import helpers
            from {APP}.api.v1.schemas import chat_schemas
            from {APP}.core.Chat import chat_service
        """,
        "api/v1/endpoints/media/upload.py": f"from {APP}.api.v1.endpoints.media import process\n",
        "core/LLM_Calls/x.py": f"from {APP}.core.Chat import a\nfrom {APP}.core.AuthNZ import b\nfrom {APP}.core.RAG import c\n",
        "core/Chat/x.py": f"from {APP}.core.LLM_Calls import a\n",
        "services/worker.py": f"def f():\n    from {APP}.api.v1.schemas import s\n",
        "core/DB_Management/db.py": f"from {APP}.core.Chat import a\n",
    }
    root = _seed(tmp_path / "app", files)

    assert {pair: rule(*pair) for pair in measure(root)} == {
        ("api.v1.endpoints.chat", "api.v1.endpoints.media"): "endpoint -> endpoint",
        ("core.LLM_Calls", "core.Chat"): "sibling ban",
        ("core.LLM_Calls", "core.AuthNZ"): "sibling ban",
        ("services", "api.v1.schemas"): "upward (services -> api)",
        ("core.DB_Management", "core.Chat"): "upward (persistence -> domain)",
    }
    assert scan_upward_edges(root) == {("services", "api.v1.schemas"), ("core.DB_Management", "core.Chat")}


def test_re_exporting_endpoint_code_through_another_api_package_fails(tmp_path: Path) -> None:
    files = {
        "api/v1/utils/media_helpers.py": f"from {APP}.api.v1.endpoints.media import upload as upload\n",
        "api/v1/router_groups/content.py": f"def f():\n    from {APP}.api.v1.endpoints import media\n",
        "api/v2/router.py": f"from {APP}.api.v2.endpoints import user_profiles\n",  # router assembly: exempt
    }

    (problem,) = violations(measure(_seed(tmp_path / "app", files)), {})

    assert problem.startswith("api.v1.utils -> api.v1.endpoints.media: 1 statement, not in the baseline"), problem
    assert "[api -> endpoint]" in problem and "don't re-export it" in problem


def test_only_main_and_the_registry_import_the_router_assembly(tmp_path: Path) -> None:
    files = {
        "api/v1/endpoints/chat.py": f"from {APP}.api.v1.router_groups import content\n",  # launders endpoints
        "api/v1/router_registry.py": f"from {APP}.api.v1.router_groups import content\n",
        "api/v2/router.py": "",
        "main.py": f"from {APP}.api.v1 import router_registry\nfrom {APP}.api.v2 import router\n",
    }

    (problem,) = violations(measure(_seed(tmp_path / "app", files)), {})

    assert problem.startswith("api.v1.endpoints.chat -> api.v1.router_groups: 1 statement, not in the baseline"), (
        problem
    )
    assert "[router assembly]" in problem


def test_package_scc_sizes_report_module_level_and_function_level_cycles(tmp_path: Path) -> None:
    files = {
        "core/Chat/a.py": f"from {APP}.core.RAG import x\n",
        "core/RAG/a.py": f"from {APP}.core.Chat import x\ndef f():\n    from {APP}.core.Notes import y\n",
        "core/Notes/a.py": f"""
            from typing import TYPE_CHECKING
            from {APP}.core.Chat import x
            if TYPE_CHECKING:
                from {APP}.core.Jobs import z
        """,
        "core/Jobs/a.py": f"from {APP}.core.Notes import x\n",
    }

    assert package_scc_sizes(_seed(tmp_path / "app", files)) == {"module_level": 2, "with_function_level": 3}


def test_tighten_only_lowers_and_drops_and_write_baseline_reports_what_it_raises(tmp_path: Path) -> None:
    source = f"from {APP}.core.Chat import a\nfrom {APP}.services import b\nfrom {APP}.services import c\nimport {APP}.main\n"
    measured = measure(_seed(tmp_path / "app", {"core/Utils/a.py": source}))
    baseline = {
        ("core.Utils", "core.Chat"): (3, "split pending (TASK-1)"),
        ("core.Utils", "core.RAG"): (1, ""),
        ("core.Utils", "services"): (1, ""),
    }

    assert rewritten(baseline, measured, tighten=True) == (
        {("core.Utils", "core.Chat"): (1, "split pending (TASK-1)"), ("core.Utils", "services"): (1, "")},
        [],
    )
    entries, raised = rewritten(baseline, measured, tighten=False)
    assert entries == {
        ("core.Utils", "core.Chat"): (1, "split pending (TASK-1)"),
        ("core.Utils", "services"): (2, ""),
        ("core.Utils", "main"): (1, ""),
    }
    assert raised == ["core.Utils -> main: 1 (new)", "core.Utils -> services: 2 (was 1)"]
    stale = violations(measured, baseline)[0]
    assert "--tighten" in stale and "take dev's side" in stale  # the quick fix and the merge recipe


def test_the_baseline_round_trips_with_its_bump_reasons_and_rejects_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "layering_baseline.txt"
    entries = {("core.Utils", "core.Chat"): (2, ""), ("services", "api.v1.schemas"): (5, "security fix (TASK-1)")}

    dump(path, entries)

    assert load(path) == entries
    text = path.read_text(encoding="utf-8")
    assert "\ncore.Utils -> core.Chat: 2\nservices -> api.v1.schemas: 5  # bump: security fix (TASK-1)\n" in text
    path.write_text("core.Utils -> core.Chat 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load(path)
    path.write_text("a -> b: 1\na -> b: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load(path)


def test_a_new_or_raised_entry_needs_a_new_reason_with_a_reference() -> None:
    base = {("a", "b"): (2, ""), ("c", "d"): (1, "old fix (TASK-1)")}

    def problems(entries: dict) -> list[str]:
        return unreasoned_bumps({**base, **entries}, base)

    assert problems({}) == []
    assert problems({("a", "b"): (1, "")}) == []  # lowered
    (no_reason,) = problems({("a", "b"): (3, "")})
    assert no_reason.startswith("a -> b: raised to 3 without a new reason") and "reviewed in the PR" in no_reason
    assert problems({("a", "b"): (3, REASON_PLACEHOLDER)})
    assert problems({("a", "b"): (3, "<why the growth is needed> (TASK-2)")})  # half-filled placeholder
    assert problems({("a", "b"): (3, "needed it")})  # no TASK-/#PR reference
    assert problems({("a", "b"): (3, "security fix (#3334)")}) == []
    assert problems({("c", "d"): (2, "old fix (TASK-1)")})  # a reused reason is not new
    assert problems({("e", "f"): (1, "")})  # a new pair is a raise
    assert problems({("e", "f"): (1, "split pending (TASK-13950.12)")}) == []
    assert unreasoned_bumps({("e", "f"): (1, "")}, None) == []  # the PR that adds the baseline


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_the_reason_check_reads_the_baseline_at_the_merge_base(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    path = repo / "layering_baseline.txt"
    _git(repo, "init", "-q", "-b", "dev")
    (repo / "README").write_text("x\n", encoding="utf-8")
    _git(repo, "add", "README")
    _git(repo, "commit", "-q", "-m", "before the baseline")
    _git(repo, "checkout", "-q", "-b", "adds-baseline")
    dump(path, {("a", "b"): (1, "")})
    _git(repo, "add", path.name)
    _git(repo, "commit", "-q", "-m", "baseline")

    assert base_version(path, "dev", repo) is None  # the PR that adds it needs no reasons

    _git(repo, "checkout", "-q", "-b", "bump")
    dump(path, {("a", "b"): (2, "")})
    _git(repo, "commit", "-q", "-am", "bump without a reason")

    assert base_version(path, "adds-baseline", repo) == {("a", "b"): (1, "")}
    assert unreasoned_bumps(load(path), base_version(path, "adds-baseline", repo))


def test_a_seeded_foundation_or_metrics_breach_is_reported_even_inside_a_function(tmp_path: Path) -> None:
    files = {
        "core/Utils/x.py": f"def f():\n    from {APP}.core.Security import egress\n",
        "core/Metrics/y.py": f"from {APP}.core.Security import egress\n",
        "core/Metrics/z.py": f"from {APP}.core.Utils import x\n",
    }
    assert foundation_breaches(scan(_seed(tmp_path / "app", files))) == [
        "core.Metrics -> core.Security (core.Metrics.y)",
        "core.Utils -> core.Security (core.Utils.x)",
    ]


def test_upper_layer_loads_reports_upper_packages_and_failed_imports() -> None:
    assert upper_layer_loads(["core.Context_Integrity.canonicalization", "core.no_such_module"]) == {
        "failed": [f"core.no_such_module: ModuleNotFoundError: No module named '{APP}.core.no_such_module'"],
        "upper": ["core.Context_Integrity"],
    }


# --- The ratchet --------------------------------------------------------------------------------


def test_imports_follow_the_layering_contract() -> None:
    problems = violations(measure(), load(BASELINE))
    assert not problems, f"{len(problems)} layering problem(s):\n" + "\n".join(problems)


def test_raised_counts_carry_a_new_reason() -> None:
    base = os.environ.get(BASE_ENV) or "origin/dev"
    try:
        old = base_version(BASELINE, base)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get(BASE_ENV):
            raise
        pytest.skip(f"cannot find the merge base with {base}; fetch it or set {BASE_ENV} to a base commit ({exc})")
    problems = unreasoned_bumps(load(BASELINE), old)
    assert not problems, "\n".join(problems)


def test_foundation_and_metrics_import_only_foundation() -> None:
    breaches = foundation_breaches(scan())
    assert not breaches, (
        "Foundation packages and core.Metrics import only foundation (TASK-13950.12); no `# bump:` "
        "admits one. Move the shared code down or inject it from above:\n" + "\n".join(breaches)
    )


def test_foundation_and_metrics_load_only_foundation_at_import_time() -> None:
    modules = foundation_modules()
    assert {"core.http_client", "core.Utils.egress"} <= set(modules)  # the pin is not vacuous
    assert upper_layer_loads(modules) == {"failed": [], "upper": []}
    assert upper_layer_loads(["core.Metrics"]) == {"failed": [], "upper": ["core.Metrics"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Report the layering contract; rewrite {_shown(BASELINE)}.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tighten", action="store_true", help="only lower fallen counts and drop gone pairs")
    mode.add_argument("--write-baseline", action="store_true", help="record every ratcheted pair, growth included")
    args = parser.parse_args(argv)
    measured = measure()
    if args.tighten or args.write_baseline:
        entries, raised = rewritten(load(BASELINE) if BASELINE.exists() else {}, measured, tighten=args.tighten)
        dump(BASELINE, entries)
        if raised:
            print(
                f"Recorded {len(raised)} raised or new pair(s); each needs '  # bump: {REASON_PLACEHOLDER}' "
                "on its line before CI passes:\n  " + "\n  ".join(raised)
            )
    records = scan()
    ratcheted = [r for r in records if rule(package_of(r.module), r.target)]
    upward = [r for r in records if layer_of(package_of(r.module)) < layer_of(r.target)]
    upward_entries = {(r.module, r.target) for r in upward}
    module_level = {(r.module, r.target) for r in upward if not r.function_level}
    print(
        f"ratcheted now: {len(measured)} pairs, {len(ratcheted)} statements "
        f"({sum(r.guarded for r in ratcheted)} under try/except ImportError) "
        f"from {len({r.module for r in ratcheted})} modules\n"
        f"upward (module -> package) entries: {len(module_level)} module-level from "
        f"{len({m for m, _ in module_level})} modules, {len(upward_entries)} with function-level from "
        f"{len({m for m, _ in upward_entries})} modules\n"
        f"largest package SCC: {package_scc_sizes()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
