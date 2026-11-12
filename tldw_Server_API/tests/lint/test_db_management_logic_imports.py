"""Ratchet: DB_Management imports of modules that load DB code at import (TASK-13950.16).

The layering ratchet (test_layering_contract.py) freezes every ``core.DB_Management -> core.*``
pair, contracts and logic alike. This one separates the cycle-closing edges. A target is
DB-loading when importing it loads ``core.DB_Management``: it, or a parent package ``__init__``
of it, imports DB code at module level, followed transitively. A DB_Management import of any
other module imports no DB code, which is what a contract is. Entries are keyed by
(importing module, target module); counts only fall, and a fallen or gone entry is stale.

Known limit: a target whose DB import is only inside a function is not followed, though the
layering pairs still count that import. TASK-13950.67 empties this baseline.

sync_contract_problems pins core.Sync_Contracts, the contracts DB_Management shares with core.Sync:
it imports no core.Sync or core.DB_Management code at any level and loads no DB code.

    python -m tldw_Server_API.tests.lint.test_db_management_logic_imports --tighten
    python -m tldw_Server_API.tests.lint.test_db_management_logic_imports --write-baseline
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections import Counter, defaultdict
from collections.abc import Mapping
from functools import cache
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, parse_tree
from tldw_Server_API.tests.lint._ceilings import BASE_ENV
from tldw_Server_API.tests.lint.test_layering_contract import (
    APP,
    REASON_PLACEHOLDER,
    Pair,
    _counts,
    _seed,
    _shown,
    _statements,
    _walk,
    base_version,
    load,
    module_names,
    package_of,
    rewritten,
    rule,
    scan,
    scan_upward_edges,
    target_modules,
    unreasoned_bumps,
)

pytestmark = pytest.mark.unit

BASELINE = Path(__file__).resolve().with_name("db_management_logic_imports.txt")
COMMAND = "python -m tldw_Server_API.tests.lint.test_db_management_logic_imports"
PERSISTENCE = "core.DB_Management"
LABEL = "loads DB code at import"
SYNC_CONTRACTS = "core.Sync_Contracts"


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


@cache
def _module_edges(root: Path) -> dict[str, set[str]]:
    """module -> the modules its module-level imports read from (app code only)."""
    names, known = module_names(root)
    edges: defaultdict[str, set[str]] = defaultdict(set)
    for path, tree in parse_tree(root).items():
        if "tests" in path.relative_to(root).parts[:-1]:
            continue
        module = names[path]
        for node, function_level, _ in _walk(tree.body, False, False, set()):
            if not function_level:
                edges[module] |= target_modules(node, module, path.name == "__init__.py", known)
    return dict(edges)


@cache
def db_loading(root: Path = APP_ROOT) -> frozenset[str]:
    """Every module that loads core.DB_Management when imported; see the module docstring."""
    root = root.resolve()
    edges = _module_edges(root)
    modules = set(module_names(root)[0].values())
    loaded = {m for m in modules if m == PERSISTENCE or m.startswith(PERSISTENCE + ".")}
    changed = True
    while changed:
        changed = False
        for module in modules - loaded:
            targets = edges.get(module, set())
            if any(p in loaded for p in _parents(module)) or any(
                t in loaded or any(p in loaded for p in _parents(t)) for t in targets
            ):
                loaded.add(module)
                changed = True
    return frozenset(loaded)


def measure(root: Path = APP_ROOT) -> dict[Pair, Counter[str]]:
    """(importing DB_Management module, DB-loading target module) -> statements per importing module."""
    root = root.resolve()
    loading = db_loading(root)
    names, known = module_names(root)
    measured: defaultdict[Pair, Counter[str]] = defaultdict(Counter)
    for path, tree in parse_tree(root).items():
        module = names[path]
        if package_of(module) != PERSISTENCE or "tests" in path.relative_to(root).parts[:-1]:
            continue
        for node, _, _ in _walk(tree.body, False, False, set()):
            for target in target_modules(node, module, path.name == "__init__.py", known):
                if target in loading and rule(PERSISTENCE, package_of(target)):
                    measured[(module, target)][module] += 1
    return dict(measured)


def sync_contract_problems(root: Path = APP_ROOT) -> list[str]:
    """core.Sync_Contracts imports no core.Sync or core.DB_Management code, at any level, and loads no DB code."""
    root = root.resolve()
    problems = [
        f"{record.module} imports {record.target}: Sync contracts must not import core.Sync or core.DB_Management"
        for record in scan(root)
        if package_of(record.module) == SYNC_CONTRACTS and record.target in {"core.Sync", PERSISTENCE}
    ]
    problems += [
        f"{module} imports DB code when it is imported"
        for module in db_loading(root)
        if package_of(module) == SYNC_CONTRACTS
    ]
    return sorted(problems)


def violations(measured: Mapping[Pair, Counter[str]], baseline: Mapping[Pair, tuple[int, str]]) -> list[str]:
    """One message per pair that is new, grew, or is stale (fell or is gone)."""
    problems = []
    for pair in sorted(measured.keys() | baseline.keys()):
        now, cap = sum(measured.get(pair, Counter()).values()), baseline.get(pair, (0, ""))[0]
        a, b = pair
        name = f"{a} -> {b}"
        if now > cap:
            grown = f" > baseline {cap}" if pair in baseline else ", not in the baseline"
            problems.append(
                f"{name}: {_statements(now)}{grown} [{LABEL}]. {b} loads core.DB_Management when it is imported, "
                "so this import closes a cycle. Inject the logic from the caller, or import a module that imports "
                f"no DB code. If it is unavoidable, set this line in {_shown(BASELINE)}; the raise and its reason "
                f"are reviewed in the PR:\n    {name}: {now}  # bump: {REASON_PLACEHOLDER}"
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


def dump(path: Path, entries: Mapping[Pair, tuple[int, str]]) -> None:
    header = (
        "# <importer module> -> <target module>: <statements>. DB_Management imports of modules that load DB code "
        f"at import. Counts only fall; see {Path(__file__).name}.\n"
    )
    lines = [
        f"{a} -> {b}: {count}" + (f"  # bump: {reason}" if reason else "") + "\n"
        for (a, b), (count, reason) in sorted(entries.items())
    ]
    path.write_text(header + "".join(lines), encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    return _seed(
        tmp_path / "app",
        {
            "core/DB_Management/store.py": (
                f"from {APP}.core.Foo import lazy, logic, models\ndef f():\n    from {APP}.core.Bar.leaf import x\n"
            ),
            "core/DB_Management/y.py": "",
            "core/Foo/__init__.py": "",
            "core/Foo/logic.py": f"from {APP}.core.DB_Management import y\n",
            "core/Foo/models.py": "X = 1\n",
            "core/Foo/lazy.py": f"def f():\n    from {APP}.core.DB_Management import y\n",
            "core/Bar/__init__.py": f"from {APP}.core.DB_Management import y\n",
            "core/Bar/leaf.py": "x = 1\n",
        },
    )


def test_a_db_management_import_of_a_db_loading_module_is_counted(tmp_path: Path) -> None:
    assert _counts(measure(_tree(tmp_path))) == {
        ("core.DB_Management.store", "core.Foo.logic"): 1,  # loads DB code at import
        ("core.DB_Management.store", "core.Bar.leaf"): 1,  # its package __init__ does; a function-level importer counts
    }  # core.Foo.models (no DB import) and core.Foo.lazy (DB import only inside a function) are contracts


def test_new_and_stale_entries_fail_with_the_fix(tmp_path: Path) -> None:
    new = violations(measure(_tree(tmp_path)), {})
    assert [p.split(" [")[0] for p in new] == [
        "core.DB_Management.store -> core.Bar.leaf: 1 statement, not in the baseline",
        "core.DB_Management.store -> core.Foo.logic: 1 statement, not in the baseline",
    ]
    assert "core.DB_Management.store -> core.Foo.logic: 1  # bump: <why" in new[1]
    (stale,) = violations({}, {("core.DB_Management.a", "core.X.y"): (1, "")})
    assert stale.startswith("core.DB_Management.a -> core.X.y: stale, the edge is gone. Delete the line")


def test_a_fallen_but_nonzero_count_is_stale_and_asks_to_lower_it(tmp_path: Path) -> None:
    pair = ("core.DB_Management.store", "core.Foo.logic")
    baseline = {pair: (3, ""), ("core.DB_Management.store", "core.Bar.leaf"): (1, "")}
    (stale,) = violations(measure(_tree(tmp_path)), baseline)
    assert stale.startswith(f"{pair[0]} -> {pair[1]}: stale, baseline 3 but 1 measured. Lower it to 1"), stale


def test_db_management_imports_of_db_loading_modules_only_fall() -> None:
    problems = violations(measure(), load(BASELINE))
    assert not problems, f"{len(problems)} problem(s):\n" + "\n".join(problems)


def test_a_sync_contract_that_imports_sync_or_db_code_fails(tmp_path: Path) -> None:
    root = _seed(tmp_path / "app", {
        "core/Sync_Contracts/models.py": f"def f():\n    from {APP}.core.Sync.v2 import store\n",
        "core/Sync_Contracts/errors.py": f"from {APP}.core.DB_Management import Sync_DB\n",
        "core/Sync/v2/store.py": "",
        "core/DB_Management/Sync_DB.py": "",
    })
    assert sync_contract_problems(root) == [
        "core.Sync_Contracts.errors imports DB code when it is imported",
        "core.Sync_Contracts.errors imports core.DB_Management: Sync contracts must not import core.Sync or core.DB_Management",
        "core.Sync_Contracts.models imports core.Sync: Sync contracts must not import core.Sync or core.DB_Management",
    ]


def test_sync_contracts_import_no_sync_or_db_code_and_db_management_no_sync() -> None:
    assert sync_contract_problems() == []
    assert ("core.DB_Management", "core.Sync") not in scan_upward_edges()


def test_raised_counts_carry_a_new_reason() -> None:
    base = os.environ.get(BASE_ENV) or "origin/dev"
    try:
        old = base_version(BASELINE, base)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get(BASE_ENV):
            raise
        pytest.skip(f"cannot find the merge base with {base}; fetch it or set {BASE_ENV} to a base commit ({exc})")
    problems = unreasoned_bumps(load(BASELINE), old, path=BASELINE)
    assert not problems, "\n".join(problems)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Report DB-loading imports; rewrite {_shown(BASELINE)}.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tighten", action="store_true", help="only lower fallen counts and drop gone pairs")
    mode.add_argument("--write-baseline", action="store_true", help="record every pair, growth included")
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
    print(f"{len(measured)} pairs, {sum(sum(c.values()) for c in measured.values())} statements")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
