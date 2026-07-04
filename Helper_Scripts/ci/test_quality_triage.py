#!/usr/bin/env python3
"""Triage: rank test files by mechanically-detectable quality problems.

The suite (~4k test files) cannot be hand-audited. This scanner parses every
test module's AST and flags the anti-patterns documented in
``audits/2026-07-04-test-suite-audit-round2.md`` (RA1-RA3, RA7):

* ``mock_density``     — an integration-marked file where a single test
                         function stacks >= 4 mock/patch/monkeypatch calls
                         ("integration" tests that don't integrate).
* ``stub_injection``   — test-module-defined fake/stub classes instantiated
                         and injected into the object under test (attribute
                         assignment or ``dependency_overrides``); catches the
                         hand-rolled-stub pattern raw mock counting misses.
* ``status_only``      — a test function whose asserts touch only
                         ``.status_code`` / ``.status`` (no body semantics).
* ``ambiguous_accept`` — ``assert status_code in (200, 5xx)``: a test that
                         passes on success AND on server error.
* ``tautology_suspect``— asserts inside test-module-defined fake classes
                         (the mock asserting itself), or the
                         collect-calls-via-monkeypatch-then-assert-the-list
                         shape where the real code never runs.
* ``skip_stale``       — skip/skipif/xfail whose reason has no issue/PR/URL
                         reference and is not a dependency/env availability
                         gate (informational only; round-1 F9 already
                         enforces that reasons exist).

Known noise (from the 10-file precision validation): the
``dependency_overrides`` arm of ``stub_injection`` fires on the standard
FastAPI TestClient wiring idiom (auth + DB overrides), so treat that flag as
a ranking input for over-mocked endpoint tests, not a per-instance verdict.

Modes (mirrors ``check_shard_coverage.py`` ratchet mechanics):

    # ranked human-readable report (exit 0 — report-only):
    python Helper_Scripts/ci/test_quality_triage.py
    # machine-readable:
    python Helper_Scripts/ci/test_quality_triage.py --json out.json
    # snapshot today's offenses (path::flag=count) for the ratchet:
    python Helper_Scripts/ci/test_quality_triage.py --write-baseline
    # ratchet mode (future CI gate): fail on NEW offenses or count INCREASES
    # (informational flags like skip_stale never gate):
    python Helper_Scripts/ci/test_quality_triage.py --enforce

Determinism: output is sorted (score desc, then path) and carries no
timestamps, so two runs over the same tree produce identical bytes.
"""
from __future__ import annotations

import argparse
import ast
import json as jsonlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_TESTS_ROOT = "tldw_Server_API/tests"
DEFAULT_BASELINE_FILE = "Helper_Scripts/ci/test_quality_baseline.txt"

MOCK_DENSITY_THRESHOLD = 4
STUB_INJECTION_THRESHOLD = 2

# Call names that indicate a mock/patch is being set up.
_MOCK_CALL_NAMES = {"MagicMock", "AsyncMock", "Mock", "PropertyMock", "create_autospec"}
_PATCH_ATTRS = {"patch", "object", "dict", "multiple", "setattr", "setitem", "setenv", "delenv"}
_PATCH_ROOTS = {"patch", "mock", "mocker", "monkeypatch"}

_STATUS_ATTRS = {"status_code", "status"}

# Weights for the ranking score (severity-ordered; skip_stale is informational).
_FLAG_WEIGHTS = {
    "parse_error": 8,
    "ambiguous_accept": 5,
    "tautology_suspect": 5,
    "mock_density": 3,
    "stub_injection": 3,
    "status_only": 2,
    "skip_stale": 0,
}

# Flags the --enforce ratchet gates on. Informational (zero-weight) flags are
# reported but never fail CI.
_ENFORCEABLE_FLAGS = frozenset(flag for flag, weight in _FLAG_WEIGHTS.items() if weight > 0)

_ISSUE_REF_TOKENS = ("#", "issue", "http://", "https://", "pr ", "pr-", "jira", "task-")

# Reasons that are dependency/environment gates, not stale skips.
_AVAILABILITY_GATE_TOKENS = (
    "not installed",
    "not available",
    "unavailable",
    "requires ",
    "required",
    "missing",
    "no docker",
    "docker not",
    "set ",  # "Set RUN_EXTERNAL_API_TESTS=1 ..." style env gates
    "skipped on",
    "only runs",
)


@dataclass
class FileReport:
    """Per-file triage result."""

    path: str
    flags: dict[str, int] = field(default_factory=dict)
    details: list[str] = field(default_factory=list)

    @property
    def score(self) -> int:
        return sum(_FLAG_WEIGHTS.get(flag, 1) * count for flag, count in self.flags.items())

    def add(self, flag: str, detail: str) -> None:
        self.flags[flag] = self.flags.get(flag, 0) + 1
        self.details.append(f"{flag}: {detail}")


def _call_name(node: ast.Call) -> str:
    """Best-effort dotted name of a call target ('mocker.patch.object', ...)."""
    parts: list[str] = []
    cur: ast.expr = node.func
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _is_mockish_call(node: ast.Call) -> bool:
    """True for patch/monkeypatch/Mock-constructor style calls.

    Matches any dotted segment against the patch roots so fully-qualified
    forms (``unittest.mock.patch``, ``mock.patch.object``) count too.
    """
    name = _call_name(node)
    if not name:
        return False
    segments = name.split(".")
    tail = segments[-1]
    if tail in _MOCK_CALL_NAMES:
        return True
    if any(seg in _PATCH_ROOTS for seg in segments) and (
        tail in _PATCH_ATTRS or tail in _PATCH_ROOTS
    ):
        return True
    return name in {"patch", "patch.object", "patch.dict", "patch.multiple"}


def _is_test_function(node: ast.AST) -> bool:
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
        "test_"
    )


def _iter_test_functions(tree: ast.Module):
    """Yield test functions at module level and inside Test* classes."""
    for node in tree.body:
        if _is_test_function(node):
            yield node
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            for sub in node.body:
                if _is_test_function(sub):
                    yield sub


def _module_fake_classes(tree: ast.Module) -> set[str]:
    """Names of module-level classes that are not pytest Test* classes."""
    return {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("Test")
    }


def _nested_fake_classes(fn: ast.AST) -> set[str]:
    """Names of classes defined inside a test function."""
    return {n.name for n in ast.walk(fn) if isinstance(n, ast.ClassDef)}


def _file_is_integration_marked(tree: ast.Module, source: str) -> bool:
    """Detect pytest.mark.integration via the AST (decorators or pytestmark).

    Substring matching would false-positive on comments/docstrings; instead
    look for any ``<...>.mark.integration`` attribute chain.
    """
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "integration"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark"
        ):
            return True
    return False


def _assert_attr_names(assert_node: ast.Assert) -> set[str]:
    """All attribute names referenced in an assert's test expression."""
    return {n.attr for n in ast.walk(assert_node.test) if isinstance(n, ast.Attribute)}


def _const_ints(node: ast.expr) -> list[int]:
    """Integer constants inside a tuple/list/set literal."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return []
    return [e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]


def _check_ambiguous_accept(fn: ast.AST, report: FileReport) -> None:
    """assert status in (2xx, ..., >=400) — passes on success and on failure."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert) or not isinstance(node.test, ast.Compare):
            continue
        cmp = node.test
        if not any(isinstance(op, ast.In) for op in cmp.ops):
            continue
        left_attrs = {
            n.attr for n in ast.walk(cmp.left) if isinstance(n, ast.Attribute)
        }
        if not (left_attrs & _STATUS_ATTRS):
            continue
        for comparator in cmp.comparators:
            values = _const_ints(comparator)
            if any(200 <= v < 300 for v in values) and any(v >= 400 for v in values):
                report.add(
                    "ambiguous_accept",
                    f"{getattr(fn, 'name', '?')}:{node.lineno} accepts {sorted(values)}",
                )


def _check_status_only(fn: ast.AST, report: FileReport) -> None:
    """Every assert in the test touches only .status_code/.status."""
    asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
    if not asserts:
        return
    for node in asserts:
        attrs = _assert_attr_names(node)
        if not attrs or not attrs.issubset(_STATUS_ATTRS):
            return
    report.add("status_only", f"{getattr(fn, 'name', '?')} ({len(asserts)} assert(s))")


def _check_mock_density(fn: ast.AST, report: FileReport, integration_marked: bool) -> None:
    """Integration-marked file with a test stacking >= threshold mock calls."""
    if not integration_marked:
        return
    count = sum(
        1 for n in ast.walk(fn) if isinstance(n, ast.Call) and _is_mockish_call(n)
    )
    if count >= MOCK_DENSITY_THRESHOLD:
        report.add("mock_density", f"{getattr(fn, 'name', '?')} has {count} mock/patch calls")


def _check_stub_injection(
    fn: ast.AST, report: FileReport, fake_classes: set[str]
) -> None:
    """Module/local fake classes injected into the object under test."""
    local_fakes = fake_classes | _nested_fake_classes(fn)
    class_injections = 0
    override_injections = 0
    samples: list[str] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            value_is_fake = (
                isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id in local_fakes
            )
            for target in node.targets:
                # obj.attr = _Stub(...): a test-defined fake wired into the
                # object under test — one occurrence is already the smell.
                if isinstance(target, ast.Attribute) and value_is_fake:
                    class_injections += 1
                    samples.append(f"line {node.lineno}")
                # app.dependency_overrides[dep] = anything
                if isinstance(target, ast.Subscript):
                    sub_attrs = {
                        n.attr for n in ast.walk(target.value) if isinstance(n, ast.Attribute)
                    }
                    if "dependency_overrides" in sub_attrs or (
                        isinstance(target.value, ast.Attribute)
                        and target.value.attr == "dependency_overrides"
                    ):
                        override_injections += 1
                        samples.append(f"line {node.lineno} (dependency_overrides)")
    if class_injections >= 1 or override_injections >= STUB_INJECTION_THRESHOLD:
        total = class_injections + override_injections
        report.add(
            "stub_injection",
            f"{getattr(fn, 'name', '?')} injects {total} fakes ({', '.join(samples[:3])})",
        )


def _check_tautology(tree: ast.Module, report: FileReport, fake_classes: set[str]) -> None:
    """Asserts inside fake classes; collect-then-assert-the-collector shape."""
    # (a) assert statements inside module-level non-Test classes
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in fake_classes:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assert):
                    report.add(
                        "tautology_suspect",
                        f"assert inside fake class {node.name}:{sub.lineno}",
                    )
    # (b) calls = [] ; nested def appends to it ; assert calls == [...]
    for fn in _iter_test_functions(tree):
        empty_lists: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and not node.value.elts:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        empty_lists.add(target.id)
            # annotated form: calls: list[object] = []
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.value, ast.List)
                and not node.value.elts
                and isinstance(node.target, ast.Name)
            ):
                empty_lists.add(node.target.id)
        if not empty_lists:
            continue
        appended: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)) and node is not fn:
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr == "append"
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id in empty_lists
                    ):
                        appended.add(sub.func.value.id)
        if not appended:
            continue
        # Only a tautology when the collector comparisons are the test's ONLY
        # asserts: interaction tests that ALSO assert real outputs are fine
        # (verified against a 10-file precision sample — see the triage report).
        all_asserts = [n for n in ast.walk(fn) if isinstance(n, ast.Assert)]
        collector_asserts = [
            n
            for n in all_asserts
            if isinstance(n.test, ast.Compare)
            and isinstance(n.test.left, ast.Name)
            and n.test.left.id in appended
            and any(isinstance(op, ast.Eq) for op in n.test.ops)
        ]
        if collector_asserts and len(collector_asserts) == len(all_asserts):
            for node in collector_asserts:
                report.add(
                    "tautology_suspect",
                    f"{getattr(fn, 'name', '?')}:{node.lineno} asserts ONLY that its own "
                    f"stub collector '{node.test.left.id}' was called",
                )


def _check_skip_stale(tree: ast.Module, report: FileReport) -> None:
    """skip/skipif/xfail whose reason lacks an issue/PR/URL reference."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        tail = name.split(".")[-1]
        if tail not in {"skip", "skipif", "xfail"}:
            continue
        if not name.startswith(("pytest.mark.", "pytest.")):
            continue
        reason = None
        for kw in node.keywords:
            if kw.arg == "reason" and isinstance(kw.value, ast.Constant):
                reason = str(kw.value.value)
        if reason is None:
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    reason = arg.value
        if reason is None:
            continue  # reason-presence is enforced elsewhere (round-1 F9)
        low = reason.lower()
        if any(tok in low for tok in _ISSUE_REF_TOKENS):
            continue
        # legitimate availability/env gates are not "stale"
        if any(tok in low for tok in _AVAILABILITY_GATE_TOKENS):
            continue
        report.add("skip_stale", f"line {node.lineno}: reason has no issue link: {reason[:60]!r}")


def analyze_file(path: Path, repo_root: Path) -> FileReport | None:
    """Run all detectors over one test module; None when nothing flagged."""
    rel = path.relative_to(repo_root).as_posix()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source)
    except SyntaxError as exc:
        report = FileReport(path=rel)
        report.add("parse_error", f"unparseable: {exc.msg} at line {exc.lineno}")
        return report

    report = FileReport(path=rel)
    fake_classes = _module_fake_classes(tree)
    integration_marked = _file_is_integration_marked(tree, source)

    for fn in _iter_test_functions(tree):
        _check_ambiguous_accept(fn, report)
        _check_status_only(fn, report)
        _check_mock_density(fn, report, integration_marked)
        _check_stub_injection(fn, report, fake_classes)
    _check_tautology(tree, report, fake_classes)
    _check_skip_stale(tree, report)

    return report if report.flags else None


def collect_test_files(tests_root: Path) -> list[Path]:
    """All pytest-collectable modules under the tests root, sorted."""
    files = [
        p
        for p in tests_root.rglob("*.py")
        if "__pycache__" not in p.parts
        and (p.name.startswith("test_") or p.name.endswith("_test.py"))
    ]
    return sorted(files)


def offense_counts(reports: list[FileReport], *, enforceable_only: bool = False) -> dict[str, int]:
    """Map of ``path::flag`` -> occurrence count for ratchet comparison."""
    out: dict[str, int] = {}
    for r in reports:
        for flag, count in r.flags.items():
            if enforceable_only and flag not in _ENFORCEABLE_FLAGS:
                continue
            out[f"{r.path}::{flag}"] = count
    return out


def load_baseline(path: Path) -> dict[str, int]:
    """Parse ``path::flag=count`` baseline lines (count defaults to a large
    sentinel for legacy count-less lines, i.e. existence-only)."""
    if not path.exists():
        return {}
    out: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token, sep, count = line.partition("=")
        out[token.strip()] = int(count) if sep and count.strip().isdigit() else 10**9
    return out


def new_offenses(current: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """Offenses that are new or worse than the baseline (the ratchet)."""
    out: list[str] = []
    for token, count in sorted(current.items()):
        allowed = baseline.get(token)
        if allowed is None:
            out.append(f"{token}={count} (new)")
        elif count > allowed:
            out.append(f"{token}={count} (baseline {allowed})")
    return out


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tests-root", default=DEFAULT_TESTS_ROOT)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE)
    parser.add_argument("--top", type=int, default=50, help="Rows in the ranked report.")
    parser.add_argument("--json", default=None, help="Also write full results as JSON.")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Snapshot current offenses (path::flag) into the baseline and exit 0.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit 1 when offenses exist that are not in the baseline (ratchet).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    tests_root = repo_root / args.tests_root
    if not tests_root.exists():
        print(f"[test-triage] tests root not found: {tests_root}", file=sys.stderr)
        return 2

    files = collect_test_files(tests_root)
    reports = [r for r in (analyze_file(p, repo_root) for p in files) if r is not None]
    reports.sort(key=lambda r: (-r.score, r.path))

    flag_totals: dict[str, int] = {}
    for r in reports:
        for flag, count in r.flags.items():
            flag_totals[flag] = flag_totals.get(flag, 0) + count

    print(
        f"[test-triage] scanned={len(files)} flagged_files={len(reports)} "
        + " ".join(f"{k}={v}" for k, v in sorted(flag_totals.items()))
    )

    if args.write_baseline:
        counts = offense_counts(reports, enforceable_only=True)
        lines = [f"{token}={count}" for token, count in sorted(counts.items())]
        out_path = repo_root / args.baseline_file
        header = (
            "# test_quality_baseline.txt — known offenses (path::flag=count) at\n"
            "# triage introduction. With --enforce the triage stays green on these\n"
            "# but fails on a NEW offense or a count INCREASE. Informational flags\n"
            "# (skip_stale) are excluded. Shrink this list by fixing tests.\n"
            "# Regenerate with: python Helper_Scripts/ci/test_quality_triage.py --write-baseline\n"
        )
        out_path.write_text(header + "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[test-triage] wrote baseline with {len(lines)} offenses -> {out_path}")
        return 0

    if args.json:
        payload = [
            {"path": r.path, "score": r.score, "flags": r.flags, "details": r.details}
            for r in reports
        ]
        Path(args.json).write_text(jsonlib.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"[test-triage] wrote JSON -> {args.json}")

    print(f"[test-triage] top {min(args.top, len(reports))} by score:")
    for r in reports[: args.top]:
        flags = ", ".join(f"{k}x{v}" for k, v in sorted(r.flags.items()))
        print(f"  {r.score:>4}  {r.path}  [{flags}]")

    if args.enforce:
        baseline = load_baseline(repo_root / args.baseline_file)
        current = offense_counts(reports, enforceable_only=True)
        regressions = new_offenses(current, baseline)
        if regressions:
            print(
                f"[test-triage] FAIL — {len(regressions)} offense(s) new or worse "
                "than the baseline:",
                file=sys.stderr,
            )
            for token in regressions:
                print(f"  {token}", file=sys.stderr)
            return 1
        print("[test-triage] OK — no offenses beyond the baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
