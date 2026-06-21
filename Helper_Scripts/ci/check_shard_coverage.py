#!/usr/bin/env python3
"""Guard: every test file must be assigned to a CI full-suite shard.

The sharded CI (`.github/workflows/ci.yml`) partitions the suite by explicit
``matrix.shard[].paths`` lists. A test file that is not under any shard path is
silently never collected -> the suite goes green while skipping it. This guard
parses the shard paths out of the workflow, enumerates the test files on disk,
and fails if any test file is covered by neither a shard, the ignore list, nor
the baseline.

Two opt-out lists, with distinct meaning:

* ``--ignore-file`` (default ``Helper_Scripts/ci/shard_coverage_ignore.txt``):
  files intentionally run by *other* workflows (e2e, jobs-suite, ...) or
  quarantined. Permanent exclusions.
* ``--baseline-file`` (default ``Helper_Scripts/ci/shard_coverage_baseline.txt``):
  the known backlog of currently-unshared files at the time the guard was
  introduced. The guard stays green on these but a *newly* unshared test fails
  CI (ratchet). Shrink this file over time by assigning files to shards.

Run:

    python Helper_Scripts/ci/check_shard_coverage.py
    # snapshot today's backlog into the baseline:
    python Helper_Scripts/ci/check_shard_coverage.py --write-baseline

Matching uses shell-glob semantics: ``*`` does not cross ``/`` (like the shell /
pytest expansion the shard paths rely on).
"""
from __future__ import annotations

import argparse
import fnmatch
import sys
from collections.abc import Iterable
from pathlib import Path

import yaml

DEFAULT_CI_FILE = ".github/workflows/ci.yml"
DEFAULT_TESTS_ROOT = "tldw_Server_API/tests"
DEFAULT_IGNORE_FILE = "Helper_Scripts/ci/shard_coverage_ignore.txt"
DEFAULT_BASELINE_FILE = "Helper_Scripts/ci/shard_coverage_baseline.txt"
_GLOB_CHARS = ("*", "?", "[")


def extract_shard_paths(ci_text: str) -> set[str]:
    """Return the union of all ``matrix.shard[].paths`` patterns in a workflow.

    Node-id suffixes (``file.py::test_x``) are reduced to the file path, since
    coverage is tracked per file. ``paths`` may be a folded string (whitespace
    separated) or a list.
    """
    doc = yaml.safe_load(ci_text) or {}
    patterns: set[str] = set()
    for job in (doc.get("jobs") or {}).values():
        if not isinstance(job, dict):
            continue
        shards = (((job.get("strategy") or {}).get("matrix") or {}).get("shard")) or []
        if not isinstance(shards, list):
            continue
        for entry in shards:
            if not isinstance(entry, dict) or "paths" not in entry:
                continue
            raw = entry["paths"]
            tokens = raw if isinstance(raw, list) else str(raw).split()
            for tok in tokens:
                tok = tok.strip()
                if not tok:
                    continue
                tok = tok.split("::", 1)[0]  # drop node-id -> file path
                patterns.add(tok.replace("\\", "/"))
    return patterns


def _glob_match(rel_path: str, pattern: str) -> bool:
    """True if ``rel_path`` is covered by ``pattern`` (shell-glob semantics)."""
    rel_path = rel_path.replace("\\", "/")
    pattern = pattern.replace("\\", "/").rstrip("/")
    if not any(ch in pattern for ch in _GLOB_CHARS):
        # Plain file or directory: exact match or directory prefix.
        return rel_path == pattern or rel_path.startswith(pattern + "/")
    # Glob: match component-wise so '*' does not cross '/'.
    p_parts = pattern.split("/")
    f_parts = rel_path.split("/")
    if len(f_parts) != len(p_parts):
        return False
    return all(fnmatch.fnmatch(f, p) for f, p in zip(f_parts, p_parts))


def is_covered(rel_path: str, patterns: Iterable[str]) -> bool:
    return any(_glob_match(rel_path, p) for p in patterns)


def collect_test_files(tests_root: Path, repo_root: Path) -> set[str]:
    """Return repo-relative posix paths of pytest-collectable test modules."""
    files: set[str] = set()
    for path in tests_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        name = path.name
        if name.startswith("test_") or name.endswith("_test.py"):
            files.add(path.relative_to(repo_root).as_posix())
    return files


def load_patterns_file(path: Path | None) -> set[str]:
    """Load newline-delimited patterns (``#`` comments, blank lines skipped)."""
    if not path or not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line.replace("\\", "/"))
    return out


def find_uncovered(
    test_files: Iterable[str],
    shard_patterns: Iterable[str],
    ignore_patterns: Iterable[str] = (),
    baseline_patterns: Iterable[str] = (),
) -> list[str]:
    shard_patterns = list(shard_patterns)
    ignore_patterns = list(ignore_patterns)
    baseline = set(baseline_patterns)
    uncovered = [
        f
        for f in test_files
        if not is_covered(f, shard_patterns)
        and not is_covered(f, ignore_patterns)
        and f not in baseline
    ]
    return sorted(uncovered)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ci-file", default=DEFAULT_CI_FILE)
    parser.add_argument("--tests-root", default=DEFAULT_TESTS_ROOT)
    parser.add_argument("--ignore-file", default=DEFAULT_IGNORE_FILE)
    parser.add_argument("--baseline-file", default=DEFAULT_BASELINE_FILE)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Rewrite the baseline file with the current unshared (non-ignored) files and exit 0.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    ci_file = repo_root / args.ci_file
    tests_root = repo_root / args.tests_root

    if not ci_file.exists():
        print(f"[shard-coverage] CI file not found: {ci_file}", file=sys.stderr)
        return 2
    if not tests_root.exists():
        print(f"[shard-coverage] tests root not found: {tests_root}", file=sys.stderr)
        return 2

    patterns = extract_shard_paths(ci_file.read_text(encoding="utf-8"))
    if not patterns:
        print(
            "[shard-coverage] No matrix.shard[].paths found — is this the sharded ci.yml?",
            file=sys.stderr,
        )
        return 2
    ignore = load_patterns_file(repo_root / args.ignore_file)
    test_files = collect_test_files(tests_root, repo_root)

    if args.write_baseline:
        backlog = find_uncovered(test_files, patterns, ignore, baseline_patterns=())
        out_path = repo_root / args.baseline_file
        header = (
            "# shard_coverage_baseline.txt — known-unshared test files at guard\n"
            "# introduction. The guard stays green on these but fails on a NEWLY\n"
            "# unshared test. Shrink this list by assigning files to ci.yml shards.\n"
            "# Regenerate with: python Helper_Scripts/ci/check_shard_coverage.py --write-baseline\n"
        )
        out_path.write_text(header + "\n".join(backlog) + ("\n" if backlog else ""), encoding="utf-8")
        print(f"[shard-coverage] wrote baseline with {len(backlog)} files -> {out_path}")
        return 0

    baseline = load_patterns_file(repo_root / args.baseline_file)
    uncovered = find_uncovered(test_files, patterns, ignore, baseline)

    print(
        f"[shard-coverage] shards={len(patterns)} test_files={len(test_files)} "
        f"ignored={len(ignore)} baseline={len(baseline)} new_uncovered={len(uncovered)}"
    )
    if uncovered:
        print(
            "[shard-coverage] FAIL — these test files are in no shard (they will be "
            "silently skipped). Add them to a shard in ci.yml, or (if intentionally run "
            "elsewhere) to the ignore file:",
            file=sys.stderr,
        )
        for f in uncovered:
            print(f"  {f}", file=sys.stderr)
        return 1
    print("[shard-coverage] OK — no newly-unshared test files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
