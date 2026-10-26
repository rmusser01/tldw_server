"""Ceiling baselines with a reviewed bump-with-reason path, shared by the size ratchets (TASK-13950.8).

A baseline is a JSON object, one entry per line, sorted by key:

    "<key>": {"<field>": int, ..., "reason": str | null}

Each int is a ceiling. Measuring above it fails; measuring below it passes, so shrinking
needs no edit. A key that is measured but not listed fails as new; a listed key that is
no longer measured fails as stale. The counter in each ratchet's test module rewrites its
baseline: ``--write-baseline`` records every measurement and ``--tighten`` only lowers
ceilings and drops stale entries. Both keep existing reasons.

Bump with reason. Raising a ceiling, or adding an entry, is allowed only when the entry
carries a new reason: non-empty, different from the reason it had at the merge base, and
not a pasted template (any unfilled ``<...>`` placeholder, as in the failure messages, is
no reason). A raised field must equal the code's measurement at the time of the raise:
a bump cannot reserve headroom for later growth.
The check compares the baseline with its version at the merge base of
$BACKLOG_TASK_FORMAT_BASE and HEAD. backend-required sets that variable to the PR base,
and a local run defaults to origin/dev. So every bump shows up in the PR diff, where
reviewers check its reason. When the baseline does not exist at the merge base (the PR
that adds it), no reasons are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

import pytest

from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT

BASE_ENV = "BACKLOG_TASK_FORMAT_BASE"
REASON_PLACEHOLDER = "<why the growth is needed, e.g. TASK-13950.7 security fix>"
_UNFILLED = re.compile(r"<[^<>]*>")

Measured = Mapping[str, Mapping[str, int]]
Baseline = dict[str, dict[str, Any]]


class Ratchet(NamedTuple):
    baseline: Path  # the JSON file
    command: str  # runs the counter, e.g. "python -m tldw_Server_API.tests.lint.test_function_size_ratchet"
    fix: str  # how to fix growth in the code
    stale: str  # why an entry is no longer measured
    new: str = ""  # extra hint for a measured key with no entry


def load(path: Path) -> Baseline:
    entries: Baseline = json.loads(path.read_text(encoding="utf-8"))
    return entries


def dump(path: Path, entries: Mapping[str, Mapping[str, Any]]) -> None:
    rows = [f"  {json.dumps(key)}: {json.dumps(dict(entries[key]))}" for key in sorted(entries)]
    path.write_text("{\n" + ",\n".join(rows) + "\n}\n", encoding="utf-8")


def _shown(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def _numbers(entry: Mapping[str, Any]) -> dict[str, int]:
    return {field: value for field, value in entry.items() if field != "reason"}


def _entry_json(key: str, numbers: Mapping[str, int], reason: str = REASON_PLACEHOLDER) -> str:
    return json.dumps({key: {**numbers, "reason": reason}})[1:-1]


def _bump(ratchet: Ratchet, key: str, numbers: Mapping[str, int]) -> str:
    return (
        f"If the growth is unavoidable (e.g. a security fix), set this entry in {_shown(ratchet.baseline)}; "
        f"the raised ceiling and its reason are reviewed in the PR:\n    {_entry_json(key, numbers)}"
    )


def violations(ratchet: Ratchet, measured: Measured, baseline: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """One message per key that grew past its ceiling, has no ceiling, or is stale."""
    problems = []
    for key in sorted(measured.keys() | baseline.keys()):
        now, entry = measured.get(key), baseline.get(key)
        if now is None:
            problems.append(
                f"{key}: stale entry in {_shown(ratchet.baseline)}. {ratchet.stale} "
                f"Delete it, or run: {ratchet.command} --tighten"
            )
        elif entry is None:
            sizes = ", ".join(f"{field} {value}" for field, value in now.items())
            hint = f" {ratchet.new}" if ratchet.new else ""
            problems.append(f"{key}: {sizes}, and no ceiling. {ratchet.fix}{hint} {_bump(ratchet, key, now)}")
        else:
            cap = _numbers(entry)
            grown = [f"{field} {now[field]} > ceiling {cap[field]}" for field in now if now[field] > cap[field]]
            if grown:
                raised = {field: max(now[field], cap[field]) for field in now}
                problems.append(f"{key}: {', '.join(grown)}. {ratchet.fix} {_bump(ratchet, key, raised)}")
    return problems


def unreasoned_raises(
    ratchet: Ratchet, baseline: Baseline, base: Baseline | None, measured: Measured | None = None
) -> list[str]:
    """Entries added or raised since ``base`` without a new reason, or raised above ``measured``.

    ``base`` None means no check; ``measured`` None skips the headroom check.
    """
    if base is None:
        return []
    problems = []
    for key, entry in sorted(baseline.items()):
        old = base.get(key, {})
        numbers = _numbers(entry)
        raised = [field for field, value in numbers.items() if field not in old or value > old[field]]
        reason = str(entry.get("reason") or "").strip()
        if raised and (
            not reason or _UNFILLED.search(reason) or reason == str(old.get("reason") or "").strip()
        ):
            problems.append(
                f"{key}: raises {', '.join(raised)} in {_shown(ratchet.baseline)} without a new reason. "
                f"Say why in its \"reason\" (fill in any <...> placeholder); the raised ceiling and its "
                f"reason are reviewed in the PR:\n    {_entry_json(key, numbers)}"
            )
        now = (measured or {}).get(key, {})
        headroom = [field for field in raised if field in now and numbers[field] > now[field]]
        if headroom:
            over = ", ".join(f"{field} to {numbers[field]}, above the code's {now[field]}" for field in headroom)
            fitted = {**numbers, **{field: now[field] for field in headroom}}
            problems.append(
                f"{key}: raises {over} in {_shown(ratchet.baseline)}. A bump raises a ceiling to the code's "
                f"current size, never above it:\n    {_entry_json(key, fitted, reason or REASON_PLACEHOLDER)}"
            )
    return problems


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(  # fixed git argv, no shell
        ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout


def base_version(path: Path, base: str, repo_root: Path = REPO_ROOT) -> Baseline | None:
    """The baseline at the merge base of ``base`` and HEAD, or None if it did not exist there."""
    merge_base = _git(repo_root, "merge-base", base, "HEAD").strip()
    relative = path.relative_to(repo_root).as_posix()
    if not _git(repo_root, "ls-tree", "--name-only", merge_base, "--", relative).strip():
        return None
    entries: Baseline = json.loads(_git(repo_root, "show", f"{merge_base}:{relative}"))
    return entries


def reasonless_bumps(ratchet: Ratchet, measured: Measured | None = None) -> list[str]:
    """The bump-with-reason check for a test: skips locally when there is no base to compare with."""
    base = os.environ.get(BASE_ENV) or "origin/dev"
    try:
        old = base_version(ratchet.baseline, base)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get(BASE_ENV):
            raise
        pytest.skip(f"cannot find the merge base with {base}; fetch it or set {BASE_ENV} to a base commit ({exc})")
    return unreasoned_raises(ratchet, load(ratchet.baseline), old, measured)


def warn_if_tightenable(ratchet: Ratchet, measured: Measured, baseline: Baseline) -> None:
    """Prompt, without failing, to lock in ceilings the code has shrunk below."""
    slack = [
        key
        for key, now in measured.items()
        if key in baseline and any(now[field] < cap for field, cap in _numbers(baseline[key]).items())
    ]
    if slack:
        warnings.warn(
            f"{len(slack)} ceiling(s) in {_shown(ratchet.baseline)} are above the code; "
            f"lock the gain in with: {ratchet.command} --tighten",
            stacklevel=2,
        )


def rewritten(baseline: Baseline, measured: Measured, *, tighten: bool) -> Baseline:
    """``--tighten``: lower ceilings and drop stale keys. Otherwise: record ``measured``. Reasons are kept."""
    if tighten:
        return {
            key: {
                **{field: min(value, measured[key][field]) for field, value in _numbers(entry).items()},
                "reason": entry.get("reason"),
            }
            for key, entry in baseline.items()
            if key in measured
        }
    return {key: {**now, "reason": baseline.get(key, {}).get("reason")} for key, now in measured.items()}


def main(ratchet: Ratchet, measure: Callable[[], Measured], argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Rewrite {_shown(ratchet.baseline)}.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-baseline", action="store_true", help="record every current measurement")
    mode.add_argument("--tighten", action="store_true", help="only lower ceilings and drop stale entries")
    args = parser.parse_args(argv)
    current = load(ratchet.baseline) if ratchet.baseline.exists() else {}
    entries = rewritten(current, measure(), tighten=args.tighten)
    dump(ratchet.baseline, entries)
    print(f"{_shown(ratchet.baseline)}: {len(entries)} entries")
    return 0
