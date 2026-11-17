"""Ratchet: no new test reloads ``app.main`` in place (TASK-15497).

``importlib.reload(app_main)`` rebinds ``app`` on the shared module. The autouse guard
(``app_main_isolated``) undoes it after each test, but a reload outside a test (collection,
a module or session fixture) is invisible to it. New tests call ``reload_app_main()``
(tests/helpers/app_main_state.py). Files listed in inplace_app_main_reload_baseline.txt
predate this; fixing one means removing its line.
"""

from __future__ import annotations

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).with_name("inplace_app_main_reload_baseline.txt")
# ``reload(app_main)``, ``reload(main_module)``, ``reload(importlib.sys.modules["...app.main"])``
PATTERN = re.compile(r"\breload\(\s*(app_main|main_module|importlib\.sys\.modules\[)")


def _reloading_files() -> set[str]:
    found = set()
    for path in TESTS.rglob("*.py"):
        rel = path.relative_to(TESTS).as_posix()
        if rel.startswith(("lint/", "helpers/")):
            continue
        if PATTERN.search(path.read_text(encoding="utf-8", errors="ignore")):
            found.add(rel)
    return found


def _baseline() -> set[str]:
    lines = BASELINE.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def test_no_new_inplace_app_main_reload() -> None:
    new = sorted(_reloading_files() - _baseline())
    assert not new, f"use reload_app_main() instead of importlib.reload(app_main): {new}"


def test_baseline_has_no_stale_entries() -> None:
    stale = sorted(_baseline() - _reloading_files())
    assert not stale, f"no longer reloads in place; remove from the baseline: {stale}"
