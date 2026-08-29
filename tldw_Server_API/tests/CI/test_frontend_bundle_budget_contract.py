"""Contracts for the WebUI shared-bundle budget check.

Every page downloads the shared ``_app`` chunk before rendering, so growth there
is paid by every route. It reached 680 KB gzip against 1-15 KB of
route-specific code, because the English locale bundles and the API client drifted
into the shell unnoticed.

Turbopack no longer prints a size table at the end of ``next build``, so there is
no incidental signal that this is happening. The budget check is the only thing
watching. These contracts keep it wired in and keep the budget honest -- a check
that is present but unreferenced, or whose ceiling has been raised to infinity,
protects nothing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

FRONTEND = Path("apps/tldw-frontend")
BUDGET_SCRIPT = FRONTEND / "scripts/check-bundle-budget.mjs"
PACKAGE_JSON = FRONTEND / "package.json"

# The budget exists to catch 100 KB-scale regressions. Much above this and it
# would stop catching the class of drift it was written for.
MAX_CREDIBLE_SHARED_BUDGET_KB = 600


@pytest.mark.unit
def test_bundle_budget_script_exists() -> None:
    """The check must be present to be enforceable."""
    assert BUDGET_SCRIPT.is_file(), f"missing {BUDGET_SCRIPT}"


@pytest.mark.unit
def test_every_production_build_runs_the_budget_check() -> None:
    """A build that skips the check lets the shell grow silently."""
    scripts = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["scripts"]

    build_scripts = {
        name: command
        for name, command in scripts.items()
        if name in {"build", "build:prod", "build:dev", "compile", "compile:prod", "compile:dev"}
    }
    assert build_scripts, "no build scripts found to check"

    missing = [
        name
        for name, command in build_scripts.items()
        if "check-bundle-budget.mjs" not in command
    ]
    assert not missing, (
        "these build scripts do not run the bundle budget check, so the shared "
        f"app-shell bundle can grow unnoticed in them: {sorted(missing)}"
    )


@pytest.mark.unit
def test_budget_check_is_invocable_standalone() -> None:
    """Keep a standalone entry point so CI can run it without a full build."""
    scripts = json.loads(PACKAGE_JSON.read_text(encoding="utf-8"))["scripts"]
    assert "check:bundle-budget" in scripts
    assert "check-bundle-budget.mjs" in scripts["check:bundle-budget"]


@pytest.mark.unit
def test_shared_budget_stays_within_a_meaningful_range() -> None:
    """A ceiling raised far enough stops being a guard at all."""
    source = BUDGET_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r"const SHARED_BUDGET_BYTES\s*=\s*(\d+)\s*\*\s*1024", source
    )
    assert match, "could not read SHARED_BUDGET_BYTES from the budget script"

    budget_kb = int(match.group(1))
    assert budget_kb <= MAX_CREDIBLE_SHARED_BUDGET_KB, (
        f"the shared bundle budget has been raised to {budget_kb} KB, above the "
        f"{MAX_CREDIBLE_SHARED_BUDGET_KB} KB ceiling this guard considers "
        "meaningful. Raising it means every page in the app got heavier; if that "
        "is intended, record why and move this ceiling deliberately."
    )


@pytest.mark.unit
def test_budget_script_fails_closed_without_a_build() -> None:
    """No manifest must not silently read as a pass."""
    source = BUDGET_SCRIPT.read_text(encoding="utf-8")
    assert "process.exit(2)" in source, (
        "the budget script must exit non-zero when the build manifest is "
        "missing, otherwise a broken build path reads as a passing budget"
    )
