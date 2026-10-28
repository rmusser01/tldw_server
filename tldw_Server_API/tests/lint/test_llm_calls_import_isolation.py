"""Guard: importing a module must not load a package it must stay independent of (TASK-13950.13).

Each case imports one module in a fresh interpreter and lists what loaded. The static
layering count (test_layering_contract.py) misses an edge laundered through a third
package (core.LLM_Calls -> core.X -> core.Chat); a fresh ``sys.modules`` does not.

TASK-13950.17 extends this file with a module-keyed ratchet on the AuthNZ modules
LLM_Calls loads, reusing ``loaded_after_import``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT

pytestmark = pytest.mark.unit

_SCRIPT = "import importlib, json, sys; importlib.import_module(sys.argv[1]); print(json.dumps(sorted(sys.modules)))"
CASES = [("tldw_Server_API.app.core.Chat", "tldw_Server_API.app.api")]


def loaded_after_import(module: str) -> list[str]:
    """Import ``module`` in a fresh interpreter and return every module name then loaded."""
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, module],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    # The traceback is at the end; startup logging can make stderr long.
    assert result.returncode == 0, f"importing {module} failed:\n{result.stderr[-4000:]}"
    loaded: list[str] = json.loads(result.stdout.strip().splitlines()[-1])
    return loaded


def forbidden_loads(module: str, forbidden: str) -> list[str]:
    """The modules of package ``forbidden`` that importing ``module`` loads."""
    return [m for m in loaded_after_import(module) if m == forbidden or m.startswith(forbidden + ".")]


@pytest.mark.parametrize(("module", "forbidden"), CASES)
def test_importing_does_not_load(module: str, forbidden: str) -> None:
    loaded = forbidden_loads(module, forbidden)
    assert loaded == [], f"importing {module} loaded {loaded[:5]}"


def test_the_guard_sees_a_forbidden_load() -> None:
    # chat_loop_store imports api.v1.schemas.chat_loop_schemas: a known violator.
    assert forbidden_loads("tldw_Server_API.app.core.Chat.chat_loop_store", "tldw_Server_API.app.api")
