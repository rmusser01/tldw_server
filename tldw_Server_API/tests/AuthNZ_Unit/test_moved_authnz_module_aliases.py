"""Each old AuthNZ path of a moved provider-credential module is that module (TASK-13950.17).

An alias, not a re-export: ``monkeypatch.setattr`` through an old path, by string or on the
module object, must patch the code that runs, whichever path a process imports first.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import REPO_ROOT
from tldw_Server_API.tests.lint.test_authnz_alias_imports import MOVED

pytestmark = pytest.mark.unit

APP = "tldw_Server_API.app.core."


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_old_authnz_path_is_the_moved_module(old: str, new: str) -> None:
    moved = importlib.import_module(APP + new)
    assert importlib.import_module(APP + old) is moved
    parent, _, leaf = (APP + old).rpartition(".")
    assert getattr(importlib.import_module(parent), leaf) is moved
    assert Path(moved.__file__).as_posix().endswith("core/" + new.replace(".", "/") + ".py")


def test_patching_the_old_path_patches_the_code_that_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Provider_Credentials import provider_credential_runtime as moved

    sentinel = object()
    monkeypatch.setattr(APP + "AuthNZ.provider_credential_runtime.load_server_config_snapshot", sentinel)
    assert moved.load_server_config_snapshot is sentinel


@pytest.mark.parametrize(("old", "new"), sorted(MOVED.items()))
def test_old_path_aliases_when_imported_first(old: str, new: str) -> None:
    code = (
        "import importlib, sys\n"
        f"first = importlib.import_module({APP + old!r})\n"
        f"assert first is importlib.import_module({APP + new!r})\n"
        f"assert sys.modules[{APP + old!r}] is sys.modules[{APP + new!r}]\n"
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_root_conftest_disables_the_moved_authnz_scheduler() -> None:
    import importlib.util

    from tldw_Server_API.app.services import authnz_scheduler

    assert authnz_scheduler.start_authnz_scheduler.__name__ == "_noop"
    assert importlib.util.find_spec("tldw_Server_API.app.core.AuthNZ.scheduler") is None


def test_the_authnz_initialize_cli_imports_no_services() -> None:
    import ast

    tree = ast.parse((REPO_ROOT / "tldw_Server_API/app/core/AuthNZ/initialize.py").read_text(encoding="utf-8"))
    names = {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert sorted(name for name in names if name.startswith("tldw_Server_API.app.services")) == []
