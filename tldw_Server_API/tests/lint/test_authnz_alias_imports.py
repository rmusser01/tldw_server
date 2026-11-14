"""Gate: app code, the CLI and Helper_Scripts do not use an AuthNZ alias path of a moved module (TASK-13950.17).

The provider-credential modules moved from ``core/AuthNZ`` to ``core/Provider_Credentials``.
Each old path is a ``sys.modules`` alias of the moved module, kept so tests and their
monkeypatch targets keep working. App code, ``tldw_Server_API/cli`` and ``Helper_Scripts`` may not use an old path from ``MOVED``: not at
module or function level, not absolute or relative, not as
``from ...AuthNZ import byok_runtime``, and not as a string equal to an alias path (for
``importlib.import_module``). In-app ``tests`` packages are skipped, as in the layering
contract.

Fix: run the K1 codemod of the TASK-13950.17 plan, or import the module named in ``MOVED``.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.tests.lint._ast_cache import APP_ROOT, REPO_ROOT, parse_tree
from tldw_Server_API.tests.lint.test_layering_contract import _module_name

pytestmark = pytest.mark.unit

APP = "tldw_Server_API.app"

# Old module -> moved module, both relative to tldw_Server_API.app.core.
MOVED: dict[str, str] = {
    **{
        f"AuthNZ.{name}": f"Provider_Credentials.{name}"
        for name in (
            "byok_config",
            "byok_helpers",
            "byok_rotation",
            "byok_runtime",
            "byok_testing",
            "user_provider_secrets",
            "provider_credential_runtime",
            "llm_provider_overrides",
        )
    },
    **{
        f"AuthNZ.repos.{name}": f"Provider_Credentials.repos.{name}"
        for name in (
            "user_provider_secrets_repo",
            "org_provider_secrets_repo",
            "llm_provider_overrides_repo",
            "byok_oauth_state_repo",
            "byok_validation_runs_repo",
        )
    },
    **{f"AuthNZ.{name}": f"Usage.{name}" for name in ("llm_budget_guard", "llm_budget_middleware")},
}

# Scanned root -> the dotted package name of that root (for resolving relative imports).
ROOTS: dict[Path, str] = {
    APP_ROOT: APP,
    REPO_ROOT / "tldw_Server_API" / "cli": "tldw_Server_API.cli",
    REPO_ROOT / "Helper_Scripts": "Helper_Scripts",
}

_OLD = {f"{APP}.core.{old}": f"{APP}.core.{new}" for old, new in MOVED.items()}


def _names(node: ast.AST, module: str, is_package: bool) -> Iterator[str]:
    """The absolute module names one node can name: imports, resolved relative imports, strings."""
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            parts = module.split(".") if module else []
            parts = parts if is_package else parts[:-1]
            if node.level - 1 > len(parts):
                return
            base = ".".join([*parts[: len(parts) - node.level + 1], *([node.module] if node.module else [])])
        else:
            base = node.module or ""
        yield base
        yield from (f"{base}.{alias.name}" for alias in node.names)
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value


def violations(roots: dict[Path, str] | None = None) -> list[str]:
    """One ``"<package>/<relative path>:<line> uses <old>; use <new>"`` per use of an alias path."""
    problems = []
    for root, package in (ROOTS if roots is None else roots).items():
        root = root.resolve()
        for path, tree in parse_tree(root).items():
            relative = path.relative_to(root)
            if "tests" in relative.parts[:-1]:
                continue  # an in-app test package, not app code
            module = ".".join(part for part in (package, _module_name(relative)) if part)
            for node in ast.walk(tree):
                for name in _names(node, module, path.name == "__init__.py"):
                    if name in _OLD:
                        problems.append(f"{root.name}/{relative.as_posix()}:{node.lineno} uses {name}; use {_OLD[name]}")
    return problems


def test_app_code_imports_no_authnz_alias_path() -> None:
    problems = violations()
    assert not problems, "\n".join(problems)


def test_a_seeded_alias_import_fails(tmp_path: Path) -> None:
    files = {
        "app/core/X/a.py": f"from {APP}.core.AuthNZ.byok_runtime import y\n",
        "app/core/X/b.py": f"def f():\n    from {APP}.core.AuthNZ import byok_runtime\n",
        "app/core/X/c.py": "from ..AuthNZ.repos.user_provider_secrets_repo import R\n",
        "app/core/X/d.py": f"import importlib\nm = importlib.import_module('{APP}.core.AuthNZ.llm_provider_overrides')\n",
        "app/core/X/ok.py": f"from {APP}.core.Provider_Credentials.byok_runtime import y\nfrom {APP}.core.AuthNZ import settings\n",
        "app/core/X/tests/test_t.py": f"from {APP}.core.AuthNZ.byok_runtime import y\n",
        "Helper_Scripts/s.py": f"from {APP}.core.AuthNZ.byok_rotation import y\n",
        "Helper_Scripts/ok.py": f"from {APP}.core.Provider_Credentials.byok_rotation import y\n",
        "cli/c.py": f"from {APP}.core.AuthNZ import byok_helpers\n",
        "cli/r.py": "from ..app.core.AuthNZ.byok_config import y\n",
        "cli/ok.py": f"from {APP}.core.AuthNZ import settings\n",
    }
    for relative, source in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    roots = {
        tmp_path / "app": APP,
        tmp_path / "Helper_Scripts": "Helper_Scripts",
        tmp_path / "cli": "tldw_Server_API.cli",
    }

    found = sorted(problem.split(":")[0] for problem in violations(roots))

    assert found == [
        "Helper_Scripts/s.py",
        "app/core/X/a.py",
        "app/core/X/b.py",
        "app/core/X/c.py",
        "app/core/X/d.py",
        "cli/c.py",
        "cli/r.py",
    ]


def test_every_scanned_root_exists() -> None:
    assert [root for root in ROOTS if not root.is_dir()] == []


# TASK-13950.17 scheduler move pins live here so backend-required runs them.
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
