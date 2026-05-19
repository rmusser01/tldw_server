from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
ENDPOINTS_ROOT = REPO_ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "endpoints"
AUTH_DEPS_PATH = (
    REPO_ROOT / "tldw_Server_API" / "app" / "api" / "v1" / "API_Deps" / "auth_deps.py"
)

BANNED_ENDPOINT_IMPORTS = {
    "tldw_Server_API.app.core.AuthNZ.User_DB_Handling": None,
    "core.AuthNZ.User_DB_Handling": None,
    "tldw_Server_API.app.core.AuthNZ.rate_limiter": {"RateLimiter"},
    "core.AuthNZ.rate_limiter": {"RateLimiter"},
}

REQUIRED_AUTH_DEPS_EXPORTS = {
    "RateLimiter",
    "User",
    "get_rate_limiter_dep",
    "get_request_user",
    "rbac_rate_limit",
    "resolve_user_id_for_request",
    "verify_jwt_and_fetch_user",
}


def _is_banned_import_module(imported_module: str) -> bool:
    return any(
        imported_module == banned_module or imported_module.startswith(f"{banned_module}.")
        for banned_module in BANNED_ENDPOINT_IMPORTS
    )


def _auth_dependency_import_offenders(module: ast.Module, rel_path: str) -> list[str]:
    offenders: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned_import_module(alias.name):
                    imported_module = (
                        f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                    )
                    offenders.append(
                        f"{rel_path}:{node.lineno}: import module {imported_module}"
                    )
            continue

        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module not in BANNED_ENDPOINT_IMPORTS:
            continue

        banned_names = BANNED_ENDPOINT_IMPORTS[node.module]
        if banned_names is None:
            imported_banned_names = sorted(alias.name for alias in node.names)
        else:
            imported_banned_names = sorted(
                alias.name for alias in node.names if alias.name in banned_names
            )
        if imported_banned_names:
            offenders.append(
                f"{rel_path}:{node.lineno}: import "
                f"{', '.join(imported_banned_names)} from {node.module}"
            )
    return offenders


def _direct_auth_dependency_imports() -> list[str]:
    offenders: list[str] = []
    for py_file in sorted(ENDPOINTS_ROOT.rglob("*.py")):
        rel_path = py_file.relative_to(REPO_ROOT).as_posix()
        module = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        offenders.extend(_auth_dependency_import_offenders(module, rel_path))
    return offenders


def test_endpoint_auth_dependency_scan_flags_direct_module_imports():
    module = ast.parse(
        "\n".join(
            [
                "import tldw_Server_API.app.core.AuthNZ.User_DB_Handling",
                "import tldw_Server_API.app.core.AuthNZ.rate_limiter as rate_limiter",
                "import tldw_Server_API.app.core.AuthNZ.User_DB_Handling.extra",
            ]
        )
    )

    assert _auth_dependency_import_offenders(module, "sample.py") == [  # nosec B101
        "sample.py:1: import module tldw_Server_API.app.core.AuthNZ.User_DB_Handling",
        "sample.py:2: import module tldw_Server_API.app.core.AuthNZ.rate_limiter as rate_limiter",
        "sample.py:3: import module tldw_Server_API.app.core.AuthNZ.User_DB_Handling.extra",
    ]


def test_endpoint_auth_dependency_symbols_come_from_auth_deps():
    offenders = _direct_auth_dependency_imports()

    assert offenders == [], (  # nosec B101
        "Endpoint files should import common auth dependency symbols "
        "from tldw_Server_API.app.api.v1.API_Deps.auth_deps instead of core.AuthNZ.\n"
        + "\n".join(offenders)
    )


def test_auth_deps_reexports_common_endpoint_auth_symbols():
    module = ast.parse(AUTH_DEPS_PATH.read_text(encoding="utf-8"), filename=str(AUTH_DEPS_PATH))
    exported_names: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                exported_names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            exported_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    exported_names.add(target.id)

    missing = sorted(REQUIRED_AUTH_DEPS_EXPORTS - exported_names)

    assert missing == [], (  # nosec B101
        "auth_deps.py must re-export common endpoint auth dependency symbols: "
        + ", ".join(missing)
    )
