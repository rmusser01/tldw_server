from __future__ import annotations

import ast
from pathlib import Path


EXPECTED_COMPAT_KEYS: set[str] = set()

# Compat paths whose sunset dates passed and were removed (TASK-13444).
# They must not be resurrected in COMPAT_PATHS or referenced by any
# log_runtime_deprecation() call site under app/.
EXPIRED_COMPAT_KEYS = {
    "auth_db_execute_compat",
    "llm_chat_legacy_session",
    "web_scraping_legacy_fallback",
}

ALLOWED_DYNAMIC_RUNTIME_MARKER_CALLS = {
    "tldw_Server_API/app/core/LLM_Calls/deprecation.py",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _registry_path() -> Path:
    return _repo_root() / "tldw_Server_API" / "app" / "core" / "deprecations" / "runtime_registry.py"


def _extract_registry_keys(path: Path) -> set[str]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        value_node = None
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "COMPAT_PATHS" for t in node.targets):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "COMPAT_PATHS":
                value_node = node.value

        if value_node is None:
            continue
        if not isinstance(value_node, ast.Dict):
            raise AssertionError("COMPAT_PATHS must remain a literal dict")
        keys: set[str] = set()
        for raw_key in value_node.keys:
            if not isinstance(raw_key, ast.Constant) or not isinstance(raw_key.value, str):
                raise AssertionError("COMPAT_PATHS keys must be string literals")
            keys.add(raw_key.value)
        return keys
    raise AssertionError("COMPAT_PATHS assignment not found in runtime_registry.py")


def _extract_runtime_marker_calls(path: Path) -> list[tuple[int, str]]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    markers: list[tuple[int, str]] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue

        func_name: str | None = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        if func_name != "log_runtime_deprecation":
            continue

        if not node.args:
            markers.append((node.lineno, "__missing_key__"))
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            markers.append((node.lineno, first_arg.value))
        else:
            markers.append((node.lineno, "__non_literal_key__"))
    return markers


def scan_for_new_compat_markers() -> list[str]:
    repo_root = _repo_root()
    app_root = repo_root / "tldw_Server_API" / "app"
    registry_keys = _extract_registry_keys(_registry_path())

    offenders: list[str] = []

    unexpected_keys = sorted(registry_keys - EXPECTED_COMPAT_KEYS)
    if unexpected_keys:
        offenders.append(
            "New runtime compatibility keys were added to COMPAT_PATHS: "
            + ", ".join(unexpected_keys)
        )

    missing_keys = sorted(EXPECTED_COMPAT_KEYS - registry_keys)
    if missing_keys:
        offenders.append(
            "Expected runtime compatibility keys were removed from COMPAT_PATHS: "
            + ", ".join(missing_keys)
        )

    resurrected_keys = sorted(EXPIRED_COMPAT_KEYS & registry_keys)
    if resurrected_keys:
        offenders.append(
            "Expired runtime compatibility keys were re-added to COMPAT_PATHS: "
            + ", ".join(resurrected_keys)
        )

    for py_file in app_root.rglob("*.py"):
        rel = py_file.relative_to(repo_root).as_posix()
        for lineno, key in _extract_runtime_marker_calls(py_file):
            if key == "__missing_key__":
                offenders.append(
                    f"{rel}:{lineno}: log_runtime_deprecation() must include a key argument."
                )
                continue
            if key == "__non_literal_key__":
                if rel in ALLOWED_DYNAMIC_RUNTIME_MARKER_CALLS:
                    continue
                offenders.append(
                    f"{rel}:{lineno}: log_runtime_deprecation() key must be a string literal."
                )
                continue
            if key not in registry_keys:
                offenders.append(
                    f"{rel}:{lineno}: runtime compatibility key '{key}' is not in COMPAT_PATHS."
                )
                continue
            if key not in EXPECTED_COMPAT_KEYS:
                offenders.append(
                    f"{rel}:{lineno}: runtime compatibility key '{key}' is new and must not be added."
                )

    return offenders


def test_no_new_runtime_compatibility_markers():
    offenders = scan_for_new_compat_markers()
    assert offenders == [], (
        "Runtime compatibility debt markers changed.\n"
        + "\n".join(offenders)
    )
