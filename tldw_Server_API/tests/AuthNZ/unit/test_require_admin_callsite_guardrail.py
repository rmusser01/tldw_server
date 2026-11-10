"""
Guardrail to keep API endpoint authorization claim-first.

Legacy `require_admin`/`require_role` shims are removed. Endpoints must keep
using claim-first dependencies (`require_roles`, `require_permissions`) only.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import pytest

from tldw_Server_API.tests.lint import _moved_endpoint_code


ENDPOINTS_ROOT = Path(__file__).resolve().parents[3] / "app" / "api" / "v1" / "endpoints"

_DEPENDS = re.compile(r"Depends\(\s*(?:auth_deps\.)?require_admin\s*\)")
_DEPENDS_ROLE = re.compile(r"Depends\(\s*(?:auth_deps\.)?require_role\s*\(")
_IMPORT = re.compile(r"from\s+.*auth_deps\s+import\s+.*\brequire_admin\b")
_IMPORT_ROLE = re.compile(r"from\s+.*auth_deps\s+import\s+.*\brequire_role\b")


def _scan_files() -> list[Path]:
    return _moved_endpoint_code.scan_files(ENDPOINTS_ROOT)


def _display(path: Path) -> str:
    return _moved_endpoint_code.display(path)


def _offenders(paths: Iterable[Path]) -> list[str]:
    offending: list[str] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.strip().startswith("#"):
                continue
            if _DEPENDS.search(line) or _DEPENDS_ROLE.search(line) or _IMPORT.search(line) or _IMPORT_ROLE.search(line):
                offending.append(f"{_display(path)}:{lineno}:{line.strip()}")
    return offending


@pytest.mark.unit
def test_api_endpoints_do_not_depend_on_legacy_admin_shims() -> None:
    assert ENDPOINTS_ROOT.exists(), f"Expected endpoints path not found: {ENDPOINTS_ROOT}"
    offending = _offenders(_scan_files())
    assert offending == [], (
        "Found legacy `require_admin`/`require_role` callsites in API endpoints: "
        f"{offending}"
    )


@pytest.mark.unit
def test_auth_deps_module_no_longer_exports_legacy_admin_shims() -> None:
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    assert not hasattr(auth_deps, "require_admin")
    assert not hasattr(auth_deps, "require_role")


@pytest.mark.unit
def test_require_admin_guardrail_flags_a_violation_in_code_moved_out_of_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = tmp_path / "api" / "v1" / "utils" / "chatops" / "seeded.py"
    seeded.parent.mkdir(parents=True)
    seeded.write_text("x = Depends(require_admin)\n", encoding="utf-8")
    monkeypatch.setattr(_moved_endpoint_code, "moved_files", lambda *_args: [seeded])
    assert _offenders(_scan_files()) == [f"{seeded}:1:x = Depends(require_admin)"]  # nosec B101
