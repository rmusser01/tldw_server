"""
Guardrail to keep API endpoint auth dependencies claim-first.

Legacy `get_current_user` should not be used directly as an endpoint dependency
inside `app/api/v1/endpoints`; endpoint auth should resolve an AuthPrincipal via
`get_auth_principal` and enforce claims from there.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import re

import pytest

from tldw_Server_API.tests.lint import _moved_endpoint_code


ENDPOINTS_ROOT = Path(__file__).resolve().parents[3] / "app" / "api" / "v1" / "endpoints"

_DEPENDS = re.compile(r"Depends\(\s*(?:auth_deps\.)?get_current_user\s*\)")
_IMPORT = re.compile(r"from\s+.*auth_deps\s+import\s+.*\bget_current_user\b")
_LOCAL_DEF = re.compile(r"^\s*async\s+def\s+get_current_user\s*\(")


def _scan_files() -> list[Path]:
    return _moved_endpoint_code.scan_files(ENDPOINTS_ROOT)


def _display(path: Path) -> str:
    return _moved_endpoint_code.display(path)


def _offenders(paths: Iterable[Path]) -> list[str]:
    offending: list[str] = []
    for path in paths:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if _DEPENDS.search(line) or _IMPORT.search(line) or _LOCAL_DEF.search(line):
                offending.append(f"{_display(path)}:{lineno}:{stripped}")
    return offending


@pytest.mark.unit
def test_api_endpoints_do_not_depend_on_legacy_get_current_user() -> None:
    assert ENDPOINTS_ROOT.exists(), f"Expected endpoints path not found: {ENDPOINTS_ROOT}"
    offending = _offenders(_scan_files())
    assert offending == [], (
        "Found legacy `Depends(get_current_user)` callsites in API endpoints: "
        f"{offending}"
    )


@pytest.mark.unit
def test_get_current_user_guardrail_flags_a_violation_in_code_moved_out_of_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seeded = tmp_path / "api" / "v1" / "utils" / "chatops" / "seeded.py"
    seeded.parent.mkdir(parents=True)
    seeded.write_text("x = Depends(get_current_user)\n", encoding="utf-8")
    monkeypatch.setattr(_moved_endpoint_code, "moved_files", lambda *_args: [seeded])
    assert _offenders(_scan_files()) == [f"{seeded}:1:x = Depends(get_current_user)"]  # nosec B101
