"""Unit tests for file artifact adapter backend error sanitization."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import FileArtifactsError
from tldw_Server_API.app.core.File_Artifacts.adapters.ical_adapter import IcalAdapter
from tldw_Server_API.app.core.File_Artifacts.adapters.xlsx_adapter import XlsxAdapter


pytestmark = pytest.mark.unit


def test_xlsx_export_unavailable_uses_sanitized_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not expose raw import failure details when the XLSX backend is unavailable."""
    original_import = builtins.__import__

    def fail_openpyxl_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openpyxl":
            raise RuntimeError("secret backend path /tmp/private/openpyxl.py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_openpyxl_import)

    with pytest.raises(FileArtifactsError) as exc_info:
        XlsxAdapter().export({"sheets": []}, format="xlsx")

    assert exc_info.value.code == "xlsx_export_unavailable"
    assert exc_info.value.detail == "xlsx export backend unavailable"


def test_ical_export_unavailable_uses_sanitized_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not expose raw import failure details when the iCalendar backend is unavailable."""
    original_import = builtins.__import__

    def fail_icalendar_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "icalendar":
            raise RuntimeError("secret backend path /tmp/private/icalendar.py")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_icalendar_import)

    with pytest.raises(FileArtifactsError) as exc_info:
        IcalAdapter().export({"calendar": {"events": []}}, format="ics")

    assert exc_info.value.code == "icalendar_library_unavailable"
    assert exc_info.value.detail == "icalendar export backend unavailable"
