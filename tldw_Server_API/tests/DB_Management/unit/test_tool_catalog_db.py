from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management import Tool_Catalog_DB as tool_catalog_db


class _CatalogNameFailureDb:
    async def fetchone(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("catalog lookup failed at /private/authnz.db")

    async def fetchall(self, *_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - trap
        raise AssertionError("entry lookup should not run after catalog name failure")


class _CatalogEntriesFailureDb:
    async def fetchall(self, *_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("catalog entries failed at /private/authnz.db")


def _capture_debug_logs(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    records: list[str] = []

    def _debug(message: str, *args: Any, **_kwargs: Any) -> None:
        records.append(message.format(*args))

    monkeypatch.setattr(tool_catalog_db, "logger", SimpleNamespace(debug=_debug))
    return records


@pytest.mark.asyncio
@pytest.mark.unit
async def test_catalog_name_lookup_failure_logs_safe_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _capture_debug_logs(monkeypatch)

    resolved = await tool_catalog_db.resolve_tool_catalog_filter_names(
        _CatalogNameFailureDb(),
        catalog_name="Team Research",
        catalog_id=None,
        metadata={"team_id": 7, "org_id": 5},
        strict=False,
    )

    joined = "\n".join(records)
    assert resolved is None
    assert "RuntimeError" in joined
    assert "catalog_name=Team Research" in joined
    assert "team_id=7" in joined
    assert "org_id=5" in joined
    assert "strict=False" in joined
    assert "/private/" not in joined
    assert "authnz.db" not in joined


@pytest.mark.asyncio
@pytest.mark.unit
async def test_catalog_entries_lookup_failure_logs_safe_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _capture_debug_logs(monkeypatch)

    resolved = await tool_catalog_db.resolve_tool_catalog_filter_names(
        _CatalogEntriesFailureDb(),
        catalog_name=None,
        catalog_id=42,
        metadata={"team_id": 7, "org_id": 5},
        strict=True,
    )

    joined = "\n".join(records)
    assert resolved == set()
    assert "RuntimeError" in joined
    assert "catalog_id=42" in joined
    assert "team_id=7" in joined
    assert "org_id=5" in joined
    assert "strict=True" in joined
    assert "/private/" not in joined
    assert "authnz.db" not in joined
