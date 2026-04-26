from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import config_admin

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_get_effective_config_sanitizes_resolution_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()

    def _raise_config_root() -> None:
        raise FileNotFoundError("config root leaked /private/config.txt")

    monkeypatch.setattr(config_admin, "logger", logger_stub)
    monkeypatch.setattr(config_admin, "resolve_config_root", _raise_config_root)

    with pytest.raises(HTTPException) as exc_info:
        await config_admin.get_effective_config(sections=None, include_defaults=True)

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to resolve effective configuration"
    assert logger_stub.debug_records == [("Effective config resolution failed", (), {})]
