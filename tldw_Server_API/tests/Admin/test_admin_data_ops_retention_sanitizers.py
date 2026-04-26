from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.admin import admin_data_ops


class _LoggerStub:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_records.append((message, args, kwargs))


@pytest.mark.asyncio
async def test_list_retention_policies_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    async def _raise_list_retention_policies() -> list[dict[str, Any]]:
        raise RuntimeError("retention list backend exploded at /private/retention.db")

    monkeypatch.setattr(
        admin_data_ops,
        "svc_list_retention_policies",
        _raise_list_retention_policies,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.list_retention_policies(principal=object())

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to list retention policies"
    assert logger_stub.error_records == [("Failed to list retention policies", (), {})]


@pytest.mark.asyncio
async def test_preview_retention_policy_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    async def _raise_preview_retention_policy(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("retention preview backend exploded at /private/retention.db")

    monkeypatch.setattr(
        admin_data_ops,
        "svc_preview_retention_policy",
        _raise_preview_retention_policy,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.preview_retention_policy(
            "audit_logs",
            admin_data_ops.RetentionPolicyPreviewRequest(current_days=180, days=90),
            principal=SimpleNamespace(principal_id="admin@example.test"),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to preview retention policy"
    assert logger_stub.error_records == [("Failed to preview retention policy", (), {})]


@pytest.mark.asyncio
async def test_update_retention_policy_sanitizes_generic_failure_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger_stub = _LoggerStub()
    monkeypatch.setattr(admin_data_ops, "logger", logger_stub)

    async def _raise_verify_retention_preview_signature(**_kwargs: Any) -> None:
        raise RuntimeError("retention update backend exploded at /private/retention.db")

    monkeypatch.setattr(
        admin_data_ops,
        "svc_verify_retention_preview_signature",
        _raise_verify_retention_preview_signature,
    )

    with pytest.raises(HTTPException) as exc_info:
        await admin_data_ops.update_retention_policy(
            "audit_logs",
            admin_data_ops.RetentionPolicyUpdateRequest(
                days=90,
                preview_signature="signed-preview",
            ),
            request=object(),
            principal=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to update retention policy"
    assert logger_stub.error_records == [("Failed to update retention policy", (), {})]
