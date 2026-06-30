from __future__ import annotations

import pytest

from tldw_Server_API.app.core.VN_Platform.errors import (
    ERROR_IDEMPOTENCY_KEY_CONFLICT,
    ERROR_INVALID_REQUEST,
    vn_error_detail,
)

pytestmark = pytest.mark.unit


def test_vn_error_detail_uses_stable_shape() -> None:
    detail = vn_error_detail(
        ERROR_IDEMPOTENCY_KEY_CONFLICT,
        "Idempotency key was already used with a different request.",
        details={"field": "idempotency_key"},
        retryable=False,
    )

    assert detail == {
        "code": "idempotency_key_conflict",
        "message": "Idempotency key was already used with a different request.",
        "details": {"field": "idempotency_key"},
        "retryable": False,
    }


def test_vn_error_detail_defaults_optional_fields() -> None:
    detail = vn_error_detail(ERROR_INVALID_REQUEST, "Invalid VN request.")

    assert detail == {
        "code": "invalid_request",
        "message": "Invalid VN request.",
        "details": {},
        "retryable": False,
    }
