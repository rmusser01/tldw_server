"""Regression coverage for shared AuthNZ contention responses."""

import pytest

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import _authnz_busy_http_exception

pytestmark = pytest.mark.unit


def test_busy_response_resolves_live_policy_on_each_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for retry_after in (17, 0, 9):
        monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", str(retry_after))

        error = _authnz_busy_http_exception()

        assert (error.status_code, error.detail, error.headers) == (
            503,
            "Authentication database is busy. Please retry shortly.",
            {"Retry-After": str(retry_after)},
        )


@pytest.mark.parametrize("retry_after", (-2, 0, 11))
def test_busy_response_preserves_explicit_retry_after(
    monkeypatch: pytest.MonkeyPatch,
    retry_after: int,
) -> None:
    monkeypatch.setenv("AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS", "17")

    error = _authnz_busy_http_exception(retry_after)

    assert error.headers == {"Retry-After": str(max(retry_after, 0))}
