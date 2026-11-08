import pytest

from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as unified_eval_svc

pytestmark = pytest.mark.unit


def test_unified_eval_inline_webhook_mode_accepts_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "y")

    assert unified_eval_svc._await_webhook_inline_in_test_mode() is True


def test_unified_eval_inline_webhook_mode_disabled_when_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    assert unified_eval_svc._await_webhook_inline_in_test_mode() is False
