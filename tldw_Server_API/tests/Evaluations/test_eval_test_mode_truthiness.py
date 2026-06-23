import pytest

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_unified as eval_unified
from tldw_Server_API.app.core.Evaluations import unified_evaluation_service as unified_eval_svc


pytestmark = pytest.mark.unit


def test_eval_test_mode_accepts_tldw_test_mode_single_letter_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TEST_MODE", "0")
    monkeypatch.setenv("TLDW_TEST_MODE", "y")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)

    assert eval_unified._is_eval_test_mode() is True


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
