"""Strict provider-failure semantics for Prompt Studio optimization runs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.bounded_daemon import DaemonCapacityError
from tldw_Server_API.app.core.Chat.Chat_Deps import (
    ChatConfigurationError,
    ChatProviderError,
    SanitizedProviderStreamError,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio import test_runner
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.test_runner import (
    TestRunner,
)

pytestmark = pytest.mark.unit

_SENTINEL = "TASK12963_STRICT_PROVIDER_SECRET"
_MODEL_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "parameters": {"temperature": 0.2, "max_tokens": 64},
    "api_key": "resolved-key",
    "app_config": {"openai_api": {"model": "gpt-4o-mini"}},
    "credentials_resolved": True,
}


class _RunnerDb:
    client_id = "strict-provider-errors"

    def __init__(
        self,
        *,
        expected: str = "expected only",
        runner_hint: str | None = None,
    ) -> None:
        self.expected = expected
        self.runner_hint = runner_hint
        self.runs: list[dict[str, Any]] = []

    def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        return {
            "id": prompt_id,
            "project_id": 71,
            "deleted": False,
            "system_prompt": "Answer precisely.",
            "user_prompt": "Question: {question}",
        }

    def get_test_case(self, test_case_id: int) -> dict[str, Any]:
        return {
            "id": test_case_id,
            "project_id": 71,
            "inputs": {"question": "q"},
            "expected_outputs": {
                "response": self.expected,
                **(
                    {"runner": self.runner_hint}
                    if self.runner_hint is not None
                    else {}
                ),
            },
        }

    def create_test_run(self, **kwargs: Any) -> dict[str, Any]:
        captured = dict(kwargs)
        self.runs.append(captured)
        return {
            "id": len(self.runs),
            "prompt_id": captured["prompt_id"],
            "test_case_id": captured["test_case_id"],
        }


def _exception_graph(exc: BaseException) -> str:
    return "".join((repr(exc), repr(exc.__cause__), repr(exc.__context__)))


_STRICT_FAILURES = [
    pytest.param(
        {
            "error": {
                "code": "invalid_provider_credentials",
                "message": _SENTINEL,
            }
        },
        "invalid_provider_credentials",
        id="canonical-in-band-credential-error",
    ),
    pytest.param(
        'data: {"error":{"code":"provider_authentication_failed",'
        f'"message":"{_SENTINEL}"}}}}\n\n',
        "provider_authentication_failed",
        id="canonical-sse-auth-error",
    ),
    pytest.param(
        ChatConfigurationError(
            provider="openai",
            message=_SENTINEL,
        ),
        "provider_configuration_invalid",
        id="raised-configuration-error",
    ),
    pytest.param(
        DaemonCapacityError(_SENTINEL),
        "provider_unavailable",
        id="capacity-exhaustion",
    ),
    pytest.param(
        TimeoutError(_SENTINEL),
        "provider_unavailable",
        id="provider-timeout",
    ),
    pytest.param(
        ChatProviderError(
            provider="openai",
            message=_SENTINEL,
            details={"private": _SENTINEL},
        ),
        "provider_unavailable",
        id="raised-provider-error",
    ),
    pytest.param(None, "provider_unavailable", id="none-result"),
    pytest.param("", "provider_unavailable", id="empty-string-result"),
    pytest.param(
        'data: {"choices":[{"delta":{"content":"unexpected stream"}}]}\n\n',
        "provider_unavailable",
        id="non-error-sse-frame-result",
    ),
    pytest.param({}, "provider_unavailable", id="empty-object-result"),
    pytest.param(
        'data: {"error":{"code":"provider_authentication_failed"}\n\n',
        "provider_unavailable",
        id="malformed-sse-error-result",
    ),
    pytest.param(
        {"choices": [{"message": {"content": []}}]},
        "provider_unavailable",
        id="empty-content-parts-result",
    ),
    pytest.param([42], "provider_unavailable", id="malformed-list-result"),
    pytest.param(
        {"choices": [{"message": {"content": {"unexpected": "shape"}}}]},
        "provider_unavailable",
        id="malformed-nested-content-result",
    ),
    pytest.param(
        {
            "choices": [
                {
                    "message": {
                        "content": {
                            "error": {
                                "code": "invalid_provider_credentials",
                                "message": _SENTINEL,
                            }
                        }
                    }
                }
            ]
        },
        "invalid_provider_credentials",
        id="nested-canonical-error-result",
    ),
    pytest.param(
        {
            "choices": [
                {"message": {"content": "apparently valid"}},
                {
                    "message": {
                        "content": {
                            "error": {
                                "code": "invalid_provider_credentials",
                                "message": _SENTINEL,
                            }
                        }
                    }
                }
            ],
        },
        "invalid_provider_credentials",
        id="mixed-success-and-error-result",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("adapter_outcome", "expected_code"), _STRICT_FAILURES)
async def test_strict_single_test_propagates_sanitized_provider_failures(
    monkeypatch: pytest.MonkeyPatch,
    adapter_outcome: Any,
    expected_code: str,
) -> None:
    db = _RunnerDb()
    runner = TestRunner(db)
    mark_count = 0

    async def _mark_success() -> None:
        nonlocal mark_count
        mark_count += 1

    def _adapter(**_kwargs: Any) -> Any:
        if isinstance(adapter_outcome, BaseException):
            raise adapter_outcome
        return adapter_outcome

    monkeypatch.setattr(runner, "_call_adapter", _adapter, raising=True)

    with pytest.raises(SanitizedProviderStreamError) as exc_info:
        await runner.run_single_test(
            prompt_id=12,
            test_case_id=3,
            model_config=_MODEL_CONFIG,
            strict_provider_errors=True,
            on_provider_success=_mark_success,
        )

    assert exc_info.value.code == expected_code
    assert _SENTINEL not in _exception_graph(exc_info.value)
    assert mark_count == 0
    assert db.runs == []


@pytest.mark.asyncio
async def test_strict_single_test_keeps_genuine_expected_output_mismatch_as_valid_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _RunnerDb(expected="expected only")
    runner = TestRunner(db)
    mark_count = 0

    async def _mark_success() -> None:
        nonlocal mark_count
        mark_count += 1
    monkeypatch.setattr(
        runner,
        "_call_adapter",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "totally different"}}],
            "usage": {"total_tokens": 2},
        },
        raising=True,
    )

    result = await runner.run_single_test(
        prompt_id=12,
        test_case_id=3,
        model_config=_MODEL_CONFIG,
        strict_provider_errors=True,
        on_provider_success=_mark_success,
    )

    assert result["success"] is True
    assert result["scores"]["aggregate_score"] == pytest.approx(0.0)
    assert mark_count == 1
    assert len(db.runs) == 1
    assert db.runs[0]["outputs"] == {"response": "totally different"}


@pytest.mark.asyncio
async def test_strict_single_test_keeps_program_evaluator_zero_as_valid_provider_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _RunnerDb(expected="program expectation", runner_hint="python")
    runner = TestRunner(db)
    mark_count = 0

    async def _mark_success() -> None:
        nonlocal mark_count
        mark_count += 1

    monkeypatch.setattr(
        runner,
        "_call_adapter",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "print('wrong')"}}],
            "usage": {"total_tokens": 2},
        },
        raising=True,
    )
    monkeypatch.setattr(
        test_runner.ProgramEvaluator,
        "evaluate",
        lambda *_args, **_kwargs: SimpleNamespace(
            success=False,
            return_code=1,
            reward=0.0,
            error="expected-output-mismatch",
            stdout="",
            stderr="",
            metrics={"score": 0.0},
        ),
        raising=True,
    )

    result = await runner.run_single_test(
        prompt_id=12,
        test_case_id=3,
        model_config=_MODEL_CONFIG,
        strict_provider_errors=True,
        on_provider_success=_mark_success,
    )

    assert result["success"] is False
    assert result["scores"]["aggregate_score"] == pytest.approx(0.0)
    assert result["scores"]["reward"] == pytest.approx(0.0)
    assert mark_count == 1
    assert len(db.runs) == 1


@pytest.mark.asyncio
async def test_non_strict_single_test_keeps_expected_output_mismatch_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _RunnerDb(expected="expected only")
    runner = TestRunner(db)
    monkeypatch.setattr(
        runner,
        "_call_adapter",
        lambda **_kwargs: {
            "choices": [{"message": {"content": "totally different"}}],
            "usage": {"total_tokens": 2},
        },
        raising=True,
    )

    result = await runner.run_single_test(
        prompt_id=12,
        test_case_id=3,
        model_config=_MODEL_CONFIG,
    )

    assert result["success"] is True
    assert result["scores"]["aggregate_score"] == pytest.approx(0.0)
    assert len(db.runs) == 1


@pytest.mark.asyncio
async def test_non_strict_single_test_preserves_scoreable_error_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _RunnerDb()
    runner = TestRunner(db)

    def _raise_provider_error(**_kwargs: Any) -> Any:
        raise ChatProviderError(provider="openai", message=_SENTINEL)

    monkeypatch.setattr(runner, "_call_adapter", _raise_provider_error, raising=True)

    result = await runner.run_single_test(
        prompt_id=12,
        test_case_id=3,
        model_config=_MODEL_CONFIG,
    )

    assert result["success"] is False
    assert result["scores"]["aggregate_score"] == pytest.approx(0.0)
    assert result["actual"]["error_code"] == "provider_unavailable"
    assert _SENTINEL not in repr(result)
    assert len(db.runs) == 1
