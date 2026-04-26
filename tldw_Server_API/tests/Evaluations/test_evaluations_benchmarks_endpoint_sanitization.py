import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.evaluations import evaluations_benchmarks
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


_BENCHMARK_SENSITIVE_MARKERS = (
    "benchmark backend leaked",
    "/private/evals-benchmark.db",
)


class _LoggerStub:
    def __init__(self):
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))


def _user() -> User:
    return User(id="tenant-user", username="tenant", email=None, is_active=True)


def _assert_sanitized_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.errors == [(expected_message, (), {})]
    rendered = " ".join(
        [
            logger_stub.errors[0][0],
            *(str(arg) for arg in logger_stub.errors[0][1]),
        ]
    )
    for marker in _BENCHMARK_SENSITIVE_MARKERS:
        assert marker not in rendered


async def test_run_benchmark_sanitizes_backend_fallback_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger_stub = _LoggerStub()

    def _raise_registry_error():
        raise RuntimeError("benchmark backend leaked /private/evals-benchmark.db")

    monkeypatch.setattr(evaluations_benchmarks, "logger", logger_stub)
    monkeypatch.setattr(evaluations_benchmarks, "get_registry", _raise_registry_error)
    monkeypatch.setattr(
        evaluations_benchmarks,
        "_get_evaluation_manager_for_user",
        lambda _identity: object(),
    )

    with pytest.raises(HTTPException) as exc_info:
        await evaluations_benchmarks.run_benchmark(
            benchmark_name="demo",
            request=evaluations_benchmarks.BenchmarkRunRequest(save_results=False),
            user_id=object(),
            current_user=_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to run benchmark: An error occurred during benchmark run"
    for marker in _BENCHMARK_SENSITIVE_MARKERS:
        assert marker not in str(exc_info.value.detail)
    _assert_sanitized_log(logger_stub, "Failed to run benchmark")
