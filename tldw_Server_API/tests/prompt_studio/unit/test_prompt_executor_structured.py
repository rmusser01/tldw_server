import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from loguru import logger

from tldw_Server_API.app.core.Prompt_Management.prompt_studio import (
    prompt_executor as prompt_executor_module,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.prompt_executor import PromptExecutor


def _make_prompt_definition_payload() -> dict:
    return {
        "schema_version": 1,
        "format": "structured",
        "variables": [
            {
                "name": "input",
                "label": "Input",
                "required": True,
                "input_type": "textarea",
            }
        ],
        "blocks": [
            {
                "id": "identity",
                "name": "Identity",
                "role": "system",
                "content": "You are a careful evaluator.",
                "enabled": True,
                "order": 10,
                "is_template": False,
            },
            {
                "id": "task",
                "name": "Task",
                "role": "user",
                "content": "Evaluate {{input}}",
                "enabled": True,
                "order": 20,
                "is_template": True,
            },
        ],
        "assembly_config": {
            "legacy_system_roles": ["system", "developer"],
            "legacy_user_roles": ["user"],
            "block_separator": "\n\n",
        },
    }


def test_legacy_build_prompt_replaces_exact_placeholder_names():
    executor = PromptExecutor(SimpleNamespace(client_id="unit-test"))

    rendered = executor._build_prompt(
        {"user_prompt": "$id $idea {id} {{id}} <id> {i-d}"},
        signature=None,
        inputs={"id": "42", "i-d": "bad"},
    )

    assert rendered == "42 $idea 42 42 42 {i-d}"


@pytest.mark.parametrize(
    "error_form",
    ["raw-prefix", "canonical", "serialized", "sse"],
)
def test_prompt_executor_rejects_list_wrapped_provider_errors_without_detail(
    error_form: str,
) -> None:
    """Prompt execution cannot reinterpret a wrapped provider error as output."""
    sentinel = "prompt-executor-list-secret-/private/provider.json"
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    payload = {
        "raw-prefix": f"Error: {sentinel}",
        "canonical": "provider_unavailable",
        "serialized": serialized,
        "sse": f"data: {serialized}\n\n",
    }[error_form]

    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response([payload])

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


def test_prompt_executor_preserves_valid_list_wrapped_text() -> None:
    """The provider-error guard preserves legacy list-returning adapters."""
    assert PromptExecutor._coerce_llm_response(["valid assistant text"]) == (
        "valid assistant text",
        3,
    )


class _OpaqueProviderResponse:
    """Malformed response whose display text must never become assistant output."""

    def __str__(self) -> str:
        return "opaque-provider-secret-/private/provider.json"

    def __repr__(self) -> str:
        return "OpaqueProviderResponse([REDACTED])"


_MALFORMED_PROMPT_EXECUTOR_RESPONSES = [
    pytest.param(None, id="none"),
    pytest.param("", id="empty-string"),
    pytest.param("  \t", id="whitespace-string"),
    pytest.param(0, id="zero-scalar"),
    pytest.param(42, id="integer-scalar"),
    pytest.param(3.5, id="float-scalar"),
    pytest.param(False, id="boolean-scalar"),
    pytest.param(b"opaque-provider-secret", id="bytes"),
    pytest.param(_OpaqueProviderResponse(), id="arbitrary-object"),
    pytest.param([], id="empty-list"),
    pytest.param([None], id="list-none"),
    pytest.param([42], id="list-scalar"),
    pytest.param((), id="empty-tuple"),
    pytest.param(("text-without-token-count",), id="short-tuple"),
    pytest.param(("text", 1, "extra"), id="long-tuple"),
    pytest.param({}, id="empty-dict"),
    pytest.param({"choices": []}, id="empty-choices"),
    pytest.param({"content": "  "}, id="blank-top-level-content"),
    pytest.param(
        {"choices": [{"message": {"content": "  "}}]},
        id="blank-message-content",
    ),
    pytest.param(
        {"choices": [{"delta": {"content": []}}]},
        id="empty-delta-content-list",
    ),
]


@pytest.mark.parametrize(
    "provider_response",
    _MALFORMED_PROMPT_EXECUTOR_RESPONSES,
)
def test_prompt_executor_rejects_nonsemantic_provider_responses(
    provider_response: object,
) -> None:
    """Only nonblank assistant output can cross the PromptExecutor boundary."""
    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response(provider_response)

    assert str(exc_info.value) == "Provider returned an empty or malformed response"
    assert "opaque-provider-secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("provider_response", "expected"),
    [
        pytest.param("valid", ("valid", 1), id="text"),
        pytest.param(["valid"], ("valid", 1), id="list-text"),
        pytest.param(("valid", 7), ("valid", 7), id="tuple-text"),
        pytest.param(
            {"choices": [{"message": {"content": "valid"}}]},
            ("valid", 1),
            id="message-text",
        ),
        pytest.param(
            {"choices": [{"delta": {"content": "valid"}}]},
            ("valid", 1),
            id="delta-text",
        ),
        pytest.param({"content": "valid"}, ("valid", 1), id="top-level-text"),
    ],
)
def test_prompt_executor_accepts_semantic_response_controls(
    provider_response: object,
    expected: tuple[str, int],
) -> None:
    """Fail-closed validation retains every supported nonblank response form."""
    assert PromptExecutor._coerce_llm_response(provider_response) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_response",
    [
        pytest.param(None, id="none"),
        pytest.param("  ", id="whitespace"),
        pytest.param(42, id="scalar"),
        pytest.param(b"opaque-provider-secret", id="bytes"),
        pytest.param(_OpaqueProviderResponse(), id="arbitrary-object"),
        pytest.param([], id="empty-list"),
        pytest.param((), id="empty-tuple"),
        pytest.param({}, id="empty-dict"),
    ],
)
async def test_prompt_executor_rejects_nonsemantic_response_before_usage_mark(
    monkeypatch: pytest.MonkeyPatch,
    provider_response: object,
) -> None:
    """Malformed adapter output cannot be certified as a successful provider use."""
    marks: list[str] = []

    class _Adapter:
        @staticmethod
        def chat(_request: dict[str, object]) -> object:
            return provider_response

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    async def _mark_success() -> None:
        marks.append("mark")

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        PromptExecutor,
        "_build_adapter_request",
        lambda _self, **kwargs: {"messages": kwargs["messages"]},
    )

    with pytest.raises(RuntimeError) as exc_info:
        await PromptExecutor(SimpleNamespace(client_id="unit-test"))._call_llm(
            provider="openai",
            model="model-a",
            prompt="hello",
            on_provider_success=_mark_success,
        )

    assert str(exc_info.value) == "Provider returned an error response"
    assert marks == []


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_executor_concurrent_nonsemantic_response_marks_only_valid_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed response cannot mark itself or a concurrent valid request."""
    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid request", "invalid request")}
    release = {key: threading.Event() for key in entered}
    marks: list[str] = []

    class _Adapter:
        def chat(self, request: dict[str, object]) -> object:
            messages = request["messages"]
            assert isinstance(messages, list)
            key = messages[-1]["content"]
            assert isinstance(key, str)
            loop.call_soon_threadsafe(entered[key].set)
            assert release[key].wait(timeout=2.0)
            return "valid output" if key == "valid request" else None

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    def _build_request(
        _self: PromptExecutor,
        **kwargs: object,
    ) -> dict[str, object]:
        return {"messages": kwargs["messages"]}

    async def _mark(key: str) -> None:
        marks.append(key)

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(PromptExecutor, "_build_adapter_request", _build_request)
    executor = PromptExecutor(SimpleNamespace(client_id="unit-test"))
    valid_task = asyncio.create_task(
        executor._call_llm(
            provider="openai",
            model="model-a",
            prompt="valid request",
            on_provider_success=lambda: _mark("valid request"),
        )
    )
    invalid_task = asyncio.create_task(
        executor._call_llm(
            provider="openai",
            model="model-a",
            prompt="invalid request",
            on_provider_success=lambda: _mark("invalid request"),
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["invalid request"].set()
        with pytest.raises(RuntimeError):
            await asyncio.wait_for(invalid_task, timeout=1.0)
        assert marks == []
        release["valid request"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, invalid_task, return_exceptions=True)

    assert valid_result == {"content": "valid output", "tokens": 2}
    assert marks == ["valid request"]


def _wrap_prompt_executor_response(
    wrapper: str,
    payload: object,
) -> object:
    if wrapper == "tuple":
        return payload, 17
    if wrapper == "message":
        return {"choices": [{"message": {"content": payload}}]}
    if wrapper == "delta":
        return {"choices": [{"delta": {"content": payload}}]}
    if wrapper == "top-level":
        return {"content": payload}
    if wrapper == "list-text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": payload}],
                    }
                }
            ]
        }
    raise AssertionError(f"Unknown response wrapper: {wrapper}")


@pytest.mark.parametrize(
    "wrapper",
    ["tuple", "message", "delta", "top-level", "list-text"],
)
@pytest.mark.parametrize(
    "error_form",
    ["raw-prefix", "canonical", "serialized", "sse"],
)
def test_prompt_executor_rejects_provider_errors_after_nested_extraction(
    wrapper: str,
    error_form: str,
) -> None:
    """Every supported response wrapper must re-enter the bounded error guard."""
    sentinel = "prompt-executor-nested-secret-/private/provider.json"
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    payload = {
        "raw-prefix": f"Error: {sentinel}",
        "canonical": "provider_unavailable",
        "serialized": serialized,
        "sse": f"data: {serialized}\n\n",
    }[error_form]

    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response(
            _wrap_prompt_executor_response(wrapper, payload)
        )

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


@pytest.mark.parametrize("wrapper", ["message", "delta", "top-level"])
def test_prompt_executor_rejects_nested_structured_provider_errors(
    wrapper: str,
) -> None:
    """A nested error object cannot be stringified into successful output."""
    sentinel = "prompt-executor-structured-secret-/private/provider.json"
    response = _wrap_prompt_executor_response(
        wrapper,
        {"error": {"message": sentinel}},
    )

    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response(response)

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


def test_prompt_executor_rejects_malformed_dict_without_stringifying_detail() -> None:
    """Malformed provider dictionaries fail closed without exposing their repr."""
    sentinel = "prompt-executor-malformed-secret-/private/provider.json"
    response = {
        "choices": [
            {
                "message": {
                    "metadata": {"provider_debug_detail": sentinel},
                }
            }
        ]
    }

    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response(response)

    assert str(exc_info.value) == "Provider returned an empty or malformed response"
    assert sentinel not in str(exc_info.value)


def test_prompt_executor_preserves_valid_tuple_tokens() -> None:
    """Tuple compatibility retains the adapter-provided token count."""
    assert PromptExecutor._coerce_llm_response(("valid assistant text", 17)) == (
        "valid assistant text",
        17,
    )


@pytest.mark.parametrize(
    "wrapper",
    ["tuple", "message", "delta", "top-level", "list-text"],
)
def test_prompt_executor_preserves_noncanonical_assistant_error_json(
    wrapper: str,
) -> None:
    """Assistant-authored noncanonical JSON remains ordinary content."""
    content = json.dumps(
        {"error": {"code": "fictional_story_error", "message": "plot device"}}
    )

    actual, tokens = PromptExecutor._coerce_llm_response(
        _wrap_prompt_executor_response(wrapper, content)
    )

    assert actual == content
    assert tokens == (17 if wrapper == "tuple" else int(len(content.split()) * 1.3))


def _mixed_prompt_executor_response(case: str, sentinel: str) -> dict[str, object]:
    serialized = json.dumps(
        {"error": {"code": "provider_unavailable", "message": sentinel}}
    )
    valid_choice = {"message": {"content": "valid assistant text"}}
    if case == "later-choice":
        return {
            "choices": [
                valid_choice,
                {"error": {"message": sentinel}},
            ]
        }
    if case == "message-error-sibling":
        return {
            "choices": [
                {
                    "message": {
                        "content": "valid assistant text",
                        "error": {"message": sentinel},
                    }
                }
            ]
        }
    if case == "message-error-block":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "error", "error": {"message": sentinel}},
                        ]
                    }
                }
            ]
        }
    if case == "message-later-error-text":
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "text", "text": f"data: {serialized}\n\n"},
                        ]
                    }
                }
            ]
        }
    if case == "delta-later-error-text":
        return {
            "choices": [
                {
                    "delta": {
                        "content": [
                            {"type": "text", "text": "valid assistant text"},
                            {"type": "text", "text": f"data: {serialized}\n\n"},
                        ]
                    }
                }
            ]
        }
    raise AssertionError(f"Unknown mixed response case: {case}")


@pytest.mark.parametrize(
    "case",
    [
        "later-choice",
        "message-error-sibling",
        "message-error-block",
        "message-later-error-text",
        "delta-later-error-text",
    ],
)
def test_prompt_executor_rejects_mixed_success_and_provider_error(
    case: str,
) -> None:
    """Earlier valid text cannot hide a sibling or later provider error."""
    sentinel = "prompt-executor-mixed-secret-/private/provider.json"

    with pytest.raises(RuntimeError) as exc_info:
        PromptExecutor._coerce_llm_response(
            _mixed_prompt_executor_response(case, sentinel)
        )

    assert str(exc_info.value) == "Provider returned an error response"
    assert sentinel not in str(exc_info.value)


def test_prompt_executor_preserves_valid_multipart_noncanonical_json() -> None:
    """Multiple valid text blocks may include assistant-authored error JSON."""
    noncanonical = json.dumps(
        {"error": {"code": "fictional_story_error", "message": "plot device"}}
    )
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "valid assistant text"},
                        {"type": "text", "text": noncanonical},
                    ]
                }
            }
        ]
    }

    actual, _tokens = PromptExecutor._coerce_llm_response(response)

    assert actual == "valid assistant text" + noncanonical


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_executor_concurrent_mixed_error_is_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later error in one response cannot affect a concurrent valid result."""
    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid request", "error request")}
    release = {key: threading.Event() for key in entered}
    sentinel = "prompt-executor-concurrent-mixed-secret-/private/provider.json"

    class _Db:
        client_id = "prompt-executor-concurrent-mixed"

        @staticmethod
        def get_prompt(prompt_id: int) -> dict[str, object]:
            return {
                "id": prompt_id,
                "deleted": False,
                "user_prompt": "valid request" if prompt_id == 1 else "error request",
            }

    class _Adapter:
        def chat(self, request: dict[str, object]) -> object:
            messages = request["messages"]
            assert isinstance(messages, list)
            key = messages[-1]["content"]
            assert isinstance(key, str)
            loop.call_soon_threadsafe(entered[key].set)
            assert release[key].wait(timeout=2.0)
            if key == "valid request":
                return "valid request output", 19
            return _mixed_prompt_executor_response("later-choice", sentinel)

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        PromptExecutor,
        "_build_adapter_request",
        lambda _self, **kwargs: {
            "messages": kwargs["messages"],
            "model": kwargs["model"],
        },
    )
    executor = PromptExecutor(_Db())  # type: ignore[arg-type]
    valid_task = asyncio.create_task(
        executor.execute_prompt(
            1,
            {},
            {"provider": "openai", "model": "model-a", "parameters": {}},
        )
    )
    error_task = asyncio.create_task(
        executor.execute_prompt(
            2,
            {},
            {"provider": "openai", "model": "model-b", "parameters": {}},
        )
    )
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error request"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        release["valid request"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)

    assert valid_result["success"] is True
    assert valid_result["raw_output"] == "valid request output"
    assert valid_result["tokens_used"] == 19
    assert error_result["success"] is False
    assert error_result["error"] == "Provider returned an error response"
    assert sentinel not in json.dumps(error_result)


@pytest.mark.asyncio
async def test_prompt_executor_detaches_raised_adapter_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raised provider exceptions stay out of both results and logs."""
    sentinel = "prompt-executor-raised-secret-/private/provider.json"
    log_messages: list[str] = []

    class _Db:
        client_id = "prompt-executor-raised-boundary"

        @staticmethod
        def get_prompt(_prompt_id: int) -> dict[str, object]:
            return {"id": 1, "deleted": False, "user_prompt": "hello"}

    class _Adapter:
        @staticmethod
        def chat(_request: dict[str, object]) -> list[str]:
            raise RuntimeError(sentinel)

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        PromptExecutor,
        "_build_adapter_request",
        lambda _self, **kwargs: {
            "messages": kwargs["messages"],
            "model": kwargs["model"],
        },
    )
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        result = await PromptExecutor(_Db()).execute_prompt(  # type: ignore[arg-type]
            1,
            {},
            {"provider": "openai", "model": "model-a", "parameters": {}},
        )
    finally:
        logger.remove(sink_id)

    assert result["success"] is False
    assert result["error"] == "Provider returned an error response"
    assert sentinel not in json.dumps(result) + "".join(log_messages)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_executor_concurrent_nested_results_remain_request_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nested provider failure handling cannot contaminate a valid tuple result."""
    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid request", "error request")}
    release = {key: threading.Event() for key in entered}
    sentinel = "prompt-executor-concurrent-nested-secret-/private/provider.json"
    log_messages: list[str] = []

    class _Db:
        client_id = "prompt-executor-concurrent-nested"

        @staticmethod
        def get_prompt(prompt_id: int) -> dict[str, object]:
            return {
                "id": prompt_id,
                "deleted": False,
                "user_prompt": "valid request" if prompt_id == 1 else "error request",
            }

    class _Adapter:
        def chat(self, request: dict[str, object]) -> object:
            messages = request["messages"]
            assert isinstance(messages, list)
            key = messages[-1]["content"]
            assert isinstance(key, str)
            loop.call_soon_threadsafe(entered[key].set)
            assert release[key].wait(timeout=2.0)
            if key == "valid request":
                return "valid request output", 19
            serialized = json.dumps(
                {"error": {"code": "provider_unavailable", "message": sentinel}}
            )
            return {
                "choices": [
                    {"delta": {"content": f"data: {serialized}\n\n"}}
                ]
            }

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(
        PromptExecutor,
        "_build_adapter_request",
        lambda _self, **kwargs: {
            "messages": kwargs["messages"],
            "model": kwargs["model"],
        },
    )
    executor = PromptExecutor(_Db())  # type: ignore[arg-type]
    valid_task = asyncio.create_task(
        executor.execute_prompt(
            1,
            {},
            {"provider": "openai", "model": "model-a", "parameters": {}},
        )
    )
    error_task = asyncio.create_task(
        executor.execute_prompt(
            2,
            {},
            {"provider": "openai", "model": "model-b", "parameters": {}},
        )
    )
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error request"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        release["valid request"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)
        logger.remove(sink_id)

    assert valid_result["success"] is True
    assert valid_result["raw_output"] == "valid request output"
    assert valid_result["tokens_used"] == 19
    assert error_result["success"] is False
    assert error_result["error"] == "Provider returned an error response"
    rendered = json.dumps(
        {"valid_result": valid_result, "error_result": error_result}
    ) + "".join(log_messages)
    assert sentinel not in rendered
    assert "valid request output" not in json.dumps(error_result)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_prompt_executor_concurrent_list_results_remain_request_local_and_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapped error is detached without contaminating concurrent valid output."""
    loop = asyncio.get_running_loop()
    entered = {key: asyncio.Event() for key in ("valid request", "error request")}
    release = {key: threading.Event() for key in entered}
    sentinel = "prompt-executor-concurrent-secret-/private/provider.json"
    log_messages: list[str] = []

    class _Db:
        client_id = "prompt-executor-boundary"

        def get_prompt(self, prompt_id: int) -> dict[str, object]:
            return {
                "id": prompt_id,
                "deleted": False,
                "user_prompt": "valid request" if prompt_id == 1 else "error request",
            }

    class _Adapter:
        def chat(self, request: dict[str, object]) -> list[str]:
            messages = request["messages"]
            assert isinstance(messages, list)
            key = messages[-1]["content"]
            assert isinstance(key, str)
            loop.call_soon_threadsafe(entered[key].set)
            assert release[key].wait(timeout=2.0)
            if key == "valid request":
                return ["valid request output"]
            return [
                "data: "
                + json.dumps(
                    {
                        "error": {
                            "code": "provider_unavailable",
                            "message": sentinel,
                        }
                    }
                )
                + "\n\n"
            ]

    class _Registry:
        @staticmethod
        def get_adapter(_provider: str) -> _Adapter:
            return _Adapter()

    def _build_request(
        _self: PromptExecutor,
        **kwargs: object,
    ) -> dict[str, object]:
        return {"messages": kwargs["messages"], "model": kwargs["model"]}

    monkeypatch.setattr(prompt_executor_module, "get_registry", lambda: _Registry())
    monkeypatch.setattr(PromptExecutor, "_build_adapter_request", _build_request)
    executor = PromptExecutor(_Db())  # type: ignore[arg-type]
    valid_task = asyncio.create_task(
        executor.execute_prompt(
            1,
            {},
            {"provider": "openai", "model": "model-a", "parameters": {}},
        )
    )
    error_task = asyncio.create_task(
        executor.execute_prompt(
            2,
            {},
            {"provider": "openai", "model": "model-b", "parameters": {}},
        )
    )
    sink_id = logger.add(log_messages.append, format="{message}")
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release["error request"].set()
        error_result = await asyncio.wait_for(error_task, timeout=1.0)
        release["valid request"].set()
        valid_result = await asyncio.wait_for(valid_task, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(valid_task, error_task, return_exceptions=True)
        logger.remove(sink_id)

    assert valid_result["success"] is True
    assert valid_result["raw_output"] == "valid request output"
    assert error_result["success"] is False
    assert error_result["error"] == "Provider returned an error response"
    rendered = json.dumps(
        {"valid_result": valid_result, "error_result": error_result}
    ) + "".join(log_messages)
    assert sentinel not in rendered
    assert "valid request output" not in json.dumps(error_result)


@pytest.mark.asyncio
async def test_execute_prompt_uses_structured_assembled_messages(isolated_db, monkeypatch):
    project = isolated_db.create_project(name="Executor Structured Project", user_id="test-user")
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        name="Structured Executor Prompt",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=_make_prompt_definition_payload(),
        few_shot_examples=[
            {
                "inputs": {"input": "Indexes"},
                "outputs": {"answer": "Use the covering index."},
            }
        ],
        modules_config=[
            {"type": "style_rules", "enabled": True, "config": {"tone": "concise"}}
        ],
    )

    captured: dict[str, object] = {}

    async def _fake_call_llm(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["prompt"] = kwargs.get("prompt")
        return {"content": '{"answer": "ok"}', "tokens": 11}

    monkeypatch.setattr(PromptExecutor, "_call_llm", staticmethod(_fake_call_llm))

    executor = PromptExecutor(isolated_db)
    result = await executor.execute_prompt(
        prompt["id"],
        {"input": "SQLite FTS"},
        {"provider": "openai", "model": "gpt-4", "parameters": {}},
    )

    assert [message["role"] for message in captured["messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert captured["messages"][1]["content"] == "Module style_rules: tone=concise"
    assert captured["messages"][2]["content"] == 'Example input: {"input": "Indexes"}'
    assert captured["messages"][3]["content"] == 'Example output: {"answer": "Use the covering index."}'
    assert captured["messages"][4]["content"] == "Evaluate SQLite FTS"
    assert captured["system_prompt"] is None
    assert result["success"] is True
    assert [message["role"] for message in result["metadata"]["assembled_messages"]] == [
        "system",
        "developer",
        "user",
        "assistant",
        "user",
    ]
    assert result["parsed_output"]["raw"] == '{"answer": "ok"}'


@pytest.mark.asyncio
async def test_execute_prompt_keeps_legacy_prompt_string_path(isolated_db, monkeypatch):
    project = isolated_db.create_project(name="Executor Legacy Project", user_id="test-user")
    prompt = isolated_db.create_prompt(
        project_id=project["id"],
        name="Legacy Executor Prompt",
        system_prompt="Stay concise.",
        user_prompt="Evaluate {input}",
    )

    captured: dict[str, object] = {}

    async def _fake_call_llm(*args, **kwargs):
        captured["messages"] = kwargs.get("messages")
        captured["system_prompt"] = kwargs.get("system_prompt")
        captured["prompt"] = kwargs.get("prompt")
        return {"content": "ok", "tokens": 3}

    monkeypatch.setattr(PromptExecutor, "_call_llm", staticmethod(_fake_call_llm))

    executor = PromptExecutor(isolated_db)
    result = await executor.execute_prompt(
        prompt["id"],
        {"input": "SQLite FTS"},
        {"provider": "openai", "model": "gpt-4", "parameters": {}},
    )

    assert captured["messages"] is None
    assert captured["system_prompt"] == "Stay concise."
    assert captured["prompt"] == "Evaluate SQLite FTS"
    assert result["success"] is True
    assert result["metadata"]["assembled_messages"] == [
        {"role": "system", "content": "Stay concise."},
        {"role": "user", "content": "Evaluate SQLite FTS"},
    ]
    assert result["parsed_output"]["raw"] == "ok"
