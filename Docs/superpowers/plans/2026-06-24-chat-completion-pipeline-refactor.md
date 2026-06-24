# Chat Completion Pipeline Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the validated Chat review findings and split the Chat completion pipeline into focused services while preserving the public Chat API and SSE shapes except for intentional safety rejections.

**Architecture:** Keep `tldw_Server_API/app/core/Chat/chat_service.py` as the import-facing compatibility facade, then move response shaping, moderation, persistence, streaming orchestration, tool execution, command authorization, and safe logging into narrower modules. A `ChatCompletionPipeline` object coordinates these services so future behavior is added behind explicit stages instead of expanding `chat_service.py`.

**Tech Stack:** Python async/await, FastAPI `HTTPException`, Pydantic/jsonable encoding, SQLite through `CharactersRAGDB`, Loguru, pytest, Bandit.

---

## Source Inputs

- Design spec: `Docs/Design/2026-06-24-chat-completion-pipeline-refactor-design.md`
- Backlog task: `TASK-12013`
- Primary module: `tldw_Server_API/app/core/Chat/chat_service.py`
- Scope boundary: `tldw_Server_API/app/core/Chat/` plus the Chat endpoint command listing in `tldw_Server_API/app/api/v1/endpoints/chat.py`

## Public Behavior Invariants

- `/api/v1/chat/completions` request and response shapes stay stable.
- SSE event names and payload shapes stay stable.
- Non-streaming multi-choice responses return all provider choices after safety processing.
- Local tool auto-execution rejects `n > 1` before provider calls.
- Persistence stores only the first assistant choice unless a future public contract explicitly adds multi-choice persistence.
- Structured response validation runs after output safety processing and before persistence.
- Logs never include raw user messages, system prompts, custom prompts, tool arguments, tool outputs, tool execution error details, API keys, or generated assistant content.

## File Structure

Create:
- `tldw_Server_API/app/core/Chat/response_processor.py` - choice extraction, content text extraction, redaction helpers, structured validation across choices, usage estimation across choices, assistant-name injection across choices.
- `tldw_Server_API/app/core/Chat/moderation_pipeline.py` - output self-monitoring, moderation, topic monitoring, audit writes, and choice content mutation.
- `tldw_Server_API/app/core/Chat/persistence_service.py` - assistant payload construction and first-choice persistence helpers.
- `tldw_Server_API/app/core/Chat/tool_execution_service.py` - tool auto-exec eligibility guard and first-choice tool execution orchestration.
- `tldw_Server_API/app/core/Chat/command_authorization.py` - normalized slash-command authorization context and fail-closed decision function.
- `tldw_Server_API/app/core/Chat/chat_logging.py` - safe logging helpers for prompts, content summaries, tool summaries, and exception summaries.
- `tldw_Server_API/app/core/Chat/streaming_pipeline.py` - streaming response factory wrapper that preserves existing `StreamingResponseHandler` behavior.
- `tldw_Server_API/app/core/Chat/completion_pipeline.py` - `ChatCompletionPipeline` coordinator that calls the focused services.

Modify:
- `tldw_Server_API/app/core/Chat/chat_service.py` - delegate to focused modules while preserving existing exported functions and request flow.
- `tldw_Server_API/app/core/Chat/chat_orchestrator.py` - replace raw content logs with safe summaries.
- `tldw_Server_API/app/core/Chat/command_router.py` - use fail-closed authorization for registered commands.
- `tldw_Server_API/app/api/v1/endpoints/chat.py` - build command authorization context and filter command listings with the same decision function.
- `tldw_Server_API/app/api/v1/schemas/chat_commands_schemas.py` - update descriptions that imply RBAC is conditional.
- `tldw_Server_API/app/core/Chat/document_generator.py` - repair `user_prompts` schema and enforce one active prompt per document type with a partial unique index.
- `tldw_Server_API/app/core/Chat/chat_history.py` - make legacy replacement save delete and insert in a single transaction.
- `tldw_Server_API/app/core/Chat/README.md` - document the new service boundaries.
- `tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md` - record the completed split and remaining limits.

Test:
- `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py`
- `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`
- `tldw_Server_API/tests/Chat/unit/test_document_generator.py`
- `tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py`
- `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py`

## Workspace Guard

The working tree may contain unrelated staged and unstaged files. Every commit in this plan uses explicit file paths and `git commit --only` so unrelated changes are not included.

---

### Task 1: Add Failing Non-Streaming Multi-Choice Safety Tests

**Files:**
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`

- [ ] **Step 1: Add shared helpers to the content test file**

Add these helpers after `_RedactingModeration`:

```python
class _NoModeration:
    class _Policy:
        enabled = False
        output_enabled = False
        output_action = "block"

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy()

    def evaluate_action_with_match(self, *_args, **_kwargs):
        return ("pass", None, None, None, None)

    def check_text(self, *_args, **_kwargs):
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return text


class _KeywordModeration:
    class _Policy:
        enabled = True
        output_enabled = True

        def __init__(self, action: str):
            self.output_action = action

    def __init__(self, *, keyword: str, action: str):
        self.keyword = keyword
        self.action = action

    def get_effective_policy(self, *_args, **_kwargs):
        return self._Policy(self.action)

    def evaluate_action_with_match(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (self.action, str(text).replace(self.keyword, "[redacted]"), "keyword", "default", (0, len(self.keyword)))
        return ("pass", None, None, None, None)

    def check_text(self, text, *_args, **_kwargs):
        if self.keyword in str(text):
            return (True, "keyword")
        return (False, None)

    def redact_text(self, text, *_args, **_kwargs):
        return str(text).replace(self.keyword, "[redacted]")


async def _run_non_stream_content_test(
    monkeypatch: pytest.MonkeyPatch,
    *,
    llm_response: dict,
    moderation,
    should_persist: bool = False,
    response_format: dict | None = None,
    metrics: object | None = None,
) -> tuple[dict, list[dict[str, object]], dict[str, object]]:
    logged_usage: dict[str, object] = {}

    async def fake_log_llm_usage(**kwargs):
        logged_usage.update(kwargs)

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    save_calls: list[dict[str, object]] = []

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_calls.append(payload)
        return f"message-{len(save_calls)}"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )
    cleaned_args = {
        "api_endpoint": "openai",
        "api_key": "test-key",
        "messages_payload": [{"role": "user", "content": "hi"}],
        "model": "gpt-4o-mini",
        "streaming": False,
    }
    if response_format is not None:
        cleaned_args["response_format"] = response_format

    response = await execute_non_stream_call(
        current_loop=asyncio.get_running_loop(),
        cleaned_args=cleaned_args,
        selected_provider="openai",
        provider="openai",
        model="gpt-4o-mini",
        request_json="{}",
        request=request,
        metrics=metrics or _DummyMetrics(),
        provider_manager=None,
        templated_llm_payload=[{"role": "user", "content": "hi"}],
        should_persist=should_persist,
        final_conversation_id="conv-multi-choice",
        character_card_for_context={"name": "Test"},
        chat_db=None,
        save_message_fn=save_message_fn,
        audit_service=None,
        audit_context=None,
        client_id="client",
        queue_execution_enabled=False,
        enable_provider_fallback=False,
        llm_call_func=lambda: llm_response,
        refresh_provider_params=lambda *_args, **_kwargs: None,
        moderation_getter=lambda: moderation,
    )
    return response, save_calls, logged_usage
```

- [ ] **Step 2: Add redaction and block tests for all choices**

Append these tests:

```python
@pytest.mark.asyncio
async def test_execute_non_stream_call_redacts_all_returned_choices(monkeypatch):
    response, _save_calls, _logged_usage = await _run_non_stream_content_test(
        monkeypatch,
        llm_response={
            "choices": [
                {"message": {"role": "assistant", "content": "first secret"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "second secret"}, "finish_reason": "stop"},
            ]
        },
        moderation=_RedactingModeration(),
    )

    assert response["choices"][0]["message"]["content"] == "REDACTED:first secret"
    assert response["choices"][1]["message"]["content"] == "REDACTED:second secret"


@pytest.mark.asyncio
async def test_execute_non_stream_call_blocks_when_later_choice_violates(monkeypatch):
    save_calls: list[dict[str, object]] = []

    async def fake_log_llm_usage(**_kwargs):
        return None

    monkeypatch.setattr(chat_service, "log_llm_usage", fake_log_llm_usage)
    monkeypatch.setattr(chat_service, "get_topic_monitoring_service", lambda: None)

    async def save_message_fn(_db, _conv_id, payload, use_transaction=True):
        save_calls.append(payload)
        return "message-1"

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/api/v1/chat/completions"),
        headers={},
        state=SimpleNamespace(user_id=None, api_key_id=None),
    )

    with pytest.raises(HTTPException) as exc_info:
        await execute_non_stream_call(
            current_loop=asyncio.get_running_loop(),
            cleaned_args={
                "api_endpoint": "openai",
                "api_key": "test-key",
                "messages_payload": [{"role": "user", "content": "hi"}],
                "model": "gpt-4o-mini",
                "streaming": False,
            },
            selected_provider="openai",
            provider="openai",
            model="gpt-4o-mini",
            request_json="{}",
            request=request,
            metrics=_DummyMetrics(),
            provider_manager=None,
            templated_llm_payload=[{"role": "user", "content": "hi"}],
            should_persist=True,
            final_conversation_id="conv-block-later-choice",
            character_card_for_context={"name": "Test"},
            chat_db=None,
            save_message_fn=save_message_fn,
            audit_service=None,
            audit_context=None,
            client_id="client",
            queue_execution_enabled=False,
            enable_provider_fallback=False,
            llm_call_func=lambda: {
                "choices": [
                    {"message": {"role": "assistant", "content": "safe"}, "finish_reason": "stop"},
                    {"message": {"role": "assistant", "content": "unsafe-token"}, "finish_reason": "stop"},
                ]
            },
            refresh_provider_params=lambda *_args, **_kwargs: None,
            moderation_getter=lambda: _KeywordModeration(keyword="unsafe-token", action="block"),
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Output violates moderation policy"
    assert save_calls == []
```

- [ ] **Step 3: Add structured validation and usage-estimate tests for all choices**

Append these tests:

```python
@pytest.mark.asyncio
async def test_execute_non_stream_call_validates_all_structured_choices_before_persist(monkeypatch):
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_schema",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
        },
    }

    with pytest.raises(HTTPException) as exc_info:
        await _run_non_stream_content_test(
            monkeypatch,
            llm_response={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer":"ok"}'}, "finish_reason": "stop"},
                    {"message": {"role": "assistant", "content": '{"answer":123}'}, "finish_reason": "stop"},
                ]
            },
            moderation=_NoModeration(),
            should_persist=True,
            response_format=response_format,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "structured_output_schema_error",
        "message": "Model output did not match the requested JSON schema.",
    }


@pytest.mark.asyncio
async def test_execute_non_stream_call_missing_usage_estimates_all_returned_choices(monkeypatch):
    metrics = _CapturingMetrics()
    response, _save_calls, logged_usage = await _run_non_stream_content_test(
        monkeypatch,
        llm_response={
            "choices": [
                {"message": {"role": "assistant", "content": "abcd"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "abcdefgh"}, "finish_reason": "stop"},
            ]
        },
        moderation=_NoModeration(),
        metrics=metrics,
    )

    assert len(response["choices"]) == 2
    assert logged_usage["completion_tokens"] == 3
    assert logged_usage["total_tokens"] == logged_usage["prompt_tokens"] + 3
    assert logged_usage["estimate_source"] == "missing_usage"
```

- [ ] **Step 4: Verify the new tests fail for the current implementation**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -q
```

Expected: the new tests fail because only the first returned choice is processed.

- [ ] **Step 5: Commit only the failing tests**

Run:

```bash
git add tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
git commit --only tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -m "test(chat): cover multi-choice response safety"
```

Expected: commit succeeds and does not include unrelated files.

---

### Task 2: Create Response Processor and Apply Safety to Every Choice

**Files:**
- Create: `tldw_Server_API/app/core/Chat/response_processor.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`

- [ ] **Step 1: Create response processor types and helpers**

Create `response_processor.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class NonStreamChoice:
    index: int
    choice: dict[str, Any]
    message: dict[str, Any]
    content: Any | None
    content_text: str
    tool_calls: Any | None
    function_call: Any | None


def extract_text_from_content(content: Any | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif "text" in item and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def collect_non_stream_choices(llm_response: Any) -> list[NonStreamChoice]:
    if not isinstance(llm_response, dict):
        return []
    raw_choices = llm_response.get("choices")
    if not isinstance(raw_choices, list):
        return []
    choices: list[NonStreamChoice] = []
    for index, raw_choice in enumerate(raw_choices):
        if not isinstance(raw_choice, dict):
            continue
        message = raw_choice.get("message")
        if not isinstance(message, dict):
            message = {}
            raw_choice["message"] = message
        content = message.get("content")
        choices.append(
            NonStreamChoice(
                index=index,
                choice=raw_choice,
                message=message,
                content=content,
                content_text=extract_text_from_content(content),
                tool_calls=message.get("tool_calls"),
                function_call=message.get("function_call"),
            )
        )
    return choices


def set_choice_content(choice: NonStreamChoice, content: Any | None) -> None:
    choice.message["content"] = content
    choice.content = content
    choice.content_text = extract_text_from_content(content)


def apply_redaction_to_content(content: Any | None, redact_text: Callable[[str], str]) -> Any | None:
    if isinstance(content, str):
        return redact_text(content)
    if isinstance(content, list):
        redacted_items: list[Any] = []
        for item in content:
            if isinstance(item, dict):
                new_item = dict(item)
                if isinstance(new_item.get("text"), str):
                    new_item["text"] = redact_text(new_item["text"])
                redacted_items.append(new_item)
            else:
                redacted_items.append(item)
        return redacted_items
    return content


def estimate_completion_tokens_from_choices(choices: list[NonStreamChoice]) -> int:
    return sum(max(0, len(choice.content_text) // 4) for choice in choices)


def primary_choice(choices: list[NonStreamChoice]) -> NonStreamChoice | None:
    return choices[0] if choices else None


def inject_assistant_name_into_choices(choices: list[NonStreamChoice], assistant_name: str | None) -> None:
    if not assistant_name:
        return
    for choice in choices:
        if not choice.message.get("name"):
            choice.message["name"] = assistant_name


def validate_structured_choices(
    *,
    choices: list[NonStreamChoice],
    structured_request_context: Any,
    validate_structured_response: Callable[..., dict[str, Any] | None],
) -> dict[str, Any] | None:
    metadata_by_choice: list[dict[str, Any]] = []
    for choice in choices:
        metadata = validate_structured_response(
            raw_text=choice.content,
            structured_request_context=structured_request_context,
        )
        if metadata is not None:
            metadata_by_choice.append({"choice_index": choice.index, **metadata})
    if not metadata_by_choice:
        return None
    if len(metadata_by_choice) == 1:
        return metadata_by_choice[0]
    return {"choices": metadata_by_choice}
```

- [ ] **Step 2: Import processor helpers in `chat_service.py` and preserve compatibility names**

Add near the other Chat imports:

```python
from tldw_Server_API.app.core.Chat.response_processor import (
    NonStreamChoice,
    apply_redaction_to_content,
    collect_non_stream_choices,
    estimate_completion_tokens_from_choices,
    extract_text_from_content,
    inject_assistant_name_into_choices,
    primary_choice,
    set_choice_content,
    validate_structured_choices,
)
```

Then replace the local `_extract_text_from_content` and `_apply_redaction_to_content` function bodies with compatibility wrappers:

```python
def _extract_text_from_content(content: Any | None) -> str:
    return extract_text_from_content(content)


def _apply_redaction_to_content(content: Any | None, moderation: Any, eff_policy: Any) -> Any | None:
    return apply_redaction_to_content(
        content,
        lambda text: moderation.redact_text(text, eff_policy),
    )
```

- [ ] **Step 3: Replace first-choice extraction with collected choices**

In `execute_non_stream_call`, replace the variables initialized after string normalization with:

```python
processed_choices: list[NonStreamChoice] = collect_non_stream_choices(llm_response)
first_choice = primary_choice(processed_choices)
content_to_save: Any | None = first_choice.content if first_choice else None
tool_calls_to_save: Any | None = first_choice.tool_calls if first_choice else None
function_call_to_save: Any | None = first_choice.function_call if first_choice else None
first_turn_tool_calls: Any | None = tool_calls_to_save
first_turn_function_call: Any | None = function_call_to_save
```

Keep the existing `llm_response is None` branch. For dict responses, use `processed_choices` for all later response processing and keep first-choice values for persistence/tool execution.

- [ ] **Step 4: Estimate missing usage from all choices**

Replace the missing-usage completion estimate block with:

```python
ct_est = estimate_completion_tokens_from_choices(processed_choices)
```

Keep prompt-token estimation and `log_llm_usage` arguments unchanged.

- [ ] **Step 5: Validate structured output across every choice**

Replace each non-continuation call to `validate_structured_response(raw_text=content_to_save, ...)` with:

```python
structured_metadata = validate_structured_choices(
    choices=processed_choices,
    structured_request_context=structured_request_context,
    validate_structured_response=validate_structured_response,
)
```

After a continuation response replaces `llm_response`, immediately refresh:

```python
processed_choices = collect_non_stream_choices(llm_response)
first_choice = primary_choice(processed_choices)
content_to_save = first_choice.content if first_choice else None
tool_calls_to_save = first_choice.tool_calls if first_choice else None
function_call_to_save = first_choice.function_call if first_choice else None
```

- [ ] **Step 6: Inject assistant names into all choices**

Replace both first-choice assistant-name injection blocks with:

```python
asst_name = sanitize_sender_name(
    character_card_for_context.get("name") if character_card_for_context else None
)
inject_assistant_name_into_choices(collect_non_stream_choices(llm_response), asst_name)
```

Run the same operation on `encoded_payload` after encoding so large and small responses match.

- [ ] **Step 7: Run the content tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -q
```

Expected: the multi-choice tests from Task 1 pass or only fail on moderation code that Task 3 moves.

- [ ] **Step 8: Commit the response processor**

Run:

```bash
git add tldw_Server_API/app/core/Chat/response_processor.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
git commit --only tldw_Server_API/app/core/Chat/response_processor.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -m "fix(chat): process non-stream choices consistently"
```

Expected: commit succeeds and the content test file is included with the processor changes.

---

### Task 3: Extract Output Moderation and Self-Monitoring

**Files:**
- Create: `tldw_Server_API/app/core/Chat/moderation_pipeline.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`

- [ ] **Step 1: Create moderation runtime dataclass**

Create `moderation_pipeline.py` with:

```python
from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.core.Chat.response_processor import (
    NonStreamChoice,
    apply_redaction_to_content,
    set_choice_content,
)
from tldw_Server_API.app.core.Audit.audit_events import AuditEventType
from tldw_Server_API.app.core.Audit.mandatory_audit import (
    MandatoryAuditWriteError,
    write_mandatory_moderation_audit,
)


_MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    IndexError,
    KeyError,
    LookupError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


@dataclass
class OutputModerationRuntime:
    request: Any | None
    client_id: str
    conversation_id: str | None
    metrics: Any
    audit_service: Any | None
    audit_context: Any | None
    moderation_getter: Callable[[], Any]
    self_monitoring_service: Any | None
    topic_monitoring_getter: Callable[[], Any | None]
    capture_review_item: Callable[..., Awaitable[None]]
    emit_completion_metric: Callable[[str], None]
```

- [ ] **Step 2: Add user and topic helpers**

Add below the dataclass:

```python
def _request_state_value(request: Any | None, name: str) -> Any | None:
    try:
        if request is not None and hasattr(request, "state"):
            return getattr(request.state, name, None)
    except _MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS:
        return None
    return None


def _moderation_user_id(runtime: OutputModerationRuntime) -> str:
    req_user_id = _request_state_value(runtime.request, "user_id")
    return str(req_user_id) if req_user_id is not None else str(runtime.client_id)
```

- [ ] **Step 3: Add self-monitoring across choices**

Add:

```python
async def apply_self_monitoring_to_choices(
    *,
    choices: list[NonStreamChoice],
    runtime: OutputModerationRuntime,
) -> None:
    if runtime.self_monitoring_service is None:
        return
    user_id = _moderation_user_id(runtime)
    loop = asyncio.get_running_loop()
    for choice in choices:
        if not choice.content_text:
            continue
        try:
            result = await loop.run_in_executor(
                None,
                lambda text=choice.content_text: runtime.self_monitoring_service.check_text(
                    text=text,
                    user_id=user_id,
                    phase="output",
                    conversation_id=runtime.conversation_id,
                ),
            )
        except _MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Self-monitoring output check skipped type={}", type(exc).__name__)
            continue
        if result.action == "block":
            runtime.emit_completion_metric("blocked")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.block_message or "Output blocked by self-monitoring rule",
            )
        if result.action == "redact" and result.redacted_text is not None:
            set_choice_content(choice, result.redacted_text)
```

- [ ] **Step 4: Add output moderation across choices**

Add:

```python
async def apply_output_moderation_to_choices(
    *,
    choices: list[NonStreamChoice],
    runtime: OutputModerationRuntime,
) -> None:
    moderation = runtime.moderation_getter()
    req_user_id = _request_state_value(runtime.request, "user_id")
    eff_policy = moderation.get_effective_policy(str(req_user_id) if req_user_id is not None else runtime.client_id)
    if not (getattr(eff_policy, "enabled", False) and getattr(eff_policy, "output_enabled", False)):
        return

    for choice in choices:
        if not choice.content_text:
            continue
        resolved_action = None
        redacted_val = None
        matched_pattern = None
        category = None
        match_span = None
        sample = None

        if hasattr(moderation, "evaluate_action_with_match"):
            eval_res = moderation.evaluate_action_with_match(choice.content_text, eff_policy, "output")
            if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                resolved_action = eval_res[0]
                redacted_val = eval_res[1]
                matched_pattern = eval_res[2]
                category = eval_res[3] if len(eval_res) >= 4 else None
                match_span = eval_res[4] if len(eval_res) >= 5 else None
        elif hasattr(moderation, "evaluate_action"):
            eval_res = moderation.evaluate_action(choice.content_text, eff_policy, "output")
            if isinstance(eval_res, tuple) and len(eval_res) >= 3:
                resolved_action = eval_res[0]
                redacted_val = eval_res[1]
                matched_pattern = eval_res[2]
                category = eval_res[3] if len(eval_res) >= 4 else None

        if match_span and hasattr(moderation, "build_sanitized_snippet"):
            with contextlib.suppress(_MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS):
                sample = moderation.build_sanitized_snippet(choice.content_text, eff_policy, match_span, matched_pattern)
        if resolved_action and resolved_action != "pass" and sample is None:
            with contextlib.suppress(_MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS):
                _flagged, sample = moderation.check_text(choice.content_text, eff_policy, "output")
        if not resolved_action:
            flagged, sample = moderation.check_text(choice.content_text, eff_policy, "output")
            if flagged:
                resolved_action = getattr(eff_policy, "output_action", "block")
                redacted_val = moderation.redact_text(choice.content_text, eff_policy) if resolved_action == "redact" else None

        await _schedule_topic_monitoring(choice, runtime)

        if resolved_action == "block":
            await _audit_moderation(runtime, "block", sample, category, matched_pattern, eff_policy)
            runtime.emit_completion_metric("blocked")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Output violates moderation policy")
        if resolved_action == "redact":
            with contextlib.suppress(_MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS):
                runtime.metrics.track_moderation_output(_moderation_user_id(runtime), "redact", streaming=False, category=(category or "default"))
            await _audit_moderation(runtime, "redact", sample, category, matched_pattern, eff_policy)
            if isinstance(redacted_val, str) and isinstance(choice.content, str):
                set_choice_content(choice, redacted_val)
            else:
                set_choice_content(
                    choice,
                    apply_redaction_to_content(
                        choice.content,
                        lambda text: moderation.redact_text(text, eff_policy),
                    ),
                )
        elif resolved_action == "warn":
            await runtime.capture_review_item(
                phase="output",
                action="warn",
                excerpt=sample,
                category=category,
                matched_pattern=matched_pattern,
                effective_policy=eff_policy,
                source_id=runtime.conversation_id,
                user_id=_moderation_user_id(runtime),
            )
```

- [ ] **Step 5: Add audit and topic helper functions**

Add:

```python
async def _audit_moderation(
    runtime: OutputModerationRuntime,
    action: str,
    sample: str | None,
    category: str | None,
    matched_pattern: str | None,
    eff_policy: Any,
) -> None:
    if runtime.audit_service and runtime.audit_context:
        await write_mandatory_moderation_audit(
            audit_service=runtime.audit_service,
            audit_context=runtime.audit_context,
            audit_event_type=AuditEventType.SECURITY_VIOLATION,
            action="moderation.output",
            result="failure" if action == "block" else "success",
            metadata={
                "phase": "output",
                "streaming": False,
                "action": action,
                "pattern": sample,
            },
        )
    await runtime.capture_review_item(
        phase="output",
        action=action,
        excerpt=sample,
        category=category,
        matched_pattern=matched_pattern,
        effective_policy=eff_policy,
        source_id=runtime.conversation_id,
        user_id=_moderation_user_id(runtime),
    )


async def _schedule_topic_monitoring(choice: NonStreamChoice, runtime: OutputModerationRuntime) -> None:
    try:
        monitor = runtime.topic_monitoring_getter()
        if monitor is None or not choice.content_text:
            return
        user_id = _moderation_user_id(runtime)
        monitor.schedule_evaluate_and_alert(
            user_id=user_id,
            text=choice.content_text,
            source="chat.output",
            scope_type="user",
            scope_id=user_id,
            team_ids=_request_state_value(runtime.request, "team_ids"),
            org_ids=_request_state_value(runtime.request, "org_ids"),
            source_id=runtime.conversation_id,
        )
    except _MODERATION_PIPELINE_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Topic monitoring skipped type={}", type(exc).__name__)
```

- [ ] **Step 6: Replace moderation code in `execute_non_stream_call`**

In `chat_service.py`, import:

```python
from tldw_Server_API.app.core.Chat.moderation_pipeline import (
    OutputModerationRuntime,
    apply_output_moderation_to_choices,
    apply_self_monitoring_to_choices,
)
```

Replace the self-monitoring and output moderation blocks with:

```python
moderation_runtime = OutputModerationRuntime(
    request=request,
    client_id=client_id,
    conversation_id=str(final_conversation_id) if final_conversation_id else None,
    metrics=metrics,
    audit_service=audit_service,
    audit_context=audit_context,
    moderation_getter=moderation_getter or get_moderation_service,
    self_monitoring_service=self_monitoring_service,
    topic_monitoring_getter=get_topic_monitoring_service,
    capture_review_item=_capture_moderation_review_item_safely_async,
    emit_completion_metric=lambda outcome: _emit_chat_run_first_completion_metric(
        metrics,
        context=run_first_metric_context,
        outcome=outcome,
    ),
)
await apply_self_monitoring_to_choices(choices=processed_choices, runtime=moderation_runtime)
await apply_output_moderation_to_choices(choices=processed_choices, runtime=moderation_runtime)
first_choice = primary_choice(processed_choices)
content_to_save = first_choice.content if first_choice else None
tool_calls_to_save = first_choice.tool_calls if first_choice else None
function_call_to_save = first_choice.function_call if first_choice else None
```

- [ ] **Step 7: Run content tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 8: Commit moderation extraction**

Run:

```bash
git add tldw_Server_API/app/core/Chat/moderation_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
git commit --only tldw_Server_API/app/core/Chat/moderation_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -m "fix(chat): moderate every non-stream choice"
```

Expected: commit succeeds with only the three listed files.

---

### Task 4: Reject Multi-Choice Tool Auto-Execution Before Provider Calls

**Files:**
- Create: `tldw_Server_API/app/core/Chat/tool_execution_service.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py`

- [ ] **Step 1: Add the failing guard test**

Append to `test_chat_service_tool_autoexec.py`:

```python
@pytest.mark.asyncio
async def test_tool_autoexec_rejects_multi_choice_before_provider_call(monkeypatch):
    provider_called = False

    def llm_call_func():
        nonlocal provider_called
        provider_called = True
        return _build_llm_response_with_tool_calls()

    async def save_message_fn(*_args, **_kwargs):
        return "message-1"

    monkeypatch.setattr(chat_service, "should_auto_execute_tools", lambda: True)

    with pytest.raises(HTTPException) as exc_info:
        await _run_execute_non_stream_call(
            llm_call_func=llm_call_func,
            save_message_fn=save_message_fn,
            cleaned_args_overrides={
                "n": 2,
                "tools": [{"type": "function", "function": {"name": "notes.search", "parameters": {}}}],
            },
        )

    assert provider_called is False
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == {
        "code": "unsupported_multi_choice_tool_autoexec",
        "message": "Local tool auto-execution supports one assistant choice per request.",
    }
```

- [ ] **Step 2: Verify the guard test fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py::test_tool_autoexec_rejects_multi_choice_before_provider_call -q
```

Expected: test fails because the provider callable is invoked.

- [ ] **Step 3: Create tool execution service guard**

Create `tool_execution_service.py` with:

```python
from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException, status


def request_choice_count(cleaned_args: dict[str, Any] | None) -> int:
    if not isinstance(cleaned_args, dict):
        return 1
    raw_n = cleaned_args.get("n", 1)
    try:
        return max(1, int(raw_n))
    except (TypeError, ValueError):
        return 1


def ensure_tool_autoexec_supports_request(
    *,
    cleaned_args: dict[str, Any] | None,
    should_run_tool_autoexec: Callable[[dict[str, Any] | None], bool],
) -> None:
    if should_run_tool_autoexec(cleaned_args) and request_choice_count(cleaned_args) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "unsupported_multi_choice_tool_autoexec",
                "message": "Local tool auto-execution supports one assistant choice per request.",
            },
        )
```

- [ ] **Step 4: Call the guard before queue/fallback/provider execution**

In `execute_non_stream_call`, after structured request preparation and before `_evaluate_chat_prompt_cost_guardrails`, add:

```python
ensure_tool_autoexec_supports_request(
    cleaned_args=cleaned_args,
    should_run_tool_autoexec=should_run_legacy_tool_autoexec,
)
```

Import the guard:

```python
from tldw_Server_API.app.core.Chat.tool_execution_service import ensure_tool_autoexec_supports_request
```

- [ ] **Step 5: Run tool autoexec tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 6: Commit tool guard**

Run:

```bash
git add tldw_Server_API/app/core/Chat/tool_execution_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py
git commit --only tldw_Server_API/app/core/Chat/tool_execution_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py -m "fix(chat): reject multi-choice tool autoexec"
```

Expected: commit succeeds with only the listed files.

---

### Task 5: Extract First-Choice Persistence

**Files:**
- Create: `tldw_Server_API/app/core/Chat/persistence_service.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`

- [ ] **Step 1: Add a test that only the first choice is persisted**

Append to `test_chat_service_content.py`:

```python
@pytest.mark.asyncio
async def test_execute_non_stream_call_persists_first_choice_only(monkeypatch):
    response, save_calls, _logged_usage = await _run_non_stream_content_test(
        monkeypatch,
        llm_response={
            "choices": [
                {"message": {"role": "assistant", "content": "persist me"}, "finish_reason": "stop"},
                {"message": {"role": "assistant", "content": "return only"}, "finish_reason": "stop"},
            ]
        },
        moderation=_NoModeration(),
        should_persist=True,
    )

    assert len(response["choices"]) == 2
    assert len(save_calls) == 1
    assert save_calls[0]["content"] == "persist me"
```

- [ ] **Step 2: Create persistence service**

Create `persistence_service.py` with:

```python
from __future__ import annotations

from typing import Any, Callable


def build_assistant_message_payload(
    *,
    character_card_for_context: dict[str, Any] | None,
    assistant_parent_message_id: str | None,
    content: Any | None,
    tool_calls: Any | None,
    function_call: Any | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sender": (
            character_card_for_context.get("name")
            if character_card_for_context and character_card_for_context.get("name")
            else "assistant"
        ),
        "content": content or "",
    }
    if assistant_parent_message_id:
        payload["parent_message_id"] = assistant_parent_message_id
    if tool_calls is not None:
        payload["tool_calls"] = tool_calls
    if function_call is not None:
        payload["function_call"] = function_call
    return payload


async def save_assistant_message(
    *,
    chat_db: Any,
    conversation_id: str,
    save_message_fn: Callable[..., Any],
    payload: dict[str, Any],
) -> str | None:
    return await save_message_fn(chat_db, conversation_id, payload, use_transaction=True)


async def save_tool_messages(
    *,
    chat_db: Any,
    conversation_id: str,
    save_message_fn: Callable[..., Any],
    tool_messages: list[dict[str, Any]],
) -> None:
    for tool_message in tool_messages:
        await save_message_fn(chat_db, conversation_id, tool_message, use_transaction=True)
```

- [ ] **Step 3: Replace payload construction in `chat_service.py`**

Import:

```python
from tldw_Server_API.app.core.Chat.persistence_service import (
    build_assistant_message_payload,
    save_assistant_message,
    save_tool_messages,
)
```

Replace `_build_assistant_message_payload` with:

```python
def _build_assistant_message_payload(
    *,
    character_card_for_context: dict[str, Any] | None,
    assistant_parent_message_id: str | None,
    content: Any | None,
    tool_calls: Any | None,
    function_call: Any | None,
) -> dict[str, Any]:
    return build_assistant_message_payload(
        character_card_for_context=character_card_for_context,
        assistant_parent_message_id=assistant_parent_message_id,
        content=content,
        tool_calls=tool_calls,
        function_call=function_call,
    )
```

Replace direct `save_message_fn` calls for assistant payloads with `save_assistant_message(...)`, and replace loops over tool messages with `save_tool_messages(...)`.

- [ ] **Step 4: Run persistence-related tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py -q
```

Expected: both test files pass.

- [ ] **Step 5: Commit persistence extraction**

Run:

```bash
git add tldw_Server_API/app/core/Chat/persistence_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py
git commit --only tldw_Server_API/app/core/Chat/persistence_service.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_chat_service_content.py -m "refactor(chat): isolate response persistence"
```

Expected: commit succeeds with only the listed files.

---

### Task 6: Replace Sensitive Logs With Safe Summaries

**Files:**
- Create: `tldw_Server_API/app/core/Chat/chat_logging.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_orchestrator.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py`

- [ ] **Step 1: Create safe logging helpers**

Create `chat_logging.py` with:

```python
from __future__ import annotations

from typing import Any


def text_summary(value: Any) -> dict[str, Any]:
    if value is None:
        return {"present": False, "chars": 0}
    text = value if isinstance(value, str) else str(value)
    return {"present": True, "chars": len(text)}


def prompt_template_summary(*, template_name: str | None, system_message: Any, character_name: str | None) -> dict[str, Any]:
    return {
        "template_name": template_name,
        "system_message": text_summary(system_message),
        "character_name": character_name,
    }


def tool_payload_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"kind": "list", "count": len(value)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(str(key) for key in value.keys())}
    return {"kind": type(value).__name__}


def exception_summary(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__}
```

- [ ] **Step 2: Add tests proving summaries do not include raw prompt text**

Append to `test_chat_service_system_messages.py`:

```python
def test_prompt_template_summary_omits_raw_prompt_text():
    from tldw_Server_API.app.core.Chat.chat_logging import prompt_template_summary

    summary = prompt_template_summary(
        template_name="raw",
        system_message="do not leak this system prompt",
        character_name="Tester",
    )

    rendered = repr(summary)
    assert "do not leak" not in rendered
    assert summary["system_message"] == {"present": True, "chars": 30}
    assert summary["character_name"] == "Tester"
```

- [ ] **Step 3: Replace raw prompt logs in `apply_prompt_templating`**

In `chat_service.py`, import:

```python
from tldw_Server_API.app.core.Chat.chat_logging import exception_summary, prompt_template_summary, text_summary
```

Replace:

```python
logger.debug(
    f"sys_msg_from_req: {sys_msg_from_req}, active_template: {active_template}, character: {character_card.get('name') if character_card else None}"
)
```

with:

```python
logger.debug(
    "Prompt templating inputs {}",
    prompt_template_summary(
        template_name=getattr(active_template, "name", None),
        system_message=sys_msg_from_req,
        character_name=character_card.get("name") if character_card else None,
    ),
)
```

Replace fallback preview logs with:

```python
logger.debug("Template empty, using payload system message summary {}", text_summary(final_system_message))
logger.debug("Template empty, using character system prompt summary {}", text_summary(final_system_message))
logger.debug("Using character system prompt summary {}", text_summary(final_system_message))
```

Replace:

```python
logger.debug(f"Final system message: {repr(final_system_message)}")
```

with:

```python
logger.debug("Final system message summary {}", text_summary(final_system_message))
```

- [ ] **Step 4: Replace raw input/custom prompt logs in `chat_orchestrator.py`**

Import:

```python
from tldw_Server_API.app.core.Chat.chat_logging import exception_summary, text_summary
```

Replace the raw user input logs at the existing INFO call sites with:

```python
logger.info("Chat input received summary={}", text_summary(user_input))
```

Replace the custom prompt DEBUG log with:

```python
logger.debug("Custom prompt received summary={}", text_summary(custom_prompt))
```

Replace exception logs that include tool execution result or provider content with the exception type:

```python
logger.warning("Chat tool auto-execution skipped error={}", exception_summary(autoexec_err))
logger.warning("Chat tool auto-continue skipped error={}", exception_summary(continue_err))
```

- [ ] **Step 5: Run logging tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py -q
```

Expected: system message tests pass and the new summary test passes.

- [ ] **Step 6: Commit safe logging**

Run:

```bash
git add tldw_Server_API/app/core/Chat/chat_logging.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py
git commit --only tldw_Server_API/app/core/Chat/chat_logging.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/core/Chat/chat_orchestrator.py tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py -m "fix(chat): redact sensitive log surfaces"
```

Expected: commit succeeds with only the listed files.

---

### Task 7: Centralize Slash Command Authorization and Fail Closed

**Files:**
- Create: `tldw_Server_API/app/core/Chat/command_authorization.py`
- Modify: `tldw_Server_API/app/core/Chat/command_router.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/chat_commands_schemas.py`
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py`
- Test: `tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py`

- [ ] **Step 1: Add test helper for authorized command contexts**

In `test_command_router.py`, add after the autouse fixture:

```python
def _authorized_ctx(user_id: str = "u1", *permissions: str) -> command_router.CommandContext:
    return command_router.CommandContext(
        user_id=user_id,
        auth_user_id=1,
        request_meta={
            "permissions": list(permissions or ("chat.commands.*",)),
            "roles": [],
            "is_admin": False,
        },
    )
```

Update command behavior tests that are not authorization tests to use `_authorized_ctx(...)`. For example:

```python
ctx = _authorized_ctx("u1", "chat.commands.time")
```

- [ ] **Step 2: Add fail-closed command authorization tests**

Append:

```python
@pytest.mark.asyncio
async def test_required_command_permission_is_enforced_without_legacy_flag(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")
    monkeypatch.delenv("CHAT_COMMANDS_REQUIRE_PERMISSIONS", raising=False)

    denied = await command_router.async_dispatch_command(
        command_router.CommandContext(user_id="anon", auth_user_id=None),
        "time",
        None,
    )

    assert not denied.ok
    assert denied.metadata["error"] == "permission_denied"
    assert denied.metadata["required_permission"] == "chat.commands.time"


@pytest.mark.asyncio
async def test_command_permission_allows_claim_without_db_lookup(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    def fail_if_called(_user_id, _permission):
        raise AssertionError("permission DB should not be called for claim hit")

    monkeypatch.setattr(command_router, "_user_has_permission", fail_if_called)

    allowed = await command_router.async_dispatch_command(
        _authorized_ctx("claim-user", "chat.commands.time"),
        "time",
        None,
    )

    assert allowed.ok


@pytest.mark.asyncio
async def test_command_permission_allows_single_user_owner(monkeypatch):
    monkeypatch.setenv("CHAT_COMMANDS_ENABLED", "1")

    allowed = await command_router.async_dispatch_command(
        command_router.CommandContext(
            user_id="owner",
            auth_user_id=None,
            request_meta={"auth_mode": "single_user", "is_single_user_owner": True},
        ),
        "time",
        None,
    )

    assert allowed.ok
```

- [ ] **Step 3: Create command authorization module**

Create `command_authorization.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CommandAuthorizationContext:
    auth_user_id: int | None
    user_id: str
    permissions: frozenset[str]
    roles: frozenset[str]
    is_admin: bool
    auth_mode: str | None
    is_single_user_owner: bool


@dataclass(frozen=True)
class CommandAuthorizationDecision:
    allowed: bool
    metadata: dict[str, Any]


def build_command_authorization_context(ctx: Any) -> CommandAuthorizationContext:
    request_meta = getattr(ctx, "request_meta", None) or {}
    permissions = frozenset(str(value) for value in request_meta.get("permissions", []) or [])
    roles = frozenset(str(value) for value in request_meta.get("roles", []) or [])
    return CommandAuthorizationContext(
        auth_user_id=getattr(ctx, "auth_user_id", None),
        user_id=str(getattr(ctx, "user_id", "anonymous")),
        permissions=permissions,
        roles=roles,
        is_admin=bool(request_meta.get("is_admin", False)),
        auth_mode=str(request_meta.get("auth_mode")) if request_meta.get("auth_mode") else None,
        is_single_user_owner=bool(request_meta.get("is_single_user_owner", False)),
    )


def _permission_in_claims(permission: str, permissions: frozenset[str]) -> bool:
    if permission in permissions or "*" in permissions:
        return True
    parts = permission.split(".")
    for end in range(len(parts), 0, -1):
        wildcard = ".".join(parts[:end]) + ".*"
        if wildcard in permissions:
            return True
    return False


def authorize_command(
    *,
    spec: Any,
    context: CommandAuthorizationContext,
    permission_checker: Callable[[int, str], bool],
) -> CommandAuthorizationDecision:
    required_permission = getattr(spec, "required_permission", None)
    rbac_required = bool(getattr(spec, "rbac_required", bool(required_permission)))
    if not required_permission or not rbac_required:
        return CommandAuthorizationDecision(True, {"checked": False})
    metadata = {"checked": True, "required_permission": required_permission}
    if context.is_admin:
        return CommandAuthorizationDecision(True, {**metadata, "source": "admin"})
    if context.auth_mode == "single_user" and context.is_single_user_owner:
        return CommandAuthorizationDecision(True, {**metadata, "source": "single_user_owner"})
    if _permission_in_claims(required_permission, context.permissions):
        return CommandAuthorizationDecision(True, {**metadata, "source": "claims"})
    if context.auth_user_id is None:
        return CommandAuthorizationDecision(False, {**metadata, "permitted": False})
    try:
        permitted = bool(permission_checker(int(context.auth_user_id), required_permission))
    except Exception:  # noqa: BLE001 - command authorization must fail closed on permission backend errors
        permitted = False
    if permitted:
        return CommandAuthorizationDecision(True, {**metadata, "source": "db"})
    return CommandAuthorizationDecision(False, {**metadata, "permitted": False})
```

- [ ] **Step 4: Use the decision in `command_router.py`**

Import:

```python
from tldw_Server_API.app.core.Chat.command_authorization import (
    authorize_command,
    build_command_authorization_context,
)
```

Replace the `CHAT_COMMANDS_REQUIRE_PERMISSIONS` block in `async_dispatch_command` with:

```python
decision = authorize_command(
    spec=spec,
    context=build_command_authorization_context(ctx),
    permission_checker=_user_has_permission,
)
if not decision.allowed:
    log_counter("chat_command_error", labels={"command": cmd, "reason": "permission_denied"})
    try:
        increment_counter("chat_command_errors_total", labels={"command": cmd, "reason": "permission_denied"})
        increment_counter("chat_command_invoked_total", labels={"command": cmd, "status": "denied"})
    except _COMMAND_ROUTER_NONCRITICAL_EXCEPTIONS:
        pass
    return _finalize_result(
        CommandResult(
            ok=False,
            command=cmd,
            content=f"Permission denied for /{cmd}",
            metadata={"error": "permission_denied", **decision.metadata},
        )
    )
```

- [ ] **Step 5: Filter command listing with the same decision**

In `chat.py`, remove the `require_perms` branch and build a `CommandContext` with request metadata:

```python
command_ctx = command_router.CommandContext(
    user_id=str(getattr(current_user, "id", "anonymous")),
    auth_user_id=getattr(current_user, "id", None),
    request_meta={
        "permissions": list(getattr(current_user, "permissions", []) or []),
        "roles": list(getattr(current_user, "roles", []) or []),
        "is_admin": bool(getattr(current_user, "is_admin", False)),
        "auth_mode": os.getenv("AUTH_MODE"),
        "is_single_user_owner": bool(getattr(current_user, "is_single_user_owner", False)),
    },
)
auth_context = build_command_authorization_context(command_ctx)
```

For each registered command, include it only when:

```python
decision = authorize_command(
    spec=spec,
    context=auth_context,
    permission_checker=user_has_permission,
)
if decision.allowed:
    items.append(_as_chat_command_from_spec(name, spec))
```

If the registry cannot be read, return `ChatCommandsListResponse(commands=[])` instead of an unfiltered list.

- [ ] **Step 6: Run command tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py -q
```

Expected: command router and command listing tests pass.

- [ ] **Step 7: Commit command authorization**

Run:

```bash
git add tldw_Server_API/app/core/Chat/command_authorization.py tldw_Server_API/app/core/Chat/command_router.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/schemas/chat_commands_schemas.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py
git commit --only tldw_Server_API/app/core/Chat/command_authorization.py tldw_Server_API/app/core/Chat/command_router.py tldw_Server_API/app/api/v1/endpoints/chat.py tldw_Server_API/app/api/v1/schemas/chat_commands_schemas.py tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py -m "fix(chat): centralize command authorization"
```

Expected: commit succeeds with only the listed files.

---

### Task 8: Repair Document Prompt Versioning

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/document_generator.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_document_generator.py`

- [ ] **Step 1: Add repeated-save regression test**

Append to `TestDocumentGeneratorService` in `test_document_generator.py`:

```python
    def test_save_user_prompt_config_allows_multiple_inactive_versions(self, service, real_db):
        assert service.save_user_prompt_config(
            DocumentType.STUDY_GUIDE,
            "system v1",
            "user v1",
            0.1,
            100,
        )
        assert service.save_user_prompt_config(
            DocumentType.STUDY_GUIDE,
            "system v2",
            "user v2",
            0.2,
            200,
        )
        assert service.save_user_prompt_config(
            DocumentType.STUDY_GUIDE,
            "system v3",
            "user v3",
            0.3,
            300,
        )

        rows = real_db.execute_query(
            "SELECT user_prompt, is_active FROM user_prompts WHERE document_type = ? ORDER BY id",
            (DocumentType.STUDY_GUIDE.value,),
        )
        assert [row["user_prompt"] for row in rows] == ["user v1", "user v2", "user v3"]
        assert sum(1 for row in rows if row["is_active"]) == 1
        assert rows[-1]["is_active"] == 1
```

- [ ] **Step 2: Verify the test fails against the current unique constraint**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_document_generator.py::TestDocumentGeneratorService::test_save_user_prompt_config_allows_multiple_inactive_versions -q
```

Expected: test fails because repeated inactive rows collide.

- [ ] **Step 3: Add schema repair helper**

In `document_generator.py`, add this method to `DocumentGeneratorService`:

```python
    def _repair_user_prompts_schema(self, conn) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'user_prompts'"
        ).fetchone()
        create_sql = row[0] if row else ""
        needs_rebuild = "UNIQUE(document_type, is_active)" in str(create_sql)
        if needs_rebuild:
            conn.execute("""
                CREATE TABLE user_prompts_new (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_type TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    user_prompt TEXT NOT NULL,
                    temperature REAL DEFAULT 0.7,
                    max_tokens INTEGER DEFAULT 2000,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                INSERT INTO user_prompts_new
                (id, document_type, system_prompt, user_prompt, temperature, max_tokens, is_active, created_at, updated_at)
                SELECT id, document_type, system_prompt, user_prompt, temperature, max_tokens, is_active, created_at, updated_at
                FROM user_prompts
            """)
            conn.execute("DROP TABLE user_prompts")
            conn.execute("ALTER TABLE user_prompts_new RENAME TO user_prompts")
        conn.execute("""
            UPDATE user_prompts
            SET is_active = 0
            WHERE is_active = 1
              AND id NOT IN (
                  SELECT MAX(id)
                  FROM user_prompts
                  WHERE is_active = 1
                  GROUP BY document_type
              )
        """)
        conn.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_user_prompts_one_active_per_type
            ON user_prompts(document_type)
            WHERE is_active = 1
        """)
```

- [ ] **Step 4: Use repaired schema during table initialization**

After the `CREATE TABLE IF NOT EXISTS user_prompts` statement, call:

```python
self._repair_user_prompts_schema(conn)
```

Keep the existing `save_user_prompt_config` update-then-insert flow; the partial unique index permits multiple inactive rows.

- [ ] **Step 5: Run document generator tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_document_generator.py -q
```

Expected: document generator tests pass.

- [ ] **Step 6: Commit document prompt repair**

Run:

```bash
git add tldw_Server_API/app/core/Chat/document_generator.py tldw_Server_API/tests/Chat/unit/test_document_generator.py
git commit --only tldw_Server_API/app/core/Chat/document_generator.py tldw_Server_API/tests/Chat/unit/test_document_generator.py -m "fix(chat): allow document prompt versions"
```

Expected: commit succeeds with only the listed files.

---

### Task 9: Make Legacy History Replacement Atomic

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_history.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py`

- [ ] **Step 1: Add transaction recording regression test**

Append to `test_chat_history_multi_image.py`:

```python
class ExistingConversationDB(DummyDB):
    def __init__(self):
        super().__init__()
        self.transaction_id = 0
        self.active_transaction_id = None
        self.soft_delete_transaction_ids = []
        self.add_message_transaction_ids = []

    @contextmanager
    def transaction(self):
        self.transaction_id += 1
        previous = self.active_transaction_id
        self.active_transaction_id = self.transaction_id
        try:
            yield
        finally:
            self.active_transaction_id = previous

    def get_conversation_by_id(self, _conversation_id):
        return {"id": "conv-1", "character_id": 1, "version": 1, "title": "Existing"}

    def get_character_card_by_id(self, _character_id):
        return {"id": 1, "name": DEFAULT_CHARACTER_NAME}

    def get_messages_for_conversation(self, *_args, **_kwargs):
        return [{"id": "old-1", "version": 1}]

    def soft_delete_message(self, *_args, **_kwargs):
        self.soft_delete_transaction_ids.append(self.active_transaction_id)

    def add_message(self, payload):
        self.add_message_transaction_ids.append(self.active_transaction_id)
        return super().add_message(payload)

    def update_conversation(self, *_args, **_kwargs):
        return True


def test_legacy_history_replacement_deletes_and_inserts_in_one_transaction():
    db = ExistingConversationDB()

    conv_id, status = save_chat_history_to_db_wrapper(
        db=db,
        chatbot_history=[{"role": "user", "content": "replacement"}],
        conversation_id="conv-1",
        media_content_for_char_assoc=None,
        media_name_for_char_assoc=None,
        character_name_for_chat=DEFAULT_CHARACTER_NAME,
    )

    assert conv_id == "conv-1"
    assert status == "Chat history saved successfully!"
    assert db.soft_delete_transaction_ids == db.add_message_transaction_ids
```

- [ ] **Step 2: Verify the test fails against separate transactions**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py::test_legacy_history_replacement_deletes_and_inserts_in_one_transaction -q
```

Expected: test fails because deletion and insertion use different transaction ids.

- [ ] **Step 3: Combine existing-conversation replacement into one transaction**

In `save_chat_history_to_db_wrapper`, remove the first transaction that only soft-deletes old messages. Keep validation of character mismatch before any delete, then perform existing-message soft deletes inside the same transaction that inserts replacement messages:

```python
existing_messages_for_replacement: list[dict] = []
if not is_new_conversation:
    existing_conv_details = db.get_conversation_by_id(current_conversation_id)
    if not existing_conv_details:
        logging.error("Cannot resave: Conversation %s not found.", current_conversation_id)
        return current_conversation_id, f"Error: Conversation {current_conversation_id} not found for resaving."
    if existing_conv_details.get("character_id") != associated_character_id:
        existing_char = db.get_character_card_by_id(existing_conv_details.get("character_id"))
        existing_char_name = existing_char.get("name") if existing_char else f"ID {existing_conv_details.get('character_id')}"
        logging.error(
            "Cannot resave: Conversation %s (for char '%s') does not match current character context '%s' (ID: %s).",
            current_conversation_id,
            existing_char_name,
            final_character_name_for_title,
            associated_character_id,
        )
        return current_conversation_id, "Error: Mismatch in character association for resaving chat. The conversation belongs to a different character."
    existing_messages_for_replacement = db.get_messages_for_conversation(
        current_conversation_id,
        limit=10000,
        order_by_timestamp="ASC",
    )
```

Then, immediately after `with db.transaction():`, add:

```python
if existing_messages_for_replacement:
    logging.info(
        "Found %s existing messages to soft-delete for conv %s.",
        len(existing_messages_for_replacement),
        current_conversation_id,
    )
    for msg in existing_messages_for_replacement:
        db.soft_delete_message(msg["id"], msg["version"])
```

- [ ] **Step 4: Run chat history tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py -q
```

Expected: chat history multi-image tests pass.

- [ ] **Step 5: Commit atomic history replacement**

Run:

```bash
git add tldw_Server_API/app/core/Chat/chat_history.py tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py
git commit --only tldw_Server_API/app/core/Chat/chat_history.py tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py -m "fix(chat): make legacy history replacement atomic"
```

Expected: commit succeeds with only the listed files.

---

### Task 10: Extract Streaming Pipeline Without Changing SSE Behavior

**Files:**
- Create: `tldw_Server_API/app/core/Chat/streaming_pipeline.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py`

- [ ] **Step 1: Create streaming pipeline wrapper**

Create `streaming_pipeline.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StreamingPipelineRequest:
    response_generator: Any
    conversation_id: str
    model: str
    provider: str | None
    timeout_seconds: float | None
    finalize_callback: Callable[..., Any] | None
    continuation_metadata: dict[str, Any] | None


def create_chat_streaming_response(
    *,
    request: StreamingPipelineRequest,
    stream_factory: Callable[..., Any],
) -> Any:
    return stream_factory(
        request.response_generator,
        request.conversation_id,
        request.model,
        provider=request.provider,
        timeout_seconds=request.timeout_seconds,
        finalize_callback=request.finalize_callback,
        continuation_metadata=request.continuation_metadata,
    )
```

- [ ] **Step 2: Route streaming construction through the wrapper**

In `chat_service.py`, import:

```python
from tldw_Server_API.app.core.Chat.streaming_pipeline import (
    StreamingPipelineRequest,
    create_chat_streaming_response,
)
```

At the existing `create_streaming_response_with_timeout(...)` call site, replace the direct call with:

```python
return create_chat_streaming_response(
    request=StreamingPipelineRequest(
        response_generator=response_generator,
        conversation_id=final_conversation_id,
        model=model,
        provider=selected_provider,
        timeout_seconds=stream_timeout,
        finalize_callback=finalize_callback,
        continuation_metadata=normalized_continuation_metadata,
    ),
    stream_factory=create_streaming_response_with_timeout,
)
```

Keep `create_streaming_response_with_timeout` imported or assigned in `chat_service.py` so existing tests that monkeypatch `chat_service.create_streaming_response_with_timeout` continue to work.

- [ ] **Step 3: Run streaming tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py -q
```

Expected: streaming utility and fallback tests pass.

- [ ] **Step 4: Commit streaming wrapper**

Run:

```bash
git add tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py
git commit --only tldw_Server_API/app/core/Chat/streaming_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py tldw_Server_API/tests/Chat/unit/test_chat_service_fallback.py -m "refactor(chat): isolate streaming response assembly"
```

Expected: commit succeeds with only the listed files.

---

### Task 11: Add ChatCompletionPipeline Coordinator

**Files:**
- Create: `tldw_Server_API/app/core/Chat/completion_pipeline.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_content.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py`
- Test: `tldw_Server_API/tests/Chat/unit/test_streaming_utils.py`

- [ ] **Step 1: Create coordinator object**

Create `completion_pipeline.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ChatCompletionPipeline:
    non_stream_executor: Callable[..., Any]
    streaming_executor: Callable[..., Any]

    async def execute_non_stream(self, **kwargs: Any) -> dict[str, Any]:
        return await self.non_stream_executor(**kwargs)

    def execute_streaming(self, **kwargs: Any) -> Any:
        return self.streaming_executor(**kwargs)
```

- [ ] **Step 2: Add a default pipeline factory in `chat_service.py`**

Import:

```python
from tldw_Server_API.app.core.Chat.completion_pipeline import ChatCompletionPipeline
```

Add near the public helper section:

```python
def get_chat_completion_pipeline() -> ChatCompletionPipeline:
    return ChatCompletionPipeline(
        non_stream_executor=_execute_non_stream_call_impl,
        streaming_executor=create_chat_streaming_response,
    )
```

Rename the current `execute_non_stream_call` implementation to `_execute_non_stream_call_impl`, then add the compatibility wrapper:

```python
async def execute_non_stream_call(**kwargs: Any) -> dict[str, Any]:
    return await get_chat_completion_pipeline().execute_non_stream(**kwargs)
```

The wrapper keeps the existing import path for tests and callers while making the pipeline object the orchestration entry point.

- [ ] **Step 3: Run core Chat tests touched by the coordinator**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_service_content.py tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py tldw_Server_API/tests/Chat/unit/test_streaming_utils.py -q
```

Expected: all three test files pass.

- [ ] **Step 4: Commit coordinator**

Run:

```bash
git add tldw_Server_API/app/core/Chat/completion_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py
git commit --only tldw_Server_API/app/core/Chat/completion_pipeline.py tldw_Server_API/app/core/Chat/chat_service.py -m "refactor(chat): add completion pipeline coordinator"
```

Expected: commit succeeds with only the listed files.

---

### Task 12: Update Chat Architecture Documentation

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/README.md`
- Modify: `tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md`
- Modify: `backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md`

- [ ] **Step 1: Document service ownership in Chat README**

Add a section:

```markdown
## Completion Pipeline Ownership

`chat_service.py` remains the compatibility facade for existing imports and endpoint wiring. New behavior should be added behind `ChatCompletionPipeline` or one of its focused services:

- `response_processor.py`: non-streaming choice extraction, response content mutation, structured validation, response usage estimates, and assistant-name injection.
- `moderation_pipeline.py`: output self-monitoring, moderation, topic-monitoring scheduling, and moderation audit/review records.
- `persistence_service.py`: assistant and tool-message persistence helpers.
- `tool_execution_service.py`: local tool auto-execution eligibility checks and execution orchestration.
- `streaming_pipeline.py`: streaming response assembly without changing SSE event shapes.
- `command_authorization.py`: slash-command authorization decisions for dispatch and command listing.
- `chat_logging.py`: safe summaries for prompt, content, tool, and exception logs.
```

- [ ] **Step 2: Update refactoring plan status**

Add:

```markdown
## 2026-06-24 Integrated Completion Pipeline Refactor

Validated review findings fixed:

- Non-streaming response safety now processes every returned choice.
- Local tool auto-execution rejects multi-choice requests before provider calls.
- Sensitive Chat logs use metadata summaries instead of raw prompt, message, tool, or assistant content.
- Document prompt versions allow repeated inactive versions while enforcing one active prompt per document type.
- Slash-command dispatch and user-visible listing use one fail-closed authorization decision.
- Legacy history replacement soft-deletes old messages and inserts replacements in one transaction.

The public Chat completion API remains compatible except for the intentional safety rejection of multi-choice local tool auto-execution.
```

- [ ] **Step 3: Update Backlog task notes and modified files**

Use Backlog MCP:

```text
task_edit TASK-12013
status: In Progress
append_notes: Implementation plan saved at Docs/superpowers/plans/2026-06-24-chat-completion-pipeline-refactor.md. Refactor execution uses focused service extraction with TDD checkpoints and path-limited commits.
modified_files: include the new Chat service modules, touched Chat tests, Chat docs, and endpoint command listing.
```

- [ ] **Step 4: Commit docs and task update**

Run:

```bash
git add tldw_Server_API/app/core/Chat/README.md tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md "backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md"
git commit --only tldw_Server_API/app/core/Chat/README.md tldw_Server_API/app/core/Chat/REFACTORING_PLAN.md "backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md" -m "docs(chat): document completion pipeline split"
```

Expected: commit succeeds with only the listed files.

---

### Task 13: Final Verification and Security Scan

**Files:**
- Modify: `backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md`

- [ ] **Step 1: Run targeted Chat unit tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_chat_service_content.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_system_messages.py \
  tldw_Server_API/tests/Chat/unit/test_streaming_utils.py \
  tldw_Server_API/tests/Chat/unit/test_document_generator.py \
  tldw_Server_API/tests/Chat/unit/test_chat_history_multi_image.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_command_router.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_chat_commands_endpoint.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run a wider Chat regression slice**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat tldw_Server_API/tests/Chat_NEW -q
```

Expected: Chat and Chat_NEW tests pass. If a pre-existing unrelated failure appears, record the exact failing test name and reason in `TASK-12013`.

- [ ] **Step 3: Run Bandit on touched Chat scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Chat \
  tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/app/api/v1/schemas/chat_commands_schemas.py \
  -f json -o /tmp/bandit_chat_completion_pipeline.json
```

Expected: command exits 0 or reports only known non-new findings. New findings in touched code are fixed before finalization.

- [ ] **Step 4: Record verification in Backlog**

Use Backlog MCP to append notes like:

```text
Verification:
- Targeted Chat unit tests: PASS
- Chat/Chat_NEW regression slice: PASS
- Bandit touched Chat scope: PASS, report /tmp/bandit_chat_completion_pipeline.json
```

If a command cannot run because the environment is missing a dependency, record the command, the environment error, and the narrower verification that did run.

- [ ] **Step 5: Final commit for task metadata**

Run:

```bash
git add "backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md"
git commit --only "backlog/tasks/task-12013 - Refactor-Chat-completion-pipeline-and-fix-validated-review-findings.md" -m "chore(chat): record pipeline refactor verification"
```

Expected: commit succeeds with only the task file.

## Self-Review Notes

- Spec coverage: every validated finding maps to a task: multi-choice safety in Tasks 1-3, tool multi-choice rejection in Task 4, sensitive logging in Task 6, command authorization in Task 7, document prompt versioning in Task 8, legacy history replacement in Task 9, streaming and coordinator split in Tasks 10-11, docs and verification in Tasks 12-13.
- Type consistency: `NonStreamChoice`, `OutputModerationRuntime`, `StreamingPipelineRequest`, `CommandAuthorizationContext`, and `ChatCompletionPipeline` are defined before later tasks reference them.
- Compatibility: `chat_service.execute_non_stream_call`, `_extract_text_from_content`, `_apply_redaction_to_content`, and `create_streaming_response_with_timeout` remain available for existing tests and callers.
- Safety order: moderation and self-monitoring mutate or block choices before structured validation and persistence.
- Commit hygiene: every commit uses explicit file paths and `git commit --only` because the workspace may contain unrelated changes.
