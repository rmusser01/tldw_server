"""The Messages API native path records one llm_usage_log row per call (TASK-13610, LLM usage accounting spec §3)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.endpoints import messages as messages_endpoint
from tldw_Server_API.app.api.v1.schemas.anthropic_messages import AnthropicMessagesRequest
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.Usage import llm_accounting, usage_tracker
from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog

pytestmark = pytest.mark.unit

NATIVE_RESPONSE = {
    "id": "msg-native-1",
    "type": "message",
    "role": "assistant",
    "model": "claude-native",
    "content": [{"type": "text", "text": "valid native output"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 21, "output_tokens": 8},
}
# Anthropic's input_tokens is only the uncached remainder; the prompt is the sum of the three input fields.
CACHED_USAGE = {"input_tokens": 3, "cache_creation_input_tokens": 2000, "cache_read_input_tokens": 60000}


def _frames(start_usage: dict[str, Any], delta_usage: dict[str, Any]) -> list[str]:
    """An Anthropic Messages SSE reply whose message_start and message_delta carry these usage blocks."""
    start = {
        "type": "message_start",
        "message": {
            "id": "msg-s",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": start_usage,
        },
    }
    delta = {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": delta_usage}
    return [
        f"event: message_start\ndata: {json.dumps(start, separators=(',', ':'))}\n\n",
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"ok"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        f"event: message_delta\ndata: {json.dumps(delta, separators=(',', ':'))}\n\n",
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]


STREAM_FRAMES = _frames({"input_tokens": 17, "output_tokens": 1}, {"output_tokens": 9})


class _Runtime:
    """A credential runtime that resolves a key and records nothing."""

    def __init__(self, **_kwargs: Any) -> None:
        """Accept the endpoint's scope arguments."""

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        """Return resolved credentials for the provider."""
        del model
        return SimpleNamespace(provider=provider, api_key="k", app_config={}, credentials_resolved=True)

    async def mark_used(self, _handle: Any) -> None:
        """Credential touches are not under test."""

    async def close(self) -> None:
        """Nothing to close."""


def _stub_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the test's loop the accounting loop, and stub the endpoint's credentials and base URL."""
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    monkeypatch.setattr(messages_endpoint, "derive_trusted_credential_scope", lambda _r, user: (user.id_int, [], [], False))
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _Runtime)
    monkeypatch.setattr(messages_endpoint, "_resolve_messages_base_url", lambda _p, _c: "http://provider.test/v1")


@pytest.fixture
async def rows(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture what would be written, and make the test's loop the accounting loop."""
    captured: list[dict[str, Any]] = []

    async def _fake_log(**kwargs: Any) -> None:
        """Record the row instead of writing it."""
        captured.append(kwargs)

    monkeypatch.setattr("tldw_Server_API.app.core.Usage.usage_tracker.log_llm_usage", _fake_log)
    _stub_endpoint(monkeypatch)
    return captured


@pytest.fixture
async def costed_rows(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Capture the llm_usage_log rows the real recorder would insert, costed by the shipped pricing catalog."""
    captured: list[dict[str, Any]] = []

    class _Repo:
        """Stands in for AuthnzUsageRepo and keeps the inserted rows."""

        def __init__(self, _pool: Any) -> None:
            """No database."""

        async def insert_llm_usage_log(self, **kwargs: Any) -> None:
            """Record the row instead of writing it."""
            captured.append(kwargs)

    async def _none() -> None:
        """No pool and no daily ledger."""

    monkeypatch.setenv("LLM_USAGE_ENABLED", "true")
    monkeypatch.delenv("PRICING_OVERRIDES", raising=False)
    reset_pricing_catalog()
    monkeypatch.setattr(usage_tracker, "get_db_pool", _none)
    monkeypatch.setattr(usage_tracker, "_get_tokens_daily_ledger", _none)
    monkeypatch.setattr(usage_tracker, "AuthnzUsageRepo", _Repo)
    _stub_endpoint(monkeypatch)
    yield captured
    reset_pricing_catalog()


async def _call(model: str, *, stream: bool) -> Any:
    """Run the Messages handler as an authenticated request for user 5 would."""
    return await messages_endpoint._handle_messages(
        AnthropicMessagesRequest(model=model, messages=[{"role": "user", "content": "hello"}], stream=stream),
        current_user=SimpleNamespace(id_int=5),
        request=SimpleNamespace(),
        anthropic_version=None,
        anthropic_beta=None,
    )


def _native_reply(usage: dict[str, Any]) -> Any:
    """A stand-in for _native_post_json whose JSON reply carries ``usage``."""

    async def _post(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """The provider's JSON reply."""
        return {**NATIVE_RESPONSE, "usage": usage}

    return _post


def _native_stream(frames: list[str]) -> Any:
    """A stand-in for _prepare_native_stream_iterator whose SSE reply is ``frames``."""

    async def _prepare(*_args: Any, **_kwargs: Any) -> Any:
        """The provider's SSE reply, split mid-frame the way a socket splits it."""

        async def _source() -> Any:
            """Yield the frames as raw bytes, the first one in two pieces."""
            first = frames[0].encode()
            yield first[:30]
            yield first[30:]
            for frame in frames[1:]:
                yield frame.encode()

        return _source()

    return _prepare


async def _stream_row(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, Any]], frames: list[str]) -> dict[str, Any]:
    """Stream ``frames`` to the client as user 5 and return the one row written."""
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(frames))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-3-5-haiku-latest", stream=True)
        body = "".join([item async for item in response.body_iterator])
    assert body == "".join(frames)
    await llm_accounting.drain_accounting()
    (row,) = rows
    return row


@pytest.mark.parametrize("model", ["anthropic/claude-3-5-haiku-latest", "llama.cpp/local-model"])
async def test_a_native_reply_records_one_row_from_its_usage_block(rows, monkeypatch, model) -> None:
    """Non-streaming: one row for the scope user, with the provider's own token counts."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
    with scoped_context(user_id=5):
        response = await _call(model, stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = rows
    provider, _, model_name = model.partition("/")
    assert (row["user_id"], row["provider"], row["model"]) == (5, provider, model_name)
    assert (row["prompt_tokens"], row["completion_tokens"]) == (21, 8)
    assert row["usage_metadata"] == NATIVE_RESPONSE["usage"]
    assert row["estimated"] is None


async def test_a_native_stream_records_one_row_with_input_and_output_tokens(rows, monkeypatch) -> None:
    """Streaming: input tokens from message_start and the final output count from message_delta."""
    row = await _stream_row(monkeypatch, rows, STREAM_FRAMES)
    assert (row["user_id"], row["provider"], row["model"]) == (5, "anthropic", "claude-3-5-haiku-latest")
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 9)


@pytest.mark.parametrize(
    "delta_usage",
    [{"input_tokens": None, "cache_read_input_tokens": None, "output_tokens": 9}, {"input_tokens": 17, "output_tokens": 9}],
    ids=["null-fields", "cumulative"],
)
async def test_message_delta_usage_keeps_the_input_count_from_message_start(rows, monkeypatch, delta_usage) -> None:
    """A null usage field in message_delta keeps the count already read; a cumulative one restates it."""
    row = await _stream_row(monkeypatch, rows, _frames({"input_tokens": 17, "output_tokens": 1}, delta_usage))
    assert (row["prompt_tokens"], row["completion_tokens"]) == (17, 9)


@pytest.mark.parametrize(
    ("model", "usage", "prompt_tokens"),
    [
        ("anthropic/claude-sonnet-4-5", {**CACHED_USAGE, "output_tokens": 100}, 62003),
        # llama.cpp sends input_tokens = n_prompt_tokens - n_prompt_tokens_cache and no cache writes.
        ("llama.cpp/local-model", {"cache_read_input_tokens": 60000, "input_tokens": 3, "output_tokens": 100}, 60003),
    ],
    ids=["anthropic", "llama.cpp"],
)
async def test_a_cached_native_reply_counts_its_cached_input_as_prompt_tokens(
    rows, monkeypatch, model, usage, prompt_tokens
) -> None:
    """input_tokens is only the uncached remainder, so cache reads and writes count as prompt tokens too."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call(model, stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["prompt_tokens"], row["completion_tokens"]) == (prompt_tokens, 100)
    assert row["usage_metadata"] == usage


async def test_a_cached_native_stream_counts_its_cached_input_as_prompt_tokens(rows, monkeypatch) -> None:
    """The same for a stream: message_start's cache fields count as prompt tokens."""
    row = await _stream_row(
        monkeypatch, rows, _frames({**CACHED_USAGE, "output_tokens": 1}, {"output_tokens": 100})
    )
    assert (row["prompt_tokens"], row["completion_tokens"]) == (62003, 100)
    assert row["usage_metadata"]["cache_read_input_tokens"] == 60000


async def test_a_cached_native_reply_is_costed_at_the_cache_rates(costed_rows, monkeypatch) -> None:
    """Cache reads and writes cost Anthropic's cache rates, not the input rate (TASK-13858)."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply({**CACHED_USAGE, "output_tokens": 100}))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-sonnet-4-6", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    assert (row["prompt_tokens"], row["cache_read_input_tokens"], row["cache_write_input_tokens"]) == (62003, 60000, 2000)
    assert row["billable_input_tokens"] == 3
    # Sonnet 4.6 per 1K tokens: input $0.003, cache read $0.0003 (0.1x), 5-minute cache write $0.00375 (1.25x).
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.003 + 60000 * 0.0003 + 2000 * 0.00375) / 1000)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.015 / 1000)
    assert row["estimated"] is False


async def test_a_native_stream_closed_early_still_records_one_row(rows, monkeypatch) -> None:
    """A client that disconnects after the first frame still costs the input tokens it was sent."""
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(STREAM_FRAMES))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-3-5-haiku-latest", stream=True)
        iterator = response.body_iterator
        assert await iterator.__anext__() == STREAM_FRAMES[0]
        await iterator.aclose()
    await llm_accounting.drain_accounting()
    (row,) = rows
    assert (row["user_id"], row["prompt_tokens"]) == (5, 17)


async def test_an_invalid_native_reply_records_nothing(rows, monkeypatch) -> None:
    """A 200 that is not a valid message is reported as a 502 and, like a failed adapter call, not recorded."""

    async def _native_bad(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """A reply with no content."""
        return {**NATIVE_RESPONSE, "content": []}

    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_bad)
    with scoped_context(user_id=5), pytest.raises(Exception) as raised:
        await _call("anthropic/claude-3-5-haiku-latest", stream=False)
    assert getattr(raised.value, "status_code", None) == 502
    await llm_accounting.drain_accounting()
    assert rows == []


async def test_with_no_user_a_native_reply_records_nothing(rows, monkeypatch) -> None:
    """No content scope and no explicit user: the call runs and nothing is written."""
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
    response = await _call("anthropic/claude-3-5-haiku-latest", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    assert rows == []


async def test_a_one_hour_cache_write_is_costed_at_twice_the_input_price(costed_rows, monkeypatch) -> None:
    """Native reply: the 1-hour share of cache writes costs 2x input, the rest 1.25x (TASK-13878)."""
    usage = {
        "input_tokens": 3,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
        "output_tokens": 100,
    }
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-sonnet-4-6", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    # Sonnet 4.6 per 1K: input $0.003, 5-minute write $0.00375, 1-hour write $0.006.
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.003 + 1000 * 0.00375 + 2000 * 0.006) / 1000)


async def test_a_streamed_one_hour_cache_write_is_costed_at_twice_the_input_price(costed_rows, monkeypatch) -> None:
    """message_start carries the split; message_delta carries only output tokens, and the merge keeps the split."""
    start = {
        "input_tokens": 3,
        "cache_creation_input_tokens": 3000,
        "cache_creation": {"ephemeral_5m_input_tokens": 1000, "ephemeral_1h_input_tokens": 2000},
        "output_tokens": 1,
    }
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(_frames(start, {"output_tokens": 100})))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-sonnet-4-6", stream=True)
        [item async for item in response.body_iterator]
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.003 + 1000 * 0.00375 + 2000 * 0.006) / 1000)


@pytest.mark.parametrize(
    ("extra", "factor"),
    [({"speed": "fast"}, 2.0), ({"inference_geo": "us"}, 1.1), ({"speed": "fast", "inference_geo": "us"}, 2.2), ({"speed": "standard", "inference_geo": "global"}, 1.0)],
)
async def test_a_native_reply_is_costed_with_its_fast_mode_and_us_only_multipliers(costed_rows, monkeypatch, extra, factor) -> None:
    """The usage block reports speed and inference_geo; the row is costed with them (TASK-13885)."""
    usage = {**CACHED_USAGE, "output_tokens": 100, **extra}
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-opus-5", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    # Opus 5 per 1K: input $0.005, cache read $0.0005, 5-minute write $0.00625, output $0.025.
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.005 + 60000 * 0.0005 + 2000 * 0.00625) / 1000 * factor)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.025 / 1000 * factor)


@pytest.mark.parametrize(("service_tier", "factor"), [("batch", 0.5), ("standard", 1.0), ("priority", 1.0)])
async def test_a_native_reply_is_costed_at_the_service_tier_its_usage_reports(
    costed_rows, monkeypatch, service_tier, factor
) -> None:
    """Batch is 50% off every token category; Priority Tier has no token price of its own (TASK-13936)."""
    usage = {**CACHED_USAGE, "output_tokens": 100, "service_tier": service_tier}
    monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(usage))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-opus-5", stream=False)
    assert response.status_code == 200
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    # Opus 5 per 1K: input $0.005, cache read $0.0005, 5-minute write $0.00625, output $0.025.
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.005 + 60000 * 0.0005 + 2000 * 0.00625) / 1000 * factor)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.025 / 1000 * factor)


async def test_a_streamed_batch_reply_is_costed_at_half_price(costed_rows, monkeypatch) -> None:
    """message_start carries service_tier; message_delta does not, and the merge keeps it."""
    start = {**CACHED_USAGE, "output_tokens": 1, "service_tier": "batch"}
    frames = _frames(start, {"output_tokens": 100})
    monkeypatch.setattr(messages_endpoint, "_prepare_native_stream_iterator", _native_stream(frames))
    with scoped_context(user_id=5):
        response = await _call("anthropic/claude-opus-5", stream=True)
        [item async for item in response.body_iterator]
    await llm_accounting.drain_accounting()
    (row,) = costed_rows
    assert row["prompt_cost_usd"] == pytest.approx((3 * 0.005 + 60000 * 0.0005 + 2000 * 0.00625) / 1000 * 0.5)
    assert row["completion_cost_usd"] == pytest.approx(100 * 0.025 / 1000 * 0.5)


class _UserKeyRuntime(_Runtime):
    """Resolves the user's own key."""

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        """Credentials whose source is the user's own key."""
        del model
        return SimpleNamespace(
            provider=provider, api_key="k", app_config={}, credentials_resolved=True, credential_source="user"
        )


@pytest.mark.parametrize("stream", [False, True])
async def test_a_native_call_on_the_users_own_key_is_recorded_as_such(rows, monkeypatch, stream) -> None:
    """The row says the user's own key paid, so the LLM quota counters leave it out (TASK-13955)."""
    monkeypatch.setattr(messages_endpoint, "ProviderCredentialRuntime", _UserKeyRuntime)
    if stream:
        row = await _stream_row(monkeypatch, rows, STREAM_FRAMES)
    else:
        monkeypatch.setattr(messages_endpoint, "_native_post_json", _native_reply(NATIVE_RESPONSE["usage"]))
        with scoped_context(user_id=5):
            assert (await _call("anthropic/claude-3-5-haiku-latest", stream=False)).status_code == 200
        await llm_accounting.drain_accounting()
        (row,) = rows
    assert row["credential_source"] == "user"
