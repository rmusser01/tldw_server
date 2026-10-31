"""Data-URI redaction stays linear on crafted input (TASK-13560).

An unbounded ``(data:image[^,]*,)`` rescans to the end of the text for every
``data:image`` with no comma after it, so a 500 KB body of such markers took
seconds to redact. Each chat call site must handle it in well under two seconds.
The call sites now share one pattern, so the endpoint's estimate also redacts a
payload outside a "url" key and the size validator an uppercase data URI.
Since TASK-13843 the chat path has one redactor (core.Utils.tokenizer.redact_data_uris)
and one estimate (estimate_segment_tokens) behind every prompt-size estimate.
"""

import json
import time

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_validators
from tldw_Server_API.app.core.Chat import chat_service, prompt_cost_envelope

# ~510 KB of markers with no comma anywhere: every one of them would rescan to the end.
_CRAFTED = '"url":"data:image' * 30_000
# Two shapes: the "url" one engages the endpoint's old "url"-anchored pattern; the quote-free one
# is a user's text inside one JSON string (its quotes are escaped), so a pattern that stops at a
# quote but drops the 128-character bound is still caught.
# The third is the worst case for the base64-header pattern: every "data:image/" makes the
# bounded header scan 128 characters and back off looking for ";base64,".
_CRAFTED_SHAPES = pytest.mark.parametrize(
    "body",
    [_CRAFTED, "data:image" * 51_000, "data:image/" * 46_000],
    ids=["url-key", "quote-free", "slash"],
)


def _assert_fast(fn, body: str) -> None:
    started = time.perf_counter()
    fn(body)
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"redaction took {elapsed:.1f}s on {len(body)} chars"


def _best_time(fn, arg) -> float:
    times = []
    for _ in range(3):
        started = time.perf_counter()
        fn(arg)
        times.append(time.perf_counter() - started)
    return min(times)


@pytest.mark.unit
@_CRAFTED_SHAPES
def test_chat_service_token_estimate_is_linear(body):
    _assert_fast(chat_service.estimate_tokens_from_json, body)


@pytest.mark.unit
@_CRAFTED_SHAPES
def test_prompt_cost_estimate_is_linear(body):
    _assert_fast(prompt_cost_envelope.estimate_segment_tokens, body)


@pytest.mark.unit
@_CRAFTED_SHAPES
def test_request_size_validator_is_linear(monkeypatch, body):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 10_000_000)
    _assert_fast(chat_validators.validate_request_size, body)


@pytest.mark.unit
def test_ordinary_data_uri_payload_is_still_redacted():
    body = '{"url": "data:image/png;base64,' + "A" * 400 + '"}'
    redacted = prompt_cost_envelope._sanitize_data_uris(body)
    assert "A" * 400 not in redacted
    assert "data:image/png;base64," in redacted


@pytest.mark.unit
def test_real_data_uri_after_crafted_markers_is_still_redacted():
    body = _CRAFTED + '"url":"data:image/png;base64,' + "A" * 400 + '"'
    assert "A" * 400 not in prompt_cost_envelope._sanitize_data_uris(body)


@pytest.mark.unit
def test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key():
    body = '{"messages":[{"role":"user","content":"data:image/png;base64,' + "A" * 4000 + '"}]}'
    assert chat_endpoint.estimate_tokens_from_json(body) < 100


@pytest.mark.unit
def test_request_size_validator_redacts_an_uppercase_data_uri(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 1000)
    body = '{"url": "DATA:IMAGE/PNG;BASE64,' + "A" * 5000 + '"}'
    assert chat_validators.validate_request_size(body) is True


@pytest.mark.unit
def test_media_type_over_128_chars_is_counted_not_redacted():
    # The bound that keeps redaction linear: such a URI is left whole, so it is over-counted.
    body = "data:image/" + "x" * 200 + ";base64,AAAA"
    assert prompt_cost_envelope._sanitize_data_uris(body) == body


@pytest.mark.unit
def test_image_bulk_redacts_within_four_times_the_json_dumps_time():
    # Ten 4 MB base64 images, about 42 MB of request JSON: the usual large body, which every
    # estimate site redacts on the event loop. Timing json.dumps of the same request in the same
    # run keeps the bound independent of machine load. Measured ratios: dev's pattern 2.0-2.5x,
    # a non-possessive base64 payload alternation 14-44x, the possessive one 0.3-0.6x.
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * (4 * 1024 * 1024)}}
    request = {"messages": [{"role": "user", "content": [image] * 10}]}
    body = json.dumps(request)

    build = _best_time(json.dumps, request)
    redact = _best_time(chat_service.estimate_tokens_from_json, body)

    assert redact <= 4 * build, f"redaction took {redact:.3f}s, json.dumps {build:.3f}s"
