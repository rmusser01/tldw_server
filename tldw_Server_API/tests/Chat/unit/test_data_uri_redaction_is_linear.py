"""Data-URI redaction stays linear on crafted input (TASK-13560).

An unbounded ``(data:image[^,]*,)`` rescans to the end of the text for every
``data:image`` with no comma after it, so a 500 KB body of such markers took
seconds to redact. Each chat call site must handle it in well under two seconds.
The call sites now share one pattern, so the endpoint's estimate also redacts a
payload outside a "url" key and the size validator an uppercase data URI.
"""

import time

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.api.v1.schemas import chat_validators
from tldw_Server_API.app.core.Chat import chat_service

# ~510 KB of markers with no comma anywhere: every one of them would rescan to the end.
_CRAFTED = '"url":"data:image' * 30_000


def _assert_fast(fn) -> None:
    started = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - started
    assert elapsed < 2.0, f"redaction took {elapsed:.1f}s on {len(_CRAFTED)} chars"


@pytest.mark.unit
def test_chat_service_token_estimate_is_linear():
    _assert_fast(lambda: chat_service.estimate_tokens_from_json(_CRAFTED))


@pytest.mark.unit
def test_chat_endpoint_rate_limit_sanitizer_is_linear():
    _assert_fast(lambda: chat_endpoint._sanitize_json_for_rate_limit(_CRAFTED))


@pytest.mark.unit
def test_request_size_validator_is_linear(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 10_000_000)
    _assert_fast(lambda: chat_validators.validate_request_size(_CRAFTED))


@pytest.mark.unit
@pytest.mark.parametrize(
    "redact",
    [
        chat_service._sanitize_data_uris,
        chat_endpoint._sanitize_json_for_rate_limit,
    ],
)
def test_ordinary_data_uri_payload_is_still_redacted(redact):
    body = '{"url": "data:image/png;base64,' + "A" * 400 + '"}'
    redacted = redact(body)
    assert "A" * 400 not in redacted
    assert "data:image/png;base64," in redacted


@pytest.mark.unit
def test_real_data_uri_after_crafted_markers_is_still_redacted():
    body = _CRAFTED + '"url":"data:image/png;base64,' + "A" * 400 + '"'
    assert "A" * 400 not in chat_service._sanitize_data_uris(body)


@pytest.mark.unit
def test_endpoint_queue_estimate_ignores_a_payload_outside_a_url_key():
    body = '{"messages":[{"role":"user","content":"data:image/png;base64,' + "A" * 4000 + '"}]}'
    assert "A" * 4000 not in chat_endpoint._sanitize_json_for_rate_limit(body)
    assert chat_endpoint._estimate_tokens_for_queue(body) < 100


@pytest.mark.unit
def test_request_size_validator_redacts_an_uppercase_data_uri(monkeypatch):
    monkeypatch.setattr(chat_validators, "MAX_REQUEST_SIZE", 1000)
    body = '{"url": "DATA:IMAGE/PNG;BASE64,' + "A" * 5000 + '"}'
    assert chat_validators.validate_request_size(body) is True


@pytest.mark.unit
def test_media_type_over_128_chars_is_counted_not_redacted():
    # The bound that keeps redaction linear: such a URI is left whole, so it is over-counted.
    body = "data:image/" + "x" * 200 + ",payload"
    assert chat_service._sanitize_data_uris(body) == body
