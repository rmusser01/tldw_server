import json

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
    FINGERPRINT_VERSION,
    build_prompt_cost_envelope,
    canonicalize_messages,
    estimate_segment_tokens,
    fingerprint_text,
)


def test_canonicalize_messages_is_stable_for_mapping_key_order() -> None:
    first = [{"role": "user", "content": {"b": 2, "a": 1}}]
    second = [{"content": {"a": 1, "b": 2}, "role": "user"}]

    assert canonicalize_messages(first) == canonicalize_messages(second)
    assert fingerprint_text(canonicalize_messages(first)) == fingerprint_text(
        canonicalize_messages(second)
    )


def test_message_order_changes_aggregate_fingerprint() -> None:
    first = build_prompt_cost_envelope(
        [
            {"role": "system", "content": "stable rules"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ]
    )
    second = build_prompt_cost_envelope(
        [
            {"role": "system", "content": "stable rules"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "first"},
        ]
    )

    assert first.aggregate_fingerprint != second.aggregate_fingerprint


def test_unknown_content_parts_are_represented_by_bounded_markers() -> None:
    large_unknown_payload = "opaque-" + ("z" * 5000)

    canonical = canonicalize_messages(
        [
            {
                "role": "user",
                "content": [
                    {
                        "text": "known shape missing type",
                        "opaque_payload": large_unknown_payload,
                    },
                ],
            }
        ]
    )

    assert large_unknown_payload not in canonical
    assert "opaque-" not in canonical
    assert "unsupported_part" in canonical
    assert "opaque_payload" in canonical


def test_data_uri_sanitization_stops_at_whitespace() -> None:
    canonical = canonicalize_messages(
        [
            {
                "role": "user",
                "content": "inspect data:image/png;base64,abc123 trailing text",
            }
        ]
    )

    assert "data:image/png;base64,<omitted> trailing text" in canonical
    assert "abc123" not in canonical


def test_build_prompt_cost_envelope_separates_expected_segments() -> None:
    envelope = build_prompt_cost_envelope(
        [
            {"role": "system", "content": "stable rules"},
            {"role": "user", "content": "prior question"},
            {"role": "assistant", "content": "prior answer"},
            {"role": "tool", "content": "retrieved citation"},
            {"role": "user", "content": "current question"},
        ],
        world_book_text="World info: clock tower is haunted",
        retrieval_text="Document chunk: clock tower history",
    )

    assert envelope.fingerprint_version == FINGERPRINT_VERSION
    assert envelope.message_count == 5
    assert envelope.total_estimated_tokens == sum(segment.estimated_tokens for segment in envelope.segments)
    assert envelope.segment_token_totals["static"] > 0
    assert envelope.segment_token_totals["history"] > 0
    assert envelope.segment_token_totals["user_turn"] > 0
    assert envelope.segment_token_totals["world_book"] > 0
    assert envelope.segment_token_totals["retrieval_tool"] > 0


def test_envelope_diagnostics_are_bounded_and_do_not_include_prompt_text() -> None:
    large_secret = "secret-" + ("x" * 5000)

    envelope = build_prompt_cost_envelope(
        [{"role": "user", "content": large_secret}],
        world_book_text="World info: " + large_secret,
    )
    diagnostics = envelope.to_diagnostics()
    diagnostics_repr = repr(diagnostics)

    assert large_secret not in diagnostics_repr
    assert "secret-" not in diagnostics_repr
    assert diagnostics["aggregate_fingerprint"].startswith(f"{FINGERPRINT_VERSION}:sha256:")
    assert diagnostics["segments"][0]["fingerprint"].startswith(f"{FINGERPRINT_VERSION}:sha256:")
    assert diagnostics["segments"][0]["text_length"] == len(large_secret)


def test_token_estimate_is_conservative_deterministic_and_non_negative() -> None:
    assert estimate_segment_tokens("") == 0
    assert estimate_segment_tokens("abcd") == 1
    assert estimate_segment_tokens("abcde") == 2
    assert estimate_segment_tokens("same text") == estimate_segment_tokens("same text")


def test_redaction_keeps_positions_when_lowercase_changes_length() -> None:
    # "İ".lower() is two characters. A scanner that finds "data:image" in text.lower() and
    # slices the original text with that index lands past the real comma: it keeps the
    # payload and puts "<omitted>" after the next comma instead.
    text = "İ" * 30 + " data:image/png;base64," + "A" * 400 + " tail, end"
    redacted = "İ" * 30 + " data:image/png;base64,<omitted> tail, end"

    canonical = json.loads(canonicalize_messages([{"role": "user", "content": text}]))
    envelope = build_prompt_cost_envelope([{"role": "user", "content": text}])

    assert canonical[0]["content"] == redacted
    assert envelope.total_estimated_tokens == (len(redacted) + 3) // 4


def test_fingerprints_of_ordinary_base64_images_do_not_change() -> None:
    # Pinned from origin/dev before TASK-13843: a well-formed base64 data URI redacts to the
    # same text under the old scanner and the shared pattern, so stored or echoed
    # fingerprints of ordinary prompts stay valid. Change this value only with a
    # FINGERPRINT_VERSION bump.
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo" + "A" * 400 + "=="}},
            ],
        },
        {"role": "assistant", "content": "It is a square."},
        {"role": "user", "content": "And this one? data:image/jpeg;base64,/9j/4AAQSkZJRg" + "B" * 400},
    ]
    envelope = build_prompt_cost_envelope(
        messages,
        world_book_text="World info: data:image/gif;base64,R0lGOD" + "C" * 100,
    )

    assert envelope.aggregate_fingerprint == (
        "prompt-v1:sha256:b21e1c8ad7bb61b6a4d7347b110559fa337f6a911c990fef890f368a06bd52ce"
    )
    assert envelope.total_estimated_tokens == 36
