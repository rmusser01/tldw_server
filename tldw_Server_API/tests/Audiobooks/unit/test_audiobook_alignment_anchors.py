import pytest

from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import AlignmentPayload, AlignmentWord
from tldw_Server_API.app.core.Audiobooks.alignment_utils import (
    AlignmentAnchor,
    apply_alignment_anchors,
    scale_alignment_payload,
    stitch_alignment_payloads,
)

pytestmark = pytest.mark.unit


def test_apply_alignment_anchors_shifts_target_block():
    words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=400, char_start=0, char_end=5),
        AlignmentWord(word="world", start_ms=450, end_ms=900, char_start=6, char_end=11),
        AlignmentWord(word="again", start_ms=1000, end_ms=1400, char_start=12, char_end=17),
    ]
    anchors = [AlignmentAnchor(offset=6, time_ms=5000)]

    adjusted = apply_alignment_anchors(words, anchors)

    assert adjusted[0].start_ms == 0
    assert adjusted[0].end_ms == 400
    assert adjusted[1].start_ms == 5000
    assert adjusted[1].end_ms == 5450
    assert adjusted[2].start_ms == 5550
    assert adjusted[2].end_ms == 5950


def test_apply_alignment_anchors_ignores_anchor_before_adjusted_previous_end():
    words = [
        AlignmentWord(word="one", start_ms=0, end_ms=100, char_start=0, char_end=3),
        AlignmentWord(word="two", start_ms=200, end_ms=300, char_start=4, char_end=7),
        AlignmentWord(word="three", start_ms=400, end_ms=500, char_start=8, char_end=13),
    ]
    anchors = [
        AlignmentAnchor(offset=4, time_ms=1000),
        AlignmentAnchor(offset=8, time_ms=350),
    ]

    adjusted = apply_alignment_anchors(words, anchors)

    assert [(word.start_ms, word.end_ms) for word in adjusted] == [
        (0, 100),
        (1000, 1100),
        (1200, 1300),
    ]


def test_scale_alignment_payload_scales_timestamps():
    payload_words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=400, char_start=0, char_end=5),
        AlignmentWord(word="world", start_ms=450, end_ms=900, char_start=6, char_end=11),
    ]
    payload = AlignmentPayload(engine="kokoro", sample_rate=24000, words=payload_words)
    scaled = scale_alignment_payload(payload, 2.0)

    assert scaled.words[0].start_ms == 0
    assert scaled.words[0].end_ms == 200
    assert scaled.words[1].start_ms == 225
    assert scaled.words[1].end_ms == 450


def test_stitch_alignment_payloads_offsets_segments():
    first_words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=400, char_start=0, char_end=5),
        AlignmentWord(word="world", start_ms=450, end_ms=900, char_start=6, char_end=11),
    ]
    second_words = [
        AlignmentWord(word="Again", start_ms=0, end_ms=300, char_start=0, char_end=5),
    ]
    payload1 = AlignmentPayload(engine="kokoro", sample_rate=24000, words=first_words)
    payload2 = AlignmentPayload(engine="kokoro", sample_rate=24000, words=second_words)

    stitched = stitch_alignment_payloads(
        [payload1, payload2],
        segment_offsets_ms=[0, 900],
        segment_offsets_chars=[0, 12],
    )

    assert len(stitched.words) == 3
    assert stitched.words[2].start_ms == 900
    assert stitched.words[2].end_ms == 1200
    assert stitched.words[2].char_start == 12
    assert stitched.words[2].char_end == 17
