import pytest

from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import AlignmentPayload, AlignmentWord
from tldw_Server_API.app.core.Audiobooks.subtitle_generator import generate_subtitles

pytestmark = pytest.mark.unit


def _make_alignment():
    words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=400),
        AlignmentWord(word="world.", start_ms=450, end_ms=900),
        AlignmentWord(word="Next", start_ms=1000, end_ms=1300),
        AlignmentWord(word="sentence.", start_ms=1350, end_ms=1800),
    ]
    return AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)


def test_generate_srt_sentence_mode_splits_on_punctuation():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="sentence",
        variant="wide",
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2
    assert "Hello world." in blocks[0]
    assert "00:00:00,000 --> 00:00:00,900" in blocks[0]
    assert "Next sentence." in blocks[1]


def test_generate_sentence_mode_uses_spacy_when_enabled(monkeypatch):
    class _FakeSpan:
        def __init__(self, start, end):
            self.start_char = start
            self.end_char = end

    class _FakeDoc:
        def __init__(self, spans):
            self.sents = spans

    def _fake_nlp(text):
        return _FakeDoc([_FakeSpan(0, 12), _FakeSpan(13, len(text))])

    from tldw_Server_API.app.core.Audiobooks import subtitle_generator as sg

    monkeypatch.setattr(sg, "_load_spacy_model", lambda: _fake_nlp)

    words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=400, char_start=0, char_end=5),
        AlignmentWord(word="world.", start_ms=450, end_ms=900, char_start=6, char_end=12),
        AlignmentWord(word="Next", start_ms=1000, end_ms=1300, char_start=13, char_end=17),
        AlignmentWord(word="sentence.", start_ms=1350, end_ms=1800, char_start=18, char_end=27),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = sg.generate_subtitles(
        alignment,
        format="srt",
        mode="sentence",
        variant="wide",
        source_text="Hello world. Next sentence.",
        enable_spacy=True,
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2
    assert "Hello world." in blocks[0]
    assert "Next sentence." in blocks[1]


def test_generate_word_count_groups_into_multiple_cues():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="word_count",
        variant="wide",
        words_per_cue=2,
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2
    assert "Hello world." in blocks[0]
    assert "Next sentence." in blocks[1]


def test_generate_vtt_includes_header_and_dot_time_format():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="vtt",
        mode="word_count",
        variant="wide",
        words_per_cue=4,
    )
    assert content.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.800" in content


def test_generate_ass_includes_dialogue_lines():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="ass",
        mode="word_count",
        variant="wide",
        words_per_cue=4,
    )
    assert "[Events]" in content
    assert "Dialogue: 0,0:00:00.00,0:00:01.80" in content
    assert "Hello world." in content


def test_generate_vtt_centered_variant_adds_alignment_setting():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="vtt",
        mode="word_count",
        variant="centered",
        words_per_cue=4,
    )
    assert "align:center" in content


def test_generate_srt_centered_variant_falls_back_to_wide():
    alignment = _make_alignment()
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="word_count",
        variant="centered",
        words_per_cue=4,
        max_chars=20,
    )
    lines = content.splitlines()
    text_lines = [line for line in lines if line and "-->" not in line and not line.isdigit()]
    assert text_lines
    assert all(not line.startswith(" ") for line in text_lines)


def test_generate_max_chars_and_max_lines_limits():
    words = [
        AlignmentWord(word="This", start_ms=0, end_ms=200),
        AlignmentWord(word="is", start_ms=250, end_ms=350),
        AlignmentWord(word="a", start_ms=360, end_ms=420),
        AlignmentWord(word="longer", start_ms=430, end_ms=600),
        AlignmentWord(word="subtitle", start_ms=610, end_ms=820),
        AlignmentWord(word="line", start_ms=830, end_ms=1000),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="word_count",
        variant="narrow",
        words_per_cue=6,
        max_chars=12,
        max_lines=2,
    )
    lines = content.splitlines()
    text_lines = [line for line in lines if line and "-->" not in line and not line.isdigit()]
    assert len(text_lines) <= 2
    assert len(text_lines[0]) <= 12
    joined = " ".join(text_lines)
    for word in ["This", "is", "a", "longer", "subtitle", "line"]:
        assert word in joined


def test_generate_vtt_highlight_adds_class_tag():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[
            AlignmentWord(word="Hello", start_ms=0, end_ms=400),
            AlignmentWord(word="world", start_ms=450, end_ms=900),
        ],
    )
    content = generate_subtitles(
        alignment,
        format="vtt",
        mode="highlight",
        variant="wide",
    )
    assert content.count("<c.hl>") == 2
    assert "<c.hl>Hello</c>" in content


def test_generate_vtt_highlight_escapes_word_markup():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[AlignmentWord(word="<script&>", start_ms=0, end_ms=400)],
    )

    content = generate_subtitles(
        alignment,
        format="vtt",
        mode="highlight",
        variant="wide",
    )

    assert "<c.hl>&lt;script&amp;&gt;</c>" in content
    assert "<script&>" not in content


def test_generate_ass_highlight_adds_karaoke_tag():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[AlignmentWord(word="Hello", start_ms=0, end_ms=500)],
    )
    content = generate_subtitles(
        alignment,
        format="ass",
        mode="highlight",
        variant="wide",
    )
    assert "{\\k50}Hello" in content


def test_generate_ass_sanitizes_override_tags_in_text():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[AlignmentWord(word="{\\pos(0,0)}Hello\\Nthere", start_ms=0, end_ms=500)],
    )

    content = generate_subtitles(
        alignment,
        format="ass",
        mode="word_count",
        variant="wide",
        words_per_cue=1,
    )

    assert "\\pos" not in content
    assert "Hello there" in content


def test_generate_srt_highlight_sanitizes_cue_breaking_text():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[
            AlignmentWord(
                word="Line\n\n2\n00:00:02,000 --> 00:00:03,000",
                start_ms=0,
                end_ms=500,
            )
        ],
    )

    content = generate_subtitles(
        alignment,
        format="srt",
        mode="highlight",
        variant="wide",
    )

    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 1
    assert "-- >" in blocks[0]
    assert "00:00:02,000 --> 00:00:03,000" not in blocks[0]


def test_generate_rejects_negative_end_times():
    alignment = AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[AlignmentWord(word="Bad", start_ms=1000, end_ms=900)],
    )
    with pytest.raises(ValueError):
        generate_subtitles(alignment, format="srt", mode="line", variant="wide")


def test_generate_line_mode_splits_on_newlines():
    words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=200),
        AlignmentWord(word="world\n", start_ms=220, end_ms=400),
        AlignmentWord(word="Next", start_ms=500, end_ms=700),
        AlignmentWord(word="line", start_ms=720, end_ms=900),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="line",
        variant="wide",
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2
    assert "Hello world" in blocks[0]
    assert "Next line" in blocks[1]


def test_generate_line_mode_uses_source_text_newlines():
    words = [
        AlignmentWord(word="Hello", start_ms=0, end_ms=200, char_start=0, char_end=5),
        AlignmentWord(word="world", start_ms=220, end_ms=400, char_start=6, char_end=11),
        AlignmentWord(word="Next", start_ms=500, end_ms=700, char_start=12, char_end=16),
        AlignmentWord(word="line", start_ms=720, end_ms=900, char_start=17, char_end=21),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="line",
        variant="wide",
        source_text="Hello world\nNext line",
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2
    assert "Hello world" in blocks[0]
    assert "Next line" in blocks[1]


def test_generate_line_mode_respects_max_lines():
    words = [
        AlignmentWord(word="Line", start_ms=0, end_ms=200),
        AlignmentWord(word="one\n", start_ms=210, end_ms=350),
        AlignmentWord(word="Line", start_ms=360, end_ms=520),
        AlignmentWord(word="two\n", start_ms=530, end_ms=700),
        AlignmentWord(word="Line", start_ms=710, end_ms=880),
        AlignmentWord(word="three", start_ms=890, end_ms=1050),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="line",
        variant="wide",
        max_lines=2,
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 3
    assert "Line one" in blocks[0]
    assert "Line two" in blocks[1]
    assert "Line three" in blocks[2]


def test_sentence_mode_falls_back_on_max_chars():
    words = [
        AlignmentWord(word="This", start_ms=0, end_ms=200),
        AlignmentWord(word="sentence", start_ms=210, end_ms=350),
        AlignmentWord(word="is", start_ms=360, end_ms=420),
        AlignmentWord(word="too", start_ms=430, end_ms=500),
        AlignmentWord(word="long.", start_ms=510, end_ms=700),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="sentence",
        variant="wide",
        words_per_cue=2,
        max_chars=10,
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 3


def test_sentence_mode_falls_back_on_duration():
    words = [
        AlignmentWord(word="Long", start_ms=0, end_ms=4000),
        AlignmentWord(word="sentence.", start_ms=4100, end_ms=9000),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="sentence",
        variant="wide",
        words_per_cue=1,
    )
    blocks = [block for block in content.strip().split("\n\n") if block]
    assert len(blocks) == 2


def test_word_count_clamps_short_duration():
    words = [
        AlignmentWord(word="Quick", start_ms=0, end_ms=200),
        AlignmentWord(word="cue", start_ms=210, end_ms=300),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="word_count",
        variant="wide",
        words_per_cue=2,
    )
    assert "00:00:00,000 --> 00:00:00,800" in content


def test_word_count_clamps_long_duration():
    words = [
        AlignmentWord(word="Long", start_ms=0, end_ms=10000),
        AlignmentWord(word="cue", start_ms=10010, end_ms=12000),
    ]
    alignment = AlignmentPayload(engine="kokoro", sample_rate=24000, words=words)
    content = generate_subtitles(
        alignment,
        format="srt",
        mode="word_count",
        variant="wide",
        words_per_cue=2,
    )
    assert "00:00:00,000 --> 00:00:06,000" in content
