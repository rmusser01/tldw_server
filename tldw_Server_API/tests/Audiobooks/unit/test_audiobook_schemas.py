import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.audiobook_schemas import (
    AlignmentPayload,
    AlignmentWord,
    AudiobookJobItem,
    AudiobookJobRequest,
    ChapterSelection,
    OutputOptions,
    SourceRef,
    SubtitleExportRequest,
    SubtitleOptions,
)

pytestmark = pytest.mark.unit


def _valid_source() -> SourceRef:
    return SourceRef(input_type="epub", upload_id="upload_1")


def _valid_chapters() -> list[ChapterSelection]:
    return [ChapterSelection(chapter_id="ch_001", include=True)]


def _valid_output() -> OutputOptions:
    return OutputOptions(formats=["mp3"])


def _valid_subtitles() -> SubtitleOptions:
    return SubtitleOptions(formats=["srt"], mode="sentence", variant="wide")


def _valid_alignment() -> AlignmentPayload:
    return AlignmentPayload(
        engine="kokoro",
        sample_rate=24000,
        words=[AlignmentWord(word="Hello", start_ms=0, end_ms=420)],
    )


def test_source_ref_requires_payload():
    with pytest.raises(ValidationError):
        SourceRef(input_type="epub")


def test_job_request_rejects_items_and_source():
    item = AudiobookJobItem(source=_valid_source())
    with pytest.raises(ValidationError):
        AudiobookJobRequest(project_title="Test", source=_valid_source(), items=[item])


def test_job_request_requires_output_and_subtitles_for_single_source():
    with pytest.raises(ValidationError):
        AudiobookJobRequest(project_title="Test", source=_valid_source(), chapters=_valid_chapters())


def test_job_request_allows_missing_subtitles_for_non_kokoro_provider():
    req = AudiobookJobRequest(
        project_title="Test",
        source=_valid_source(),
        chapters=_valid_chapters(),
        output=_valid_output(),
        tts_provider="openai",
    )
    assert req.tts_provider == "openai"


def test_job_request_rejects_subtitles_for_non_kokoro_provider():
    with pytest.raises(ValidationError):
        AudiobookJobRequest(
            project_title="Test",
            source=_valid_source(),
            chapters=_valid_chapters(),
            output=_valid_output(),
            subtitles=_valid_subtitles(),
            tts_provider="openai",
        )


def test_job_request_batch_requires_defaults_or_item_overrides():
    item = AudiobookJobItem(source=_valid_source(), chapters=_valid_chapters())
    with pytest.raises(ValidationError):
        AudiobookJobRequest(project_title="Batch", items=[item])


def test_job_request_batch_allows_missing_subtitles_for_non_kokoro_items():
    item = AudiobookJobItem(source=_valid_source(), chapters=_valid_chapters(), tts_provider="openai")
    req = AudiobookJobRequest(project_title="Batch", items=[item], output=_valid_output())
    assert req.items


def test_job_request_batch_rejects_non_kokoro_items_with_default_subtitles():
    item = AudiobookJobItem(source=_valid_source(), chapters=_valid_chapters(), tts_provider="openai")
    with pytest.raises(ValidationError):
        AudiobookJobRequest(
            project_title="Batch",
            items=[item],
            output=_valid_output(),
            subtitles=_valid_subtitles(),
        )


def test_job_request_batch_allows_explicit_null_subtitles_for_non_kokoro_items():
    item = AudiobookJobItem(
        source=_valid_source(),
        chapters=_valid_chapters(),
        tts_provider="openai",
        subtitles=None,
    )
    req = AudiobookJobRequest(
        project_title="Batch",
        items=[item],
        output=_valid_output(),
        subtitles=_valid_subtitles(),
    )
    assert req.items


def test_output_formats_must_not_be_empty():
    with pytest.raises(ValidationError):
        OutputOptions(formats=[])


def test_subtitle_export_defaults_words_per_cue():
    req = SubtitleExportRequest(
        format="srt",
        mode="word_count",
        variant="wide",
        alignment=_valid_alignment(),
    )
    assert req.words_per_cue == 12


def test_job_request_round_trips_gateway_fields_at_all_supported_scopes():
    req = AudiobookJobRequest(
        project_title="Gateway book",
        source=_valid_source(),
        chapters=[
            ChapterSelection(
                chapter_id="ch_001",
                include=True,
                tts_backend="gateway:chapter",
                tts_allow_fallback=False,
            )
        ],
        output=_valid_output(),
        tts_backend="openrouter",
        tts_allow_fallback=True,
        tts_model="Vendor/Exact-TTS",
    )

    payload = req.model_dump()

    assert payload["tts_backend"] == "openrouter"
    assert payload["tts_allow_fallback"] is True
    assert payload["chapters"][0]["tts_backend"] == "gateway:chapter"
    assert payload["chapters"][0]["tts_allow_fallback"] is False


def test_batch_item_round_trips_gateway_fields_without_changing_legacy_fields():
    item = AudiobookJobItem(
        source=_valid_source(),
        chapters=_valid_chapters(),
        tts_backend="company",
        tts_allow_fallback=False,
        tts_provider="openai",
        tts_model="Vendor/Exact-TTS",
    )
    req = AudiobookJobRequest(
        project_title="Batch",
        items=[item],
        output=_valid_output(),
    )

    assert req.items is not None
    assert req.items[0].tts_backend == "company"
    assert req.items[0].tts_allow_fallback is False
    assert req.items[0].tts_provider == "openai"
    assert req.items[0].tts_model == "Vendor/Exact-TTS"


@pytest.mark.parametrize("scope", ["chapter", "item", "request"])
def test_gateway_backend_rejects_blank_values_at_every_scope(scope):
    with pytest.raises(ValidationError, match="tts_backend must not be blank"):
        if scope == "chapter":
            ChapterSelection(chapter_id="ch_001", include=True, tts_backend=" \t")
        elif scope == "item":
            AudiobookJobItem(source=_valid_source(), tts_backend=" \t")
        else:
            AudiobookJobRequest(
                project_title="Gateway book",
                source=_valid_source(),
                chapters=_valid_chapters(),
                output=_valid_output(),
                tts_backend=" \t",
            )
