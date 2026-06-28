import pytest

from tldw_Server_API.app.core.TTS.fish_s2_reference_imports import (
    FishS2ReferenceImportError,
    parse_fish_s2_reference_import,
)


def test_parse_json_single_existing_voice_import():
    items = parse_fish_s2_reference_import(
        filename="voice.json",
        content=b'{"voice_id": "voice-1", "reference_text": "hello there", "force": true}',
    )

    assert len(items) == 1
    item = items[0]
    assert item.voice_id == "voice-1"
    assert item.reference_text == "hello there"
    assert item.force is True


def test_parse_json_references_array_import():
    items = parse_fish_s2_reference_import(
        filename="voices.json",
        content=(
            b'{"references": ['
            b'{"voice_id": "voice-1", "reference_text": "one"},'
            b'{"voice_id": "voice-2", "text": "two"}'
            b"]}"
        ),
    )

    assert [item.voice_id for item in items] == ["voice-1", "voice-2"]
    assert [item.reference_text for item in items] == ["one", "two"]


def test_parse_json_embedded_audio_import():
    items = parse_fish_s2_reference_import(
        filename="voice.json",
        content=(
            b"{"
            b'"audio_b64": "QUJD",'
            b'"filename": "voice.wav",'
            b'"name": "Voice One",'
            b'"reference_text": "hello there"'
            b"}"
        ),
    )

    item = items[0]
    assert item.audio_base64 == "QUJD"
    assert item.filename == "voice.wav"
    assert item.name == "Voice One"


def test_parse_markdown_frontmatter_uses_body_as_reference_text():
    items = parse_fish_s2_reference_import(
        filename="voice.md",
        content=(
            b"---\n"
            b"voice_id: voice-1\n"
            b"name: Voice One\n"
            b"description: Private clone\n"
            b"---\n"
            b"Hello from the transcript.\n"
        ),
    )

    item = items[0]
    assert item.voice_id == "voice-1"
    assert item.name == "Voice One"
    assert item.description == "Private clone"
    assert item.reference_text == "Hello from the transcript."


def test_parse_markdown_frontmatter_can_override_body_text():
    items = parse_fish_s2_reference_import(
        filename="voice.markdown",
        content=(
            b"---\n"
            b"voice_id: voice-1\n"
            b"reference_text: Frontmatter transcript\n"
            b"---\n"
            b"Body transcript.\n"
        ),
    )

    assert items[0].reference_text == "Frontmatter transcript"


@pytest.mark.parametrize("filename", ["voice.txt", "voice.yaml", "voice"])
def test_parse_rejects_unsupported_extensions(filename):
    with pytest.raises(FishS2ReferenceImportError, match="Unsupported Fish S2 import file type"):
        parse_fish_s2_reference_import(filename=filename, content=b"{}")


def test_parse_rejects_markdown_without_voice_or_audio():
    with pytest.raises(FishS2ReferenceImportError, match="voice_id or audio_base64 is required"):
        parse_fish_s2_reference_import(
            filename="voice.md",
            content=b"Transcript only is not enough.\n",
        )


def test_parse_rejects_item_with_voice_and_audio():
    with pytest.raises(FishS2ReferenceImportError, match="Provide either voice_id or audio_base64"):
        parse_fish_s2_reference_import(
            filename="voice.json",
            content=b'{"voice_id": "voice-1", "audio_base64": "QUJD", "reference_text": "text"}',
        )


def test_parse_rejects_embedded_audio_without_required_fields():
    with pytest.raises(FishS2ReferenceImportError, match="filename, name, and reference_text"):
        parse_fish_s2_reference_import(
            filename="voice.json",
            content=b'{"audio_base64": "QUJD"}',
        )
