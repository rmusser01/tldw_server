from pathlib import Path

EXPECTED_GUIDE_LINKS = [
    "../User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md",
    "../User_Guides/WebUI_Extension/TTS_Getting_Started.md",
    "../User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md",
]
QWEN3_ASR_GUIDE = "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md"


def test_tts_api_docs_include_user_guide_map() -> None:
    for path in [
        "Docs/API-related/TTS_API.md",
        "Docs/Published/API-related/TTS_API.md",
    ]:
        text = Path(path).read_text()
        assert "## User Guide Map" in text
        for link in EXPECTED_GUIDE_LINKS:
            assert link in text


def test_audio_transcription_docs_include_user_guide_map() -> None:
    for path in [
        "Docs/API-related/Audio_Transcription_API.md",
        "Docs/Published/API-related/Audio_Transcription_API.md",
    ]:
        text = Path(path).read_text()
        assert "## User Guide Map" in text
        for link in EXPECTED_GUIDE_LINKS:
            assert link in text
        assert QWEN3_ASR_GUIDE in text
