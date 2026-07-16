from pathlib import Path

TTS_RUNBOOK_URL = "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/TTS-SETUP-GUIDE.md"


def test_stt_tts_quickstart_declares_scope() -> None:
    for path in [
        "Docs/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md",
        "Docs/Published/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md",
    ]:
        text = Path(path).read_text()
        assert "Use this guide for first successful STT + TTS requests end-to-end." in text
        assert "TTS Providers Getting Started" in text
        assert "TTS Provider Setup Guide" in text


def test_tts_provider_guide_declares_scope() -> None:
    for path in [
        "Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md",
        "Docs/Published/User_Guides/WebUI_Extension/TTS_Getting_Started.md",
    ]:
        text = Path(path).read_text()
        assert "Use this page for provider selection and first successful synthesis." in text
        assert "Use [TTS Provider Setup Guide]" in text


def test_tts_setup_guide_is_runbook_index() -> None:
    for path in [
        "Docs/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md",
        "Docs/Published/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md",
    ]:
        text = Path(path).read_text()
        assert "# TTS Provider Setup Guide (Runbook Index)" in text
        assert "intentionally avoids duplicating provider setup details" in text
        assert TTS_RUNBOOK_URL in text
        assert "../../STT-TTS/TTS-SETUP-GUIDE.md" not in text
        assert len(text.splitlines()) < 220
