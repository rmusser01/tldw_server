import re
from pathlib import Path

STT_TTS_BLOB_PREFIX = "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/"
AUDITED_TTS_GETTING_STARTED_RUNBOOKS = frozenset(
    {
        f"{STT_TTS_BLOB_PREFIX}TTS-SETUP-GUIDE.md",
        f"{STT_TTS_BLOB_PREFIX}NEUTTS_TTS_SETUP.md",
        f"{STT_TTS_BLOB_PREFIX}CHATTERBOX_SETUP.md",
        f"{STT_TTS_BLOB_PREFIX}VIBEVOICE_GETTING_STARTED.md",
        f"{STT_TTS_BLOB_PREFIX}LUXTTS_TTS_SETUP.md",
    }
)
STT_TTS_BLOB_URL = re.compile(re.escape(STT_TTS_BLOB_PREFIX) + r"[A-Za-z0-9_-]+\.md")


def test_readme_tts_onboarding_path_points_to_webui_extension() -> None:
    text = Path("README.md").read_text()
    assert "Docs/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md" in text
    assert "Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md" in text
    assert "Docs/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md" in text
    assert "Docs/User_Guides/TTS_Getting_Started.md" not in text


def test_tts_getting_started_uses_only_frozen_verified_runbook_links() -> None:
    legacy_stt_tts = "https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting-Started-STT_and_TTS.md"
    for path in [
        "Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md",
        "Docs/Published/User_Guides/WebUI_Extension/TTS_Getting_Started.md",
    ]:
        text = Path(path).read_text()
        assert legacy_stt_tts not in text
        assert frozenset(STT_TTS_BLOB_URL.findall(text)) == AUDITED_TTS_GETTING_STARTED_RUNBOOKS
        assert "./Getting-Started-STT_and_TTS.md" in text
