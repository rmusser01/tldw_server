from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("Helper_Scripts/docs/check_speech_docs_link_hygiene.py")


def _load_hygiene_module() -> ModuleType:
    spec = spec_from_file_location("check_speech_docs_link_hygiene", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speech_docs_link_hygiene_script_passes() -> None:
    module = _load_hygiene_module()
    assert module.main() == 0


def test_hygiene_allows_audited_stt_tts_blob_and_rejects_remaining_patterns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            (
                "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
                "https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting-Started-STT_and_TTS.md",
                "Docs/User_Guides/TTS_Getting_Started.md",
                "Installation-Setup-Guide.md",
            )
        ),
        encoding="utf-8",
    )
    module = _load_hygiene_module()
    module.PROJECT_ROOT = tmp_path
    module.MONITORED_ENTRYPOINTS = [Path("README.md")]
    module.MONITORED_DIRS = []

    assert module.main() == 1
    output = capsys.readouterr().out
    assert "legacy_stt_tts_blob_link" in output
    assert "bad_tts_user_guide_path" in output
    assert "removed_installation_setup_guide" in output
    assert "QWEN3_ASR_SETUP.md" not in output
