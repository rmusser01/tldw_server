from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path("Helper_Scripts/docs/check_speech_docs_link_hygiene.py")
QWEN3_ASR_URL = "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md"
TASK3_AUDITED_STT_TTS_URLS = (
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_TTS_SETUP.md#runtime-modes",
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/TTS-SETUP-GUIDE.md#commercial-providers",
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/NEUTTS_TTS_SETUP.md#prerequisites",
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/CHATTERBOX_SETUP.md#requirements",
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/VIBEVOICE_GETTING_STARTED.md#1-prerequisites",
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/LUXTTS_TTS_SETUP.md#prerequisites",
)
TASK3_GETTING_STARTED_ENTRYPOINTS = (
    Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
)
UNREVIEWED_STT_TTS_URL = (
    "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/NOT_A_REVIEWED_OR_EXISTING_GUIDE.md"
)


def _load_hygiene_module() -> ModuleType:
    spec = spec_from_file_location("check_speech_docs_link_hygiene", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_speech_docs_link_hygiene_script_passes() -> None:
    module = _load_hygiene_module()
    assert module.main() == 0


def _run_hygiene(
    lines: tuple[str, ...],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    readme = tmp_path / "README.md"
    readme.write_text("\n".join(lines), encoding="utf-8")
    module = _load_hygiene_module()
    module.PROJECT_ROOT = tmp_path
    module.MONITORED_ENTRYPOINTS = [Path("README.md")]
    module.MONITORED_DIRS = []
    result = module.main()
    return result, capsys.readouterr().out


@pytest.mark.parametrize("entrypoint", TASK3_GETTING_STARTED_ENTRYPOINTS)
def test_hygiene_monitors_task3_getting_started_entrypoints(
    entrypoint: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_hygiene_module()
    assert entrypoint in module.MONITORED_ENTRYPOINTS

    target = tmp_path / entrypoint
    target.parent.mkdir(parents=True)
    target.write_text(UNREVIEWED_STT_TTS_URL, encoding="utf-8")
    module.PROJECT_ROOT = tmp_path
    module.MONITORED_ENTRYPOINTS = [entrypoint]
    module.MONITORED_DIRS = []

    assert module.main() == 1
    output = capsys.readouterr().out
    assert "unaudited_stt_tts_blob_target" in output
    assert str(entrypoint) in output


def test_hygiene_allows_audited_qwen_stt_tts_blob(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene(
        (
            QWEN3_ASR_URL,
            f"{QWEN3_ASR_URL}#manual-model-download",
            f"{QWEN3_ASR_URL}?plain=1#manual-model-download",
        ),
        tmp_path,
        capsys,
    )

    assert result == 0
    assert "Speech docs link hygiene check passed." in output


@pytest.mark.parametrize(
    "line",
    (
        f"[Qwen setup]({QWEN3_ASR_URL}#manual-model-download)",
        f"<{QWEN3_ASR_URL}?plain=1>",
        f'Reference: "{QWEN3_ASR_URL}".',
    ),
    ids=("markdown-parentheses", "angle-brackets", "quoted-plain-url"),
)
def test_hygiene_allows_exact_audited_urls_with_delimiters(
    line: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene((line,), tmp_path, capsys)

    assert result == 0
    assert "Speech docs link hygiene check passed." in output


@pytest.mark.parametrize("url", TASK3_AUDITED_STT_TTS_URLS)
def test_hygiene_allows_frozen_verified_task3_stt_tts_blobs(
    url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene((url,), tmp_path, capsys)

    assert result == 0
    assert "Speech docs link hygiene check passed." in output


def test_hygiene_rejects_unreviewed_stt_tts_blob(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene((UNREVIEWED_STT_TTS_URL,), tmp_path, capsys)

    assert result == 1
    assert "unaudited_stt_tts_blob_target" in output
    assert "NOT_A_REVIEWED_OR_EXISTING_GUIDE.md" in output


@pytest.mark.parametrize(
    ("line", "rejected_url"),
    (
        (
            "[guide](http://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md)",
            "http://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
        (
            "<https://GitHub.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md>",
            "https://GitHub.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
        (
            "https://gItHuB.cOm/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
            "https://gItHuB.cOm/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
        (
            'Reference: "https://github.com/rmusser01/tldw_server/blob/main/Docs/%53TT-TTS/QWEN3_ASR_SETUP.md".',
            "https://github.com/rmusser01/tldw_server/blob/main/Docs/%53TT-TTS/QWEN3_ASR_SETUP.md",
        ),
        (
            "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md%2fPRIVATE.md",
            "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md%2fPRIVATE.md",
        ),
        (
            "HTTPS://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
            "HTTPS://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
    ),
    ids=(
        "http-scheme-markdown",
        "uppercase-host-angle-brackets",
        "mixed-case-host",
        "percent-encoded-segment-quoted",
        "percent-encoded-path-suffix",
        "uppercase-scheme",
    ),
)
def test_hygiene_rejects_same_repo_candidate_normalization_bypasses(
    line: str,
    rejected_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene((line,), tmp_path, capsys)

    assert result == 1
    assert "unaudited_stt_tts_blob_target" in output
    assert rejected_url in output


@pytest.mark.parametrize(
    "url",
    (
        f"{QWEN3_ASR_URL}@NOT_A_REVIEWED_TARGET",
        f"{QWEN3_ASR_URL};NOT_A_REVIEWED_TARGET",
        "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/",
    ),
    ids=("at-path-suffix", "semicolon-path-suffix", "bare-prefix"),
)
def test_hygiene_rejects_audited_target_parser_bypasses(
    url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene((url,), tmp_path, capsys)

    assert result == 1
    assert "unaudited_stt_tts_blob_target" in output
    assert url in output


def test_hygiene_rejects_remaining_legacy_patterns(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result, output = _run_hygiene(
        (
            "https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting-Started-STT_and_TTS.md",
            "Docs/User_Guides/TTS_Getting_Started.md",
            "Installation-Setup-Guide.md",
        ),
        tmp_path,
        capsys,
    )

    assert result == 1
    assert "legacy_stt_tts_blob_link" in output
    assert "bad_tts_user_guide_path" in output
    assert "removed_installation_setup_guide" in output
