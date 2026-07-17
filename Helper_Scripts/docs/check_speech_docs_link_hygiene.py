#!/usr/bin/env python3
"""Fail when speech docs drift back to deprecated link patterns."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import SplitResult, unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORED_ENTRYPOINTS = [
    Path("README.md"),
    Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
]

MONITORED_DIRS = [
    Path("Docs/API-related"),
    Path("Docs/Published/API-related"),
    Path("Docs/User_Guides"),
    Path("Docs/Published/User_Guides"),
]

BLOCKED_PATTERNS: dict[str, re.Pattern[str]] = {
    "legacy_stt_tts_blob_link": re.compile(
        r"https://github\.com/rmusser01/tldw_server/blob/main/Docs/Getting-Started-STT_and_TTS\.md"
    ),
    "bad_tts_user_guide_path": re.compile(r"Docs/User_Guides/TTS_Getting_Started\.md"),
    "removed_installation_setup_guide": re.compile(r"Installation-Setup-Guide\.md"),
}
URL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s<>()\[\]{}\"'`]+")
STT_TTS_REPO_PATH_PREFIX = "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/"
AUDITED_STT_TTS_BLOB_TARGETS = frozenset(
    {
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_TTS_SETUP.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/TTS-SETUP-GUIDE.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/NEUTTS_TTS_SETUP.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/CHATTERBOX_SETUP.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/VIBEVOICE_GETTING_STARTED.md",
        ),
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/LUXTTS_TTS_SETUP.md",
        ),
    }
)


def iter_stt_tts_blob_candidates(line: str) -> Iterator[tuple[str, SplitResult, str]]:
    """Yield same-repo STT-TTS URL tokens, including encoded or case-varied forms."""
    for match in URL_TOKEN.finditer(line):
        url = match.group(0)
        try:
            parsed = urlsplit(url)
            hostname = parsed.hostname
        except ValueError:
            continue
        decoded_path = unquote(parsed.path)
        if (
            hostname is not None
            and hostname.lower() == "github.com"
            and decoded_path.startswith(STT_TTS_REPO_PATH_PREFIX)
        ):
            yield url, parsed, decoded_path


def iter_monitored_files() -> tuple[list[Path], list[Path]]:
    files: list[Path] = []
    missing_roots: list[Path] = []

    for rel in MONITORED_ENTRYPOINTS:
        files.append(rel)

    for rel_dir in MONITORED_DIRS:
        abs_dir = PROJECT_ROOT / rel_dir
        if not abs_dir.exists():
            missing_roots.append(rel_dir)
            continue
        for path in abs_dir.rglob("*.md"):
            files.append(path.relative_to(PROJECT_ROOT))

    deduped_sorted = sorted(set(files))
    return deduped_sorted, missing_roots


def main() -> int:
    failures: list[str] = []
    monitored_files, missing_roots = iter_monitored_files()
    for root in missing_roots:
        failures.append(f"{root}: missing monitored docs root")

    for rel in monitored_files:
        path = PROJECT_ROOT / rel
        if not path.exists():
            failures.append(f"{rel}: missing monitored file")
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for name, pattern in BLOCKED_PATTERNS.items():
                if pattern.search(line):
                    failures.append(f"{rel}:{line_no}: [{name}] {line.strip()}")
            for url, parsed, decoded_path in iter_stt_tts_blob_candidates(line):
                target = (parsed.scheme, parsed.netloc, parsed.path)
                if (
                    not url.startswith("https://")
                    or parsed.path != decoded_path
                    or target not in AUDITED_STT_TTS_BLOB_TARGETS
                ):
                    failures.append(f"{rel}:{line_no}: [unaudited_stt_tts_blob_target] {url}")

    if failures:
        print("Speech docs link hygiene violations found:")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Speech docs link hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
