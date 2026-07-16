#!/usr/bin/env python3
"""Fail when speech docs drift back to deprecated link patterns."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MONITORED_FILES = [
    Path("README.md"),
    Path("Docs/API-related/Audio_Transcription_API.md"),
    Path("Docs/Published/API-related/Audio_Transcription_API.md"),
    Path("Docs/API-related/TTS_API.md"),
    Path("Docs/Published/API-related/TTS_API.md"),
    Path("Docs/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md"),
    Path("Docs/Published/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md"),
    Path("Docs/User_Guides/WebUI_Extension/TTS_Getting_Started.md"),
    Path("Docs/Published/User_Guides/WebUI_Extension/TTS_Getting_Started.md"),
    Path("Docs/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md"),
    Path("Docs/Published/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md"),
]

MONITORED_ENTRYPOINTS = [
    Path("README.md"),
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
STT_TTS_BLOB_PREFIX = "https://github.com/rmusser01/tldw_server/blob/main/Docs/STT-TTS/"
STT_TTS_BLOB_URL = re.compile(re.escape(STT_TTS_BLOB_PREFIX) + r"[^\s)\]}>\"'`]*")
AUDITED_STT_TTS_BLOB_TARGETS = frozenset(
    {
        (
            "https",
            "github.com",
            "/rmusser01/tldw_server/blob/main/Docs/STT-TTS/QWEN3_ASR_SETUP.md",
        ),
    }
)


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
            for match in STT_TTS_BLOB_URL.finditer(line):
                url = match.group(0)
                parsed = urlsplit(url)
                target = (parsed.scheme, parsed.netloc, parsed.path)
                if target not in AUDITED_STT_TTS_BLOB_TARGETS:
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
