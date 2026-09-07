"""Fail when guide and contributor docs reference missing Docs paths."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE_DIRS = (
    REPO_ROOT / "Docs/User_Guides",
    REPO_ROOT / "Docs/Getting_Started",
    REPO_ROOT / "Docs/Deployment",
    REPO_ROOT / "Docs/Published/User_Guides",
    REPO_ROOT / "Docs/Published/Getting_Started",
    REPO_ROOT / "Docs/Published/Deployment",
    REPO_ROOT / "Docs/Published/Monitoring",
)
GUIDE_FILES = (
    REPO_ROOT / "Helper_Scripts/Samples/Grafana/README.md",
    REPO_ROOT / "New-User-Guide.md",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / ".github/pull_request_template.md",
    REPO_ROOT / "tldw_Server_API/README.md",
    REPO_ROOT / "tldw_Server_API/app/core/MCP_unified/README.md",
    REPO_ROOT / "tldw_Server_API/app/core/RAG/README.md",
    REPO_ROOT / "tldw_Server_API/app/core/Resource_Governance/README.md",
    REPO_ROOT / "tldw_Server_API/app/core/TTS/README.md",
    REPO_ROOT / "tldw_Server_API/app/core/TTS/TTS-DEPLOYMENT.md",
    REPO_ROOT / "Docs/STT-TTS/VIBEVOICE_GETTING_STARTED.md",
    REPO_ROOT / "Docs/STT-TTS/QWEN3_TTS_SETUP.md",
)
DOC_PATH_PATTERN = re.compile(r"Docs/[A-Za-z0-9_./-]+")
EXTERNAL_URL_PATTERN = re.compile(r"(?<![A-Za-z0-9_./-])(?:https?:)?//[^\s<>\[\]()`\"']+", re.IGNORECASE)


def _collect_missing_paths() -> list[tuple[str, str]]:
    """Collect missing local paths without treating remote URL paths as local."""
    missing: list[tuple[str, str]] = []
    markdown_files: list[Path] = []
    for root in GUIDE_DIRS:
        if not root.exists():
            continue
        markdown_files.extend(sorted(root.rglob("*.md")))
    for guide_file in GUIDE_FILES:
        if guide_file.exists():
            markdown_files.append(guide_file)

    for guide_file in markdown_files:
        text = EXTERNAL_URL_PATTERN.sub("", guide_file.read_text(encoding="utf-8"))
        for match in DOC_PATH_PATTERN.finditer(text):
            raw_path = match.group(0).rstrip("`),.:;")
            candidate = REPO_ROOT / raw_path
            if not candidate.exists():
                missing.append((str(guide_file.relative_to(REPO_ROOT)), raw_path))
    return sorted(set(missing))


def main() -> int:
    """Report broken repository paths and return a failing exit code if any exist."""
    missing = _collect_missing_paths()
    if missing:
        print("Top guide docs reference missing paths:")
        for source_file, missing_path in missing:
            print(f"- {source_file} -> {missing_path}")
        return 1

    print("Top guide docs path hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
