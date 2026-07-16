from __future__ import annotations

import subprocess  # nosec B404 - fixed local Bash command executes a repository script
from pathlib import Path

import pytest

_REQUIRED_PUBLISHED_FILES = (
    "Docs/Published/Getting_Started/Profile_Local_Single_User.md",
    "Docs/Published/API-related/TTS_API.md",
    "Docs/Published/API-related/Audio_Transcription_API.md",
    "Docs/Published/User_Guides/WebUI_Extension/Getting-Started-STT_and_TTS.md",
    "Docs/Published/User_Guides/WebUI_Extension/TTS_Getting_Started.md",
    "Docs/Published/User_Guides/WebUI_Extension/TTS-SETUP-GUIDE.md",
    "Docs/Published/Wiki/User_Wiki.md",
    "Docs/Published/Wiki/Developer_Wiki.md",
)


@pytest.fixture(scope="session", autouse=True)
def ensure_docs_published_mirror() -> None:
    """
    Ensure docs tests have a generated Docs/Published mirror available.

    The published tree is curated/generated from Docs/* and can be absent in
    clean local checkouts. Tests that assert published parity should operate on
    the generated view rather than fail due to a missing pre-step.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if all((repo_root / rel_path).exists() for rel_path in _REQUIRED_PUBLISHED_FILES):
        return

    _refresh_docs_published(repo_root)


def _refresh_docs_published(repo_root: Path) -> None:
    result = subprocess.run(  # nosec B603
        [
            "/bin/bash",
            str(repo_root / "Helper_Scripts" / "refresh_docs_published.sh"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr or result.stdout)
