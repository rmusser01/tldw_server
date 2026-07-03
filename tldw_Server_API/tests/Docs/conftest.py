from __future__ import annotations

import shutil
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
    src_dir = repo_root / "Docs"
    dest_dir = src_dir / "Published"
    dest_dir.mkdir(parents=True, exist_ok=True)

    _preserve_and_copy(src_dir / "API-related", dest_dir / "API-related")
    _preserve_and_copy(src_dir / "Code_Documentation", dest_dir / "Code_Documentation")
    _preserve_and_copy(src_dir / "Wiki", dest_dir / "Wiki")
    _preserve_and_copy(src_dir / "ADR", dest_dir / "ADR")
    architecture = src_dir / "Architecture.md"
    if architecture.exists():
        shutil.copy2(architecture, dest_dir / "Architecture.md")
    _preserve_and_copy(src_dir / "Deployment", dest_dir / "Deployment", skip_monitoring=True)
    _preserve_and_copy(src_dir / "Deployment" / "Monitoring", dest_dir / "Monitoring")
    _preserve_and_copy(src_dir / "Evaluations", dest_dir / "Evaluations")
    _preserve_and_copy(src_dir / "Getting_Started", dest_dir / "Getting_Started")
    _preserve_and_copy(src_dir / "User_Guides", dest_dir / "User_Guides")

    logo = src_dir / "Logo.png"
    if logo.exists():
        assets_dir = dest_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(logo, assets_dir / "logo.png")
        shutil.copy2(logo, assets_dir / "favicon.png")


def _preserve_and_copy(src: Path, dest: Path, skip_monitoring: bool = False) -> None:
    dest.mkdir(parents=True, exist_ok=True)

    # Preserve section landing pages while refreshing curated content.
    for child in dest.iterdir():
        if child.name == "index.md":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    if not src.is_dir():
        return

    for item in src.iterdir():
        if skip_monitoring and item.name == "Monitoring":
            continue
        if (dest / "index.md").exists() and item.name in {"README", "README.md"}:
            continue
        target = dest / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
