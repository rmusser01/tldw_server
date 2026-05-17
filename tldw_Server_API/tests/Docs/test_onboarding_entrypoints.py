"""Contract tests for the top-level onboarding entrypoints."""

from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_readme_start_here_links_to_profile_index() -> None:
    """README should keep linking to the canonical onboarding profiles."""
    text = _read_text("README.md")
    _require(
        "Docs/Getting_Started/README.md" in text,
        "README should link to the Getting Started index",
    )
    _require(
        "Local single-user" in text,
        "README should still mention the local single-user profile",
    )
    _require(
        "Docker single-user" in text,
        "README should mention the Docker single-user profile",
    )
    _require("make quickstart" in text, "README should mention make quickstart")
    _require(
        "make quickstart-docker-webui" in text,
        "README should mention the explicit WebUI Docker path",
    )
    _require(
        "apps/DEVELOPMENT.md" in text,
        "README should link developers to apps/DEVELOPMENT.md",
    )
    _require(
        "First_Time_Audio_Setup_CPU.md" in text,
        "README should link to the CPU first-time audio guide",
    )
    _require(
        "First_Time_Audio_Setup_GPU_Accelerated.md" in text,
        "README should link to the GPU/accelerated first-time audio guide",
    )


def test_readme_quickstart_defaults_to_webui_before_local_dev() -> None:
    """README quickstart should lead with the Docker WebUI path."""
    text = _read_text("README.md")
    _, separator, quickstart_section = text.partition("## Quickstart")
    _require(separator == "## Quickstart", "README should include a Quickstart section")
    _require(
        "make quickstart" in quickstart_section,
        "Quickstart section should mention make quickstart",
    )
    _require(
        "make quickstart-install" in quickstart_section,
        "Quickstart section should still mention make quickstart-install",
    )
    _require(
        quickstart_section.index("make quickstart") < quickstart_section.index("make quickstart-install"),
        "Quickstart section should present make quickstart before make quickstart-install",
    )
    _require(
        "docker multi-user + postgres" in quickstart_section.lower(),
        "Quickstart section should mention the multi-user + Postgres deployment path",
    )


def test_getting_started_index_lists_profiles_and_audio_guides() -> None:
    """The Getting Started index should enumerate base profiles and audio guides."""
    text = _read_text("Docs/Getting_Started/README.md")
    _require(
        "Choose exactly one base setup profile" in text,
        "Getting Started index should explain the base-profile model",
    )
    _require(
        "Canonical base profiles" in text,
        "Getting Started index should list canonical base profiles",
    )
    _require("Optional add-ons" in text, "Getting Started index should keep the optional add-ons section")
    for label in [
        "Local single-user",
        "Docker single-user",
        "Docker multi-user + Postgres",
        "First-time audio setup: CPU systems",
        "First-time audio setup: GPU/accelerated systems",
        "GPU/STT Add-on",
    ]:
        _require(label in text, f"Getting Started index should include {label}")


def test_getting_started_index_calls_out_default_webui_path() -> None:
    """The Getting Started index should call out the WebUI Docker default."""
    text = _read_text("Docs/Getting_Started/README.md")
    _require("make quickstart" in text, "Getting Started index should mention make quickstart")
    _require(
        "quickstart-docker-webui" in text,
        "Getting Started index should mention the WebUI Docker path",
    )
    _require(
        "Docker multi-user + Postgres" in text,
        "Getting Started index should keep the team/public deployment callout",
    )


def test_no_make_local_quick_launch_shortcuts_are_documented() -> None:
    """No-Make local self-hosters should be able to find the quick-launch scripts."""
    docs = {
        "README.md": _read_text("README.md"),
        "Docs/Getting_Started/README.md": _read_text("Docs/Getting_Started/README.md"),
        "Docs/Getting_Started/Profile_Local_Single_User.md": _read_text(
            "Docs/Getting_Started/Profile_Local_Single_User.md"
        ),
        "Docs/Published/Getting_Started/README.md": _read_text(
            "Docs/Published/Getting_Started/README.md"
        ),
        "Docs/Published/Getting_Started/Profile_Local_Single_User.md": _read_text(
            "Docs/Published/Getting_Started/Profile_Local_Single_User.md"
        ),
    }

    for path, text in docs.items():
        for script in ("quick-launch.sh", "quick-launch.command", "quick-launch.ps1"):
            _require(script in text, f"{path} should document {script}")
