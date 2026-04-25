from __future__ import annotations

from pathlib import Path


PROFILE_DOCS = [
    Path("Docs/Getting_Started/Profile_Docker_Single_User.md"),
    Path("Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
    Path("Docs/Getting_Started/Profile_Local_Single_User.md"),
]

LIFECYCLE_HEADINGS = [
    "## Prepare",
    "## Start",
    "## Verify",
    "## First Value",
    "## Audio Path",
    "## Troubleshoot",
    "## Optional Add-ons",
]

WINDOWS_WSL_NOTE = (
    "> **Windows:** Use WSL2 for the documented make commands. If you prefer "
    "PowerShell, run the equivalent tldw-setup command shown under each step "
    "and start Docker Desktop before Docker profiles."
)


def test_profile_docs_use_same_lifecycle_headings() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        for heading in LIFECYCLE_HEADINGS:
            assert heading in text, f"{path} missing {heading}"


def test_profile_docs_include_windows_wsl_guidance() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        assert WINDOWS_WSL_NOTE in text, f"{path} should include Windows/WSL guidance"


def test_audio_docs_show_single_user_and_multi_user_auth() -> None:
    for path in (
        Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
        Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
        Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md"),
        Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "X-API-KEY" in text, f"{path} missing single-user API key example"
        assert (
            "Authorization: Bearer" in text
        ), f"{path} missing multi-user bearer token example"


def test_readme_lists_peer_profiles_in_order() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    expected_order = [
        "Docker single-user + WebUI",
        "Docker multi-user + Postgres",
        "Local single-user",
    ]
    positions = [readme.index(profile) for profile in expected_order]
    assert positions == sorted(positions)


def test_public_docs_use_new_profile_commands() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    for command in (
        "make setup-docker-single",
        "make start-docker-single",
        "make verify-docker-single",
        "make setup-docker-multi",
        "make start-docker-multi",
        "make verify-docker-multi",
        "make install-local",
        "make setup-local-single",
        "make start-local-single",
        "make verify-local-single",
    ):
        assert command in readme


def test_readme_resources_list_peer_profiles_in_order() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    _, separator, resources = readme.partition("## Documentation & Resources")
    assert separator == "## Documentation & Resources"

    expected_order = [
        "Docker Single-User + WebUI Profile",
        "Docker Multi-User + Postgres Profile",
        "Local Single-User Profile",
    ]
    positions = [resources.index(label) for label in expected_order]
    assert positions == sorted(positions)
