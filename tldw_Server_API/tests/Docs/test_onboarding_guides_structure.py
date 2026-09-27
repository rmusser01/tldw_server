"""Contract tests for the Getting Started onboarding documentation."""

from pathlib import Path

import pytest

REQUIRED = [
    "## Prepare",
    "## Start",
    "## Verify",
    "## First Value",
    "## Audio Path",
    "## Troubleshoot",
    "## Optional Add-ons",
]


def _require(condition: bool, message: str) -> None:
    """Fail a docs contract with a readable message."""
    if not condition:
        pytest.fail(message)


@pytest.mark.unit
def test_each_profile_has_required_sections() -> None:
    """Ensure each canonical onboarding profile keeps required sections."""
    guides = [
        "Docs/Getting_Started/Profile_Local_Single_User.md",
        "Docs/Getting_Started/Profile_Docker_Single_User.md",
        "Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md",
    ]
    for guide in guides:
        text = Path(guide).read_text(encoding="utf-8")
        positions = []
        for heading in REQUIRED:
            _require(heading in text, f"{guide} missing required heading: {heading}")
            positions.append(text.index(heading))
        _require(
            positions == sorted(positions),
            f"{guide} should present lifecycle headings in order",
        )
        _require(
            "First-Time Audio Setup: CPU Systems" in text,
            f"{guide} should point to the CPU audio guide",
        )
        _require(
            "First-Time Audio Setup: GPU/Accelerated Systems" in text,
            f"{guide} should point to the GPU/accelerated audio guide",
        )


@pytest.mark.unit
def test_getting_started_presents_peer_solo_paths_and_first_chat_journey() -> None:
    """Ensure the Getting Started index presents peer solo setup paths."""
    text = Path("Docs/Getting_Started/README.md").read_text(encoding="utf-8")

    for phrase in (
        "Solo setup chooser",
        "Docker single-user",
        "Local single-user",
        "shared server/operator path",
        "open the WebUI",
        "first successful chat",
        "add your first source",
    ):
        _require(phrase in text, f"Getting Started should mention: {phrase}")

    docker_idx = text.index("Docker single-user")
    local_idx = text.index("Local single-user")
    _require(
        abs(docker_idx - local_idx) < 2000,
        "Docker and local single-user paths should be presented as peer solo choices",
    )


@pytest.mark.unit
def test_single_user_profiles_handoff_to_webui_first_chat_and_first_source() -> None:
    """Ensure single-user profiles guide users into first WebUI value."""
    for guide in (
        "Docs/Getting_Started/Profile_Docker_Single_User.md",
        "Docs/Getting_Started/Profile_Local_Single_User.md",
    ):
        text = Path(guide).read_text(encoding="utf-8")
        for phrase in (
            "open the WebUI",
            "first successful chat",
            "add your first source",
        ):
            _require(phrase in text, f"{guide} should mention: {phrase}")
        before_troubleshoot, _, _ = text.partition("## Troubleshoot")
        forbidden = (
            "Add provider API keys to",
            "edit `.env`",
            "edit `tldw_Server_API/Config_Files/.env`",
            "edit `Config_Files/config.txt`",
        )
        for phrase in forbidden:
            _require(
                phrase not in before_troubleshoot,
                f"{guide} should not require manual config editing before troubleshooting: {phrase}",
            )


@pytest.mark.unit
def test_multi_user_profile_routes_operators_out_of_solo_wizard() -> None:
    """Ensure the multi-user profile directs operators away from solo setup."""
    text = Path("Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md").read_text(encoding="utf-8")

    for phrase in (
        "shared server/operator path",
        "solo wizard is not the multi-user path",
        "multi-user setup guide",
        "operator checklist",
    ):
        _require(phrase in text, f"Multi-user profile should mention: {phrase}")


@pytest.mark.unit
def test_gpu_addon_is_legacy_pointer_to_hardware_guides() -> None:
    """Ensure the GPU add-on points users to current hardware guides."""
    text = Path("Docs/Getting_Started/GPU_STT_Addon.md").read_text(encoding="utf-8")
    _require("legacy pointer" in text, "GPU_STT_Addon should be marked as a legacy pointer")
    _require(
        "First-Time Audio Setup: GPU/Accelerated Systems" in text,
        "GPU_STT_Addon should point to the accelerated guide",
    )
    _require(
        "First-Time Audio Setup: CPU Systems" in text,
        "GPU_STT_Addon should point to the CPU guide",
    )


@pytest.mark.unit
def test_first_time_audio_guides_have_core_sections() -> None:
    """Ensure first-time audio setup guides keep their core sections."""
    guide_requirements = {
        "Docs/Getting_Started/First_Time_Audio_Setup_CPU.md": [
            "## Before You Start",
            "## Step 1: Choose Your Base Setup Path",
            "## Step 2: Set the CPU STT Defaults",
            "## Step 3: Set Up the Recommended CPU TTS Path (`supertonic`)",
            "## Step 4: First Successful Verification",
            "## Troubleshooting",
            "parakeet-onnx",
            "supertonic",
        ],
        "Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md": [
            "## Before You Start",
            "## Step 1: Choose Your Base Setup Path",
            "## Step 2: Configure Accelerated STT",
            "## Step 3: Configure the Recommended TTS Path (`supertonic`)",
            "## Step 4: First Successful Verification",
            "## Troubleshooting",
            "faster-whisper",
            "parakeet-mlx",
            "supertonic",
        ],
    }
    for guide, required_content in guide_requirements.items():
        text = Path(guide).read_text(encoding="utf-8")
        for item in required_content:
            _require(item in text, f"{guide} missing expected content: {item}")


@pytest.mark.unit
def test_readme_positions_quickstart_prereqs_after_clone() -> None:
    """Ensure Makefile prerequisite checks are optional post-clone helpers."""
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "After cloning, you can run the optional Makefile helper checks" in readme


@pytest.mark.unit
def test_quickstart_runs_wizard_init_before_uvicorn() -> None:
    """Ensure manual local setup initializes auth before server startup."""
    quickstart = Path("Docs/Getting_Started/QUICKSTART.md").read_text(encoding="utf-8")

    init_at = quickstart.find("python -m tldw_Server_API.cli.wizard.cli init --profile local-single")
    uvicorn_at = quickstart.find("python -m uvicorn tldw_Server_API.app.main:app --reload")

    assert 0 <= init_at < uvicorn_at


@pytest.mark.unit
def test_quickstart_mentions_lower_level_auth_init_equivalent() -> None:
    """Ensure the lower-level AuthNZ initializer remains documented."""
    quickstart = Path("Docs/Getting_Started/QUICKSTART.md").read_text(encoding="utf-8")

    assert "python -m tldw_Server_API.app.core.AuthNZ.initialize --non-interactive" in quickstart


@pytest.mark.unit
def test_minimal_deployment_uses_supported_local_profile_commands() -> None:
    """Keep the minimal local path aligned with the maintained Make targets."""
    guide = Path("Docs/Deployment/minimal-deploy.md").read_text(encoding="utf-8")

    for command in (
        "make install-local",
        "make setup-local-single",
        "make start-local-single",
        "make verify-local-single",
    ):
        assert command in guide


@pytest.mark.unit
def test_minimal_deployment_uses_supported_docker_compose_file() -> None:
    """Keep the Docker path on the maintained production image definition."""
    guide = Path("Docs/Deployment/minimal-deploy.md").read_text(encoding="utf-8")

    assert "Dockerfiles/docker-compose.single-user.yml" in guide
    assert "build: ." not in guide


@pytest.mark.unit
def test_minimal_deployment_published_copy_matches_source() -> None:
    """Keep the hosted minimal deployment guide aligned with its source."""
    source = Path("Docs/Deployment/minimal-deploy.md").read_text(encoding="utf-8")
    published = Path("Docs/Published/Deployment/minimal-deploy.md").read_text(encoding="utf-8")

    assert published == source


@pytest.mark.unit
def test_minimal_deployment_documents_observable_reversible_recovery() -> None:
    """Keep silent-exit capture and backup-first invariant recovery discoverable."""
    guide = Path("Docs/Deployment/minimal-deploy.md").read_text(encoding="utf-8")

    assert "2>&1 | tee" in guide
    assert "Single-user bootstrap invariant check failed" in guide
    assert "--non-interactive" in guide


@pytest.mark.unit
def test_local_profile_runs_webui_setup_before_bun_dev() -> None:
    """Ensure WebUI setup commands precede starting the dev server."""
    local_profile = Path("Docs/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    copy_at = local_profile.find("cp .env.local.example .env.local")
    install_at = local_profile.find("bun install")
    dev_at = local_profile.find("bun run dev -- -p 8080")

    assert 0 <= copy_at < install_at < dev_at


@pytest.mark.unit
def test_local_profile_documents_loopback_api_url() -> None:
    """Ensure the local profile shows the expected WebUI API URL."""
    local_profile = Path("Docs/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" in local_profile


@pytest.mark.unit
def test_local_profile_documents_api_version() -> None:
    """Ensure the local profile shows the expected WebUI API version."""
    local_profile = Path("Docs/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_API_VERSION=v1" in local_profile


@pytest.mark.unit
def test_local_profile_documents_single_user_key_hint() -> None:
    """Ensure the local profile shows the optional single-user key hint."""
    local_profile = Path("Docs/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "# NEXT_PUBLIC_X_API_KEY=your_single_user_api_key" in local_profile


@pytest.mark.unit
def test_published_local_profile_documents_loopback_api_url() -> None:
    """Ensure the published local profile mirrors the WebUI API URL."""
    published_profile = Path("Docs/Published/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" in published_profile


@pytest.mark.unit
def test_published_local_profile_documents_api_version() -> None:
    """Ensure the published local profile mirrors the WebUI API version."""
    published_profile = Path("Docs/Published/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_API_VERSION=v1" in published_profile


@pytest.mark.unit
def test_published_local_profile_documents_single_user_key_hint() -> None:
    """Ensure the published local profile mirrors the single-user key hint."""
    published_profile = Path("Docs/Published/Getting_Started/Profile_Local_Single_User.md").read_text(encoding="utf-8")

    assert "# NEXT_PUBLIC_X_API_KEY=your_single_user_api_key" in published_profile


@pytest.mark.unit
def test_frontend_env_example_uses_loopback_api_url() -> None:
    """Ensure the copied WebUI env example matches the local profile docs."""
    env_example = Path("apps/tldw-frontend/.env.local.example").read_text(encoding="utf-8")

    assert "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" in env_example


@pytest.mark.unit
def test_frontend_env_example_describes_single_user_api_key() -> None:
    """Ensure the WebUI env example describes local single-user bootstrap auth."""
    env_example = Path("apps/tldw-frontend/.env.local.example").read_text(encoding="utf-8")

    assert "single-user API key" in env_example


@pytest.mark.unit
def test_frontend_env_example_references_backend_single_user_key() -> None:
    """Ensure the WebUI env example names the backend single-user key."""
    env_example = Path("apps/tldw-frontend/.env.local.example").read_text(encoding="utf-8")

    assert "SINGLE_USER_API_KEY" in env_example


@pytest.mark.unit
def test_frontend_env_example_allows_backend_key_match() -> None:
    """Ensure the WebUI env example does not contradict single-user setup."""
    env_example = Path("apps/tldw-frontend/.env.local.example").read_text(encoding="utf-8")

    assert "NOT the same as SINGLE_USER_API_KEY" not in env_example


@pytest.mark.unit
def test_quickstart_frames_setup_as_operator_recovery() -> None:
    """Ensure /setup is documented as a recovery surface, not the default path."""
    quickstart = Path("Docs/Getting_Started/QUICKSTART.md").read_text(encoding="utf-8")

    assert "backend/operator recovery surface" in quickstart


@pytest.mark.unit
def test_troubleshooting_explains_next_public_x_api_key_bootstrap() -> None:
    """Ensure troubleshooting explains browser-visible single-user bootstrap auth."""
    troubleshooting = Path("Docs/Getting_Started/TROUBLESHOOTING.md").read_text(encoding="utf-8")

    assert "browser-visible copy of the single-user API key" in troubleshooting


@pytest.mark.unit
def test_troubleshooting_uses_current_single_user_compose_file() -> None:
    """Ensure Windows/no-make troubleshooting uses single-user compose."""
    troubleshooting = Path("Docs/Getting_Started/TROUBLESHOOTING.md").read_text(encoding="utf-8")

    assert "Dockerfiles/docker-compose.single-user.yml" in troubleshooting


@pytest.mark.unit
def test_troubleshooting_uses_current_webui_compose_file() -> None:
    """Ensure Windows/no-make troubleshooting uses WebUI compose."""
    troubleshooting = Path("Docs/Getting_Started/TROUBLESHOOTING.md").read_text(encoding="utf-8")

    assert "Dockerfiles/docker-compose.webui.yml" in troubleshooting


@pytest.mark.unit
def test_troubleshooting_omits_stale_cmd_continuation_compose() -> None:
    """Ensure Windows/no-make troubleshooting avoids stale cmd syntax."""
    troubleshooting = Path("Docs/Getting_Started/TROUBLESHOOTING.md").read_text(encoding="utf-8")

    assert "Dockerfiles/docker-compose.yml ^" not in troubleshooting


@pytest.mark.unit
def test_troubleshooting_omits_stale_webui_build_command() -> None:
    """Ensure Windows/no-make troubleshooting avoids stale build syntax."""
    troubleshooting = Path("Docs/Getting_Started/TROUBLESHOOTING.md").read_text(encoding="utf-8")

    assert "docker-compose.webui.yml up -d --build" not in troubleshooting
