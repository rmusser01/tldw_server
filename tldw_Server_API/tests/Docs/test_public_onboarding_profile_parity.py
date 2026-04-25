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

PUBLIC_ONBOARDING_DOCS = [
    Path("README.md"),
    Path("Docs/Getting_Started/README.md"),
    Path("Docs/Getting_Started/QUICKSTART.md"),
    Path("Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
    Path("Docs/Published/Getting_Started/README.md"),
    Path("Docs/Published/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
    Path("Dockerfiles/README.md"),
    Path("Docs/Website/index.html"),
]

AUDIO_DOCS = [
    Path("Docs/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_CPU.md"),
    Path("Docs/Published/Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md"),
]

PROFILE_DOCS_WITH_PUBLISHED = [
    *PROFILE_DOCS,
    Path("Docs/Published/Getting_Started/Profile_Docker_Single_User.md"),
    Path("Docs/Published/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
    Path("Docs/Published/Getting_Started/Profile_Local_Single_User.md"),
]


def test_profile_docs_use_same_lifecycle_headings() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        positions = []
        for heading in LIFECYCLE_HEADINGS:
            assert heading in text, f"{path} missing {heading}"
            positions.append(text.index(heading))
        assert positions == sorted(positions), f"{path} lifecycle headings are out of order"


def test_profile_docs_include_windows_wsl_guidance() -> None:
    for path in PROFILE_DOCS:
        text = path.read_text(encoding="utf-8")
        for term in ("Windows", "WSL2", "PowerShell", "tldw-setup", "Docker Desktop"):
            assert term in text, f"{path} should include Windows/WSL guidance term {term}"


def test_audio_docs_show_single_user_and_multi_user_auth() -> None:
    for path in (
        *AUDIO_DOCS,
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


def test_multi_user_docs_generate_and_persist_admin_password() -> None:
    for path in PUBLIC_ONBOARDING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "replace-with-a-long-password" not in text, f"{path} uses a static admin password"
        assert "ADMIN_USERNAME" in text, f"{path} should persist ADMIN_USERNAME for login examples"
        assert "ADMIN_PASSWORD" in text, f"{path} should persist ADMIN_PASSWORD for login examples"
        assert "secrets.token_urlsafe(24)" in text, f"{path} should generate ADMIN_PASSWORD"


def test_multi_user_external_postgres_docs_use_override_vars() -> None:
    paths = [
        Path("Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
        Path("Docs/Published/Getting_Started/Profile_Docker_Multi_User_Postgres.md"),
        Path("Docs/Getting_Started/QUICKSTART.md"),
        Path("Dockerfiles/README.md"),
    ]
    forbidden = (
        "Use an external Postgres instance by setting `DATABASE_URL`",
        "point `DATABASE_URL`",
        "Point `DATABASE_URL`",
        "DATABASE_URL to your instance",
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text, f"{path} contains stale external Postgres guidance"
        assert "TLDW_DATABASE_URL_OVERRIDE" in text, f"{path} missing auth DB override var"
        assert "TLDW_JOBS_DB_URL_OVERRIDE" in text, f"{path} missing jobs DB override var"


def test_audio_verification_examples_cover_bearer_auth_for_all_audio_endpoints() -> None:
    command_markers = (
        "curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/speech",
        'curl -sS "http://127.0.0.1:8000/api/v1/audio/transcriptions/health',
        "curl -sS -X POST http://127.0.0.1:8000/api/v1/audio/transcriptions",
    )
    for path in AUDIO_DOCS:
        text = path.read_text(encoding="utf-8")
        assert "AUTH_HEADER" in text, f"{path} should define a reusable auth header"
        assert "Authorization: Bearer" in text, f"{path} should show bearer-token auth"
        for marker in command_markers:
            position = text.find(marker)
            assert position >= 0, f"{path} missing audio verification command: {marker}"
            while position >= 0:
                window = text[max(0, position - 1200) : position + 1800]
                assert "AUTH_HEADER" in window, (
                    f"{path} should use reusable auth guidance near {marker}"
                )
                position = text.find(marker, position + len(marker))


def test_profile_first_value_examples_are_provider_independent() -> None:
    for path in PROFILE_DOCS_WITH_PUBLISHED:
        text = path.read_text(encoding="utf-8")
        _, separator, remainder = text.partition("## First Value")
        assert separator == "## First Value", f"{path} missing First Value section"
        first_value, _, _ = remainder.partition("## Audio Path")
        assert "gpt-4o-mini" not in first_value, f"{path} uses provider-dependent model"
        assert "/api/v1/chat/completions" not in first_value, f"{path} uses provider-dependent chat"
        assert "first-value ingest/search" in first_value or (
            "/api/v1/media/add" in first_value and "/api/v1/media/search" in first_value
        ), f"{path} should describe provider-independent ingest/search first value"
