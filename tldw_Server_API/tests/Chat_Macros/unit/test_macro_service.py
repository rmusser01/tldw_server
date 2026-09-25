from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError, MacroValidationError
from tldw_Server_API.app.core.Chat_Macros.output_profiles import (
    DEFAULT_OUTPUT_PROFILE,
    normalize_output_profile,
    render_output_profile,
)
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chat_Macros.service import ChatMacrosService
from tldw_Server_API.app.core.Chat_Macros.settings import normalize_settings
from tldw_Server_API.app.core.Chat_Macros.storage import ChatMacroStorage
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture()
def raw_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(str(tmp_path / "macros.db"), client_id="test_client")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def service(tmp_path: Path, raw_db: CharactersRAGDB) -> ChatMacrosService:
    return ChatMacrosService(
        user_id="1",
        storage=ChatMacroStorage(tmp_path / "user"),
        repository=ChatMacroRepository(raw_db),
        core_commands={"weather", "time", "skill", "skills"},
    )


def _user_macro_yaml(name: str = "daily_digest", command: str | None = None) -> str:
    return (
        "schema_version: 1\n"
        f"name: {name}\n"
        f"command: {command or name}\n"
        "steps:\n"
        "  - id: prompt\n"
        "    type: prompt\n"
        "    output: answer\n"
        "    prompt: Say hi.\n"
    )


def test_lists_builtin_wrapup_and_can_disable_it(service: ChatMacrosService) -> None:
    wrapup = next(item for item in service.list_macros() if item.name == "wrapup")

    assert wrapup.command == "wrapup"
    assert wrapup.source == "builtin"
    assert wrapup.enabled is True
    assert wrapup.immutable is True

    disabled = service.set_builtin_enabled("wrapup", False)
    assert disabled.enabled is False
    assert next(item for item in service.list_macros() if item.name == "wrapup").enabled is False
    assert service.repository.list_registry_entries("1")[0]["enabled"] == 0

    with pytest.raises(MacroStorageError, match="built-in"):
        service.delete_macro("wrapup")


def test_clone_builtin_creates_user_macro_with_non_conflicting_command(service: ChatMacrosService) -> None:
    cloned = service.clone_builtin("wrapup", new_name="my_wrapup", command="my_wrapup")

    assert cloned.name == "my_wrapup"
    assert cloned.command == "my_wrapup"
    assert cloned.source == "user"
    assert cloned.immutable is False
    assert service.get_macro("my_wrapup").definition.command == "my_wrapup"

    with pytest.raises(MacroValidationError, match="core command"):
        service.clone_builtin("wrapup", new_name="weather_wrapup", command="weather")


def test_create_update_delete_user_macro_and_validate_without_saving(service: ChatMacrosService) -> None:
    validated = service.validate_macro(_user_macro_yaml())
    assert validated.command == "daily_digest"
    assert service.storage.list() == []

    with pytest.raises(MacroValidationError, match="macro name"):
        service.validate_macro(_user_macro_yaml("BadName", "bad_name"))

    created = service.create_macro("daily_digest", _user_macro_yaml())
    assert created.source == "user"

    with pytest.raises(MacroValidationError, match="core command"):
        service.create_macro("bad_weather", _user_macro_yaml("bad_weather", "weather"))
    with pytest.raises(MacroValidationError, match="another macro"):
        service.create_macro("wrapup", _user_macro_yaml("wrapup", "wrapup"))
    with pytest.raises(MacroValidationError, match="macro name"):
        service.create_macro("wrapup", _user_macro_yaml("wrapup", "my_wrapup"))

    updated = service.update_macro("daily_digest", _user_macro_yaml("daily_digest", "team_digest"))
    assert updated.command == "team_digest"
    registry_commands = {row["command"] for row in service.repository.list_registry_entries("1")}
    assert "team_digest" in registry_commands
    assert "daily_digest" not in registry_commands

    service.delete_macro("daily_digest")
    assert service.storage.list() == []
    registry_commands = {row["command"] for row in service.repository.list_registry_entries("1")}
    assert registry_commands == {"wrapup"}


def test_create_macro_rejects_definition_name_mismatch_before_storage(service: ChatMacrosService) -> None:
    """Reject a mismatched resource name before creating either definition."""
    with pytest.raises(
        MacroValidationError,
        match="macro definition name 'other_name' must match resource name 'daily_digest'",
    ):
        service.create_macro("daily_digest", _user_macro_yaml("other_name"))

    assert not (service.storage.macros_dir / "daily_digest").exists()
    assert not (service.storage.macros_dir / "other_name").exists()


def test_update_macro_rejects_definition_rename_before_storage(service: ChatMacrosService) -> None:
    """A rejected rename leaves the original definition unchanged."""
    original_raw = _user_macro_yaml("daily_digest")
    service.create_macro("daily_digest", original_raw)

    with pytest.raises(
        MacroValidationError,
        match="macro definition name 'renamed' must match resource name 'daily_digest'",
    ):
        service.update_macro("daily_digest", _user_macro_yaml("renamed"))

    assert service.storage.read("daily_digest").raw == original_raw
    assert not (service.storage.macros_dir / "renamed").exists()


def test_user_enabled_override_preserves_authored_yaml(service: ChatMacrosService) -> None:
    raw = _user_macro_yaml() + "# keep this comment\n"
    service.create_macro("daily_digest", raw)

    disabled = service.set_macro_enabled("daily_digest", False)

    assert disabled.enabled is False
    assert service.storage.read("daily_digest").raw == raw
    assert service.get_macro("daily_digest").enabled is False

    enabled = service.set_macro_enabled("daily_digest", True)

    assert enabled.enabled is True
    assert service.storage.read("daily_digest").raw == raw


def test_normalize_settings_preserves_unknown_keys_without_aliasing() -> None:
    """Preserve future settings without sharing mutable values with the input."""
    raw = {
        "disabled_builtins": ["wrapup"],
        "future_authoring": {"options": ["keep"]},
    }

    normalized = normalize_settings(raw)
    raw["future_authoring"]["options"].append("changed")

    assert normalized["future_authoring"] == {"options": ["keep"]}
    assert normalized["disabled_builtins"] == ["wrapup"]
    assert "default" in normalized["output_profiles"]


def test_collision_validation_does_not_sync_registry(
    service: ChatMacrosService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.list_macros()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("collision validation must be read-only")

    monkeypatch.setattr(service.repository, "upsert_registry_entry", fail_write)
    monkeypatch.setattr(service.repository, "mark_registry_entries_deleted_except", fail_write)

    with pytest.raises(MacroValidationError, match="another macro"):
        service.create_macro("wrapup", _user_macro_yaml("wrapup", "wrapup"))


def test_service_rejects_non_empty_future_permissions(service: ChatMacrosService) -> None:
    raw = (
        "schema_version: 1\n"
        "name: bad\n"
        "command: bad\n"
        "permissions:\n"
        "  skills: [python]\n"
        "steps: []\n"
    )

    with pytest.raises(MacroValidationError, match="skills"):
        service.validate_macro(raw)


def test_output_profiles_resolve_from_settings_and_render_default_order(service: ChatMacrosService) -> None:
    assert DEFAULT_OUTPUT_PROFILE.sections == [
        "summary",
        "decisions",
        "action_items",
        "open_questions",
        "failed_branches",
    ]

    service.save_settings(
        {
            "output_profiles": {
                "questions_first": {
                    "format": "structured_sections",
                    "sections": ["open_questions", "summary"],
                }
            }
        }
    )

    profile = service.resolve_output_profile("questions_first", local_overrides={"include_branch_outputs": True})
    assert profile.sections == ["open_questions", "summary"]
    assert profile.include_branch_outputs is True

    rendered = render_output_profile(
        DEFAULT_OUTPUT_PROFILE,
        {
            "summary": "S",
            "decisions": "D",
            "action_items": "A",
            "open_questions": "Q",
        },
        failed_branches=[{"label": "Research", "error": "timed out"}],
    )

    assert rendered.index("## Summary") < rendered.index("## Decisions")
    assert rendered.index("## Decisions") < rendered.index("## Action Items")
    assert rendered.index("## Action Items") < rendered.index("## Open Questions")
    assert rendered.index("## Open Questions") < rendered.index("## Failed Branches")


def test_output_profile_local_overrides_are_bounded(service: ChatMacrosService) -> None:
    with pytest.raises(MacroValidationError, match="too many sections"):
        service.resolve_output_profile("default", local_overrides={"sections": [f"s{i}" for i in range(20)]})

    with pytest.raises(MacroValidationError, match="invalid output profile format"):
        normalize_output_profile("bad", {"format": "multiple_messages"})

    with pytest.raises(MacroValidationError, match="unknown output profile keys"):
        normalize_output_profile("bad", {"sectons": ["summary"]})


def test_output_profile_renders_custom_section_titles() -> None:
    """Render configured headings in place of generated section titles."""
    profile = normalize_output_profile(
        "handoff",
        {
            "format": "structured_sections",
            "sections": ["summary", "action_items"],
            "section_titles": {
                "summary": "Executive brief",
                "action_items": "Owners and dates",
            },
        },
    )

    rendered = render_output_profile(profile, {"summary": "S", "action_items": "A"})

    assert "## Executive brief" in rendered
    assert "## Owners and dates" in rendered


def test_output_profile_rejects_titles_for_unknown_sections() -> None:
    """Reject headings that do not belong to a configured output section."""
    with pytest.raises(MacroValidationError, match="unknown section"):
        normalize_output_profile(
            "bad",
            {"sections": ["summary"], "section_titles": {"risks": "Risk register"}},
        )


@pytest.mark.parametrize("title", ["", "   ", "\t\n", "x" * 129])
def test_output_profile_rejects_blank_or_oversized_titles(title: str) -> None:
    """Reject unusable headings at the backend validation boundary."""
    with pytest.raises(MacroValidationError, match="section title"):
        normalize_output_profile("bad", {"sections": ["summary"], "section_titles": {"summary": title}})


def test_output_profile_trims_custom_heading() -> None:
    """Persist a trimmed heading so rendered Markdown has useful text."""
    profile = normalize_output_profile("brief", {"section_titles": {"summary": "  Brief  "}})
    assert profile.section_titles == {"summary": "Brief"}


@pytest.mark.parametrize(
    "legacy_profile",
    [
        {"sections": []},
        {"sections": ["summary"], "section_titles": {"summary": " \t "}},
    ],
)
def test_legacy_empty_profiles_remain_readable_and_editable(
    service: ChatMacrosService,
    legacy_profile: dict,
) -> None:
    """Upgrade previously accepted empty values without relaxing new writes."""
    original = {"output_profiles": {"default": legacy_profile}, "future_setting": True}
    service.repository.save_settings("1", original)

    profile = service.get_settings()["output_profiles"]["default"]
    assert profile["sections"]
    assert profile["section_titles"] == {}
    assert service.repository.get_settings("1") == original
    assert service.list_macros()
    service.create_macro("daily_digest", _user_macro_yaml())
    service.set_macro_enabled("daily_digest", False)
    service.set_builtin_enabled("wrapup", False)
    service.delete_macro("daily_digest")
    assert service.get_settings()["future_setting"] is True
    with pytest.raises(MacroValidationError):
        service.save_output_profiles({"default": legacy_profile})


@pytest.mark.parametrize("format_name", ["single_response", "structured_sections"])
def test_output_profile_requires_at_least_one_section(format_name: str) -> None:
    """Neither response format may silently suppress all selected output."""
    with pytest.raises(MacroValidationError, match="at least one section"):
        normalize_output_profile("empty", {"format": format_name, "sections": []})


def test_save_profiles_preserves_current_settings_and_other_users(service: ChatMacrosService) -> None:
    """Profile-only saves retain current toggles and remain user scoped."""
    service.save_settings({"future_authoring": {"enabled": True}})
    service.set_builtin_enabled("wrapup", False)
    service.repository.save_settings("2", {"future_authoring": "other user"})

    saved = service.save_output_profiles({"Review-Notes": {"sections": ["summary"]}})

    assert saved["disabled_builtins"] == ["wrapup"]
    assert saved["future_authoring"] == {"enabled": True}
    assert "Review-Notes" in saved["output_profiles"]
    assert service.repository.get_settings("2") == {"future_authoring": "other user"}


def test_single_response_output_includes_failed_branches() -> None:
    profile = normalize_output_profile(
        "single",
        {"format": "single_response", "sections": ["summary", "failed_branches"]},
    )

    rendered = render_output_profile(
        profile,
        {"summary": "Done."},
        failed_branches=[{"label": "Risks", "error": "timed out"}],
    )

    assert "Done." in rendered
    assert "Risks: timed out" in rendered


def test_list_macros_does_not_rewrite_an_unchanged_registry(
    service: ChatMacrosService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.list_macros()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unchanged catalog should not write registry rows")

    monkeypatch.setattr(service.repository, "upsert_registry_entry", fail_write)
    monkeypatch.setattr(service.repository, "mark_registry_entries_deleted_except", fail_write)

    assert [item.name for item in service.list_macros()] == ["wrapup"]


def test_get_macro_does_not_rewrite_an_unchanged_registry(
    service: ChatMacrosService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service.list_macros()

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unchanged macro reads should not write registry rows")

    monkeypatch.setattr(service.repository, "upsert_registry_entry", fail_write)

    assert service.get_macro("wrapup").name == "wrapup"
