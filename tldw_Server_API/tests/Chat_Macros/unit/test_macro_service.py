from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Chat_Macros.exceptions import MacroStorageError, MacroValidationError
from tldw_Server_API.app.core.Chat_Macros.output_profiles import (
    DEFAULT_OUTPUT_PROFILE,
    normalize_output_profile,
    render_output_profile,
)
from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.Chat_Macros.service import ChatMacrosService
from tldw_Server_API.app.core.Chat_Macros.storage import ChatMacroStorage
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture()
def raw_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "macros.db"), client_id="test_client")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture()
def service(tmp_path, raw_db):
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


def test_lists_builtin_wrapup_and_can_disable_it(service):
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


def test_clone_builtin_creates_user_macro_with_non_conflicting_command(service):
    cloned = service.clone_builtin("wrapup", new_name="my_wrapup", command="my_wrapup")

    assert cloned.name == "my_wrapup"
    assert cloned.command == "my_wrapup"
    assert cloned.source == "user"
    assert cloned.immutable is False
    assert service.get_macro("my_wrapup").definition.command == "my_wrapup"

    with pytest.raises(MacroValidationError, match="core command"):
        service.clone_builtin("wrapup", new_name="weather_wrapup", command="weather")


def test_create_update_delete_user_macro_and_validate_without_saving(service):
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


def test_service_rejects_non_empty_future_permissions(service):
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


def test_output_profiles_resolve_from_settings_and_render_default_order(service):
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


def test_output_profile_local_overrides_are_bounded(service):
    with pytest.raises(MacroValidationError, match="too many sections"):
        service.resolve_output_profile("default", local_overrides={"sections": [f"s{i}" for i in range(20)]})

    with pytest.raises(MacroValidationError, match="invalid output profile format"):
        normalize_output_profile("bad", {"format": "multiple_messages"})
