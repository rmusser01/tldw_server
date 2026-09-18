"""Self-contained saved behavior projection, without live source lookup."""

from copy import deepcopy

import pytest

from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
from tldw_Server_API.tests.DB_Management.test_character_behavior_snapshot_migration import _snapshot


def state():
    return {
        "conversation": {"character_id": 1, "assistant_kind": "character", "assistant_id": "1"},
        "settings": None,
        "behavior_snapshot": {
            "status": "valid",
            "payload": _snapshot().payload,
            "schema_version": 1,
            "digest": _snapshot().digest,
        },
    }


def test_saved_character_projection_preserves_all_self_contained_prompt_fields():
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    captured = state()
    result = project_history_context(captured, character_override="1")
    prompt = result[0]["system_prompt"]
    assert all(
        text in prompt
        for text in (
            "Stay in character.",
            "A stored character.",
            "Careful and precise.",
            "A migration test.",
            "Character: Hello",
            "Preserve history.",
        )
    )
    captured["behavior_snapshot"]["payload"]["participants"][0]["identity"]["name"] = "changed"
    assert result[0]["name"] == "Legacy Character"


@pytest.mark.parametrize("field", ["world_books", "exemplars", "default_memory", "generation_defaults"])
def test_unbound_saved_behavior_is_explicitly_unsupported(field):
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    captured = deepcopy(state())
    captured["behavior_snapshot"]["payload"]["participants"][0][field] = (
        [{"required": True}] if field in {"world_books", "exemplars"} else {"required": True}
    )
    with pytest.raises(HistorySelectionError, match="unsupported_history_context"):
        project_history_context(captured)


def test_neutral_projection_and_character_override_mismatch():
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    neutral = {"conversation": {}, "settings": {}, "behavior_snapshot": {"status": "missing"}}
    assert project_history_context(neutral) == (None, None, {"assistant_kind": None, "assistant_id": None})
    with pytest.raises(HistorySelectionError, match="assistant_mismatch"):
        project_history_context(state(), character_override="2")


def test_saved_materialized_custom_preset_and_sampling_supersede_base():
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    captured = state()
    captured["settings"] = {"participantCharacterIds": [1], "presetScope": "chat", "roleplayBehaviorV1": {}}
    captured["materialized_settings"] = {
        "values": {
            "base_snapshot": {"schema_version": 1, "digest": captured["behavior_snapshot"]["digest"]},
            "behavior_controls": {"turn_taking_mode": "single"},
            "prompt_preset": {
                "section_order": ["only"],
                "section_templates": {"only": "SAVED {{char}}: {{description}}"},
            },
            "effective_completion": {"provider": "openai", "model": "saved", "sampling": {"temperature": 0.2}},
        }
    }
    card, _, context = project_history_context(captured)
    assert card["system_prompt"] == "SAVED Legacy Character: A stored character."
    assert context["history_sampling"] == {"temperature": 0.2}


@pytest.mark.asyncio
async def test_current_input_image_loss_is_rejected_before_admission():
    from fastapi import HTTPException

    from tldw_Server_API.app.core.Chat.persistence_service import prepare_native_history_message

    async def dropping_processor(*args):
        return ["text survives"], [(b"one", "image/png")]

    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "text survives"},
            {"type": "image_url", "image_url": {"url": "https://example.test/one.png"}},
            {"type": "image_url", "image_url": {"url": "https://example.test/two.png"}},
        ],
    }
    with pytest.raises(HTTPException) as failure:
        await prepare_native_history_message(message, "conversation", dropping_processor)
    assert failure.value.status_code == 422


@pytest.mark.parametrize("layout", ["absent", "disabled", "hidden", "nested", "symlink"])
def test_read_only_skill_absence_ignores_nonvisible_layouts(tmp_path, layout):
    from tldw_Server_API.app.core.Chat.history_context import require_history_skill_absence

    root = tmp_path / "skills"
    if layout != "absent":
        directory = root / (".trash/ignored" if layout == "hidden" else "example")
        directory.mkdir(parents=True)
        if layout == "nested":
            directory = directory / "resources"
            directory.mkdir()
        content = "---\nuser-invocable: false\n---\ntext" if layout == "disabled" else "instructions"
        if layout == "symlink":
            target = tmp_path / "target.md"
            target.write_text(content)
            (directory / "SKILL.md").symlink_to(target)
        else:
            (directory / "SKILL.md").write_text(content)
    require_history_skill_absence(tmp_path, registry_may_be_visible=False)
    assert root.exists() is (layout != "absent")


@pytest.mark.parametrize("cause", ["installed", "registry", "recovery"])
def test_potentially_visible_or_unresolved_skills_gate_without_sync(tmp_path, cause):
    from tldw_Server_API.app.core.Chat.history_context import require_history_skill_absence

    if cause != "registry":
        directory = tmp_path / "skills" / ("example" if cause == "installed" else ".replacing-example.v1")
        directory.mkdir(parents=True)
        (directory / "SKILL.md").write_text("instructions")
    with pytest.raises(HistorySelectionError, match="unsupported_history_context_skills"):
        require_history_skill_absence(tmp_path, registry_may_be_visible=cause == "registry")


@pytest.mark.parametrize(
    "binding", [{"schema_version": 1, "digest": "wrong"}, {"schema_version": 2, "digest": "same"}, None]
)
def test_materialized_context_requires_exact_validated_base_snapshot(binding):
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    captured = state()
    captured["materialized_settings"] = {"values": {"base_snapshot": binding}}
    with pytest.raises(HistorySelectionError, match="materialized_binding"):
        project_history_context(captured)


@pytest.mark.parametrize(
    "note, unsupported",
    [
        ({"enabled": True, "text": "REQUIRED NOTE"}, True),
        ({"enabled": False, "text": "disabled"}, False),
        ({"enabled": True, "exclude_from_prompt": True, "text": "excluded"}, False),
        ({"enabled": True, "gm_only": False, "exclude_from_prompt": False, "position": "before_system"}, False),
    ],
)
def test_materialized_author_note_gates_only_substantive_active_state(note, unsupported):
    from tldw_Server_API.app.core.Chat.history_context import project_history_context

    captured = state()
    captured["materialized_settings"] = {
        "values": {
            "base_snapshot": {"schema_version": 1, "digest": captured["behavior_snapshot"]["digest"]},
            "behavior_controls": {"author_note": note},
        }
    }
    if unsupported:
        with pytest.raises(HistorySelectionError, match="author_note"):
            project_history_context(captured)
    else:
        assert project_history_context(captured)[0]["name"] == "Legacy Character"
