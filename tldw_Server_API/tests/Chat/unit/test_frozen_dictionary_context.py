"""Detached assigned dictionary plans retain the real replacement semantics."""

from copy import deepcopy

import pytest


def plan(**entry_overrides):
    return {"root_ids": [1], "passes": [{"root_id": 1, "sources": [{"id": 1, "version": 1}],
        "token_budget": None, "entries": [{"id": 10, "dictionary_id": 1, "key": "IN", "content": "OUT",  # nosec B105 - None budget, not a credential
            "is_regex": False, "probability": 1.0, "max_replacements": 1, "group_name": None,
            "timed_effects": {}, "enabled": True, "case_sensitive": True,
            "created_at": "2020-01-01T00:00:00+00:00", **entry_overrides}]}]}


@pytest.mark.parametrize("overrides,text,expected", [
    ({}, "IN IN", "OUT OUT"),
    ({"case_sensitive": False}, "in", "OUT"),
    ({"probability": 0.0}, "IN", "IN"),
    ({"enabled": False}, "IN", "IN"),
    ({"is_regex": True, "key": r"IN-(\d+)", "content": r"OUT-\1"}, "IN-742", "OUT-742"),
    ({"timed_effects": {"delay": 2147483647}}, "IN", "IN"),
])
def test_frozen_current_user_transform_reuses_literal_regex_probability_delay_and_iteration_semantics(overrides, text, expected):
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import transform_frozen_user_content

    incoming = [{"role": "system", "content": "IN"}, {"role": "assistant", "content": "IN"},
                {"role": "user", "content": [{"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]
    before = deepcopy(incoming)
    transformed = transform_frozen_user_content(incoming, plan(**overrides))
    assert transformed == [[{"type": "text", "text": expected},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]]  # nosec B101
    assert incoming == before  # nosec B101


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(root_ids=[True]),
    lambda p: p["passes"][0].update(root_id=2),
    lambda p: p["passes"][0].update(root_id=True),
    lambda p: p["passes"][0].update(root_id=1.0),
    lambda p: p["passes"][0].update(token_budget=0),
    lambda p: p["passes"][0]["entries"][0].update(dictionary_id=2),
    lambda p: p["passes"][0]["entries"][0].update(probability="invalid"),
    lambda p: p["passes"][0]["entries"][0].update(is_regex="false"),
    lambda p: p["passes"][0]["entries"][0].update(timed_effects={"future_effect": 2}),
    lambda p: p["passes"][0]["entries"][0].update(is_regex=True, key="(a+)+$"),
    lambda p: p["passes"][0]["entries"][0].update(key=""),
])
def test_malformed_frozen_plan_cannot_become_executable(mutate):
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import validate_frozen_dictionary_plan
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    data = plan()
    mutate(data)
    with pytest.raises(InputError):
        validate_frozen_dictionary_plan(data, [1])


def test_frozen_default_budget_stops_after_first_oversized_replacement(monkeypatch):
    from tldw_Server_API.app.core.Character_Chat import chat_dictionary
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import transform_frozen_user_content

    data = plan(content="OUT LONG", max_replacements=0)
    data["passes"][0]["token_budget"] = 1
    monkeypatch.setattr(chat_dictionary, "_count_tokens", lambda text: len(text.split()))
    with pytest.warns(chat_dictionary.TokenBudgetExceededWarning):
        transformed = transform_frozen_user_content([{"role": "user", "content": "IN IN"}], data)
    assert transformed == ["OUT LONG IN"]  # nosec B101


@pytest.mark.parametrize("settings,expected", [
    ({"chat_dictionary_id": "2"}, [2]),
    ({"chat_dictionary_ids": ["2", 1, 2], "chat_dictionary_id": 3}, [2, 1]),
    ({"chat_dictionary_ids": [], "chat_dictionary_id": 3}, []),
    ({"conversationContext": {"chat_dictionary_ids": [3]}, "chat_dictionary_ids": [2]}, [3]),
])
def test_saved_assignment_writer_precedence_and_clear_are_preserved(settings, expected):
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import assigned_dictionary_ids

    assert assigned_dictionary_ids(settings) == expected  # nosec B101


@pytest.mark.parametrize("settings", [
    {"chat_dictionary_ids": [True]}, {"chat_dictionary_ids": [0]},
    {"chat_dictionary_ids": [1.5]}, {"chat_dictionary_ids": "1"},
    {"chat_dictionary_id": "missing"}, {"conversationContext": {"chat_dictionary_ids": None}},
    {"chat_dictionary_id": "²"}, {"chat_dictionary_ids": ["9" * 5000]},
])
def test_malformed_authoritative_assignment_is_refused(settings):
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import assigned_dictionary_ids
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    with pytest.raises(InputError):
        assigned_dictionary_ids(settings)


def test_oversized_frozen_plan_is_refused_before_processing():
    from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import DEFAULT_MAX_SNAPSHOT_BYTES
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import validate_frozen_dictionary_plan
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    with pytest.raises(InputError, match="snapshot byte limit"):
        validate_frozen_dictionary_plan(plan(content="x" * DEFAULT_MAX_SNAPSHOT_BYTES), [1])


@pytest.mark.parametrize("target,field,value", [
    ("metadata", "included_dictionary_ids", "not-json"),
    ("metadata", "included_dictionary_ids", "[true]"),
    ("metadata", "is_active", "false"),
    ("metadata", "default_token_budget", -1),
    ("entry", "probability", "not-numeric"),
    ("entry", "timed_effects", "not-json"),
    ("entry", "is_regex", 1.0),
])
def test_malformed_stored_catalog_is_refused_at_capture_boundary(target, field, value):
    """Inject a corrupt database row only; retain the real capture and validator."""
    from unittest.mock import Mock

    from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    metadata = {"id": 1, "version": 1, "is_active": 1, "default_token_budget": None,  # nosec B105 - None budget, not a credential
                "included_dictionary_ids": "[]"}
    entry = plan()["passes"][0]["entries"][0]
    entry["timed_effects"] = "{}"
    (metadata if target == "metadata" else entry)[field] = value
    conn = Mock()
    conn.execute.side_effect = [Mock(fetchone=lambda: metadata),
                               Mock(fetchone=lambda: {"entry_count": 1, "entry_bytes": 256}),
                               Mock(fetchall=lambda: [entry])]
    with pytest.raises(InputError):
        ChatDictionaryService.snapshot_assigned_dictionaries(conn, [1])
