import json

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.moderation_schemas import ModerationUserOverride
from tldw_Server_API.app.core.Moderation import moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService


@pytest.mark.unit
def test_set_user_override_rejects_invalid_action(tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "overrides.json"
    svc._user_overrides_path = str(overrides_path)

    res = svc.set_user_override("user1", {"input_action": "blok"})
    assert res["ok"] is False
    assert "invalid input_action" in (res.get("error") or "")
    assert res.get("error_type") == "validation"
    assert overrides_path.exists() is False


@pytest.mark.unit
def test_load_user_overrides_sanitizes_invalid_action(tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(json.dumps({"user1": {"input_action": "blok", "output_action": "warn"}}))

    svc = ModerationService()
    svc._user_overrides_path = str(overrides_path)
    loaded = svc._load_user_overrides()

    user_override = loaded.get("user1") or {}
    assert "input_action" not in user_override
    assert user_override.get("output_action") == "warn"


@pytest.mark.unit
def test_sanitize_user_override_invalid_action_warning_sanitizes_value():
    svc = ModerationService()

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        sanitized = svc._sanitize_user_override(
            {
                "input_action": "exfiltrate /private/user-overrides.json",
                "output_action": "warn",
            }
        )
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "input_action" not in sanitized
    assert sanitized.get("output_action") == "warn"
    assert "Invalid moderation override action" in joined
    assert "exfiltrate" not in joined
    assert "input_action" not in joined
    assert "user-overrides.json" not in joined


@pytest.mark.unit
def test_set_user_override_rejects_invalid_rule_action(tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "overrides.json"
    svc._user_overrides_path = str(overrides_path)

    res = svc.set_user_override(
        "user1",
        {
            "rules": [
                {
                    "id": "bad",
                    "pattern": "x",
                    "is_regex": False,
                    "action": "redact",
                    "phase": "both",
                }
            ]
        },
    )
    assert res["ok"] is False
    assert "invalid rule action" in (res.get("error") or "")
    assert res.get("error_type") == "validation"
    assert overrides_path.exists() is False


@pytest.mark.unit
def test_set_user_override_rejects_non_boolean_rule_is_regex(tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "overrides.json"
    svc._user_overrides_path = str(overrides_path)

    res = svc.set_user_override(
        "user1",
        {
            "rules": [
                {
                    "id": "bad-type",
                    "pattern": "x",
                    "is_regex": "false",
                    "action": "warn",
                    "phase": "both",
                }
            ]
        },
    )
    assert res["ok"] is False
    assert "invalid rule is_regex" in (res.get("error") or "")
    assert res.get("error_type") == "validation"
    assert overrides_path.exists() is False


@pytest.mark.unit
def test_set_user_override_persist_failure_does_not_mutate_live_state(monkeypatch, tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "overrides.json"
    svc._user_overrides_path = str(overrides_path)
    svc._user_overrides = {"user1": {"input_action": "warn", "output_action": "redact"}}
    original_override = json.loads(json.dumps(svc._user_overrides["user1"]))

    def _raise_disk_full(*_args, **_kwargs):
        raise OSError("disk full at /private/moderation-overrides.json")

    monkeypatch.setattr(ModerationService, "_write_json_atomic", _raise_disk_full, raising=False)

    res = svc.set_user_override("user1", {"input_action": "block"})

    assert res["ok"] is False
    assert res["error_type"] == "persistence"
    assert res.get("error") == "Failed to persist user override."
    assert "disk full" not in (res.get("error") or "")
    assert "/private/moderation-overrides.json" not in (res.get("error") or "")
    assert svc._user_overrides["user1"] == original_override


@pytest.mark.unit
def test_delete_user_override_persist_failure_does_not_mutate_live_state(monkeypatch, tmp_path):
    svc = ModerationService()
    overrides_path = tmp_path / "overrides.json"
    svc._user_overrides_path = str(overrides_path)
    svc._user_overrides = {"user1": {"input_action": "warn", "output_action": "redact"}}
    original_override = json.loads(json.dumps(svc._user_overrides["user1"]))

    def _raise_disk_full(*_args, **_kwargs):
        raise OSError("disk full at /private/moderation-overrides.json")

    monkeypatch.setattr(ModerationService, "_write_json_atomic", _raise_disk_full, raising=False)

    res = svc.delete_user_override("user1")

    assert res["ok"] is False
    assert res["persisted"] is False
    assert res.get("error_type") == "persistence"
    assert res.get("error") == "Failed to delete user override."
    assert "disk full" not in (res.get("error") or "")
    assert "/private/moderation-overrides.json" not in (res.get("error") or "")
    assert svc._user_overrides["user1"] == original_override


@pytest.mark.unit
def test_load_user_overrides_drops_invalid_rules_but_keeps_valid_entries(tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "alice": {
                    "rules": [
                        {
                            "id": "bad",
                            "pattern": "(",
                            "is_regex": True,
                            "action": "block",
                            "phase": "both",
                        },
                        {
                            "id": "ok",
                            "pattern": "safe",
                            "is_regex": False,
                            "action": "warn",
                            "phase": "both",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    svc = ModerationService()
    svc._user_overrides_path = str(overrides_path)
    loaded = svc._load_user_overrides()

    assert loaded["alice"]["rules"] == [
        {
            "id": "ok",
            "pattern": "safe",
            "is_regex": False,
            "action": "warn",
            "phase": "both",
        }
    ]


@pytest.mark.unit
def test_load_user_overrides_parses_string_boolean_for_is_regex(tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text(
        json.dumps(
            {
                "alice": {
                    "rules": [
                        {
                            "id": "ok",
                            "pattern": "safe",
                            "is_regex": "false",
                            "action": "warn",
                            "phase": "both",
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    svc = ModerationService()
    svc._user_overrides_path = str(overrides_path)
    loaded = svc._load_user_overrides()

    assert loaded["alice"]["rules"] == [
        {
            "id": "ok",
            "pattern": "safe",
            "is_regex": False,
            "action": "warn",
            "phase": "both",
        }
    ]


@pytest.mark.unit
def test_load_user_overrides_sanitizes_read_failure_log(monkeypatch, tmp_path):
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_text("{}", encoding="utf-8")

    svc = ModerationService()
    svc._user_overrides_path = str(overrides_path)

    def _raise_read_failure(*_args, **_kwargs):
        raise OSError("user overrides load failed at /private/moderation-overrides.json")

    monkeypatch.setattr(moderation_service_module, "open", _raise_read_failure, raising=False)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        loaded = svc._load_user_overrides()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert loaded == {}
    assert "Failed to load user overrides" in joined
    assert "user overrides load failed" not in joined
    assert "moderation-overrides.json" not in joined


@pytest.mark.unit
def test_load_user_overrides_missing_file_sanitizes_info_path(tmp_path):
    overrides_path = tmp_path / "private_user_overrides.json"

    svc = ModerationService()
    svc._user_overrides_path = str(overrides_path)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="INFO")
    try:
        loaded = svc._load_user_overrides()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert loaded == {}
    assert "Moderation user overrides file not found" in joined
    assert "private_user_overrides.json" not in joined
    assert str(tmp_path) not in joined


@pytest.mark.unit
def test_compile_user_rule_sanitizes_rule_id_warning_logs():
    svc = ModerationService()
    rules = [
        {
            "id": "secret_invalid_bool_/private/user-overrides.json",
            "pattern": "safe",
            "is_regex": "maybe",
            "action": "warn",
            "phase": "both",
        },
        {
            "id": "secret_dangerous_regex_/private/user-overrides.json",
            "pattern": "(a+)+$",
            "is_regex": True,
            "action": "warn",
            "phase": "both",
        },
        {
            "id": "secret_invalid_regex_/private/user-overrides.json",
            "pattern": "(",
            "is_regex": True,
            "action": "warn",
            "phase": "both",
        },
    ]

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        compiled = [svc._compile_user_rule(rule) for rule in rules]
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert compiled == [None, None, None]
    assert "Skipped per-user rule with invalid is_regex" in joined
    assert "Skipped dangerous per-user regex rule" in joined
    assert "Skipped invalid per-user regex rule" in joined
    assert "secret_" not in joined
    assert "user-overrides.json" not in joined


@pytest.mark.unit
def test_user_override_rules_schema_accepts_block_and_warn_with_phase():
    model = ModerationUserOverride(
        enabled=True,
        rules=[
            {
                "id": "r1",
                "pattern": "bad phrase",
                "is_regex": False,
                "action": "block",
                "phase": "both",
            },
            {
                "id": "r2",
                "pattern": "warn\\s+me",
                "is_regex": True,
                "action": "warn",
                "phase": "input",
            },
        ],
    )
    assert len(model.rules or []) == 2


@pytest.mark.unit
def test_user_override_rules_schema_rejects_invalid_phase():
    with pytest.raises(ValidationError):
        ModerationUserOverride(
            rules=[
                {
                    "id": "r1",
                    "pattern": "x",
                    "is_regex": False,
                    "action": "block",
                    "phase": "sideways",
                }
            ]
        )
