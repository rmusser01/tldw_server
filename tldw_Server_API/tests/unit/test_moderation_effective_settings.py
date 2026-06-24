import re

import pytest

import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    ModerationService,
    PatternRule,
)


def _tmp_moderation_config(tmp_path, blocklist_path):
    return {
        "moderation": {
            "enabled": "true",
            "input_enabled": "true",
            "output_enabled": "true",
            "input_action": "block",
            "output_action": "redact",
            "redact_replacement": "[REDACTED]",
            "per_user_overrides": "true",
            "categories_enabled": "confidential",
            "pii_enabled": "false",
            "blocklist_file": str(blocklist_path),
            "user_overrides_file": str(tmp_path / "moderation_user_overrides.json"),
            "runtime_overrides_file": str(tmp_path / "moderation_runtime_overrides.json"),
        }
    }


@pytest.mark.unit
def test_effective_pii_respects_categories_enabled():
    svc = ModerationService()
    pii_rule = PatternRule(
        regex=re.compile(r"\\S+@\\S+"),
        action="redact",
        replacement="[PII]",
        categories={"pii", "pii_email"},
    )

    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[pii_rule],
        categories_enabled={"confidential"},
    )
    settings = svc.get_settings()
    assert settings["effective"]["pii_enabled"] is False

    svc._global_policy = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[pii_rule],
        categories_enabled={"pii_email"},
    )
    settings = svc.get_settings()
    assert settings["effective"]["pii_enabled"] is True


@pytest.mark.unit
def test_category_reporting_respects_allowlist():
    svc = ModerationService()
    rule = PatternRule(
        regex=re.compile(r"secret", re.IGNORECASE),
        action="block",
        categories={"pii", "confidential"},
    )
    pol = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[rule],
        categories_enabled={"pii"},
    )
    act, _red, _pattern, cat = svc.evaluate_action("secret", pol, "input")
    assert act == "block"
    assert cat == "pii"


@pytest.mark.unit
def test_update_settings_recompiles_global_policy_with_runtime_categories(monkeypatch, tmp_path):
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    blocklist_path.write_text("runtime-secret -> block #runtime\n", encoding="utf-8")
    monkeypatch.setattr(
        moderation_service_module,
        "load_and_log_configs",
        lambda: _tmp_moderation_config(tmp_path, blocklist_path),
    )

    svc = ModerationService()

    assert svc._global_policy.categories_enabled == {"confidential"}
    act_before, _red_before, _pattern_before, _cat_before = svc.evaluate_action(
        "runtime-secret",
        svc._global_policy,
        "input",
    )
    assert act_before == "pass"

    settings = svc.update_settings(categories_enabled=["runtime"], persist=False)

    assert settings["effective"]["categories_enabled"] == ["runtime"]
    assert svc._global_policy.categories_enabled == {"runtime"}
    act_after, _red_after, _pattern_after, cat_after = svc.evaluate_action(
        "runtime-secret",
        svc._global_policy,
        "input",
    )
    assert act_after == "block"
    assert cat_after == "runtime"
    assert not (tmp_path / "moderation_runtime_overrides.json").exists()
