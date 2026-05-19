import os
import re
import tempfile

import pytest

from tldw_Server_API.app.core.Moderation.moderation_service import (
    ModerationPolicy,
    ModerationService,
    PatternRule,
)


@pytest.mark.unit
def test_redact_text_respects_categories_enabled():
    svc = ModerationService()
    lines = [
        "secret -> redact:[MASK] #pii",
        "confidential -> redact:[CENSORED] #confidential",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        path = tmp.name
    try:
        rules = svc._load_block_patterns(path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
            categories_enabled={"pii"},
        )
        text = "secret and confidential info"
        red = svc.redact_text(text, pol)
        # Only the PII-tagged rule should apply
        assert "[MASK]" in red
        assert "confidential" in red  # should not be redacted
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_input_only_rule_does_not_redact_output_phase():
    svc = ModerationService()
    rule = PatternRule(
        regex=re.compile(r"danger", re.IGNORECASE),
        action="redact",
        replacement="[MASK]",
        phase="input",
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
        categories_enabled=None,
    )

    assert svc.redact_text("danger", pol, phase="output") == "danger"
    assert svc.redact_text_with_count("danger", pol, phase="output") == ("danger", 0)
