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
def test_check_text_returns_sanitized_snippet_not_pattern():
    svc = ModerationService()
    lines = [
        "/token\\s*[=:]\\s*([A-Za-z0-9_-]{8,})/ -> block #confidential",
        "secret -> redact:[MASK] #pii",
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
            categories_enabled={"pii", "confidential"},
        )
        text = "please do not reveal token=ABCDEFGH and also keep this secret safe"
        flagged, sample = svc.check_text(text, pol)
        assert flagged is True
        assert sample is not None
        # The sample is sanitized; should not include the actual token or word 'secret'
        assert "ABCDEFGH" not in sample
        assert "secret" not in sample
        # But it should include the redaction marker
        assert ("[MASK]" in sample) or ("[REDACTED]" in sample)
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_evaluate_text_returns_structured_result_with_sample():
    svc = ModerationService()
    lines = [
        "/token\\s*[=:]\\s*([A-Za-z0-9_-]{8,})/ -> block #confidential",
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
            categories_enabled={"confidential"},
        )
        text = "please do not reveal token=ABCDEFGH in logs"
        result = svc.evaluate_text(text, pol, phase="input")
        assert result.action == "block"
        assert result.matched_pattern
        assert result.match_span == (21, 35)
        assert result.sample is not None
        assert "[REDACTED]" in result.sample
        assert "ABCDEFGH" not in result.sample
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_check_text_does_not_compute_full_redaction_for_redact_rule(monkeypatch):
    svc = ModerationService()
    lines = [
        "secret -> redact:[MASK] #pii",
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

        def _fail_redact(*_args, **_kwargs):
            raise AssertionError("check_text should not call redact_text")

        monkeypatch.setattr(svc, "redact_text", _fail_redact)

        flagged, sample = svc.check_text("secret", pol, phase="input")
        assert flagged is True
        assert sample is not None
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_check_text_respects_phase_enablement():
    svc = ModerationService()
    lines = [
        "secret -> block",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        path = tmp.name
    try:
        rules = svc._load_block_patterns(path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=False,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
            categories_enabled=None,
        )
        flagged_in, _ = svc.check_text("secret", pol, phase="input")
        flagged_out, _ = svc.check_text("secret", pol, phase="output")
        assert flagged_in is False
        assert flagged_out is True
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_check_text_detects_long_match_across_window():
    svc = ModerationService()
    lines = [
        "/A.*B/ -> block",
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
            categories_enabled=None,
        )
        svc._max_scan_chars = 50
        svc._match_window_chars = 5
        text = ("x" * 40) + "A" + ("x" * 500) + "B"
        flagged, _ = svc.check_text(text, pol, phase="input")
        assert flagged is True
    finally:
        try:
            os.unlink(path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_rule_phase_input_only_does_not_trigger_output():
    svc = ModerationService()
    rule = PatternRule(
        regex=re.compile(r"danger", re.IGNORECASE),
        action="block",
        phase="input",
    )
    pol = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="warn",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[rule],
        categories_enabled=None,
    )

    flagged_input, _ = svc.check_text("danger", pol, phase="input")
    flagged_output, _ = svc.check_text("danger", pol, phase="output")
    input_action, _, _, _ = svc.evaluate_action("danger", pol, "input")
    output_action, _, _, _ = svc.evaluate_action("danger", pol, "output")

    assert flagged_input is True
    assert flagged_output is False
    assert input_action == "block"
    assert output_action == "pass"
