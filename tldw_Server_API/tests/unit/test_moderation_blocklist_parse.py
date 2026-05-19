import os
import re
import tempfile
from typing import Optional, Set

import pytest

import tldw_Server_API.app.core.Moderation.moderation_service as moderation_service_module
from tldw_Server_API.app.core.Moderation.moderation_service import ModerationService, ModerationPolicy, PatternRule


@pytest.mark.unit
def test_parse_line_with_categories_suffix_after_action():
    svc = ModerationService()
    # Format: pattern -> action #cats
    expr, action, repl, cats = svc._parse_rule_line("/leak\\d+/ -> block #pii,confidential")
    assert expr == "/leak\\d+/"
    assert action == "block"
    assert repl is None
    assert cats == {"pii", "confidential"}

    expr2, action2, repl2, cats2 = svc._parse_rule_line("secret token -> redact:[MASK] #pii")
    assert expr2 == "secret token"
    assert action2 == "redact"
    assert repl2 == "[MASK]"
    assert cats2 == {"pii"}


@pytest.mark.unit
def test_parse_regex_with_arrow_inside_pattern():
    svc = ModerationService()
    expr, action, repl, cats = svc._parse_rule_line(r"/a->b/ -> block")
    assert expr == "/a->b/"
    assert action == "block"
    assert repl is None
    assert cats is None


@pytest.mark.unit
def test_literal_starting_with_slash_is_not_regex():
    svc = ModerationService()
    lines = [
        "/etc/passwd",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(enabled=True, block_patterns=rules)
        flagged, _ = svc.check_text("path /etc/passwd here", pol)
        assert flagged is True
        flagged2, _ = svc.check_text("etc passwd", pol)
        assert flagged2 is False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_load_block_patterns_and_evaluate_actions():
    svc = ModerationService()
    # Create a temporary blocklist with categories after action
    lines = [
        "/forbidden/ -> block #confidential",
        "secret -> redact:[MASK] #pii",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name

    try:
        rules = svc._load_block_patterns(tmp_path)
        assert isinstance(rules, list) and len(rules) == 2
        # Ensure actions and categories are bound
        r0: PatternRule = rules[0]
        r1: PatternRule = rules[1]
        assert isinstance(r0, PatternRule) and isinstance(r1, PatternRule)
        assert r0.action == "block" and r0.categories == {"confidential"}
        assert r1.action == "redact" and r1.replacement == "[MASK]" and r1.categories == {"pii"}

        # Build a policy to evaluate
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

        # Block action fires on input
        act, red, sample, cat = svc.evaluate_action("this is forbidden", pol, "input")
        assert act == "block" and sample and (cat == "confidential")

        # Redact action applies on output
        act2, red2, sample2, cat2 = svc.evaluate_action("found secret token", pol, "output")
        assert act2 == "redact" and red2 and "[MASK]" in red2
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None

@pytest.mark.unit
def test_warn_with_categories_and_category_label():
    svc = ModerationService()
    # Warning rule with category
    lines = [
        "/minor issue/ -> warn #confidential",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name

    try:
        rules = svc._load_block_patterns(tmp_path)
        assert len(rules) == 1
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
        act, red, sample, cat = svc.evaluate_action("this is a minor issue in docs", pol, "input")
        assert act == "warn" and sample and cat == "confidential"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_invalid_and_dangerous_regex_lines_are_skipped():
    svc = ModerationService()
    lines = [
        "validliteral",
        "/(unclosed/ -> block #pii",  # invalid regex
        "/(a+)+$/ -> block",           # nested quantifiers (dangerous)
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        # Only the literal should survive
        assert len(rules) == 1
        # Ensure it actually matches
        pol = ModerationPolicy(enabled=True, block_patterns=rules)
        flagged, sample = svc.check_text("This contains validliteral term", pol)
        assert flagged and sample
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_blocklist_pattern_warnings_sanitize_raw_regex_details(tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    blocklist_path.write_text(
        "\n".join(
            [
                "/(private_secret/ -> block #pii",
                "/(private_secret+)+$/ -> block #pii",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        rules = svc._load_block_patterns(str(blocklist_path))
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert rules == []
    assert "Invalid blocklist pattern" in joined
    assert "Skipped dangerous regex in blocklist" in joined
    assert "private_secret" not in joined
    assert "missing )" not in joined


@pytest.mark.unit
def test_blocklist_invalid_action_warning_sanitizes_line_details(tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    blocklist_path.write_text(
        "private_secret -> exfiltrate:/private/moderation_blocklist.txt #pii\n",
        encoding="utf-8",
    )

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        rules = svc._load_block_patterns(str(blocklist_path))
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert rules == []
    assert "Invalid moderation action in blocklist; skipping line" in joined
    assert "exfiltrate" not in joined
    assert "private_secret" not in joined
    assert "moderation_blocklist.txt" not in joined


@pytest.mark.unit
def test_load_block_patterns_missing_file_sanitizes_warning_path(tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "private_moderation_blocklist.txt"

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        rules = svc._load_block_patterns(str(blocklist_path))
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert rules == []
    assert "Moderation blocklist file not found" in joined
    assert "private_moderation_blocklist.txt" not in joined
    assert str(tmp_path) not in joined


@pytest.mark.unit
def test_load_block_patterns_sanitizes_outer_failure_log(monkeypatch, tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    blocklist_path.write_text("secret\n", encoding="utf-8")

    def _raise_load_failure(*_args, **_kwargs):
        raise OSError("blocklist load failed at /private/moderation_blocklist.txt")

    monkeypatch.setattr(moderation_service_module, "open", _raise_load_failure, raising=False)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        rules = svc._load_block_patterns(str(blocklist_path))
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert rules == []
    assert "Failed to load moderation blocklist" in joined
    assert "blocklist load failed" not in joined
    assert "moderation_blocklist.txt" not in joined


@pytest.mark.unit
def test_build_block_patterns_sanitizes_builtin_pii_failure_log(monkeypatch):
    svc = ModerationService()
    svc._pii_enabled = True

    def _raise_pii_failure():
        raise RuntimeError("builtin pii load failed at /private/pii-rules.py")

    monkeypatch.setattr(svc, "_load_builtin_pii_rules", _raise_pii_failure)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        patterns = svc._build_block_patterns(None)
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert patterns == []
    assert "Failed to load builtin PII rules" in joined
    assert "builtin pii load failed" not in joined
    assert "pii-rules.py" not in joined


@pytest.mark.unit
def test_get_blocklist_lines_sanitizes_read_failure_log(monkeypatch, tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    blocklist_path.write_text("secret\n", encoding="utf-8")
    svc._blocklist_path = str(blocklist_path)

    def _raise_read_failure(*_args, **_kwargs):
        raise OSError("blocklist read failed at /private/moderation_blocklist.txt")

    monkeypatch.setattr(moderation_service_module, "open", _raise_read_failure, raising=False)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        lines = svc.get_blocklist_lines()
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert lines == []
    assert "Failed to read blocklist" in joined
    assert "blocklist read failed" not in joined
    assert "moderation_blocklist.txt" not in joined


@pytest.mark.unit
def test_set_blocklist_lines_sanitizes_write_failure_log(monkeypatch, tmp_path):
    svc = ModerationService()
    blocklist_path = tmp_path / "moderation_blocklist.txt"
    svc._blocklist_path = str(blocklist_path)

    def _raise_write_failure(*_args, **_kwargs):
        raise OSError("blocklist write failed at /private/moderation_blocklist.txt")

    monkeypatch.setattr(moderation_service_module.os, "replace", _raise_write_failure)

    messages: list[str] = []
    sink_id = moderation_service_module.logger.add(lambda message: messages.append(str(message)), level="ERROR")
    try:
        ok = svc.set_blocklist_lines(["secret"])
    finally:
        moderation_service_module.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert ok is False
    assert "Failed to write blocklist" in joined
    assert "blocklist write failed" not in joined
    assert "moderation_blocklist.txt" not in joined


@pytest.mark.unit
def test_lint_warns_on_invalid_regex_flags():
    svc = ModerationService()
    res = svc.lint_blocklist_lines(["/foo/z"])
    item = res["items"][0]
    assert item["ok"] is True
    assert item["pattern_type"] == "literal"
    assert "invalid regex flags" in (item.get("warning") or "")


@pytest.mark.unit
def test_lint_rejects_invalid_action():
    svc = ModerationService()
    res = svc.lint_blocklist_lines(["secret -> blok"])
    item = res["items"][0]
    assert item["ok"] is False
    assert "invalid action" in (item.get("error") or "")


@pytest.mark.unit
def test_replacement_limits_are_enforced():
    svc = ModerationService()
    # Create a simple redact rule
    lines = [
        "secret -> redact:[MASK]",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        # Limit to one replacement per pattern
        svc._max_replacements_per_pattern = 1
        text = "secret and another secret and one more secret"
        red = svc.redact_text(text, pol)
        # Exactly one replacement expected with [MASK]
        assert red.count("[MASK]") == 1
        assert red.count("secret") >= 2
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_user_override_empty_categories_clears_gating():
    svc = ModerationService()
    lines = [
        "secret -> block #confidential",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        svc._global_policy = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=True,
            block_patterns=rules,
            categories_enabled={"pii"},
        )
        svc._user_overrides = {"user1": {"categories_enabled": ""}}
        pol = svc.get_effective_policy("user1")
        assert pol.categories_enabled == set()
        act, _, _, cat = svc.evaluate_action("secret", pol, "input")
        assert act == "block"
        assert cat == "confidential"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_blocklist_update_preserves_pii_rules(monkeypatch):
    svc = ModerationService()
    # Force a deterministic PII rule for this test
    pii_rule = PatternRule(regex=re.compile(r"pii", re.IGNORECASE), action="redact", replacement="[PII]", categories={"pii"})
    monkeypatch.setattr(svc, "_load_builtin_pii_rules", lambda: [pii_rule])
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        svc.update_settings(pii_enabled=True)
        svc._blocklist_path = tmp_path
        ok = svc.set_blocklist_lines(["secret -> block"])
        assert ok is True
        assert any(
            isinstance(rule, PatternRule) and rule.categories and "pii" in rule.categories
            for rule in svc._global_policy.block_patterns
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_compile_user_rule_rejects_dangerous_regex():
    svc = ModerationService()
    compiled = svc._compile_user_rule(
        {
            "id": "danger",
            "pattern": "(a+)+$",
            "is_regex": True,
            "action": "block",
            "phase": "both",
        }
    )
    assert compiled is None


@pytest.mark.unit
def test_compile_user_rule_sets_phase_and_action():
    svc = ModerationService()
    compiled = svc._compile_user_rule(
        {
            "id": "warn-1",
            "pattern": "heads up",
            "is_regex": False,
            "action": "warn",
            "phase": "input",
        }
    )
    assert isinstance(compiled, PatternRule)
    assert compiled is not None
    assert compiled.action == "warn"
    assert compiled.phase == "input"
    assert compiled.categories == {"*"}
    assert compiled.regex.search("please heads up now")


@pytest.mark.unit
def test_uncategorized_category_allows_untagged_rules():
    svc = ModerationService()
    lines = [
        "secret -> block",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
            categories_enabled={"uncategorized"},
        )
        act, _red, _pattern, cat = svc.evaluate_action("secret", pol, "input")
        assert act == "block"
        assert cat == "uncategorized"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_evaluate_action_redacts_all_matching_rules():
    svc = ModerationService()
    lines = [
        "secret -> redact:[MASK]",
        "token -> redact:[TOK]",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        act, red, _, _ = svc.evaluate_action("secret token", pol, "output")
        assert act == "redact"
        assert red is not None
        assert "[MASK]" in red
        assert "[TOK]" in red
        assert "secret" not in red
        assert "token" not in red
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_evaluate_action_block_precedence_over_warn():
    svc = ModerationService()
    lines = [
        "secret -> warn #pii",
        "secret -> block #confidential",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="warn",
            output_action="warn",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        act, _, _, cat = svc.evaluate_action("secret", pol, "input")
        assert act == "block"
        assert cat == "confidential"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chunk_scanning_detects_matches_past_max_chars():
    svc = ModerationService()
    lines = [
        "secret -> block",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        svc._max_scan_chars = 10
        text = ("a" * 25) + "secret"
        flagged, sample = svc.check_text(text, pol)
        assert flagged is True
        assert sample is not None
        act, _, _, _ = svc.evaluate_action(text, pol, "input")
        assert act == "block"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chunk_scanning_limits_search_window():
    class _FakeMatch:
        def __init__(self, start: int, end: int):
            self._start = start
            self._end = end

        def start(self):
            return self._start

        def end(self):
            return self._end

        def span(self):
            return self._start, self._end

    class _LenCheckingPattern:
        def __init__(self, needle: str, max_window: int):
            self.needle = needle
            self.max_window = max_window
            self.calls = []

        def search(self, text: str, pos: int = 0, endpos: int | None = None):
            if endpos is None:
                endpos = len(text)
            self.calls.append((pos, endpos))
            assert (endpos - pos) <= self.max_window
            idx = text.find(self.needle, pos, endpos)
            if idx == -1:
                return None
            return _FakeMatch(idx, idx + len(self.needle))

        @property
        def pattern(self):
            return self.needle

    svc = ModerationService()
    svc._max_scan_chars = 20
    svc._match_window_chars = 5
    pat = _LenCheckingPattern("secret", max_window=25)
    pol = ModerationPolicy(
        enabled=True,
        input_enabled=True,
        output_enabled=True,
        input_action="block",
        output_action="redact",
        redact_replacement="[REDACTED]",
        per_user_overrides=False,
        block_patterns=[PatternRule(regex=pat, action="block")],
    )
    text = ("a" * 100) + "secret"
    flagged, _ = svc.check_text(text, pol)
    assert flagged is True
    assert pat.calls


@pytest.mark.unit
def test_set_blocklist_lines_empty_writes_empty_file():
    svc = ModerationService()
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        svc._blocklist_path = tmp_path
        ok = svc.set_blocklist_lines([])
        assert ok is True
        lines = svc.get_blocklist_lines()
        assert lines == []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_append_blocklist_line_rejects_newlines():
    svc = ModerationService()
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        svc._blocklist_path = tmp_path
        expected_version = svc._compute_version([])
        ok, state = svc.append_blocklist_line(expected_version, "secret\nanother")
        assert ok is False
        assert "single-line" in str(state.get("error", ""))
        assert svc.get_blocklist_lines() == []
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_replacement_limit_zero_is_unlimited():
    svc = ModerationService()
    lines = [
        "secret -> redact:[MASK]",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        svc._max_replacements_per_pattern = 0
        text = "secret secret"
        red = svc.redact_text(text, pol)
        assert red.count("[MASK]") == 2
        assert "secret" not in red
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_chunk_boundary_match_window_detects_long_match():
    svc = ModerationService()
    lines = [
        "ABCDEFGHIJKL -> block",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        svc._max_scan_chars = 10
        svc._match_window_chars = 20
        text = "xxxxxABCDEFGHIJKL"
        flagged, sample = svc.check_text(text, pol)
        assert flagged is True
        assert sample is not None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_redaction_replacement_treated_as_literal():
    svc = ModerationService()
    lines = [
        r"secret -> redact:\1",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        text = "secret secret"
        red = svc.redact_text(text, pol)
        assert r"\1" in red
        assert "secret" not in red
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None


@pytest.mark.unit
def test_redact_text_with_count_tracks_replacements():
    svc = ModerationService()
    lines = [
        "secret -> redact:[MASK]",
        "token -> redact:[TOK]",
    ]
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write("\n".join(lines) + "\n")
        tmp_path = tmp.name
    try:
        rules = svc._load_block_patterns(tmp_path)
        pol = ModerationPolicy(
            enabled=True,
            input_enabled=True,
            output_enabled=True,
            input_action="block",
            output_action="redact",
            redact_replacement="[REDACTED]",
            per_user_overrides=False,
            block_patterns=rules,
        )
        red, count = svc.redact_text_with_count("secret token secret", pol)
        assert count == 3
        assert "[MASK]" in red
        assert "[TOK]" in red
        assert "secret" not in red
        assert "token" not in red
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            _ = None
