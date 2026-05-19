# Stage 2 Policy and Rule Parsing

## Scope
Review moderation policy loading, override merging, blocklist parsing, rule application, redaction behavior, and sanitized snippet handling inside `moderation_service.py`.

## Files Reviewed
- `Docs/Code_Documentation/Moderation-Guardrails.md`
- `tldw_Server_API/app/core/Moderation/README.md`
- `tldw_Server_API/Config_Files/moderation_blocklist.txt`
- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- `tldw_Server_API/tests/unit/test_moderation_env_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`

## Tests Reviewed
- `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- `tldw_Server_API/tests/unit/test_moderation_env_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- Narrow policy and parsing verification slice passed: `36 passed, 86 warnings in 2.26s`

## Validation Commands
- `nl -ba tldw_Server_API/app/core/Moderation/moderation_service.py | sed -n '840,1135p'`
- `nl -ba tldw_Server_API/app/core/Moderation/moderation_service.py | sed -n '1170,1205p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py | sed -n '90,160p'`
- `nl -ba tldw_Server_API/tests/unit/test_moderation_redact_categories.py | sed -n '1,220p'`
- `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py tldw_Server_API/tests/unit/test_moderation_effective_settings.py tldw_Server_API/tests/unit/test_moderation_env_parse.py tldw_Server_API/tests/unit/test_moderation_redact_categories.py tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- `source ../../.venv/bin/activate && python -c '<reproduce output-phase redaction with an input-only rule and an output-only redact rule>'`
- `source ../../.venv/bin/activate && python -c '<reproduce long-span /A.*B/ miss with _max_scan_chars=10 and _match_window_chars=5>'`

## Confirmed Findings
- Medium, high confidence: phase-specific rule gating is enforced during detection, but not during redaction transforms. `check_text()` and `_evaluate_action_internal()` both call `_rule_applies_to_phase()` before matching rules, but `redact_text()` and `redact_text_with_count()` only enforce category gating and then redact every matching pattern in `policy.block_patterns` regardless of rule phase. The mismatch is visible in [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) at lines 847-854, 902-949, 997-1065, and 1068-1132. Reproduction: evaluating output moderation for `"danger secret"` with an input-only `danger` rule and an output-only `secret -> redact:[MASK]` rule returns `('redact', '[REDACTED] [MASK]', 'secret', 'uncategorized')`, so the input-only rule is still redacted during the output transform. Impact: rule phase contracts are internally inconsistent, and output moderation can mutate text with input-only rules or vice versa. Existing coverage is misleading here: [`test_moderation_check_text_snippet.py`](../../../../tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py) lines 121-148 only proves that `check_text()` and `evaluate_action()` respect phase for the decision result, while [`test_moderation_redact_categories.py`](../../../../tldw_Server_API/tests/unit/test_moderation_redact_categories.py) only exercises category gating, not phase-aware redaction.
- Medium, high confidence: `_find_match_span()` silently misses valid long-span regex matches once the text length grows beyond `4 * _max_scan_chars`. For long payloads it scans chunked windows and then only falls back to a full-text `search()` when `text_len <= chunk_limit * 4`; after that threshold it returns `None` even if the configured regex would match the full input. The behavior is in [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 1173-1197. Reproduction: with `/A.*B/ -> block`, `_max_scan_chars=10`, `_match_window_chars=5`, and text length `81`, `check_text(..., phase="input")` returns `(False, None)` even though the pattern matches the full string and the internal fallback limit is only `40`. Impact: detection paths that rely on `_find_match_span()` can under-enforce configured moderation rules on larger payloads. Existing coverage does not catch this edge: [`test_moderation_check_text_snippet.py`](../../../../tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py) lines 90-112 uses `text_len=131` with `_max_scan_chars=50`, which stays under the fallback limit of `200` and therefore never exercises the miss.

## Probable Risks
- None recorded in this pass beyond the confirmed defects above.

## Improvements
- Add a regression test that drives `evaluate_action(..., phase="output")` through a redaction path where one matching rule is phase-ineligible, and assert both the returned action and the final redacted text stay phase-consistent.
- Add the same phase-aware regression for `redact_text_with_count()`, which currently shares the same missing `_rule_applies_to_phase()` guard as `redact_text()`.
- Add a long-span regex regression where `len(text) > 4 * _max_scan_chars` so `_find_match_span()` cannot silently stop matching rules that span farther than the chunk window.

## Open Questions
- None.

## Exit Note
This pass confirmed two service-level correctness bugs in the policy and parsing surface: one in phase-aware redaction and one in long-span regex detection. Outside those gaps, the inspected unit slice aligned with the current implementation for category normalization, environment-value fallback handling, per-user category clearing, and sanitized snippet behavior. The policy and parsing pass is complete and ready to hand off to the endpoint and caller-path review.
