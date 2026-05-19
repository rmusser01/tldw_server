# Stage 5 Test Gaps and Final Synthesis

## Scope
Assess moderation test coverage gaps and synthesize the final findings-first review output.

## Files Reviewed
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage1-baseline-and-inventory.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- `Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`
- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py`
- `tldw_Server_API/app/core/Chat/chat_service.py`
- `tldw_Server_API/app/core/Moderation/supervised_policy.py`

## Tests Reviewed
- Stage 2 policy and parsing slice: `36 passed, 86 warnings in 2.26s`
- Stage 3 endpoint and caller-path slice: `14 passed, 228 warnings in 14.60s`
- Stage 4 persistence and concurrency slice: `24 passed, 62 warnings in 2.18s`
- Representative files: `test_moderation_blocklist_parse.py`, `test_moderation_check_text_snippet.py`, `test_moderation_redact_categories.py`, `test_moderation_test_endpoint_sample.py`, `test_moderation_permissions_claims.py`, `test_moderation_etag_handling.py`, `test_moderation_user_override_contract.py`, `test_moderation_user_override_validation.py`, `test_moderation_runtime_overrides_bool.py`, `test_moderation.py`, `test_moderation_categories.py`

## Validation Commands
- `sed -n '1,260p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage2-policy-and-rule-parsing.md`
- `sed -n '1,280p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage3-endpoints-caller-and-permissions.md`
- `sed -n '1,280p' Docs/superpowers/reviews/moderation-backend/2026-04-07-stage4-persistence-concurrency-and-verification.md`

## Confirmed Findings
- High, high confidence: failed per-user override writes still change the live moderation state. [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 1243-1295 mutate `self._user_overrides` before attempting persistence, so `set_user_override()` and `delete_user_override()` can return persistence errors after the running process has already changed enforcement behavior. This was directly reproduced in Stage 4. Fix direction: stage the new override JSON first or roll back in-memory state on failure.
- High, high confidence: runtime settings persistence fails open. [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 562-632 let `update_settings(persist=True)` return success after `_save_runtime_overrides_file()` logs and suppresses a write failure, so the caller can believe a setting is durable when it only exists in memory. This was directly reproduced in Stage 4. Fix direction: make persistence failure observable to callers and avoid mutating durable state until the save succeeds.
- Medium, high confidence: the admin `/moderation/test` endpoint is not representative for supervised accounts. [`moderation.py`](../../../../tldw_Server_API/app/api/v1/endpoints/moderation.py) lines 402-489 evaluate only the base moderation service, while live chat input and output paths overlay guardian policies in [`chat_service.py`](../../../../tldw_Server_API/app/core/Chat/chat_service.py) lines 2325-2359, 3721-3819, 4155-4254, and 4850-4951, with output wrapping performed in [`chat.py`](../../../../tldw_Server_API/app/api/v1/endpoints/chat.py) lines 3852-3859. Impact: the admin tester can report pass or weaker effective policy than live chat enforces for supervised users. Fix direction: either document the tester as base-policy-only or make it use the same guardian overlay path.
- Medium, high confidence: phase-specific rules are ignored during redaction transforms. [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 847-854 and 902-949 respect `_rule_applies_to_phase()`, but `redact_text()` and `redact_text_with_count()` at lines 997-1065 only gate on categories. As a result, output redaction can still apply input-only rules and vice versa. This was directly reproduced in Stage 2. Fix direction: pass phase into the redaction helpers and apply the same phase filter used by detection.
- Medium, high confidence: long-span regex matches can be silently missed on larger payloads. [`moderation_service.py`](../../../../tldw_Server_API/app/core/Moderation/moderation_service.py) lines 1173-1197 only fall back to a full-text search when `len(text) <= 4 * _max_scan_chars`, so valid matches beyond the chunk-window threshold can disappear from detection paths. This was directly reproduced in Stage 2. Fix direction: remove the correctness cutoff or document and enforce a narrower supported regex model.

## Probable Risks
- Medium, medium confidence: governance `scope_chat_types` may not propagate through `/chat/completions`. The supervised-policy engine and governance utilities support chat-type filtering, but the main chat endpoint does not appear to pass `chat_type` into the moderation path while still accepting `character_id`. If non-`regular` moderation contexts are intended here, character-scoped guardian policies may not be enforced as expected.
- Medium, medium confidence: user-override and runtime-override persistence are non-atomic. Unlike blocklist writes, which use temp files plus `os.replace()`, these JSON writes use direct `open(..., "w")`. I did not simulate crash or kill during write, but partial-file corruption is a realistic failure mode.

## Improvements
- Add a real-service regression for failed `set_user_override()` and `delete_user_override()` persistence that asserts live memory does not drift when the write fails.
- Add a real-service regression for `update_settings(persist=True)` that forces a save failure and asserts the API surfaces it instead of silently succeeding.
- Add a phase-aware redaction regression that exercises both `redact_text()` and `redact_text_with_count()` with phase-ineligible matching rules.
- Add a long-span regex regression where `len(text) > 4 * _max_scan_chars` so `_find_match_span()` cannot silently drop real matches.
- Add a caller-path regression for supervised accounts that compares `/moderation/test` against live chat behavior, and add at least one authz regression on a mutating moderation route.

## Open Questions
- Does `/chat/completions` intentionally treat all moderation governance contexts as `"regular"`, even when character context is present, or is chat-type propagation simply incomplete?

## Exit Note
The final synthesis is ready. The dominant pattern across the backend moderation surface is not parser fragility alone; it is contract drift between what callers are told and what the running process actually enforces. That shows up in persistence failure handling, the admin test surface, and a smaller chat-type scoping question. The final user-facing review should lead with the five confirmed findings above, then keep the two confidence-limited risks separate so they are not overstated.
