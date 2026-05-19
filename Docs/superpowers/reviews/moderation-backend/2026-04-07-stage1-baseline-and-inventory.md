# Stage 1 Baseline and Inventory

## Scope
Create the moderation review scaffold, record the baseline workspace state, and freeze the scoped source/test inventory before any deep review begins.

## Files Reviewed
- `Docs/superpowers/specs/2026-04-07-moderation-backend-review-design.md`
- `Docs/superpowers/plans/2026-04-07-moderation-backend-review-execution-plan.md`
- `Docs/superpowers/reviews/moderation-backend/README.md`

## Tests Reviewed
- No tests were executed for Stage 1.
- The scoped moderation test surface was inventoried for later stages only.

## Validation Commands
- `git status --short`
- `git rev-parse --short HEAD`
- `git log --oneline -n 20 -- tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/api/v1/endpoints/chat.py`
- `{ rg --files tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1 | rg 'Moderation|moderation|chat\\.py$'; rg --files tldw_Server_API/tests | rg 'moderation'; } | sort | tee /tmp/moderation_review_inventory.txt`
- `git diff --name-only | rg 'tldw_Server_API/app/core/Moderation|tldw_Server_API/app/api/v1/endpoints/moderation.py|tldw_Server_API/app/api/v1/schemas/moderation_schemas.py|tldw_Server_API/app/api/v1/endpoints/chat.py|tldw_Server_API/tests/.+moderation'`

## Confirmed Findings
- The Step 4 `git status --short` baseline output was empty, so the worktree was clean before any review artifacts were created.
- The Step 4 `git rev-parse --short HEAD` baseline hash was `a2a10c601`.
- The Step 4 `git log --oneline -n 20 -- tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/api/v1/endpoints/chat.py` baseline output is recorded below and anchors the moderation history used by later stages.
- The approved execution-plan input was read from `Docs/superpowers/plans/2026-04-07-moderation-backend-review-execution-plan.md`, and Stage 1 remained limited to scaffold, baseline, and inventory capture.

## Baseline Output
### `git status --short`
```text

```

### `git rev-parse --short HEAD`
```text
a2a10c601
```

### `git log --oneline -n 20 -- tldw_Server_API/app/core/Moderation tldw_Server_API/app/api/v1/endpoints/moderation.py tldw_Server_API/app/api/v1/schemas/moderation_schemas.py tldw_Server_API/app/api/v1/endpoints/chat.py`
```text
54ebf26b8 Merge branch 'dev' into codex/mcp-virtual-cli-phase2c
a4fe3dc72 fix(chat,acp): address PR #967 review feedback
de9523507 fix(security): address PR #973 review — API key leak, rate limit, auth fallback, init marker
2d3209a6b fix: return structured 503 error when no LLM provider configured in chat (#18)
d9ba0009d progress
c710e6978 progress
b0f8c1d61 merge: bring main into dev preferring dev conflicts
b949b6b01 fixes
61b81e3b0 Merge origin/dev into codex/deep-research-collecting-dev-pr
c07ab95f6 fixres
aa4c96c39 fixes
24c62f7c2 fix: live-proof workspace mind map outputs
eb0f9fc2b fix: close remaining auto-router review gaps
161737e7e fix: address router review feedback
6ec1b1a43 fix: harden router parsing and share endpoint helpers
82fc6adaa fix: honor runtime auto-router policy
378a02520 feat: route auto model selections in chat completions
7656c6147 Merge origin/dev into feature/workspace-chat-isolation-full-sync
e2e62b356 fix: address PR review feedback (types, docstrings, scope enforcement, PG migrations)
78081bec1 Merge origin/dev into codex/deep-research-collecting-dev-pr
```

## Probable Risks
- Review-stage notes should continue to be treated as the canonical trace in the isolated worktree so later stages do not drift from the plan that was inspected.

## Improvements
- Keep the stage notes and README as the single source of truth for the review contract so future stages do not depend on re-discovering the task instructions.

## Open Questions
- None.

## Exit Note
Baseline capture is complete. The review workspace is ready for deep inspection with the following frozen final response structure:
```markdown
## Findings
### Confirmed findings
- severity, confidence, file references, impact, and fix direction when clear

### Probable risks
- material issues not fully proven, with explicit confidence limits

## Open Questions
- only unresolved ambiguities that materially affect confidence

## Improvements
- lower-priority hardening or maintainability suggestions

## Verification
- files inspected, tests run, and what remains unverified
```

## Source Inventory
The scoped source inventory was captured with this command:

```text
tldw_Server_API/app/api/v1/endpoints/chat.py
tldw_Server_API/app/api/v1/endpoints/moderation.py
tldw_Server_API/app/api/v1/schemas/moderation_schemas.py
tldw_Server_API/app/core/Moderation/README.md
tldw_Server_API/app/core/Moderation/__init__.py
tldw_Server_API/app/core/Moderation/category_taxonomy.py
tldw_Server_API/app/core/Moderation/conflict_resolution.py
tldw_Server_API/app/core/Moderation/family_wizard_materializer.py
tldw_Server_API/app/core/Moderation/governance_io.py
tldw_Server_API/app/core/Moderation/governance_utils.py
tldw_Server_API/app/core/Moderation/moderation_service.py
tldw_Server_API/app/core/Moderation/semantic_matcher.py
tldw_Server_API/app/core/Moderation/supervised_policy.py
```

## Test Inventory
The scoped moderation-focused test inventory was captured with this command:

```text
tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py
tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py
tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py
tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py
tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py
tldw_Server_API/tests/unit/test_moderation_effective_settings.py
tldw_Server_API/tests/unit/test_moderation_env_parse.py
tldw_Server_API/tests/unit/test_moderation_etag_handling.py
tldw_Server_API/tests/unit/test_moderation_redact_categories.py
tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py
tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py
tldw_Server_API/tests/unit/test_moderation_user_override_contract.py
tldw_Server_API/tests/unit/test_moderation_user_override_validation.py
```
