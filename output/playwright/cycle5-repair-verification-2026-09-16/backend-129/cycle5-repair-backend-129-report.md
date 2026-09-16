# UAT129 / TASK13260.69 — ready for independent review

Canonical AuthNZ AUTH_MODE now solely selects Prompt authentication mode; canonical settings errors propagate and cannot synthesize single-user/admin identity through legacy APP_MODE defaults. Credential comparison, unified resolution, claim-first admin checks and DB dependency remain unchanged.

## Verification
- Permanent actual collections/create/list router + actual per-user SQLite databases: RED **5 failed / 7 passed**, then **12 passed**. Covers contradictory legacy flags, Alice/Bob isolation, valid/invalid/missing credentials, supported header forwarding, admin gate and settings-error denial. The unified credential resolver is controlled; this is not a claim of end-to-end JWT verification in the route fixture.
- Broader command: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/AuthNZ_Unit/test_prompts_canonical_auth_mode.py tldw_Server_API/tests/AuthNZ_Unit/test_prompts_admin_claims.py tldw_Server_API/tests/Prompt_Management/test_prompts_api.py tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py tldw_Server_API/tests/AuthNZ/unit/test_jwt_service.py -q` — **87 passed**.
- Final test-only refinement removes redundant Authorization headers from API-key/legacy Token controls; rerun new suite **12 passed** (`-final-focused.log`).
- `python -m ruff check` touched production/test: **0 findings**, production baseline also0.
- `python -m bandit tldw_Server_API/app/api/v1/endpoints/prompts.py -f json`: **0 findings**.
- Existing import/deprecation warnings and pytest temporary-directory cleanup warning remain; no clean-warning claim.

Artifacts use this report prefix: `-red.log`, `-green.log`, `-final-focused.log`, `-ruff-before.json`, `-ruff-after.json`, `-bandit.json`, `-manifest.json`. No browser/runtime/inference, frontend edits, staging or commit. Native acceptance and independent review remain pending.

## Required PostgreSQL follow-up
Root restored the official PostgreSQL18.6 fixture. Bounded backend32 and AuthNZ2 checks passed with REQUIRED=1/NO_DOCKER=1 and zero skips. The two formerly skipped nodes now execute. The bootstrap test assertion defect was corrected separately under TASK13260.75.1; see /private/tmp/cycle5-postgres-136-report.md for exact commands, scope and redacted evidence. This resolves the recorded fixture-execution gap for these controls, not native full-workflow acceptance. Product code for this unit remains unchanged.
