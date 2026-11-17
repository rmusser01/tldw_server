# C01 prompt-library cookie authentication — 2026-10-08

Associated task: TASK-13260.278.18.83.45 (root-owned; new failure recorded before edits). Reviewed Prompt Studio and Character patches remain frozen. Parent owns task updates, runtime replacement and actual UI validation.

## Diagnosis and ADR assessment

Actual composer/server-title library GET `/api/v1/prompts?page=1&per_page=100&sort_by=id&sort_order=asc` returns401 `Missing authentication token`. `server-prompt-library.ts` uses this exact supported list contract. Shared endpoint `_resolve_prompts_auth_user` rejects absent Token/X-API-KEY/Authorization before canonical `get_request_user`; `get_prompts_db_for_user` already derives its owner from canonical authentication. The resolver is shared by list/search/detail/persistence/admin guards.

Minimal repair: let real headerless requests authenticate through canonical `get_request_user`, passing explicit None dependency arguments. Preserve existing missing-auth401 detail if canonical authentication rejects401; propagate other canonical errors. Keep existing header/legacy compatibility, claim/admin gates, CSRF and DB owner boundaries intact. No frontend/client change or auth fallback based on profile/mode.

ADR check: ADR required:no. Searched `Docs/ADR/README.md`; this restores established canonical cookie authentication and owner-scoped storage, without new API/security/module policy. Existing ADR-044 governs cookie-owner resolution, ADR-020 per-user storage, ADR-006 scoped Bandit. No new ADR.

Separate share-links401 comes from a different chat dependency chain; no shared prompt resolver. No edits/tests for that separate observation.

## Stage 1: Establish causal regressions
**Goal**: Real supported cookie/session fixtures reproduce headerless prompt list/search failure.
**Success Criteria**: Cookie list slash/no-slash and cookie persistence/search/detail fail at the observed guard while no-auth/invalid/revoked/invalid-header controls reject; canonical errors and claims remain visible.
**Tests**: Canonical minted cookie, real prompt router/official DB dependency, exact composer list query, CSRF write guard, search query contract, per-user DB isolation; narrow headerless resolver claim/error tests.
**Status**: Complete

## Stage 2: Delegate the shared headerless auth path
**Goal**: Minimal shared resolver repair after causal red.
**Success Criteria**: Authenticated cookie requests pass list/search/detail/write; explicit invalid headers reject; missing/invalid/revoked sessions and non-admin claims keep denial.
**Tests**: Only fresh affected scope, supported existing Python3.12, normal configured plugins,600-second outer/300-second case caps; maximum3 distinct attempts per issue.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Scoped compile/Ruff and baseline/current AST-only Bandit, connected-source review and root handoff.
**Success Criteria**: No new lint/security findings, exact files/hashes/results; root reviews/freezes source and performs actual loaded-runtime UI retry.
**Tests**: Touched Python only; primary existing Python3.11/Bandit only as static scanner (app tests stay3.12), whitespace/source review.
**Status**: In Progress

## Boundaries

No runtime/browser/build/protected requests, parent task edits, changes to frozen units, environment/model/install/network changes, custom SQL/DB setup, proof/log files, or Git index/history/publication. Test fixtures own ephemeral settings/session DBs; application imports are real existing supported fixture imports.

## Initial fixture attempt (separate from causal red)

First run observed6 causal failures/6 passing rejection controls,1315 warnings,42.15s pytest/49.132s outer,natural1;9 fixture teardown errors occurred because test cleanup awaited the existing Prompts close-worker task from pytest's event loop instead of TestClient's owner portal loop. Corrected fixture cleanup and official DB helper/probe calls to use the existing TestClient portal; no application lifecycle guard was changed. Clean red qualification pending before production edits.

Baseline static scans: existing primary Python3.11.13/Bandit1.9.3 confirmed AST-only; prompts.py0 findings/0errors. Supported Python3.12 Ruff baseline0 findings.

## Clean causal red

Supported Python3.12.13/pytest9.1.1, normal configured plugins, timeout300/outer600:6 failed,6 passed,1163 warnings,no skips/no fixture errors,40.48s pytest/47.823s outer,natural1. Exact composer list query with slash/no-slash and cookie+CSRF persistence return401 `Missing authentication token`. Canonical503 and headerless owner-claim checks are also blocked by that guard. No-auth/invalid/revoked-cookie and invalid Token/API-key/Bearer controls reject. Root received source ruling and clean red before the one-file production edit.

## Final scoped qualification and freeze

- Fresh affected module12 passed,no skips/no fixture errors,1316 warnings,36.17s pytest/42.106s outer,natural0. Same supported Python3.12.13/pytest9.1.1, configured plugins and300case/600outer caps; no old/frozen-unit test replay.
- Nine real HTTP cases cover exact composer list query slash/no-slash; cookie+CSRF persistence, search_query search, detail and official per-user DB/foreign UUID absence; missing/invalid/revoked cookies and invalid explicit API-key/Bearer/legacy Token. Three direct resolver/claim cases cover canonical503 propagation and regular-owner PROMPTS_REQUIRE_ADMIN/admin-only claim gates.
- Compile2 passed; Ruff baseline/current0; Bandit baseline/current0 findings/0scan errors using verified existing primary static-only scanner. Own whitespace checks clean. Production diff is11 added resolver lines plus explanatory docstring only; no header/legacy branch, authorization policy, CSRF, owner or endpoint shape change.
- Connected-source self-review found no substantiated Critical/Important/Minor finding in this repair. Production prompts.py frozen and evidence sent to root for independent review. Stage3 remains In Progress solely for root independent review/live C01 retry/final task evidence; source tests do not establish loaded-runtime UI success, full-app/UAT matrix or other auth/DB profiles.
- Separate share-links401: chat list share-links declares `TokenScopeGuard("any", require_if_present=True, ...)`; `auth_deps.py:2776` raises401 `Authentication required` without a Bearer/API key. This is a distinct dependency chain; no chat/auth_deps edits, test replay or live claims made.
- Parent task untouched; reviewed Prompt Studio and Character changes untouched; no runtime/browser/build/install/network/proof/log/Git index/history/publication work.
