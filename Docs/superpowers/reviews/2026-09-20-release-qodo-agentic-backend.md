# Agentic Qodo backend findings 5, 6, 12, 16, 17

Task TASK-13263.1. Shared worktree `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/release-main-0.1.42`; starting candidate 254165e7b5. No commits, branch changes, release-record edits, or delegation by this agent.

## Outcomes

### 5 — fixed: image detail preservation on failed retry/regeneration

Confirmed that initial Chat persistence discarded valid `high`/`low` image options while retry rejected them with 409. Save one ordered detail per successfully decoded attachment in existing message extra metadata (`image_details`), restore those options for history and saved-turn regeneration, and include detail in strict content comparison. This requires no DB migration. Changed details, changed image bytes/order, malformed options and incomplete saved option arrays still conflict. Absent legacy metadata defaults to `auto`; an old saved non-default option already discarded by previous versions cannot safely be reconstructed and remains a conflict if the caller supplies that non-default value. Metadata-inclusive message listing retains the options for clients.

Regression coverage: real SQLite/API initial failed save -> retry and successful save -> regeneration, for both high and low, with two differently detailed images in order; only one user row remains and provider receives exact image URLs/details. Existing changed-detail conflicts, omitted/auto legacy retry, image completeness, ownership and provider order checks remain green. New helper tests cover legacy defaults and malformed/partial options.

Source: `app/api/v1/endpoints/chat.py`, `app/core/Chat/chat_service.py`.
Tests: `tests/Chat/integration/test_chat_image_recovery.py`, `tests/Chat/unit/test_chat_persistence_content.py`.

### 6 — fixed: OSCE endpoint helper contracts

Added concise contract docstrings to the five named helpers and the adjacent transition helper: lookup failures, activity-type guard, candidate/revealed response projection, authoring allowlist, error mapping, and versioned transitions. No behavior change from docstrings.

Source: `app/api/v1/endpoints/quizzes_osce.py`.

### 12 — fixed: OSCE schema helper contracts

Documented normalized rubric-label uniqueness, collection of assigned nested IDs, and whole-document nested-ID uniqueness. Existing schema behavior unchanged; full schema contract suite passes.

Source: `app/api/v1/schemas/osce.py`.

### 16 — fixed: mandatory OSCE ingress dependency

Confirmed neither OSCE nor parent quizzes router nor router registry supplies a mandatory limiter. Global RG ingress is optional (`main.py` enables it only when RG is enabled); its quizzes policy alone does not cover disabled deployments. Merely adding `rbac_rate_limit("quizzes.read")` would not guarantee a finite fallback because that resource has no privilege-catalog entry. Added a router-wide Request-only wrapper around existing `check_rate_limit`, following repository finite-ingress policy (default fallback 120 per 60 seconds, configurable; honors existing validated single-user/test-mode exemptions and RG enforcement metadata). All OSCE station and attempt operations inherit it.

Tests exercise actual routing with test-mode bypass disabled, finite capacity 1, and RG disabled: first admission reaches body validation, second returns 429 with Retry-After before handler execution. Tests also assert every OSCE operation carries the guard and the internal legacy `rate_limiter` hook is absent from OpenAPI query parameters. No new public query parameters.

Source: `app/api/v1/endpoints/quizzes_osce.py`.
Tests: `tests/Quizzes/test_osce_endpoints.py`.

### 17 — fixed: anonymous setup model-path disclosure

Concrete path: saved local provider `default_model` and first-chat model flow through public first-run-state projections, which previously reused write-side path allowances. GET is unauthenticated only for clients allowed by `require_local_setup_access`; remote callers require admin authorization. This is not a claim of arbitrary remote unauthenticated access.

Invariant: anonymous progress responses must not disclose configured local filesystem identifiers. Legitimate local model identifiers must remain exact in saved state and calls to local inference servers, and authenticated configuration administrators must be able to resume onboarding.

Patch: projection-only field checks recognize absolute POSIX, Windows drive, UNC, short rooted, home-relative and dot-relative forms. Public GET/POST/skip state projections hide model paths. GET uses the existing verified principal resolver and restores only these local model fields when SYSTEM_CONFIGURE or wildcard permission is present. Missing/rejected authentication (401/403) remains on safe public projection; unrelated auth errors are not swallowed. Validated cookie-session authentication remains supported by the existing resolver. Other sensitive fields still use normal sanitization even for administrators. Writes, persisted model IDs, and first-chat invocation/response to caller-supplied IDs remain intact. Ordinary model IDs such as `org/model`, `llama3:8b`, and `local-model.gguf` remain visible. Authenticated non-admin users do not receive paths.

Tests cover six path classes, anonymous/admin/unprivileged callers, public state save and skip responses, raw persisted exact IDs, backend first-chat calls and authenticated GET resume; include ordinary local identifiers. All existing unified first-run setup tests pass.

Security workflow: skill read and applied; parent-dispatched investigator independently confirmed source-to-sink path, local-only guard, exact-ID resume requirement, core resolver behavior and alternate path forms before the patch. Parent reported agent thread-limit/pending-init preventing fresh candidate review and performs independent final candidate review. This agent additionally performed a separate post-patch pass over all `_public_first_run_state` callers (GET, update, skip), helper callers and both authorization outcomes, including hosted/local distinction and unsupported/secret-shaped values. No surviving path within this finding's projection boundary identified.

Source: `app/api/v1/endpoints/setup.py`.
Tests: `tests/integration/test_unified_first_run_setup_api.py`.

## Verification evidence

All Python commands first activated `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate`.

Red evidence:
- Original saved high/low retry regression returned 409; `/tmp/qodo-agentic-chat-red.log` (regeneration test metadata key was corrected before final successful regression).
- Anonymous path response regressions: 6 failures, 6 authenticated controls passed; `/tmp/qodo-agentic-setup-red.log`.
- Missing OSCE dependency yielded second 422 rather than 429; `/tmp/qodo-agentic-osce-red.log`. Initial green test subsequently required disabling the test-mode bypass; OpenAPI test also corrected to inspect query parameters rather than rejecting pre-existing authentication headers. No runtime failures remain.

Final checks:
1. `python -m pytest tldw_Server_API/tests/Chat/integration/test_chat_image_recovery.py tldw_Server_API/tests/Chat/unit/test_chat_persistence_content.py tldw_Server_API/tests/Chat/unit/test_failed_retry_provider_order.py -q -rs`: **100 passed, 1 skipped**, 114.80s. Existing PostgreSQL attachment case skipped by official fixture because PostgreSQL was unreachable. `/tmp/qodo-agentic-chat-green.log`.
2. `python -m pytest tldw_Server_API/tests/Quizzes/test_osce_endpoints.py tldw_Server_API/tests/Quizzes/test_osce_schema_contract.py -q`: **88 passed**, 32.00s. `/tmp/qodo-agentic-osce-green.log`.
3. `python -m pytest tldw_Server_API/tests/integration/test_unified_first_run_setup_api.py -q`: **139 passed**, 9.27s. `/tmp/qodo-agentic-setup-green.log`.
4. `python -m bandit` against all five changed source files, JSON output `/tmp/bandit_qodo_agentic_backend.json`: **0 findings, 0 errors**.
5. Ruff fully clean on changed core service, OSCE source/schema, setup source, unit persistence tests and OSCE tests. Pre-existing import-order findings in untouched chat endpoint import blocks remain outside changed scope. Changed imports fixed. Scoped `git diff --check` passes.

Total final focused checks: **327 passed, 1 existing PostgreSQL skip**. No live PostgreSQL success claimed.

OpenAPI: limiter wrapper avoids adding any internal query parameter. New GET state contract docstring may alter its description; parent owns final full OpenAPI fingerprint/types regeneration.

## Parent candidate-review follow-up

A fresh independent reviewer became available after the implementation agent completed. It found that the real setup domain always forced noAuth, so authenticated endpoint controls alone did not preserve manual-key administrator resume. The domain now attempts normal credential-bearing GET and falls back to public progress only on401. A public state with its private model redacted routes to provider reselection instead of rendering a blank first-chat step. Three new regressions failed before these changes;105 setup-domain/hook/wizard/transport tests pass afterward, full TypeScript passes and touched-file ESLint has zero errors. Parent independently reran24 setup privacy/control tests. The new description is the sole additional OpenAPI delta; fingerprint check passes with CI generator versions.
