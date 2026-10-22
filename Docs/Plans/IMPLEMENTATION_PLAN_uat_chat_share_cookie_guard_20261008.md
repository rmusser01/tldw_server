# UAT C01: Cookie owner at the chat share-list token guard

Associated task: `TASK-13260.278.18.83.45`; parent recorded the actual cookie-owner share-list 401 before edits, then recorded the same-cookie B03 dictionary-validation POST 401 before its regression. Existing all-43 repair authorization covers this bounded fix.

## Contract and bounded design

Chat share-list/create/revoke routes retain canonical `get_request_user`, owner-scoped ChaCha DB access, conversation owner/scope validation, RBAC limits, and CSRF middleware for cookie mutations. The actual `/api/v1/chat/dictionaries/validate` POST declares the same guard, RBAC limit, and canonical user dependency. `TokenScopeGuard` currently attempts canonical claim-based admin resolution only when bearer credentials exist; a cookie-only request can therefore hit its missing-header 401 before the route resolves the legitimate owner.

Resolve the configured cookie through the existing canonical principal dependency during the existing admin-bypass phase, only when both explicit Authorization and X-API-KEY headers are absent. Keep the verified role/permission bypass decision, `allow_admin_bypass`, explicit-header branches, scoped JWT/API-key constraints, quotas, schedule checks, and every share route unchanged. A cookie is never itself an admin claim or API token. No auth rewrite, UI change, live public share, runtime, index, build, or parent-task edit belongs to this unit.

Production scope: `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`; test scope: new real-auth `tests/AuthNZ/integration/test_chat_share_cookie_session.py` and focused claim-boundary additions to existing scoped-token tests only if needed.

## ADR check

**ADR required: no.** This restores canonical cookie authentication at an existing guard while preserving its claim-based authority and constraints. Governing decisions: [ADR-019](../ADR/019-security-request-edge-middleware.md) and [ADR-044](../ADR/044-cookie-session-governance-owner-preflight.md). Parent receives this assessment for the associated task.

## Stage 1: Trace and causal red

**Goal**: Confirm supported cookie ownership, all guard callers and flags, header precedence, claim bypass, share owner/scope checks, settings writes, and client consumers; reproduce the actual share GET and dictionary-validation POST failures with real session authentication.
**Success Criteria**: Real existing cookie fixture plus actual share routes and isolated real ChaCha SQLite produce the missing-header 401 on the unchanged guard; rejection controls use real missing/invalid/expired/revoked sessions and explicit invalid/empty headers.
**Tests**: New module covers primary/alias share GET, dictionary-validation POST with valid and invalid dictionaries, owner/global/workspace boundaries, claim-based bypass, `allow_admin_bypass=False`, constrained explicit tokens, settings retention, and CSRF on validation/revocation without creating a live share.
**Status**: Complete

## Stage 2: Canonical cookie principal at the guard

**Goal**: Add only the missing canonical resolution trigger.
**Success Criteria**: Cookie-only principal resolution occurs under existing `allow_admin_bypass`; absent/invalid credentials still deny; explicit headers never gain cookie fallback; claim-less administrative booleans cannot bypass; token scope and revocation remain enforced.
**Implementation**: Extend `if principal is None and credentials` with a cookie-only condition requiring absent Authorization/X-API-KEY headers and the configured session cookie, then reuse the existing canonical resolver and `_principal_has_admin_bypass_claims` unchanged.
**Tests**: Causal red-to-green and complete scoped-token/share/session affected suites; Python 3.12.13 existing qualified UAT venv, normal plugins, `--timeout=300`, outer 600-second command cap, at most three bounded attempts per issue.
**Status**: Complete

## Stage 3: Static security, self-review, freeze

**Goal**: Validate affected contracts, compare static baseline/current lint and Bandit, review the exact owned diff, and freeze for the parent.
**Success Criteria**: Natural affected-suite exits with exact counts; no new production security/lint findings; owned new test assertions use targeted B101 annotations. Parent owns independent review and real runtime/UAT acceptance.
**Tests**: Scoped-token enforcement and route auth-chain, real single-user-cookie integration, new chat-share cookie integration, existing chat-share API/signing/token security tests. Static Ruff/Bandit compare the touched scope with its pre-unit baseline using the already installed primary-project tools; no application imports in the Python 3.11 static environment.
**Status**: Complete

## Verification record

Clean causal red: 3 expected failures (primary and alias share-list GET plus dictionary-validation POST), each `401 Authentication required`, 29 other cases deselected; normal plugins, 300-second case/600-second outer bounds, natural exit 1, pytest 20.01 seconds/outer 26.34 seconds. Two preceding bounded runs had the same causal failures plus fixture shutdown Jobs DB errors; the fixture now supplies an isolated temporary shutdown Jobs DB, restores startup test flags before teardown, and resets the API-key manager in the TestClient portal. No production change occurred before the clean red run. No runtime refresh or C01/B03 acceptance is claimed here.

- Final new module: **34 passed**, no skips/errors, natural exit 0; pytest 188.60 seconds, outer 195.340 seconds. Its first full green attempt had 28 passes and four test-setup failures: three missing workspace FK rows and a persisted virtual key accepted by the guard but intentionally rejected by single-user legacy user authentication. The final tests create the real workspace row, test persisted virtual-key constraints at a guarded probe, and separately check actual share routes with primary-key and Bearer credentials. No production change followed that attempt.
- Existing affected scope: **97 unique cases passed across bounded runs**, preserving passing cases. Initial nine-module run: 86 passed, six failures and five setup errors, with default AuthNZ database-path failures and two scoped-JWT cases that obtain the real guard-side session manager despite their legacy user-resolution stubs. Private supported `DATABASE_URL`, `USER_DB_BASE_DIR`, and `JOBS_DB_PATH` plus synthetic test primary/JWT values yielded nine targeted passes in 17.66 seconds (outer 22.070 seconds), retaining two scoped-JWT failures. Final check added the explicit synthetic `SESSION_ENCRYPTION_KEY` used by the existing cookie fixture: the two remaining cases passed in 7.77 seconds (outer 11.821 seconds), natural exit 0. Existing auth stubs, plugins and application security were unchanged. An intervening broader retry was interrupted at the parent's instruction (exit 130), contributes no result, and was replaced with failed-case-only checks.
- Existing modules: `AuthNZ/unit/test_scoped_token_enforcement.py`, `AuthNZ/unit/test_scoped_token_route_auth_chain.py`, `AuthNZ/integration/test_single_user_cookie_session.py`, `Chat/unit/test_chat_share_links_api.py`, `Chat/unit/test_knowledge_qa_share_token_security.py`, `Chat/unit/test_share_token_malformed_signature.py`, `Chat/unit/test_share_link_signing_key_fail_closed.py`, `Chat_NEW/integration/test_chat_dictionary_validate_endpoint.py`, and `Chat_NEW/unit/test_dictionary_validator.py`. **131 total unique new/affected cases qualified**; this is not a full backend/UAT claim.
- Static validation with installed primary-project tools only: Bandit 1.9.3 baseline guard **0 findings/0 scan errors**; final guard plus new test module **0 findings/0 scan errors**, 2,793 lines scanned. Assertions use targeted `# nosec B101`. Ruff baseline/current production has the same one inherited `I001` import-order finding; the new module has zero findings. Initial Ruff cache access was denied; subsequent checks used `--no-cache` and identical explicit project config. Static `ast.parse` and owned `git diff --check` passed. Python 3.11 tooling performed AST/static analysis only; application tests used the qualified Python 3.12.13 environment.
- Source coverage: canonical resolver/session validation and cache activation; legacy request-user precedence and verified claims; shared guard JWT scope, revocation, API-key scope/endpoint/method/path constraints, quotas/deferred counts and schedule checks; all production guard caller families and flags; CSRF middleware; both share router aliases and create/list/revoke owner, workspace and settings consumers; dictionary validator route/schema/client consumers. Full new test bodies and the owned hunk were self-reviewed. No route, ownership, sharing setting, guard enforcement or runtime code changed beyond the canonical resolution trigger.
- Independent actual-source review by the parent's Prompt-cookie agent: **CLEAR**, no substantiated Critical/Important/Minor findings; parent also read the complete final new module and guard and approved controlled runtime load. Tests/runtime acceptance remain separate.
- Frozen production SHA256: `0165972b5674d2243e7061092bb6794572a32d61e362d63bec1b51b776e11c43`; new test SHA256: `2b385f9ab64c5e6c40e0562f9d2e892c026dcc915a1ba201d8e747040bcef2f9`. No commit, index, build, browser, runtime, production credential inspection or public-share operation belongs to this source unit. Parent owns the combined normal API reload and actual C01/B03 acceptance.
