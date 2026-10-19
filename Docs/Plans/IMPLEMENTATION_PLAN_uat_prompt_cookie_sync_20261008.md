# Prompt cookie-session sync repair — 2026-10-08

Associated task: TASK-13260.278.18.83.45 (root-owned unit note precedes edits).

Scope: repair the actual C01 Prompt save/sync failure. Prompt Studio must authenticate cookie sessions through the existing shared AuthNZ resolver. Project discovery must distinguish a successful empty list from transport or payload failures, preserving the local draft and avoiding unintended project creation. Existing header precedence, CSRF, claim-based roles, ownership, quota and rate-limit guards remain governing contracts.

ADR check: ADR required: no. ADR-044 (`Docs/ADR/044-cookie-session-governance-owner-preflight.md`) and ADR-018 govern canonical cookie ownership and claim-based authorization. This fix restores those existing contracts and changes no durable auth, persistence or API rule.

## Stage 1: Causal regressions
**Goal**: reproduce cookie rejection and invalid project-response handling on unchanged production source.
**Success Criteria**: supported real single-user session/CSRF fixture exercises slash/no-slash Prompt Studio requests; absent/unusable credentials remain rejected; client failure stops project creation and preserves the draft.
**Tests**: new AuthNZ cookie integration cases; focused prompt-sync response cases; existing relevant claim/owner controls.
**Status**: Complete

## Stage 2: Minimal shared repair
**Goal**: remove Prompt Studio's redundant header-only barriers and validate project discovery responses at their boundary.
**Success Criteria**: canonical AuthNZ authentication and all downstream guards remain intact; client displays the server failure without an array-method crash or implicit create.
**Tests**: causal cases green; affected Prompt Studio auth/ownership and client consumers.
**Status**: Complete

## Stage 3: Scoped qualification and handoff
**Goal**: review and verify the touched scope, then return the exact patch and evidence to root for actual runtime validation.
**Success Criteria**: scoped compile/Ruff/Bandit comparison and affected UI/WebUI tests complete; no new touched-code diagnostics; plan records inherited findings and limitations.
**Tests**: normal configured supported Python environment and UI/WebUI configurations, bounded to 600 seconds per command and 300 seconds per pytest case.
**Status**: In Progress

Live API/frontend/browser validation, build/startup, staging/commit, canonical exports and task edits remain root-owned. No unrelated Character or Regenerate edits.

Causal RED: backend real session module 4 intended failures/6 rejection controls passed, 57.50s pytest/64.007s outer, natural exit 1, no timeout. UI service selection 5 failures/49 selection-excluded (1.039s outer); selector 1 failure/1 selection-excluded (0.952s outer). No fixture failure.

Scoped qualification complete; Stage 3 remains open for root's independent review and live C01 acceptance.

- Production candidate 1 removes only the redundant header-only barriers. The unified resolver, explicit-header precedence, claims, CSRF, owner guards, quotas and rate limits remain intact. Client discovery rejects transport errors, rejected envelopes and non-array responses; selector displays query errors instead of offering empty-list creation.
- Final supported Python 3.12.13 / pytest 9.1.1 configured run: 25 passed, no skips, 1323 observed warnings, 51.82s pytest / 58.355s outer, natural exit 0, no 600/300-second deadline. Includes nine real cookie/CSRF/credential cases, one real database owner-guard case, existing header/claims/hook/quota controls and the corrected original no-auth endpoint case.
- Initial green exposed one unsupported test assumption: the existing canonical single-user cookie deliberately uses the bootstrapped admin singleton; removing its database role cannot create a non-admin cookie. A focused recovery confirmed the legitimate admin result. The test now exercises unchanged regular-owner list/read/write guards directly through the official managed database abstraction; no canonical auth or profile override was added. A subsequent unconditional fixture cleanup-only correction passed the one affected case (6.06s pytest / 9.412s outer, natural 0). No passed unrelated controls were replayed for cleanup.
- UI two affected modules: 56 passed, no skips, 0.900s outer, natural 0. WebUI same modules: 56 passed, no skips, 1.254s outer, natural 0. Owned test formatting preserves emitted JavaScript tokens. The existing localStorage experimental warning is retained.
- Configured UI and WebUI TypeScript checks: natural 0 in 42.129s and 31.801s respectively, installed tooling, 8192 MB Node allowance, no emit/incremental state and 600-second caps.
- Five touched Python files compile. Ruff baseline/current has seven inherited diagnostics: two imports in the project endpoint and five existing header-test import/type-style diagnostics; no new findings. New cookie test module is Ruff clean. Four TS files have zero ESLint errors and thirteen inherited warnings (0 service, 1 auto-sync test, 3 selector, 9 selector test); no new diagnostics. Scoped diff whitespace check passes.
- Existing approved primary .venv Python 3.11.13 / Bandit 1.9.3 was confirmed and used only for static AST scanning. Baseline/current: 187 inherited B101 test assertions (176 existing endpoint suite, 11 header unit suite), zero production findings, zero scan errors, no new findings. Final cleanup-changed new test separately compiles and scans with zero findings. The supported app/test env lacked Bandit; the suggested older static env path did not exist. No environment was installed or rebuilt.
- One earlier operator delivery combined verbose lint and pytest telemetry and truncated; that aggregate is not used as qualification. Final pytest stdout and compact static results were recovered in separate readable calls. Diagnostic stderr sizes are recorded, but noisy standard telemetry was not written to proof/log files.

Limitations: live API/browser, full application UAT, actual C01 acceptance, other auth/database matrix cells, build/restart, canonical export, commit/publication and independent review remain root-owned and unclaimed. No Character or Regenerate source, parent task/plan, live profile, secrets, runtime processes, dependencies or Git history/index were changed by this unit.
