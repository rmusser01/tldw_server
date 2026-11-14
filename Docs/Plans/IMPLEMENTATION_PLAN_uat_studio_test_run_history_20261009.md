# C02: Read immutable Prompt Studio test-run history

Associated task: TASK-13260.278.18.83.45. Root recorded and read back the approved scope before edits. Ordinary Execute is ephemeral; this unit reads existing durable test runs and does not backfill or rerun them.

## ADR assessment

No new ADR is required. [ADR-055](../ADR/055-prompt-studio-db-single-implementation.md) governs the existing backend-neutral repositories; [ADR-020](../ADR/020-db-management-per-user-paths-and-content-backend.md) governs per-user paths and the explicit shared PostgreSQL backend. Both complete ADRs and the current repository/session implementation were read. Existing prompt-version inserts retain exact original IDs; that is current implementation behavior rather than an invented architecture rule. This adds bounded owner-authorized reads of existing records and preserves storage, execution, provider budgets, and version identities. Historical configuration is projected through known nonsecret fields; arbitrary stored configuration is never exposed.

## Stage 1: Establish durable history contracts
**Goal**: Causal regressions for exact version/snapshot reads, pagination, real-owner refusal, and evaluation provenance.
**Success Criteria**: Real production routers and official authentication fixtures establish missing read/projection behavior; repository reads use the official dual-backend fixture.
**Tests**: Real TestRunner through installed Ollama transport injection, stored failed/nullable records, owner/header/page controls, historical model config dict/list safety.
**Status**: Complete

## Stage 2: Add minimal reads and history UI
**Goal**: Backend-neutral repository/facade reads, authenticated GET endpoints, typed responses, existing Test Cases History drawer.
**Success Criteria**: Page size is bounded at 100; ordering is created_at then id; stored project authority and exact original prompt ID are retained. Open/Refresh/reload use GET only. Project/principal/selection changes retire stale responses. Existing create and provider payloads remain unchanged.
**Tests**: New backend and mounted UI contracts plus genuinely affected existing tests; no migration, raw SQL outside DB abstractions, or provider runtime action.
**Status**: Complete

## Stage 3: Qualify and freeze
**Goal**: Scoped normal-config tests, static baseline/current security, self-review, peer/root source review handoff.
**Success Criteria**: Accurate attempt ledger and natural exits; unavailable PostgreSQL is explicitly unqualified; source/test hashes and unit-only deltas are delivered directly, with pre-unit bytes retained only in memory. Root owns final types/build/live UAT.
**Tests**: Activated Python 3.12 pytest (600s outer/300s case), affected UI/Web Vitest, scoped lint/syntax, Python compile/Ruff/Bandit baseline/current.
**Status**: In Progress

## Attempt ledger

- Backend attempt 1: real-auth causal module naturally exited 1 in 112.410s; 14 failed, no setup/teardown errors. The actual TestRunner POST completed through injected installed Ollama transport once and the subsequent new detail GET returned 404. Missing history reads and evaluation provenance/owner/header assertions failed. No red replay for cleaner output; noisy captured logging displaced some detailed failure heads, so the aggregate and confirmed request statuses are retained without inventing per-case attribution.
- Official dual-backend selected test-runs case naturally exited 0 in 33.325s: one skipped because PostgreSQL was unavailable; 11 deselected. This does not qualify PostgreSQL or SQLite reads. A separate real SQLite scenario is added for qualification while preserving the official parity case.
- Six Python production paths: pre-unit Bandit AST-only exit 0, zero findings/errors (4,183 lines). Pre-unit source bytes retained only in memory.
- Root released the frontend edit hold after the owned Web startup; current implementation is a separate unloaded candidate. Backend attempt 2 is running the new reads and genuine unchanged auth controls with normal configured fixtures.
- Backend attempt 2 naturally exited 0 in 147.681s: 23 passed (22 new real-auth cases plus one real SQLite history case), 12 deselected. PostgreSQL remains unqualified. Later immutable-template projection changes only the guarded detail JOIN/DTO; root approved that connected owner-safe preview seam before edits.
- UI attempt 1: causal missing History action, 12 failed; natural 1, Vitest duration 14.48s. This first Node invocation omitted the formal outer wrapper; case waits were bounded, and it is not claimed as outer-qualified.
- UI attempt 2: 600s outer, naturally 1 in 23.273s; 14 failed and one empty-case path passed. Normal-case failures occurred at dialog naming before substantive assertions.
- UI attempt 3: one failed normal-case diagnostic, naturally 1 in 3.061s, 14 unselected. STOP/reassessment: actual native Drawer is open with correct title, row and two GETs; installed test-mode IDs collide (`aria-labelledby=test-id` with existing Select ID). The empty branch lacks that Select and passes. Failure-only observation is removed; no substantive assertion or production behavior changed. Root recorded and approved a distinct fixture phase: role lookup plus scoped visible-title assertion.
- Corrected Drawer phase naturally exited 1 in 38.137s: 13 formerly failed cases passed; the earlier empty-case pass was unselected. Only the cookie-owner preview case failed before the row loaded. Its fixture incorrectly declared a multi-user cookie at a different origin. The actual browser transport contract and existing caller-capabilities cookie fixture require single-user, quickstart and the exact browser origin. Correct only that fixture configuration and environment cleanup, retaining all owner/header/GET/refusal assertions; select only this remaining failed UI ID and first Web scope.
- Supported cookie fixture attempt naturally exited 1 in 2.959s before the row: the external fetch seam parsed URLs without a browser base, but canonical quickstart correctly supplies relative URLs. Use browser-faithful `new URL(input, window.location.origin)` in that fixture; preserve all assertions and qualify only the failed ID, without a separate diagnostic replay. No production authentication or transport behavior changes.
- Guarded original-template projection qualification naturally exited 0 in 20.996s: seven changed/affected cases passed (real TestRunner/revision/template, genuine authenticated owner-switch numeric collision, current SQLite scenario and four existing DB/evaluation error-mapping controls). Earlier 23 passes remain; 28 distinct backend cases passed, not one combined 28-case run. No supported prompt-delete writer exists; missing/deleted historical template metadata is covered through faithful mounted DTO controls rather than invented raw SQL mutation.
- Final failed cookie-preview ID naturally passed in 4.568s (14 unselected). UI has 15 distinct new passes from the retained empty case, corrected Drawer 13, and this final cookie case; no combined 15-case UI run. This is mounted cookie transport evidence, separate from genuine multi-user backend authentication.
- First affected Web run naturally exited 0 in 39.596s: all 15 new history cases and two existing TestCaseRunModal design-system controls passed, no skips. Existing run creation, prompt selection and result contracts remain unchanged.
- Final static scope: six production Python files plus two test files compile; five TypeScript/TSX files parse. Ruff reports only the inherited facade I001 and ten inherited repository UP045 diagnostics, zero new findings; the two test files are clean. Initial static commands could not write the worktree Ruff cache; the final normal configured read uses `--no-cache`, with no source or lint rule change. Configured ESLint exits 0 for all five files, retaining exactly 24 service and six TestCasesTab warnings and zero new warnings/errors.
- Bandit production baseline/current: zero findings/errors. Full eight-file current scan retains only four identical inherited parity-test B101 assertions; new pytest contract assertions and synthetic scrub markers have precise false-positive annotations, preserving their AST exactly. No application security guard changed for qualification.
- Root independently read all history hunks, complete new backend/mounted bodies and connected owner/caller/GET contracts and found no finding in that scope. Source is frozen for root's final configured types/build/live read of existing records. No live acceptance, PostgreSQL acceptance or all-family acceptance is claimed. Existing immutable producers, storage and migrations are untouched.
