# Task 8 Report: Shared WebUI Data Layer, Generate, Create, And Manage

## Status

Complete on `TASK-12102.3.5.4` from head `60382f75f0`.

## Baseline

- The supplied linked worktree and branch were clean at the requested prior head.
- `apps/node_modules` was absent, so the first neighboring Vitest command failed while loading `vitest/config` and `@vitejs/plugin-react`.
- Ran `bun install --frozen-lockfile` from `apps/`; no tracked lockfile change was produced.
- Neighboring service and Generate/Create/Manage baseline: 4 files, 31 tests passed. Existing Ant `Alert.message` deprecation and Node localStorage warnings remain.

## TDD Evidence

RED:

- Exact focused command: 4 files failed. Service, query-hook, and editor suites failed to resolve the intentionally absent Task 8 modules; all three Generate/Create tests failed because the OSCE profile and activity selector were unavailable.
- A first run also exposed JSX in the required `.ts` service-test filename. The test harness was corrected to use `React.createElement`, then RED was rerun so the recorded failures are attributable only to missing production behavior.

GREEN:

- Exact focused command: 4 files, 18 tests passed.
- Neighboring Quiz service and Generate/Create/Manage command: 16 files, 90 tests passed.
- QuizPlayground navigation regression: 1 file, 22 tests passed.

## Implementation

- Added the typed OSCE service and React Query hooks for compact station pages, authoring CRUD, discriminated candidate/revealed attempts, repeated state filters, optimistic-version mutations, stable keys, and affected cache invalidation.
- Added the shared full-width `OsceStationEditor` with candidate content, bounded duration, ordered checklist/rubric/key-point controls, citations, candidate preview, explicit save, dirty navigation signaling, and 409 reload/keep/confirm recovery.
- Added catalog-aware OSCE Generate controls, preserving source, difficulty, focus, generation-provider, and verification-provider inputs while suppressing question-only controls and routing successful generation to Manage.
- Added segmented Quiz/OSCE Create composition and the shared explicit-save station workflow. Question draft autosave remains isolated to question quizzes.
- Added compact OSCE station management with verification labels, explicit-save editing, dirty selection/close guards, OSCE-incompatible action suppression, and ordinary Quiz behavior retained.
- Added client portability that keeps question-only exports at v1, emits v2 for mixed/OSCE exports, recursively removes sensitive provenance keys, and passes v1/v2 imports to the Task 7 endpoint.

## Verification

- `bun run lint`: exit 0 with 0 errors and the existing 169-warning frontend baseline. The new Task 8 files also pass the frontend ESLint config with no file diagnostics.
- `bun run typecheck`: blocked by existing errors in Presentation Studio and skills-certification test files. No diagnostic references a Task 8 or Quiz service/component path.
- `git diff --check`: passed.
- Bandit: not applicable; this slice changes TypeScript/TSX and task documentation only, with no Python touched.
- Prettier check is not a passing repository baseline: an untouched Quiz file (`ResultsTab.tsx`) and the existing shared surface fail the configured check. No broad formatting rewrite was applied.
- Commit: `feat(webui): add OSCE authoring workspace` (this completed slice).

## Concerns

- Practice autosave, timer, Take, and Results remain deferred to Task 9.
- The OSCE fallback/catalog remains planned; Task 8 only exposes controls when a test or server catalog marks the profile available.
- Repository-wide typecheck remains blocked by unrelated baseline errors listed above.

## Review Fix Round 1

Status: complete from Task 8 head `58f0491602`; committed separately as `fix(webui): harden OSCE authoring workflows`.

RED evidence:

- Expanded Task 8 command: 6 files, 11 failed and 44 passed. Failures covered OSCE launch suppression, partial-create reuse, ambiguous response handling, conflict order preservation, pagination, citation/rubric validation, and Manage reset guarding.
- The first full Quiz run also identified five older CreateTab suites whose complete hook mocks needed the newly consumed OSCE mutation hook. After those fixtures were corrected, all 13 affected Create tests passed.

GREEN evidence:

- Exact Task 8 focused command: 4 files, 25 tests passed.
- Take and navigation regressions: 2 files, 31 tests passed.
- Full Quiz component suite with bounded concurrency: 38 files, 259 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors. Scoped shared UI lint also reports no errors.
- Frontend typecheck remains blocked by the existing Presentation Studio and skills-certification diagnostics. No diagnostic references a changed Task 8, Quiz, or OSCE path.
- `git diff --check`: passed after the final report/task updates.
- Bandit remains not applicable because this review fix changes TypeScript, TSX, tests, and task documentation only.

Fixes:

- Hid question-attempt Start, Practice, and Review entry points for OSCE rows and blocked direct OSCE auto-start until Task 9 adds its practice route.
- Retained a confirmed manual OSCE quiz shell ID across station-save retries, routed station creation through the shared mutation/invalidation contract, and locked shell metadata while retrying. Ambiguous shell-create responses now fail closed and direct the author to inspect Manage instead of risking an automatic duplicate; the current API has no create idempotency key that could safely resolve that ambiguity.
- Carried the latest server `order_index` and version together through conflict recovery.
- Propagated Manage editor dirtiness to the playground so global reset requires confirmation.
- Added complete bounded station pagination for Manage and v2 export, including non-advancing-offset and maximum-page failure behavior instead of silent truncation.
- Aligned citation URL, rubric-label uniqueness, and offset pagination types with the backend/OpenAPI contract.

Concerns:

- The OSCE generation profile remains planned/hidden unless the server catalog explicitly marks it available.
- Task 9 practice, autosave, and results behavior was not started.
- The first unconstrained full Quiz run produced load-related five-second timeouts. The timed-out tests passed in focused reruns, and the complete suite passed with `--maxWorkers=2`.

## Review Fix Round 2

Status: complete from review-fix-round-1 head `1a3a032962`; committed separately as `fix(webui): close OSCE authoring retry and delete gaps`.

RED evidence:

- Initial three-file regression run: 8 failed and 14 passed. The failures demonstrated that ambiguous station creates remained retryable, recovered question drafts survived an OSCE switch, Manage had no station-delete action, and new station order used list length instead of the maximum stored order.

GREEN evidence:

- Expanded Task 8 suite: 5 files, 36 tests passed.
- Dedicated Take suite: 7 files, 64 tests passed.
- QuizPlayground navigation suite: 1 file, 23 tests passed.
- Full Quiz component suite with `--maxWorkers=2`: 38 files, 267 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors; `eslint --quiet` also exits 0.
- Frontend typecheck remains blocked by 80 existing diagnostics across six Presentation Studio, presentation E2E, and skills-certification files. No diagnostic references a changed Quiz or OSCE path.
- `git diff --check`: passed. Bandit remains not applicable because no Python changed.

Fixes:

- Classify the request client's concrete `error.status` shape so missing status, 408, and 5xx station-create failures fail closed after a confirmed shell. The local station draft remains visible, Save is disabled, and the author is directed to inspect Manage before creating another station. Definitive 4xx rejection remains retryable against the retained shell.
- Clear persisted, pending-recovery, and in-memory question content after a confirmed Questions-to-OSCE switch, preventing discarded questions from returning on remount.
- Add an accessible station delete control in Manage with unsaved-draft and destructive-action confirmation, expected-version deletion, cache-backed refresh, success selection clearing, and failure-state preservation.
- Compute new station order as the maximum existing `order_index` plus one, with zero for an empty station list, so sparse or colliding imported indexes are handled safely.

Concerns:

- The OSCE generation profile remains planned/hidden unless the server catalog explicitly marks it available.
- Task 9 practice, autosave, and results behavior was not started.

## Review Fix Round 3

Status: complete from review-fix-round-2 head `8f2eee0223`; committed separately as `fix(webui): preserve OSCE authoring state`.

RED evidence:

- Initial three-file regression run: 9 failed and 21 passed. The failures reproduced status-zero shell/station retry exposure, duplicate custom-create requests, dirty editor resets on same-station refresh, and missing dirty-draft guards for single and bulk parent-quiz deletion.

GREEN evidence:

- Focused round-3 suite: 3 files, 31 tests passed.
- Expanded Task 8 suite: 5 files, 45 tests passed.
- Dedicated Take suite: 7 files, 64 tests passed.
- QuizPlayground navigation suite: 1 file, 23 tests passed.
- Full Quiz component suite with `--maxWorkers=2`: 38 files, 276 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors; `eslint --quiet` also exits 0.
- Frontend typecheck remains blocked by 80 existing diagnostics across six Presentation Studio, presentation E2E, and skills-certification files. No diagnostic references a changed Task 8, Quiz, or OSCE path.
- `git diff --check`: passed. Bandit remains not applicable because no Python changed.

Fixes:

- Treat the production request client's `status: 0` transport failure as ambiguous for quiz-shell and station creation, preserving fail-closed behavior while keeping definitive 4xx station rejections retryable against the retained shell.
- Add a synchronous local save lock plus visible pending state around every station editor save path, including custom `onCreate`, so rapid repeated activation cannot start concurrent POSTs.
- Preserve dirty local station content and its acknowledged version across same-station background refreshes, allowing explicit save to reach normal 409 recovery. Clean refreshes and station identity changes still adopt server content.
- Guard single and bulk parent-quiz deletion when the active managed OSCE quiz has a dirty station draft. Cancellation leaves the quiz and draft intact; confirmation clears the manager and permits deletion.

Concerns:

- The OSCE generation profile remains planned/hidden unless the server catalog explicitly marks it available.
- Task 9 practice, autosave, and results behavior was not started.

## Review Fix Round 4

Status: complete from review-fix-round-3 head `7b496b41f3`; committed separately as `fix(webui): reconcile OSCE authoring operations`.

RED evidence:

- Initial four-file regression run: 10 failed and 39 passed. The failures reproduced retry exposure after every ambiguous direct station-create shape, value-derived citation row remounts, backend casefold mismatch, ambiguous-delete state drift, successful-create draft retention, and the stalled off-page ordinary start intent.
- Complete-range self-review added a shell-composition regression that initially failed because a quiz-shell transport failure also produced the station-create ambiguity warning.

GREEN evidence:

- Expanded Task 8 suite: 6 files, 63 tests passed.
- Dedicated Take suite: 7 files, 65 tests passed.
- QuizPlayground navigation suite: 1 file, 23 tests passed.
- Full Quiz component suite with `--maxWorkers=2`: 38 files, 286 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors; `eslint --quiet` also exits 0.
- Frontend typecheck remains blocked by 80 existing diagnostics across six unrelated files: 42 in skills-certification runner tests, 14 in lifecycle tests, 10 in profile tests, 3 in evidence tests, 10 in the Presentation Studio standalone HTML E2E test, and 1 in Presentation Studio. No diagnostic references a changed Task 8, Quiz, or OSCE path.
- `git diff --check`: passed. Bandit remains not applicable because no Python changed.

Fixes:

- Centralized request-error status and ambiguity classification, then applied editor-level fail-closed handling to every station-create path. Missing status, status zero, 408, and 5xx preserve the draft, block another POST, and direct the author to Manage or reload; definitive 4xx remains retryable. A typed shell-create wrapper lets CreateTab compose with the editor without duplicate station warnings or retry logic.
- Allowed a different direct ordinary quiz target to load while an existing attempt remains active, so off-page start or retake intent reaches its confirmation action and clears deterministically without replacing the active attempt first.
- Added UI-only citation row identities that survive source ID edits and removals without entering API payloads.
- Reconciled ambiguous station deletes against fresh list and detail reads, refreshed React Query state, treated confirmed 404 as deletion, and retained an actionable selected station when the server still reports it.
- Reset the shared editor to a blank station only after successful manual OSCE creation. Failed saves continue to preserve the local draft.
- Replaced locale lowercase comparison with dependency-free upper-then-lower Unicode normalization, matching backend casefold for `Straße` and `STRASSE` and other common special folds. JavaScript does not expose Python's complete Unicode casefold table, so exact parity for every code point is not guaranteed without shipping a mapping or dependency.

Complete Task 8 range self-review:

- Reviewed the Task 8 range from `60382f75f0` through the current changes for create/update/delete retry behavior, dirty-state identity/version handling, station pagination, OSCE action suppression, and query invalidation.
- Added and fixed the shell-versus-station ambiguity composition regression found during that review. No Task 9 practice, autosave, or results paths were introduced, and the OSCE profile remains planned.

Concerns:

- The OSCE generation profile remains planned/hidden unless the server catalog explicitly marks it available.
- Task 9 practice, autosave, and results behavior was not started.
- Exact full-Unicode Python casefold parity remains a documented JavaScript platform limitation beyond the tested backend-relevant fold cases.

## Review Fix Round 5

Status: complete from review-fix-round-4 head `03d223a890`; committed separately as `fix(webui): finalize OSCE authoring safeguards`.

RED evidence:

- Initial two-file regression run: 4 failed and 29 passed. The failures reproduced loss of ambiguous-create protection when the editor unmounted, the missing failed/successful list-reconciliation flow, the false duplicate result for `i`/`ı`, and missing fractional duration/page validation.
- The `ß`/`ẞ` regression passed before implementation because the existing approximation happened not to conflate that pair; it still proves the editor submits the values and presents the authoritative server 422. After the first implementation pass, 32 tests passed and the reconciliation retry test exposed an unstable loading-prefixed accessible button name.

GREEN evidence:

- Focused editor and Manage suite: 2 files, 33 tests passed.
- Expanded Task 8 suite: 6 files, 67 tests passed.
- Dedicated Take suite: 7 files, 65 tests passed.
- QuizPlayground navigation suite: 1 file, 23 tests passed.
- Full Quiz component suite with `--maxWorkers=2`: 38 files, 290 tests passed.
- Full frontend lint: exit 0 with the unchanged 169-warning baseline and no errors; `eslint --quiet` also exits 0.
- Frontend typecheck remains blocked by the same 80 existing diagnostics across six unrelated files: 42 in skills-certification runner tests, 14 in lifecycle tests, 10 in profile tests, 3 in evidence tests, 10 in the Presentation Studio standalone HTML E2E test, and 1 in Presentation Studio. No diagnostic references a changed Task 8, Quiz, or OSCE path.
- `git diff --check`: passed. Bandit remains not applicable because no Python changed.

Fixes:

- Lifted ambiguous station-create uncertainty into a per-quiz Manage state set. Selection changes and manager close/reopen no longer permit a second POST; Add remains disabled until an explicit station-list refetch with `throwOnError` succeeds. Failed/offline reconciliation remains blocked with a clear message, and the reload action has a stable accessible name while loading.
- Removed client-side rubric-label equivalence checking. Required labels and descriptions remain validated locally, while exact duplicate equivalence is server-authoritative and a definitive 422 remains visible and retryable. Tests submit both Python-equivalent `ß`/`ẞ` and Python-distinct `i`/`ı` pairs.
- Added integer precision and steps to duration/document-page inputs plus `Number.isInteger` validation before create or update submission.

Final Task 8 range self-review:

- Re-read the Task 8 brief and reviewed all 26 files changed since `60382f75f0`, covering wire types, repeated-state encoding, expected-version mutations, query invalidation, bounded pagination, portability redaction, Generate/Create/Manage composition, Take suppression, dirty guards, conflict recovery, and create/delete ambiguity handling.
- Confirmed the OSCE generation profile remains planned, no OSCE practice/autosave/results behavior was introduced, and no backend file changed.
- No additional Task 8 code defect was found in the final audit.

Residual risk:

- Station creation has no backend idempotency key or request-status lookup. The Manage guard is intentionally in-memory, so a hard page/app reload can lose the uncertainty marker, and a successful fresh list cannot prove which station corresponds to a lost POST response. Resolving that fully requires a backend contract change; the Task 8 UI now fails closed during the live Manage session and requires explicit fresh-list inspection.
- Rubric duplicate equivalence now requires a server round trip by design because JavaScript has no native Python-compatible full Unicode casefold operation.
- The repository-wide frontend typecheck baseline remains red in the unrelated files listed above.
- The OSCE profile remains planned and Task 9 was not started.
