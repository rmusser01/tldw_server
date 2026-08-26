# WebUI Tier 1-3 Live-Backend UAT Results

Date: 2026-08-25

## Certification outcome

The complete Tier 1, Tier 2, and Tier 3 Playwright inventory passed against an isolated real FastAPI backend and the repository's deterministic OpenAI-compatible mock service.

- Run ID: `task13124-pr2822-review-162e8a4b`
- Tested commit: `162e8a4bb2a8ce08d83c3e30357b3599c6dd14b7`
- Command: `bun run uat:live-tiers -- --run-id=task13124-pr2822-review-162e8a4b`
- Execution: one worker, zero retries, offline fallback disabled, skips rejected
- Health: healthy before and after the tests; all spawned services stopped
- Cleanup: disposable backend profile and isolated Next build directory removed
- Egress: retained backend log contains no non-loopback HTTP request
- Source-tree isolation: no generated Watchlist templates, ACP audit database, or live-tier database files remain in the repository

| Project | Listed | Passed | Failed | Skipped | Interrupted | Intercepted tests | Live tests |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tier 1 | 34 | 34 | 0 | 0 | 0 | 0 | 34 |
| Tier 2 | 104 | 104 | 0 | 0 | 0 | 4 | 100 |
| Tier 3 | 37 | 37 | 0 | 0 | 0 | 1 | 36 |
| **Total** | **175** | **175** | **0** | **0** | **0** | **5** | **170** |

The intercepted classification is test-level: the report's route table contains every intercepted matcher installed by those five tests. Intercepted cases count as deterministic UI/contract coverage, not live-backend evidence.

Retained evidence:

- `apps/tldw-frontend/test-results/live-tier-uat/task13124-pr2822-review-162e8a4b/summary.json`
- `apps/tldw-frontend/test-results/live-tier-uat/task13124-pr2822-review-162e8a4b/report.md`
- `apps/tldw-frontend/test-results/live-tier-uat/task13124-pr2822-review-162e8a4b/playwright-results.json`
- `apps/tldw-frontend/test-results/live-tier-uat/task13124-pr2822-review-162e8a4b/backend.log`

## Defect and improvement loop

Every confirmed issue received a child task under `TASK-13124` before its repository fix. The remediation covered:

- false outage popups and cancelled-request classification;
- canonical notes-to-chat routing and media delete/review error recovery;
- media analysis persistence, exact refresh verification, and deterministic first/second model responses;
- multi-review content extraction and concurrent permalink hydration;
- TTS provider readiness, chat settings, startup auth readiness, and themed chat-save notifications;
- tracked character/persona selection, chat creation/persistence, and picker refetch behavior;
- retained WebUI/extension real-server gate parity, readiness, background-ping behavior, fallback removal, and fixture cleanup;
- Chatbooks export/import, Document Workspace shortcuts, Kanban empty state, and Sources live sync/duplicate-CTA handling;
- Jobs, ACP session/audit, Evaluations, ingestion source, monitoring, system-log, Watchlist, and Next artifact isolation;
- strict inventory/list accounting, skip/failure/interruption rejection, deterministic certification labeling, and grep parity;
- best-effort process teardown, signal-safe child-group cleanup, setup-preflight cleanup, and truthful stopped/report evidence;
- host provider endpoint/credential scrubbing plus a fail-closed non-loopback request guard.

The 17-case legacy real-server suite was reviewed before reduction. Four unique cross-surface live workflows remain; 13 redundant cases map to narrower maintained coverage in `Docs/superpowers/reviews/2026-08-25-real-server-workflow-coverage-map.md`.

## Verification gates

- Frontend touched regression suite: 34 files, 522 tests passed.
- Runner lifecycle suites after final cleanup hardening: 65 tests passed; the live-tier suite alone passed 30/30 after the final type fixture correction.
- Backend/mocked-provider focused suite: 178 passed, 2 environment-dependent skips.
- Production build: passed with `NEXT_PUBLIC_API_URL=http://127.0.0.1:8000`. Existing broad documentation-trace and stale Browserslist warnings remain non-blocking; two trace-copy warnings named stale extension-test browser profiles that had no open owner and were removed by exact path afterward.
- Touched-scope ESLint: passed with no errors.
- `git diff --check`: passed.
- Bandit touched Python scope: no finding intersects a changed hunk. The report contains nine pre-existing findings outside the edited lines (eight SQL-construction warnings and one hardcoded-password false positive).
- Full frontend typecheck: no diagnostic in a touched file. The command remains non-zero because of existing Presentation Studio and skills-certification test diagnostics outside this task.

## Diagnostic history

Earlier full runs were deliberately not certified until the evidence was trustworthy. They exposed and drove fixes for intercepted-helper undercounting, stale/missing Playwright JSON acceptance, incomplete teardown, non-isolated mutable paths, host provider egress, signal interruption leaks, deterministic analysis generation, and the Sources duplicate-locator failure. The final exact-commit run supersedes those diagnostic attempts.

The superseding PR-review run also includes fixes for connection-hysteresis outage confirmation, version-only media-analysis persistence verification, removed persisted model fallback, phase-specific test registration, and an order-dependent Research Workspace controller assertion. All 21 review threads were answered and resolved before the final merge gate.
