---
id: TASK-13209
title: Expose Writing Agent mode prompts through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-06 15:38'
updated_date: '2026-09-06 16:05'
labels: []
dependencies: []
documentation:
  - Docs/Design/writing-agent-service-prompts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded slice: make Quick, Planning, and Brainstorm Writing Playground AI Agent instructions editable in shared WebUI/extension Settings using existing owner-scoped Service Prompts. Preserve defaults, model settings, manuscript bounds and older-server compatibility; prevent cross-scope context and stale replies after account/server, project, or mode changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three mode-specific prompts support existing Settings save/reset and preserve packaged defaults.
- [x] #2 Each send captures one owner-scoped prompt configuration and uses that scope for manuscript reads and model dispatch.
- [x] #3 Stale context, replies and errors are discarded after scope, project or mode changes; unscoped service callers remain compatible.
- [x] #4 Affected frontend/backend tests, lint, Bandit and code review pass with known skips documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish baseline and add regression tests. 2. Register three prompts and wire scoped context/generation using existing helpers. 3. Verify Settings, old-server fallback, request lifecycle, and review the bounded diff.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Baseline: 90 registry/API and 91 frontend checks passed. RED: component 11 failures/1 pass, snapshot compatibility 4 failures/79 passes, Settings 3 expected failures, backend registry 2 expected failures. Implemented 3 literal mode definitions, shared Settings labels, explicit packaged fallback allowlist, optional scoped manuscript reads, and retained history scope lease. Additional regression reproduced unverified failed-load text leaking into later request history; now excluded. Backend registry/API: 93 passed. Broad frontend, WebUI shim, lint/typecheck and independent read-only review in progress. Bandit on touched Python runtime and Ruff on touched Python runtime/tests pass.

Independent review found and verified two issues: first-load errors without a retained lease could remain visible across account changes; scoped manuscript GETs were rejected by the existing transport allowlist. Both addressed test-first. Unbound config/auth cleanup regressions: 2 RED then 17 component tests GREEN. Scoped transport regressions: 6 RED then 70 real-transport/guard tests GREEN. HTTP expected-user regressions: 3 RED, 6 compatible cases passed; now adding the existing expected-user dependency only to those three GET endpoints. Follow-up review approved with no actionable findings. Broader scope suite: 286 passed. Shared Settings/save-reset regression and other affected suites passed, with final totals pending. WebUI Settings 87 passed and component rerun underway. Bandit on both Python runtime files reports zero findings. Five Ruff findings in preexisting manuscript endpoint/test code were verified against HEAD; other touched Python files pass. Frontend ESLint has no errors; 10 existing any warnings in tldw-server.ts. Full shared TypeScript check reports 158 existing diagnostics outside changed runtime code; corrected a preexisting over-broad fixture Record annotation. Locale generator run, retaining only this slice’s seven generated keys to avoid unrelated prior locale drift.

Final backend verification: 132 passed (manuscript HTTP integration plus Service Prompts registry/API), including all nine optional expected-user cases. Current frontend coverage: 438 shared UI tests and 104 WebUI-harness tests pass across focused runs; consolidated shared UI rerun pending. Bandit: zero findings on both touched runtime files. Independent follow-up review approved, no remaining actionable findings. Full repository suite, production frontend builds and live-browser smoke were not run; focused suites match the approved design. Existing TypeScript diagnostics (158; none on changed lines), five preexisting Ruff findings in manuscript files, and existing frontend lint warnings are not newly introduced. No PR exists yet; branch is ready for integration choice after commit.

Consolidated final shared frontend run completed: 438 passed across 12 files (65.78 seconds), zero failures. Final backend run: 132 passed, zero failures. WebUI harness: Settings 87 and current component 17 passed. Fresh Bandit and Python compilation also pass. Temporary dependency symlinks removed before staging; no dependency changes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Quick, Planning and Brainstorm prompt customization through existing shared Service Prompts Settings. Preserved exact defaults and generation settings, added narrow scoped manuscript GET support and optional server-side identity assertions, and retained a scope lease to clear stale requests/history. Older-server fallback remains explicit and auth/non-404 errors propagate. Verified actionable review findings with failing regressions before minimal fixes. Implementation and review complete; awaiting PR/integration choice.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
