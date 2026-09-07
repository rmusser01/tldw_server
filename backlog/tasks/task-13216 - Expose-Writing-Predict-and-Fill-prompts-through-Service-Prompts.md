---
id: TASK-13216
title: Expose Writing Predict and Fill prompts through Service Prompts
status: Done
assignee: []
created_date: 2026-09-07 21:52
updated_date: 2026-09-07 23:16
labels: []
dependencies: []
documentation:
- Docs/Design/writing-continuation-service-prompts.md
references:
- https://github.com/rmusser01/tldw_server/pull/2931
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved bounded continuation slice: expose Writing Playground non-chat Predict and Fill system prompts in the existing shared Service Prompts settings. Preserve defaults, explicit chat behavior, context/templates, stopping/provider options, insertion and undo semantics; bind requests to their starting account/server and discard stale output. No new settings system or continuation-engine redesign.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Predict and Fill have independently editable literal system parts in the existing backend registry and shared Settings, with packaged-default compatibility.
- [x] #2 Both streaming and non-streaming non-chat continuation use one scope-bound prompt snapshot; chat mode keeps its existing explicit system precedence and does not load these prompts.
- [x] #3 Cancellation, scope changes, session/scene changes and unmount cannot apply stale text, logprobs, history, errors or finalizers to a newer operation.
- [x] #4 Default payloads, context/template and stop behavior, provider controls, manual-stop partial output and insertion/undo behavior remain compatible.
- [x] #5 Focused regression tests, lint, Bandit and independent review pass for the touched scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record approved design and concrete implementation plan. 2. Test-first registry/defaults/settings support. 3. Test-first scoped Predict/Fill request integration and lifecycle guards. 4. Focused verification, independent review, tracking update and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approved design and three-stage plan remain linked.

Stage 1 complete at ae91269a78: exact independent literal Predict/Fill definitions, packaged 404 fallback, Settings save/reset and mirrored locale support. Backend 101/101 and shared UI 225/225 passed; Ruff clean; production Bandit zero findings; ESLint zero errors with 10 baseline no-explicit-any warnings. Independent Task 1 review approved.

Stage 2 complete at d95697c95c: non-chat continuation consumes one scope-bound snapshot, while chat retains explicit message precedence. Ownership guards cover lookup, transport, binding invalidation, manual-stop partial output, stale callbacks and cleanup. Focused Task 2 union passed 185/185; independent Task 2 review approved. Preserved rulings: manual Stop remains streaming-only, and early target mismatch uses the canonical scope-change error.

Stage 3 focused verification at d95697c95c: seven-file UI union 322/322 passed; backend registry/API 101/101 passed with 14 existing warnings; Ruff clean; production Python Bandit zero findings and zero errors across 748 lines; repository-pinned ESLint zero errors with 37 baseline warnings; five English locale values match. Controller fresh typecheck has 158 diagnostics, exactly matching the 158-diagnostic pre-change baseline after line/column normalization, so zero new diagnostics.

Known limitations and baseline output: full frontend builds and live-browser checks were not run. Vitest retains Ant Design Drawer deprecation, Node experimental localStorage, jsdom navigation and expected abort/timeout logs. ESLint retains 27 Writing Playground unused-variable/React-hook warnings, 10 tldw-server no-explicit-any warnings and the root pages-directory notice. Two incidental TTS failures reproduced unchanged at base 6cd2745f69; no TTS changes were made.

Stage 3 remains in progress pending controller full base-to-head review, plan removal and temporary dependency-symlink cleanup. This record does not claim whole-branch approval.

Final review found and test-first fixed provisional continuation autosave through revision controls in 3f67f71fa4. Scoped re-review approved all findings addressed. Final combined UI run: 388/388 across 13 files; backend 101/101. Post-fix TypeScript 158 diagnostics exactly matches baseline. Production Bandit zero findings; Ruff clean; changed-file ESLint zero errors with baseline warnings. All implementation stages complete; task-owned implementation plan removed and dependency symlinks cleaned. Adjacent unchanged idle Apply persistence issue tracked separately as TASK-13217. Earlier in-progress notes are superseded by this final closeout.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented independent literal Predict/Fill Service Prompts in shared WebUI/extension Settings using existing registry, storage and 404-compatible defaults. Non-chat generation captures one scope-bound snapshot; chat precedence and provider/context/stop behavior remain unchanged. Operation ownership and revision mutation guards prevent stale or invalidated continuation output from reaching editor state, history or autosave. Task reviews and final scoped fix review approved. Final verification: 388 client tests across 13 files plus 101 backend tests pass; Ruff clean; production Bandit zero findings; ESLint zero errors with baseline warnings; five locale entries match; post-fix TypeScript158 diagnostics exactly matches baseline. Full builds/live-browser checks not run. Two unrelated TTS failures reproduced original base. Separate unchanged idle-Apply save issue tracked as TASK-13217. Implementation plan and temporary dependency symlinks removed; branch ready for user-selected integration.
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
