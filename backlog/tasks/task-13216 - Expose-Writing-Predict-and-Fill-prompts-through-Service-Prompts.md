---
id: TASK-13216
title: Expose Writing Predict and Fill prompts through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-07 21:52'
updated_date: '2026-09-07 22:50'
labels: []
dependencies: []
documentation:
  - Docs/Design/writing-continuation-service-prompts.md
  - Docs/superpowers/plans/2026-09-07-writing-continuation-service-prompts.md
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
- [ ] #5 Focused regression tests, lint, Bandit and independent review pass for the touched scope.
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
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
In-progress handoff at d95697c95c: implementation stages 1 and 2 are complete and independently approved; focused final verification is green with 322 UI and 101 backend tests, Ruff clean, production Bandit zero findings, ESLint zero errors, locale parity, and zero new TypeScript diagnostics versus baseline. TASK-13216 remains In Progress until the controller completes the full-branch review and final cleanup. Full frontend builds and live-browser checks were not run; baseline warning categories and reproduced base TTS failures are documented in the implementation notes and design verification results.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
