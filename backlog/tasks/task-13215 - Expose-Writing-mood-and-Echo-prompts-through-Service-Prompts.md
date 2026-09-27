---
id: TASK-13215
title: Expose Writing mood and Echo prompts through Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-07 20:14'
updated_date: '2026-09-07 20:54'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2930'
documentation:
  - Docs/Design/writing-feedback-service-prompts.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded follow-up to Writing Agent prompts: expose mood classifier semantics and five Echo persona instructions through existing Service Prompts in both clients, preserving defaults and scope isolation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Mood and all five Echo instructions are editable through existing Settings
- [x] #2 Default requests, output contracts, limits, rotation and enablement remain compatible
- [x] #3 Account or server changes abort stale requests and clear old-scope feedback
- [x] #4 Focused regressions, lint and Bandit pass with documented verification
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Register literal parts and shared Settings/fallback definitions test-first. 2. Capture request-scoped snapshots in feedback hook with stale-result protection test-first. 3. Verify regression suites, security checks and review; update design and task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented and independently reviewed on codex/writing-feedback-service-prompts from dev e3174f1ad9. Test-first red/green captured for missing definitions/fallbacks, customized/scoped hook requests, Settings classifier label, and stale scope-error race. Verification: 97 backend registry/API tests; 281 focused shared-client tests; 47 direct-browser transport tests; 12 locale mirrors match; Bandit zero findings; Ruff clean; ESLint zero errors and 10 preexisting explicit-any warnings. Shared-UI tsc completes with 158 existing diagnostics, none in changed files, with 8 GiB Node heap. Full-repo tests/build/live browser smoke not run. Ready for integration choice; no PR pushed yet.

PR #2930 created against dev at user request. Feature commit ef699229c6 pushed. Awaiting remote review/checks and integration; worktree retained.

Qodo review at ef699229c6 reported three findings: silent unexpected-error diagnostics, missing registry-test docstring, and retaining new scope leases for failed/invalid feedback. Verified against current head; addressing with targeted regressions while preserving leases for existing visible feedback.

Addressed all three Qodo findings: request-local snapshots now release on invalid/failed/empty/cancelled results and only usable feedback replaces a retained visible-state lease; unique callbacks preserve listener ownership. Unexpected failures log static kind and phase only, with cancellation/scope control flow quiet. Added Python test docstring. Ten added regression cases reproduced failures before the fix. Verification: 249 affected client tests and 97 backend tests pass; touched-scope ESLint/Ruff clean, Bandit zero findings; independent review approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exposes mood classifier guidance and five Echo persona prompts through existing shared Service Prompts settings. Preserves exact default chat payloads and fixed output/rotation/limits, uses existing older-server fallbacks, and binds async generation plus displayed feedback to its starting account/server with bounded retained scope leases. No separate storage or settings system.
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
