---
id: TASK-13194
title: Expose synchronous video summarization in Service Prompts
status: In Progress
assignee: []
created_date: '2026-09-05 21:35'
updated_date: '2026-09-05 21:53'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2904'
documentation:
  - Docs/Design/video-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Approved bounded slice: expose video system instructions and recursive final-summary instructions through existing shared Service Prompts Settings. Resolve one owner-scoped configuration before processing, preserve request overrides and default behavior, and leave queued/direct legacy ingestion unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared WebUI and extension can edit/reset the atomic video system/final-summary pair
- [x] #2 One owner snapshot is reused across videos and recursive passes with explicit-part precedence
- [x] #3 Defaults, disabled analysis, chunking/transcription and queued/direct callers retain behavior
- [x] #4 Focused regression tests, security checks and independent review pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Record approved scope and establish baseline. 2. Add failing registry, multipart and video integration tests; implement minimal resolution and stage-specific prompt forwarding. 3. Add shared Settings metadata and editor test. 4. Verify backend/UI, Bandit and API generation; obtain independent review and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Baseline141 tests passed. RED evidence: new video integration10failed/8passed; video API404 test failed; shared Settings video editor test failed; canonical-provider cases2failed. Implemented owner snapshot plus separate final_summary forwarding; corrected import error and switched integration fixture from remote download to real MP4 uploads to avoid unrelated quota-user setup. Independent reviewer approved original diff and canonical normalization follow-up. Broader backend276passed/11warnings; shared UI199passed; WebUI-specific2passed/73deselected. Bandit zero findings/errors; changed-scope Ruff/compileall/diff checks pass. Existing Video_DL_Ingestion_Lib import sorting violation verified unchanged at base, not swept. OpenAPI export/typegen/fingerprint check passed with unchanged contract2073paths/3142schemas. Full repo suite, live browser/STT/provider runs and full frontend build/typecheck not run. Final post-normalization video/request-contract suite pending.

Post-normalization contract run reported43passed/8warnings in86.78s; process shutdown still being checked. Main backend run276passed, sharedUI199passed, WebUI2passed. No API fingerprint change required. Implementation ready for local commit and user integration choice; no PR created yet.

Final contract test process exited successfully (exit0), including shutdown. Temporary frontend dependency symlinks removed. All related local verification complete.

Opened PR2904 against dev from codex/video-summary-service-prompt. Implementation commitfb882adccc. Awaiting remote review and CI; worktree retained for follow-up.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented synchronous video Service Prompts using the existing registry, owner storage and shared Settings editor. A separate final-summary argument preserves stage-specific semantics while the owner-scoped system prompt applies throughout. Explicit empty request prompts remain explicit; canonical provider normalization avoids accidentally bypassing analysis. Existing direct/queued defaults remain unchanged. Independent review approved; focused backend, shared UI and WebUI regressions plus Bandit and OpenAPI validation passed.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
