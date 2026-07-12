---
id: TASK-12109
title: Design per-video YouTube playlist ingestion visibility for WebUI and extension
status: In Progress
labels:
- webui
- browser-extension
- media-ingestion
- design
priority: high
modified_files:
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
references:
- Docs/superpowers/specs/2026-05-16-bulk-conference-ingest-workflow-design.md
documentation:
- Docs/superpowers/specs/2026-07-12-youtube-playlist-per-item-ingest-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review the shared WebUI/browser-extension playlist ingestion flow and specify a fail-closed, server-owned preflight and per-video queue/progress/results contract. This task covers the approved design specification only; implementation planning follows after human review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document the current single-row playlist failure mode across shared frontend and backend worker paths.
- [x] #2 Specify mandatory playlist preflight, per-video identity, queue, progress, cancellation, persistence, and recovery behavior for both clients.
- [x] #3 Specify scalable snapshot pagination, run/chunk/job contracts, error handling, security, and testing strategy.
- [x] #4 Commit the reviewed design specification and link it from this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming/design workflow only. A separate implementation plan will be written after the requester approves the committed specification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Completed the brainstorming design and three independent written-spec review iterations. The final two review issues were resolved with requester approval: queue materialization preserves source identity only; Start Processing supplies validated Review-time overrides after a fresh duplicate lookup; every selected occurrence resolves once, while only processing-required actions create a media job; file_reattach_required is client presentation over server awaiting_upload.

Verification: git diff --check passed for the specification and task record. A targeted placeholder and stale-language scan returned no matches. Positive contract checks confirmed review_required/review_overrides, exactly-once resolution with processing-only jobs, and client-derived file reattachment wording. Bandit was not run because this task changes documentation only. No implementation blocker is known; requester file review remains the gate before implementation planning.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
