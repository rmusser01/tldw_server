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
- [ ] #1 Document the current single-row playlist failure mode across shared frontend and backend worker paths.
- [ ] #2 Specify mandatory playlist preflight, per-video identity, queue, progress, cancellation, persistence, and recovery behavior for both clients.
- [ ] #3 Specify scalable snapshot pagination, run/chunk/job contracts, error handling, security, and testing strategy.
- [ ] #4 Commit the reviewed design specification and link it from this task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Brainstorming/design workflow only. A separate implementation plan will be written after the requester approves the committed specification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
