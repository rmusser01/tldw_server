---
id: TASK-415
title: Design media viewer read-along TTS interaction
status: Done
labels:
- design
- webui
- extension
- tts
- media
modified_files:
- Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md
- Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md
- backlog/tasks/task-415 - Design-media-viewer-read-along-TTS-interaction.md
- backlog/tasks/task-416 - Plan-media-viewer-read-along-TTS-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for selection-initiated read-along functionality in the shared media viewer, covering WebUI and extension behavior, TTS reuse, segmentation, highlighting, local audio cache, error states, testing, and rollout boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the approved selection-first interaction model.
- [x] #2 Spec anchors the design to shared ContentViewer/WebUI/extension paths and existing TTS services.
- [x] #3 Spec covers hybrid transcript-line/sentence segmentation, range expansion actions, active highlighting, chunked lookahead, browser-local cache, error handling, and v1 exclusions.
- [x] #4 Spec includes testing and staged rollout guidance.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design spec drafted at Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md.

Spec review loop:
- Pass 1 found planning conflicts around active transport visibility, full-item large-content behavior, and markdown/html fallback.
- Pass 2 approved after those fixes.
- Pass 3 approved after advisory clarifications for canonical Read from here behavior and mid-session TTS settings behavior.

Human-requested critique pass:
- Hardened the spec around the existing annotation selection capture path in ContentViewer so read-along and annotations share mediated selection actions instead of competing listeners.
- Added lazy/cancellable segmentation requirements for large media items and explicit abort/stale-result suppression for in-flight TTS lookahead.
- Clarified provider synthesis reuse, request cap splitting, Dexie schema/type migration requirements, storage quota/privacy constraints, autoplay rejection handling, and embedded media preview pause behavior.

Verification:
- Marker scan found no unfinished-work markers in the spec or task after the critique patch.
- git diff --check passed for the spec and task files.
- Bandit is not applicable because this change is docs/task tracking only.

2026-05-23 closeout review:
- Re-reviewed the design spec against all acceptance criteria and current `origin/dev`.
- Re-hardened the implementation plan at Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md with a status note showing the plan has already been executed by TASK-417 and post-PR review fixes were recorded in TASK-425.
- Confirmed current code ownership for shared `ContentViewer` integration, mediated annotation/read-along selection, read-along modules, TTS provider abort support, Dexie cache schema, and WebUI/extension route parity.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
The design spec remains the authoritative v1 product contract. No spec changes were required during the closeout pass because the existing document already covers the selection-first entry model, shared UI ownership, existing TTS reuse, segmentation/highlighting/lookahead/cache/error behavior, v1 exclusions, staged rollout, and testing guidance.

Backlog note: this repository currently has duplicate `TASK-416` IDs, so MCP task lookup resolves `TASK-416` to an unrelated chat task. The media read-along plan file was updated directly by path to avoid modifying the wrong task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the media viewer read-along TTS design task. The approved spec documents the selection-first interaction model, anchors the work to shared `ContentViewer`/WebUI/extension surfaces and existing TTS services, covers hybrid transcript/sentence segmentation, range expansion actions, active highlighting, chunked lookahead, browser-local cache, error handling, and v1 exclusions, and includes staged rollout plus testing guidance. The implementation plan was re-reviewed and updated with current execution status: implementation landed through TASK-417 and PR review feedback through TASK-425. Verification for this closeout used current-code ownership `rg` checks, Backlog evidence review, unfinished-marker checks, and `git diff --check`; Bandit is not applicable because only Markdown/Backlog metadata changed.
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
