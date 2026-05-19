---
id: TASK-415
title: Design media viewer read-along TTS interaction
status: In Progress
labels:
- design
- webui
- extension
- tts
- media
modified_files:
- Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for selection-initiated read-along functionality in the shared media viewer, covering WebUI and extension behavior, TTS reuse, segmentation, highlighting, local audio cache, error states, testing, and rollout boundaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Spec documents the approved selection-first interaction model.
- [ ] #2 Spec anchors the design to shared ContentViewer/WebUI/extension paths and existing TTS services.
- [ ] #3 Spec covers hybrid transcript-line/sentence segmentation, range expansion actions, active highlighting, chunked lookahead, browser-local cache, error handling, and v1 exclusions.
- [ ] #4 Spec includes testing and staged rollout guidance.
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
