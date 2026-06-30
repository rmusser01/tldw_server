---
id: TASK-2352
title: Add Audio Studio timeline editor slice
status: Done
labels:
- audio
- backlog
priority: medium
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/plans/2026-06-23-audio-studio-timeline-editor-slice-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up non-MVP task for a waveform timeline editor in Audio Studio, including clip dragging, trim/fade controls, and live preview after the server-backed MVP is complete.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Waveform timeline design is defined after MVP route and render/export services exist.
- [x] #2 Timeline editor supports clip dragging, trim/fade controls, and live preview.
- [x] #3 Implementation preserves the MVP server-backed revision/artifact model.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement the first bounded timeline editing slice from Docs/superpowers/plans/2026-06-23-audio-studio-timeline-editor-slice-plan.md now that TASK-2351 is complete. Keep backend schema unchanged, persist edits through existing Audio Studio track/clip APIs, and record the artifact-audio playback limitation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added Docs/superpowers/plans/2026-06-23-audio-studio-timeline-editor-slice-plan.md to bound the first editor slice after TASK-2351.
- Added a full-width Audio Studio timeline panel that renders server-backed tracks and clips from the active project.
- Added selected-clip editing for start, duration, volume, fade-in, fade-out, and mute state.
- Added horizontal pointer dragging for selected clips. Dragging updates the draft start time and can be persisted through the existing clip upsert API.
- Added `useUpsertAudioStudioClip(projectId)` and kept persistence on the existing `PUT /api/v1/audio-studio/projects/{project_id}/clips/{clip_id}` contract with `base_revision_id`.
- Added a lightweight preview transport and scrubber over current timeline state.
- Updated E2E page object/spec coverage to assert the Timeline panel appears on `/audio-studio`.
- Known limitation: the preview transport scrubs timeline state only. Actual artifact audio playback remains blocked until the WebUI exposes safe artifact playback/download URLs.
- Verification on 2026-06-23:
  - `bunx vitest run ../packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx ../packages/ui/src/services/__tests__/audio-studio.test.ts` -> 2 files passed, 27 tests.
  - Shared UI ESLint for touched Audio Studio files -> 0 errors, 0 warnings; existing Next pages-directory notice only.
  - E2E ESLint for touched Audio Studio page object/spec -> 0 errors, 0 warnings.
  - `bunx playwright test e2e/workflows/tier-2-features/audio-studio.spec.ts e2e/workflows/tier-2-features/audiobook-studio.spec.ts --reporter=line` -> 5 passed.
  - `git diff --check` on touched task paths -> clean.
  - `bunx tsc --noEmit` still fails in unrelated baseline files (`TaskActivityNotice`, Evaluations embeddings config, WritingPlayground, persona visuals, VN play API, calendar route import). No Audio Studio paths appeared in the typecheck errors.
- Bandit is not applicable to this slice because it changed frontend TypeScript, E2E tests, docs, and Backlog metadata only.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first Audio Studio timeline editor slice: track/clip rendering, clip selection, pointer drag start-time editing, trim/fade/volume/mute controls, revision-pinned clip persistence through the existing API, and a lightweight preview scrubber. Browser and unit coverage now include the timeline panel and editor behaviors. Actual artifact audio playback remains a follow-up pending safe artifact playback URLs.
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
