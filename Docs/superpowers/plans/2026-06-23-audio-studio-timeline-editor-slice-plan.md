# Audio Studio Timeline Editor Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first usable Audio Studio timeline editor slice for arranging server-backed tracks and clips.

**Architecture:** The frontend renders timeline state from the active server-backed Audio Studio project and persists clip edits through the existing `PUT /api/v1/audio-studio/projects/{project_id}/clips/{clip_id}` service. This slice does not add a new backend model; it keeps all timeline mutations pinned to the active project revision and existing clip/track ids.

**Tech Stack:** React, Ant Design, Zustand Audio Studio store, React Query mutations, existing Audio Studio service helpers, Vitest/Testing Library, Playwright smoke coverage where useful.

---

## Scope

- Add a timeline editor component below the workflow editor where it has horizontal space.
- Show track rows and positioned clips using a stable time scale.
- Select a clip and edit start, duration, volume, fade-in, fade-out, and muted state.
- Persist selected clip edits via `useUpsertAudioStudioClip`.
- Add a lightweight live-preview transport that scrubs a playhead over the current timeline state. Actual artifact audio playback remains a later slice because the WebUI does not yet expose artifact download/playback URLs.
- Keep render/export controls separate.

## Out Of Scope

- Backend schema changes.
- True waveform rendering from decoded artifact audio.
- Split/crossfade editing.
- Multi-clip drag/drop across tracks.
- Artifact URL/download API work.

## Task 1: Hook Support For Clip Persistence

**Files:**
- Modify: `apps/packages/ui/src/hooks/useAudioStudioProjects.ts`
- Test: `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`

- [ ] Add `useUpsertAudioStudioClip(projectId)` mirroring the existing section hook.
- [ ] Invalidate Audio Studio project queries after a successful clip mutation.
- [ ] Mock the hook in existing Audio Studio page tests.

## Task 2: Timeline Editor Component

**Files:**
- Create: `apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx`
- Modify: `apps/packages/ui/src/components/Option/AudioStudio/AudioStudioPage.tsx`
- Test: `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`

- [ ] Write a failing test that renders active project tracks/clips in a timeline panel.
- [ ] Write a failing test that selecting a clip, changing trim/fade/volume controls, and saving calls `useUpsertAudioStudioClip` with `base_revision_id`, `track_id`, `clip_type`, `start_ms`, `duration_ms`, `volume`, `fade_in_ms`, and `fade_out_ms`.
- [ ] Implement a compact track/clip timeline with stable row heights and a time ruler.
- [ ] Add numeric controls for start, duration, volume, fade-in, fade-out, and muted.
- [ ] Use existing clip settings to preserve clip type/title/fade values when available.

## Task 3: Preview Transport

**Files:**
- Modify: `apps/packages/ui/src/components/Option/AudioStudio/TimelineEditor.tsx`
- Test: `apps/packages/ui/src/components/Option/AudioStudio/__tests__/AudioStudioPage.test.tsx`

- [ ] Write a failing test for play/pause and scrubber movement state.
- [ ] Implement a deterministic interval-based playhead that loops/stops at the timeline end.
- [ ] Keep preview transport client-only; do not persist playhead position.

## Task 4: Focused Verification And Backlog Closure

**Files:**
- Modify: `backlog/tasks/task-2352 - Add-Audio-Studio-timeline-editor-slice.md`

- [ ] Run focused Vitest for Audio Studio page tests.
- [ ] Run ESLint on touched Audio Studio files.
- [ ] Run scoped `git diff --check`.
- [ ] Record known limitation: preview scrubber exists, artifact audio playback waits for artifact URL API.
- [ ] Commit only the scoped timeline files and `TASK-2352`.
