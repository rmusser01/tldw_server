---
id: TASK-417
title: Implement media viewer read-along TTS
status: Done
labels:
- implementation
- webui
- extension
- tts
- media
references:
- Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md
- Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md
modified_files:
- apps/packages/ui/src/services/tts-provider.ts
- apps/packages/ui/src/services/__tests__/tts-provider.read-along.test.ts
- apps/packages/ui/src/components/Media/read-along/media-read-along-segments.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-segments.test.ts
- apps/packages/ui/src/components/Media/read-along/media-read-along-cache-key.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts
- apps/packages/ui/src/components/Media/read-along/media-read-along-cache.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/media-read-along-cache.test.ts
- apps/packages/ui/src/components/Media/read-along/useContentSelectionActions.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx
- apps/packages/ui/src/components/Media/read-along/useMediaReadAlongSession.ts
- apps/packages/ui/src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx
- apps/packages/ui/src/components/Media/read-along/MediaReadAlongPopover.tsx
- apps/packages/ui/src/components/Media/read-along/MediaReadAlongTransport.tsx
- apps/packages/ui/src/components/Media/ContentViewer.tsx
- apps/packages/ui/src/components/Media/hooks/useTranscriptDisplay.tsx
- apps/packages/ui/src/components/Media/__tests__/ContentViewer.read-along.test.tsx
- apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx
- apps/packages/ui/src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx
- apps/packages/ui/src/routes/__tests__/option-media-route-guards.test.tsx
- apps/tldw-frontend/__tests__/extension/entry-shell-performance.test.ts
- Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved selection-initiated media viewer read-along TTS feature in shared apps/packages/ui surfaces using the committed design spec and implementation plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-05-17-media-viewer-read-along-tts-implementation-plan.md with subagent-driven development.

Implementation constraints:
- Stay in the isolated worktree at .worktrees/media-read-along-tts on branch codex/media-read-along-tts.
- Follow TDD: write focused failing tests, verify red, implement, verify green, commit per task.
- Preserve existing annotation workflows through mediated selection actions.
- Keep work scoped to shared apps/packages/ui surfaces unless the plan explicitly says otherwise.
- Do not introduce backend APIs or duplicate extension-specific implementations.
- Record verification and known skips before final completion.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 baseline completed.

Dependency setup:
- Initial worker baseline could not run because the clean worktree had broken node_modules symlinks for Vitest.
- Ran `bun install` from `apps/` to restore dependencies. The command populated `apps/node_modules` enough for Vitest to run, but the install session did not exit cleanly through the tool after dependency resolution. Subsequent Vitest commands are usable.

Focused baseline results before read-along code edits:
- Command: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx src/services/__tests__/tts.defaults.test.ts src/db/dexie/__tests__/stt-recordings.test.ts --maxWorkers=1`
- Result: 17 passed, 1 failed.
- Reproduced with: `bunx vitest run src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx --maxWorkers=1`
- Pre-existing failure: `ContentViewer.stage14.annotations.test.tsx` first test cannot find `data-testid="media-intelligence-tab-annotations"` immediately after clicking `media-intelligence-toggle`. The selected-content annotation test passes.

Guardrail anchors:
- `rg` confirmed `ContentViewer` still binds `onMouseUp`/`onKeyUp` to `modals.handleCaptureAnnotationSelection`.
- `rg` confirmed embedded audio/video elements still assign through `modals.mediaPlayerRef`.

Known baseline concern:
- The annotation panel create/update/sync/delete test is already failing before implementation changes. Task 4 will intentionally update this annotation-selection surface, so keep this failure visible in verification until the mediated selection task replaces or fixes the expectation.

Task 2 completed:
- Added pure read-along segment types, canonical segmentation/scope helpers, and focused tests under `apps/packages/ui/src/components/Media/read-along/`.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/media-read-along-segments.test.ts --maxWorkers=1` failed because `../media-read-along-segments` did not exist.
- Green verification: same command passed with 7 tests after implementation.
- `git diff --check` passed for the Task 2 files.
- Spec review found transient fallback segment IDs were too collision-prone; fixed in `be51074a4` by including derived media/source identity or a stable text hash.
- Task 2 spec and code-quality re-reviews approved after the fix.

Task 3 completed:
- Added read-along cache-key helpers, Dexie-backed audio cache helpers, Dexie schema/type support, and TTS provider abort-signal surface.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts src/components/Media/read-along/__tests__/media-read-along-cache.test.ts src/services/__tests__/tts-provider.read-along.test.ts --maxWorkers=1` failed before implementation because cache modules/signatures were missing and `tldw` synthesis did not forward `signal`.
- Initial green verification passed 5 files / 18 tests, then code-quality review found two issues: weak fake SHA fallback and oversized cache writes exceeding the cap.
- Fixed in `31180c28` by making `sha256Hex` fail closed without Web Crypto and skipping oversized cache writes before eviction/write attempts.
- Final green verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/media-read-along-cache-key.test.ts src/components/Media/read-along/__tests__/media-read-along-cache.test.ts src/services/__tests__/tts-provider.read-along.test.ts src/services/__tests__/tts.defaults.test.ts src/db/dexie/__tests__/stt-recordings.test.ts --maxWorkers=1` passed 5 files / 21 tests.
- `git diff --check` passed for the Task 3 files.
- Task 3 spec and code-quality re-reviews approved after the fix.
- Bandit skipped for Task 3 because the touched slice is TypeScript frontend/Dexie code only.

Task 4 completed:
- Added DOM-safe content-selection helpers, a mediated `useContentSelectionActions` hook, and a temporary annotation-only selection action popover in `ContentViewer`.
- Split annotation capture into explicit `captureAnnotationSelection(selectionText, location)` while preserving `handleCaptureAnnotationSelection` as a backwards-compatible wrapper.
- Updated the stage 14 annotation test so text selection opens `media-selection-actions-popover`, keeps `media-annotation-selection-preview` absent until `media-selection-action-annotate` is clicked, then saves the selected highlight.
- Fixed the pre-existing stage 14 lazy intelligence-tab timing failure by waiting for `media-intelligence-tab-annotations` after expanding the section.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx --maxWorkers=1` failed before implementation because `../useContentSelectionActions` was missing and the popover was absent.
- Green verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useContentSelectionActions.test.tsx src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx src/components/Media/__tests__/ContentViewer.stage15.accessibility.test.tsx --maxWorkers=1` passed 3 files / 9 tests.
- `git diff --check` passed for the Task 4 files.
- Bandit skipped for Task 4 because the touched slice is TypeScript/React frontend code only.

Task 5 completed:
- Added `useMediaReadAlongSession` with session-token guarded async state, frozen TTS provider context/signature, generated-audio cache read/write, browser SpeechSynthesis fallback, current/lookahead abort controllers, media-preview pause, object URL cleanup, stop/retry/skip/pause/resume controls, and media/content-change stale completion suppression.
- Added focused renderHook coverage for cached selection playback, full-content from-here queueing beyond rendered windows, bounded 4-segment lookahead, browser provider no-cache behavior, abort/cancel stop behavior, stale completion suppression, frozen settings, audio.play errors, and embedded media pause.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx --maxWorkers=1` failed before implementation because `../useMediaReadAlongSession` did not exist.
- Green verification: `cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx src/components/Media/read-along/__tests__/media-read-along-segments.test.ts src/components/Media/read-along/__tests__/media-read-along-cache.test.ts --maxWorkers=1` passed 3 files / 23 tests.
- `git diff --check` passed.
- Bandit skipped for Task 5 because the touched slice is TypeScript/React frontend code only.
Task 5 review findings fixed:
- Segment load/play failures now mark the failed pending segment as active so retry and skip target the correct parent segment.
- Lookahead tracks in-flight segment audio, reuses a pending lookahead request when it becomes current, and aborts stale lookahead on retry/skip/current changes while preserving the bounded window.
- Provider synthesis now splits over-cap parent segments into deterministic TTS request parts, plays all parts sequentially under the same highlighted/counting parent segment, and keeps lookahead bounded.
- Read-along uses the session-captured TTS text normalizer for provider/browser synthesis and cache text hashes; browser provider contexts also capture the configured browser voice name.
- Generated-audio cache keys now include a sanitized active server/auth-scope identity from tldwClient.getConfig() instead of the hard-coded media-read-along scope.
- Recursive play-next calls now go through a playSegment ref to avoid stale hook callback dependencies.

Verification:
- cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx src/components/Media/read-along/__tests__/media-read-along-segments.test.ts src/components/Media/read-along/__tests__/media-read-along-cache.test.ts src/services/__tests__/tts-provider.read-along.test.ts --maxWorkers=1 -> passed, 4 files / 44 tests.
- cd apps/packages/ui && bunx vitest run src/components/Media/read-along/__tests__/useMediaReadAlongSession.test.tsx src/services/__tests__/tts-provider.read-along.test.ts --maxWorkers=1 -> passed, 2 files / 27 tests.
- git diff --check -> passed.
- Bandit not run because this review-fix slice touched TypeScript frontend files only.

Task 6 completed:
- Added ContentViewer read-along UI integration with selection popover actions, inline transport, read-along session wiring, and active segment wrappers for plain/timestamped transcript rendering.
- Preserved mediated annotation selection; Annotate remains explicit, and read-along actions clear the popover/document selection after starting.
- Markdown/html fallback starts playback through the existing session without mutating rich HTML; full-item playback uses the session hook and does not expand the lazy plain-content window.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx --maxWorkers=1` failed before implementation because `media-selection-action-read-selection` was missing.
- Green verification: same command passed 1 file / 7 tests.
- Regression verification: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx --maxWorkers=1` passed 4 files / 20 tests.
- `git diff --check` passed.
- Bandit skipped for Task 6 because the touched slice is TypeScript/React frontend code only.

Task 6 review findings fixed:
- Lazy plain rendering now builds highlight wrappers only from the currently visible plain-content window and initializes large plain content at the first chunk instead of the full item.
- Full-item/from-here playback no longer forces the lazy plain-content window to the full content. When the active segment advances outside the rendered chunk, the visible window expands only through that active segment so highlight and scroll can catch up.
- Markdown/html text-only selections now hide unsupported mapped-scope actions (`Read from here`, `Read current section`) while preserving `Read selection`, `Read full item`, and `Annotate`; exact plain selections still expose all scopes.
- Selection popover and read-along transport are clamped to the content viewport when available, including right-edge placement, and transport progress/error text uses polite status live regions.
- Red verification: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx --maxWorkers=1` failed with 4 expected review-regression failures: unsupported markdown/html scopes visible, full display content segmented during lazy rendering, active large-content highlight missing after playback advanced, and right-edge popover unclamped.
- Green verification: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx --maxWorkers=1` passed 1 file / 11 tests.
- Regression verification: `cd apps/packages/ui && bunx vitest run src/components/Media/__tests__/ContentViewer.read-along.test.tsx src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx src/components/Media/__tests__/ContentViewer.stage14.annotations.test.tsx --maxWorkers=1` passed 4 files / 24 tests.
- `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented shared WebUI/extension media-viewer read-along TTS. The feature adds selectable read-along scopes, explicit annotation mediation, TTS-provider reuse including browser SpeechSynthesis, generated-audio caching, cancellation/race hardening, active segment rendering, route parity coverage, accessibility coverage, and browser-discovered selection/viewport hardening. Final verification recorded: focused read-along Vitest suite passed (12 files, 104 tests), route parity/connection suite passed (2 files, 6 tests), browser render smoke passed, and git diff --check passed. OpenAPI passed earlier in Task 8; design-system verification remains blocked by unrelated baseline findings outside the touched read-along files; Bandit skipped because no Python files were touched.
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
