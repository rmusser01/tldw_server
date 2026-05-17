---
id: TASK-417
title: Implement media viewer read-along TTS
status: In Progress
labels:
- implementation
- webui
- extension
- tts
- media
references:
- Docs/superpowers/specs/2026-05-17-media-viewer-read-along-tts-design.md
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
