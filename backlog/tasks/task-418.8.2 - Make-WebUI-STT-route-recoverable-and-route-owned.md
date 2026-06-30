---
id: TASK-418.8.2
title: Make WebUI STT route recoverable and route-owned
status: Done
labels:
- webui
- ux-audit
- audio
- wp11a
- stt
priority: medium
parent_task_id: TASK-418.8
documentation:
- Docs/superpowers/plans/2026-05-17-webui-audio-routes-implementation-plan.md
modified_files:
- apps/packages/ui/src/routes/option-stt.tsx
- apps/packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx
- apps/packages/ui/src/components/Option/STT/SttPlaygroundPage.tsx
- apps/packages/ui/src/components/Option/STT/RecordingStrip.tsx
- apps/packages/ui/src/components/Option/STT/HistoryPanel.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx
- apps/packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx
- apps/tldw-frontend/extension/routes/option-stt.tsx
- apps/tldw-frontend/extension/__tests__/audio-route-parity.guard.test.ts
- apps/tldw-frontend/e2e/utils/page-objects/STTPage.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute WP11A Task 3 from the WebUI audio routes implementation plan. Make /stt a recoverable dedicated transcription route by adding route-boundary coverage, preserving hosted behavior, verifying extension route parity, and improving STT readiness/empty/error states using existing STT components and APIs only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /stt route-boundary coverage exists and verifies routeId=stt and routeLabel=STT Playground around the dedicated SttPlaygroundPage.
- [x] #2 Hosted behavior remains intact through HostedAudioFeatureMessage.
- [x] #3 Extension /stt ownership/parity is verified before any ownership change; any exception is documented in route metadata/tests.
- [x] #4 STT readiness coverage includes catalog loading/failure retry, no-model setup language, recording source state, comparison partial failure preservation, save-to-notes failure preservation, and history empty state.
- [x] #5 Focused STT component and E2E verification is recorded; no backend APIs or unrelated routes are changed.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented WP11A Task 3 for the dedicated `/stt` route.

- Added shared route-boundary coverage and wrapped shared `/stt` in `RouteErrorBoundary routeId="stt" routeLabel="STT Playground"` while preserving the hosted `HostedAudioFeatureMessage` path.
- Aligned extension `/stt` to the dedicated `SttPlaygroundPage` owner under the same route boundary and extended parity coverage.
- Added STT readiness/recovery tests for catalog loading, no-model blocking, recording source status, partial comparison failure, save-to-notes failure, and history empty state.
- Added readiness-driven recording/upload disabling and a visible recording source status to `RecordingStrip`.
- Updated the STT E2E page object to the route-owned `Speech to Text` heading.

Verification:
- `bunx vitest run ../packages/ui/src/routes/__tests__/option-audio-route-identity.test.tsx extension/__tests__/audio-route-parity.guard.test.ts` passed: 2 files, 5 tests.
- `bunx vitest run ../packages/ui/src/components/Option/STT/__tests__/SttPlaygroundPage.test.tsx ../packages/ui/src/components/Option/STT/__tests__/RecordingStrip.test.tsx ../packages/ui/src/components/Option/STT/__tests__/InlineSettingsPanel.test.tsx ../packages/ui/src/components/Option/STT/__tests__/ComparisonPanel.test.tsx ../packages/ui/src/components/Option/STT/__tests__/HistoryPanel.test.tsx ../packages/ui/src/components/Option/STT/__tests__/keyboard-shortcuts.test.tsx` passed: 6 files, 36 tests.
- `bunx playwright test e2e/workflows/tier-2-features/stt-transcription.spec.ts --reporter=line` passed: 2 tests.
- Browser DOM QA at `http://localhost:8080/stt` observed `Speech to Text`, `STT readiness`, `Recording source: Audio source: no audio selected`, disabled `Transcribe All`, and the updated history empty state.
- `bunx eslint extension/routes/option-stt.tsx extension/__tests__/audio-route-parity.guard.test.ts e2e/utils/page-objects/STTPage.ts` passed.
- `bunx tsc --noEmit --pretty false -p ../packages/ui/tsconfig.json` failed on inherited repo-wide TypeScript debt outside the touched STT/route files; no reported errors referenced touched files.
- Bandit: skipped (no Python code touched).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the /stt recoverability slice with route ownership parity, readiness-driven recording blocking, visible recording source status, stronger transcript/history recovery tests, and focused Vitest/Playwright verification. No backend APIs were changed.
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
