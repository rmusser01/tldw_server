---
id: TASK-432
title: Implement TTS/STT comparison provenance and repeat controls
status: Done
labels:
- implementation
- webui
- extension
- audio
- tts
- stt
- ux
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 4 from Docs/superpowers/plans/2026-05-19-tts-stt-webui-extension-workflows-implementation-plan.md. Scope: frontend-only comparison provenance and repeat-use controls for visible TTS/STT workflows, including safe metadata, client-measured labels, retry/duplicate/disable controls, and privacy-aware text previews/hashes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Shared provenance helpers format created time, byte size, client-measured latency, text preview, and local text hash consistently.
- [x] #2 STT comparison results preserve request configuration and available response metadata without inventing missing backend values.
- [x] #3 STT result cards and history expose model, language/task/format settings, source metadata, client latency, word/segment/duration metadata where available, and retry/duplicate/disable controls.
- [x] #4 TTS render rows preserve provider/model/voice/format/speed, created time, input length/preview/hash, audio size, and client latency metadata.
- [x] #5 TTS render rows expose retry/duplicate/disable controls without breaking existing generate, edit, play, remove, copy, save, download, or history workflows.
- [x] #6 Focused tests cover helpers, STT provenance rendering/history, TTS render provenance, and repeat controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add tested shared comparison provenance helpers for preview/hash, byte size, latency, created time, and metadata labels. 2. Extend STT comparison result config/metadata from existing request and response values, then render metadata rows and repeat controls. 3. Extend TTS render state metadata for provider/model/voice/format/speed, created time, input length/preview/hash, audio size, and client latency. 4. Verify focused hook/component suites and document known broader TypeScript baseline failures if unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `comparison-provenance.ts` for privacy-aware text preview/hash, created time, byte size, latency, STT config capture, STT response normalization, and TTS metadata.
- Extended `useComparisonTranscribe` with stable row IDs, request options, response metadata, retry-original-configuration behavior, duplicate rows, and disabled rows skipped by repeat runs.
- Rendered STT provenance and repeat controls in `ComparisonPanel`, and persisted/displayed the same config/metadata in `HistoryPanel`.
- Extended `useMultiRenderState` and `RenderStrip` so TTS rows show created time, input length/preview/hash, audio size, client latency, duplicate, and disable/enable controls.
- Updated `SpeechPlaygroundPage` to pass TTS render metadata and repeat-control handlers into `RenderStrip`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 4 comparison provenance and repeat-use controls for visible TTS/STT workflows. Focused Vitest coverage for helpers, STT hook/cards/history, TTS render state, render rows, and speech page render path passes. Full package TypeScript and Playwright audio smoke remain blocked by known broader baseline/environment failures documented in the plan; this slice is frontend-only, so Bandit was not applicable.
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
