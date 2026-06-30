# WebUI Stage 10 TTS No-Provider Capability Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show one shared, user-language setup state in the TTS playground when the selected tldw server provider has no TTS engine available, while preserving normal TTS defaults, provider panel rendering, and browser/provider workflows.

**Architecture:** Reuse `StatePanel` for the no-provider setup state. Keep ffmpeg warnings, ElevenLabs-specific setup, generation behavior, and voice selection out of this slice unless a focused test catches a regression. The no-provider state has no request diagnostics because it is based on capability data, not a failed request.

**Tech Stack:** React, TypeScript, existing TTS hooks, shared WebUI state primitives, Vitest, React Testing Library.

---

## Stage 1: Failing Coverage
**Goal**: Capture the current duplicate local-alert no-provider gap.
**Success Criteria**: Focused tests fail because the no-provider state is rendered as duplicate local AntD alerts instead of one shared setup state.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/TTS/__tests__/TtsPlaygroundPage.defaults.test.tsx`
**Status**: Complete

- [x] Add a failing assertion that no-provider renders one shared `StatePanel`.
- [x] Add a failing assertion that no-provider does not render duplicate local alert states.
- [x] Preserve existing default model/voice and page heading assertions.
- [x] Run the focused TTS test file and confirm the new assertions fail for the current local-alert implementation.

## Stage 2: TTS No-Provider Recovery UI
**Goal**: Replace duplicate no-provider alerts with one shared setup state.
**Success Criteria**: No-provider renders a single `StatePanel`; normal has-audio rendering remains unchanged.
**Tests**: Focused TTS defaults test.
**Status**: Complete

- [x] Import and use `StatePanel` in `TtsPlaygroundPage`.
- [x] Replace duplicate no-provider alerts with one setup state.
- [x] Preserve existing no-provider title/body guidance and Speech Settings link.
- [x] Keep ffmpeg and ElevenLabs notices unchanged in this slice.
- [x] Keep normal TTS defaults, provider panel, and voice picker behavior intact.

## Stage 3: Verification And Closeout
**Goal**: Prove the slice works and record the result.
**Success Criteria**: Focused tests pass, lint/whitespace checks are clean for touched files, and Backlog reflects verification and known skips.
**Tests**: Focused Vitest, ESLint touched files, `git diff --check`.
**Status**: Complete

- [x] Run the focused TTS test file.
- [x] Run ESLint on touched TS/TSX files.
- [x] Run `git diff --check`.
- [x] Record Bandit as not applicable for TS/TSX/docs-only changes.
- [x] Update `TASK-12039` acceptance criteria, notes, touched files, and final summary.
- [x] Commit the Stage 10 slice.
