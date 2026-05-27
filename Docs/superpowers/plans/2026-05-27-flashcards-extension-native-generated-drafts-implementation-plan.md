# Flashcards Extension Native Generated Drafts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the extension sidepanel generate a small batch of editable draft flashcards from selected page text.

**Architecture:** Reuse the existing sidepanel active-page selection reader, the existing `useGenerateFlashcardsMutation`, and the existing generated-card normalizer from the full Flashcards Generate panel. Generated drafts append into the current sidepanel draft queue so deck selection, editing, delete, save-one, save-all, source provenance, and partial save recovery stay on the established path.

**Tech Stack:** React, Ant Design, WXT browser APIs, Vitest, Testing Library, existing Flashcards hooks/services.

---

## Stage 1: Native Generation Tests
**Goal**: Prove selected page text can generate editable sidepanel drafts without opening the full workspace.
**Success Criteria**:
- `Generate draft cards` is visible alongside capture and full-workspace generation handoff.
- Clicking it reads selected page text through the existing injected helper.
- The flashcards generation mutation receives selected text, a compact default card count, and basic mixed-difficulty generation options.
- Generated cards append to the existing editable sidepanel draft queue and keep source URL/title provenance.
- Generation failures stay inline and do not clear queued manual drafts.
**Tests**:
- `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
**Status**: Complete

## Stage 2: Sidepanel Implementation
**Goal**: Add native sidepanel draft generation without adding native templates or in-extension review.
**Success Criteria**:
- `CaptureDraft` can represent manual and generated drafts with model type, tags, notes, and extra fields.
- Generated draft saves preserve model type, cloze/reverse flags, tags, notes, extra, and page source id.
- Capture, full-workspace generation handoff, save-one, save-all, and partial failure behavior remain unchanged.
**Tests**:
- `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
**Status**: Complete

## Stage 3: Source List And Task Closeout
**Goal**: Keep the master fix list, user docs, and TASK-519 aligned with the completed F12 sub-slice.
**Success Criteria**:
- `Flashcards-UX-Fix-List.md` marks native sidepanel generated drafts as completed while keeping templates and in-extension review deferred.
- Extension flashcards docs describe `Generate draft cards` and distinguish compact native generation from the full Flashcards generation workflow.
- TASK-519 records acceptance criteria, verification, known skips, and final summary.
**Tests**:
- `git diff --check`
**Status**: Complete
