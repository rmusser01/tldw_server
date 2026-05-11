# Persona Chat Context Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded effective Persona Chat context preview to the existing prompt-preview path.

**Architecture:** Reuse `POST /api/v1/chats/{chat_id}/prompt-preview` and the existing persona exemplar assembly helper. The endpoint will continue returning the existing `sections` array while adding a compact `persona_context` envelope for persona-backed conversations.

**Tech Stack:** FastAPI, `CharactersRAGDB`, pytest, Backlog.md.

---

## Stage 1: Regression Coverage
**Goal**: Lock the missing `persona_context` behavior before implementation.
**Success Criteria**: Persona-backed prompt preview test fails because `persona_context` is missing; non-persona preview remains compatible.
**Tests**: `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py`
**Status**: Complete

- [x] Add a persona-backed prompt-preview test that expects `persona_context.assistant_kind == "persona"`, a stable `assistant_id`, `persona_memory_mode`, selected/rejected exemplar IDs, and current turn source.
- [x] Add a non-persona prompt-preview test that verifies no active persona context is introduced.
- [x] Run the new focused tests and confirm the persona-backed case fails for the missing preview envelope.

## Stage 2: Preview Envelope
**Goal**: Build the preview envelope from existing conversation and exemplar assembly data.
**Success Criteria**: `persona_context` is bounded, deterministic, and does not alter provider payload construction.
**Tests**: Focused prompt-preview tests from Stage 1.
**Status**: Complete

- [x] Extend `_build_persona_preview_sections` or an adjacent helper to return selected/rejected exemplar diagnostics alongside sections.
- [x] Add `persona_context` to prompt-preview responses only when the conversation is persona-backed.
- [x] Cap string/list fields to keep the response bounded.
- [x] Preserve existing section names and content.

## Stage 3: Verification And Closeout
**Goal**: Verify the slice and package it for review.
**Success Criteria**: Focused tests, compile check, Bandit, and diff hygiene pass; Backlog task is updated.
**Tests**: Focused pytest for prompt assembly/preview.
**Status**: Complete

- [x] Run focused pytest for prompt assembly and prompt-preview regression tests.
- [x] Run `py_compile` on touched backend modules.
- [x] Run Bandit on touched backend files.
- [x] Run `git diff --check`.
- [ ] Update `TASK-253`, commit, push, and open a PR linked to #1560.
