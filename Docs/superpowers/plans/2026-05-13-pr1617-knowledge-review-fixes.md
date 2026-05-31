# PR 1617 Knowledge Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve the still-actionable review comments on PR #1617 without expanding the `/knowledge` scope beyond QA-only source selection and retrieval diagnostics.

**Architecture:** Keep fixes local to the reviewed files. Frontend changes should reuse existing i18n/source metadata patterns; backend changes should centralize source/path and workspace filtering behavior so all retrieval variants stay aligned.

**Tech Stack:** FastAPI/Pydantic, Python RAG retrieval service, React/TypeScript, Vitest, Playwright, Bun, pytest, Bandit.

---

## Stage 1: Frontend And Extension Review Fixes
**Goal**: Address i18n, brittle tests, source-picker state, and streaming diagnostics comments.
**Success Criteria**: Valid UI/review issues are patched with focused tests updated or added.
**Tests**: Focused Vitest/Bun tests for Knowledge QA, route parity, source metadata, and copilot lazy import.
**Status**: Complete

- [x] Localize the knowledge sidebar title and update English locale entries.
- [x] Update RAG source translation entries for chats and characters to match new labels.
- [x] Replace brittle exact-string route assertions with behavior-focused regex/block checks.
- [x] Tighten persisted media ID validation and include selected IDs in profile-save dependencies.
- [x] Restore item IDs as secondary source-picker identifiers.
- [x] Preserve source-status diagnostics in streaming search details.
- [x] Wrap extension copilot dynamic import errors with contextual cause.

## Stage 2: Backend Retrieval And Pipeline Fixes
**Goal**: Address prompts wiring, retriever correctness/performance, workspace filtering, cache scoping, type hints, and docstrings.
**Success Criteria**: All unified search variants have consistent prompts/source wiring, final documents are workspace-filtered, cache behavior is workspace-aware, and retriever review findings are resolved.
**Tests**: Focused RAG/source-contract pytest, with added or updated tests for changed behavior where practical.
**Status**: Complete

- [x] Type `_validate_batch_sources` and `PromptsDBRetriever.retrieve`.
- [x] Add docstrings for new workspace/source-status helpers.
- [x] Use keyword arguments for `search_prompts` and precompute query terms.
- [x] Clamp world-book scores and add `ChatDictionariesRetriever` adapter initialization.
- [x] Avoid N+1 chat metadata lookups when direct SQL fallback can retrieve conversation/card context.
- [x] Centralize complete DB path mapping and reuse it in re-retrieval paths.
- [x] Scope cache keys by `workspace_id` and apply final workspace filtering/status normalization before return.
- [x] Pass `prompts_db`/path consistently through stream, batch, and resume paths.

## Stage 3: Verification, Commit, Push, Thread Closeout
**Goal**: Prove the review fixes, push them to PR #1617, and clear or explain review threads.
**Success Criteria**: Local focused verification passes, branch is pushed, PR threads are resolved or have a concrete explanation, and pending remote CI is reported separately.
**Tests**: `git diff --check`, focused pytest, focused Vitest/Bun, extension compile if affected, and Bandit on touched Python production files.
**Status**: Complete

- [x] Run focused backend/frontend/extension tests.
- [x] Run `git diff --check`.
- [x] Run Bandit on touched Python production files.
- [x] Commit and push the review-fix patch.
- [x] Requery PR #1617 review threads/checks and resolve addressed threads.
