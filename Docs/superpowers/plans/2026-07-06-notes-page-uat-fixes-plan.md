# Notes Page UAT Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run full WebUI notes-page UAT against a live backend and patch root causes of blocking or user-visible issues found.

**Architecture:** Keep fixes in the existing notes route/components and current test stack. Add the smallest regression coverage that fails on the confirmed bug and passes after the patch.

**Tech Stack:** Next.js WebUI, React, existing frontend tests, FastAPI backend, live llama.cpp-compatible provider when chat-adjacent notes actions need generation.

---

### Task 1: Map And Exercise Notes Functionality

**Files:**
- Inspect: `apps/packages/ui/src/routes/option-notes.tsx`
- Inspect: existing notes tests under `apps/tldw-frontend` and `apps/packages/ui`

- [x] List visible notes-page workflows and controls.
- [x] Start isolated backend/frontend configured for llama.cpp on `127.0.0.1:9099`.
- [x] Exercise create, save, reload, select, search, filter, view toggles, trash, restore/delete, collections/tags, editor modes, split/preview, keyboard shortcuts, mobile layout.
- [x] Record reproducible issues with exact steps and evidence.

### Task 2: Patch Confirmed Root Causes

**Files:**
- Modify only the notes route/component files required by confirmed bugs.
- Test with the smallest existing frontend test command that covers the touched behavior.

- [x] For each confirmed bug, trace the event/data flow to the shared root cause.
- [x] Write a failing regression check first.
- [x] Implement the shortest fix at the root cause.
- [x] Verify the regression check passes.

### Task 3: Final Verification

**Files:**
- Update: `backlog/tasks/task-12903 - Full-notes-page-UAT-and-root-cause-fixes.md`

- [x] Re-run live notes UAT for fixed flows.
- [x] Run focused frontend tests.
- [x] Run Bandit only if Python code is touched; otherwise document skip.
- [x] Save screenshot evidence outside the repo.
- [x] Summarize findings, fixes, verification, and remaining risk.
