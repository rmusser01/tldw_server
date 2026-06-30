# PR 2309 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2309 on latest `dev` and address actionable review feedback without broadening Knowledge QA scope.

**Architecture:** Keep fixes local to the reviewed code paths: Knowledge QA trust persistence, scoped RAG retrieval, extension live-gate utilities, test metadata, and malformed Backlog task markers. Treat optional review comments as actionable only when the change is low-risk and directly improves the touched code.

**Tech Stack:** FastAPI/Python RAG service, pytest, Bandit, React/TypeScript Knowledge QA UI, Vitest, Playwright extension tests, Backlog.md.

---

## Stage 1: Review Inventory And Tracking

**Goal:** Confirm latest `dev`, current PR comments, and task tracking.

**Success Criteria:** Branch rebased on `origin/dev`; unresolved review threads and check state inventoried; `TASK-2316` exists.

**Status:** Complete

## Stage 2: Red Tests For Behavioral Feedback

**Goal:** Add or tighten tests before production fixes where review feedback changes behavior.

**Success Criteria:** Tests fail for stale retry-sync trust persistence, missing note source metadata, oversized SQL allowlist handling, async offload call path, explicit include source fallback, effective fallback mode, classification skip-search no-doc generation, manifest parse errors, and extension launch gate bypass.

**Status:** Complete

## Stage 3: Minimal Production Fixes

**Goal:** Fix the reviewed issues without adding unrelated Knowledge QA behavior.

**Success Criteria:** Retry sync persists recomputed trust state; selected note metadata includes stable IDs and source type; explicit-note lookups are offloaded; chat source exclusion caches duplicate lookups; scoped requests bypass cache and do not collapse to empty source scope; fallback uses effective retrieval mode; no-document generation gate preserves classification skip-search; extension launch gate no longer uses unconditional expected failure; manifest JSON errors include the manifest path.

**Status:** Complete

## Stage 4: Low-Risk Review Cleanups

**Goal:** Apply low-risk style/tooling comments that are within touched files.

**Success Criteria:** Test markers are added where requested, mock test state uses narrower union types, the unreachable web category check is removed if verified, duplicate constants/docstrings are cleaned up, and malformed Backlog section markers are fixed.

**Status:** Complete

## Stage 5: Verification, Push, And PR Replies

**Goal:** Prove fixes and update PR #2309.

**Success Criteria:** Focused backend/frontend/extension tests pass; TypeScript check passes; Bandit runs on touched backend scope; live Knowledge QA gate is rerun if backend changes affect it; branch is pushed with `--force-with-lease`; actionable review comments are replied to or resolved.

**Status:** Complete
