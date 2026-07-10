# PR 2706 Review Follow-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #2706 on the latest `dev` and address every actionable review finding with minimal, regression-tested changes.

**Architecture:** Preserve the existing `extractImageUrl` precedence (HTML before Markdown), but scan each candidate class until a URL passes the shared `safeImageUrl` boundary and accept valid whitespace around `src =`. Preserve the server-side group filter design while deriving the optional group list with the store type's explicit null contract.

**Tech Stack:** TypeScript, React 18, Vitest, Testing Library, Bun.

---

## Stage 1: Inventory and rebase

**Goal:** Establish the complete review scope and replay the branch on current `origin/dev`.

**Success Criteria:** All issue comments, reviews, and inline threads are inventoried; `git rebase origin/dev` succeeds.

**Status:** Complete

- [x] Fetch all PR reviews, issue comments, and inline review comments.
- [x] Verify each suggestion against current code and backend constraints.
- [x] Fetch `origin/dev` and rebase the PR branch (no-op because it was already current).

## Stage 2: Fix image candidate extraction

**Goal:** Preserve valid preview images when earlier candidates are unsafe and accept valid HTML attribute whitespace.

**Success Criteria:** The first safe HTML candidate is returned; if no safe HTML candidate exists, the first safe Markdown candidate is returned; `src = "..."` and `src= "..."` are recognized.

**Tests:** `apps/packages/ui/src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts`

**Status:** In Progress

- [ ] Add failing tests for both `src = "..."` and `src= "..."`, unsafe HTML followed by safe HTML, unsafe Markdown followed by safe Markdown, no safe HTML followed by safe Markdown, and an earlier safe Markdown candidate that must not outrank a later safe HTML candidate.
- [ ] From `apps/packages/ui`, run `bunx vitest run src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts --maxWorkers=1 --no-file-parallelism` and confirm the new tests fail for the expected reasons.
- [ ] Replace the single-match early returns with global candidate scans; keep HTML-before-Markdown precedence and reuse `safeImageUrl`.
- [ ] Re-run the Stage 2 focused command and confirm it passes.
- [ ] Commit the focused image-extraction fix.

## Stage 3: Preserve the optional group-ID contract

**Goal:** Avoid truthiness when converting `number | null` selection state into the `groups` query.

**Success Criteria:** `null` omits the group filter and numeric selections, including defensive value `0`, produce a one-element groups array.

**Tests:** `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx`

**Status:** Not Started

- [ ] Add a failing test with `selectedGroupId: 0` expecting `groups: [0]`.
- [ ] From `apps/packages/ui`, run `bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx --maxWorkers=1 --no-file-parallelism` and confirm the new test fails because the filter is omitted.
- [ ] Change only the query construction to `selectedGroupId != null`.
- [ ] Re-run the Stage 3 focused command and confirm it passes.
- [ ] Commit the focused group-filter fix.

## Stage 4: Verify, respond, and update the PR

**Goal:** Prove the rebased branch is clean, respond precisely in every review thread, and update task/PR records.

**Success Criteria:** Focused tests and frontend typecheck pass; diff and source checks are clean; review threads are replied to; the branch is pushed and PR checks are inspected.

**Status:** Not Started

- [ ] From `apps/packages/ui`, run both focused suites together:
  `bunx vitest run src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx --maxWorkers=1 --no-file-parallelism`.
- [ ] From `apps/packages/ui`, run the full affected CodeQL regression set:
  `bunx vitest run src/utils/__tests__/image-utils.test.ts src/types/__tests__/assistant-selection.test.ts src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.test.ts src/components/Option/Watchlists/ItemsTab/__tests__/items-utils.domless.test.ts src/utils/__tests__/assistant-overlay.test.ts src/components/Sidepanel/Chat/__tests__/CharacterSelect.persona-avatar.test.tsx src/components/Option/Watchlists/SourcesTab/__tests__/SourcesTab.advanced-details.test.tsx src/components/Quiz/tabs/__tests__/ManageTab.bulk-duplicate.test.tsx src/utils/__tests__/provider-registry-tts.test.ts src/utils/__tests__/codeql-source-contracts.test.ts --maxWorkers=1 --no-file-parallelism`.
- [ ] From `apps/tldw-frontend`, run `NODE_OPTIONS=--max-old-space-size=8192 bun run typecheck`; from the repository root, run `git diff --check origin/dev...HEAD` and `git status --short`.
- [ ] Record Bandit as not applicable because no Python source changed.
- [ ] Use `superpowers:requesting-code-review` to review the complete diff for regressions and unresolved comments; after each correction, rerun the directly affected focused suite and re-review.
- [ ] Update the Backlog task with verification/final summary, mark every plan stage complete, remove this task-specific plan, and commit the task finalization plus plan removal before pushing.
- [ ] Push the updated branch, reply in the original inline threads, address the summary-only finding in a PR comment, and inspect checks. State the existing branch-analysis limitation and do not claim JavaScript CodeQL passed unless GitHub emits a JavaScript analysis.
