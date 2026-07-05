# Chat, Extension Chat, Character Chat, and Notes Dark Theme Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or follow this task list directly with TDD and rendered browser verification.

**Goal:** Remove light-theme visual leaks from WebUI chat, extension chat, character-card chat, and Notes in dark mode.

**Architecture:** Prefer shared token fixes in `apps/packages/ui/src/assets/tailwind-shared.css` and Ant Design theme bridging before local component patches. Only patch route/component files when a hardcoded light surface is confirmed in the target flow.

**Tech Stack:** Next.js, React, Tailwind, Ant Design, Vitest, Playwright/Browser.

---

## Stage 1: Map and Reproduce
**Goal:** Identify the exact light-theme leaks in the target dark-mode flows.
**Success Criteria:** Screenshots or DOM/CSS evidence for `/chat`, `/chat?mode=character`, extension sidepanel chat/options, and `/notes`.
**Tests:** Browser walkthrough with dark mode set before route load.
**Status:** Complete

## Stage 2: Add Minimal Regression Coverage
**Goal:** Capture the shared dark-theme invariant that failed.
**Success Criteria:** A focused test fails before the fix and checks relevant shared CSS/component output.
**Tests:** Targeted Vitest or CSS contract test under `apps/tldw-frontend` or `apps/packages/ui`.
**Status:** Complete

## Stage 3: Fix Root Cause
**Goal:** Replace confirmed light-only colors/classes with shared semantic tokens.
**Success Criteria:** The target surfaces use `bg/surface/elevated/text/border` tokens in dark mode, with no new local theme abstraction.
**Tests:** Targeted regression test passes.
**Status:** Complete

## Stage 4: Rendered Walkthrough
**Goal:** Walk through menus/options in normal chat, character chat, extension chat, and Notes.
**Success Criteria:** Desktop and one narrow viewport show no jarring light panels in dark mode; console has no relevant errors.
**Tests:** Browser screenshots plus interaction checks for menus, selectors, modals, sidebars, and Notes overlays.
**Status:** Complete

## Stage 5: Final Verification
**Goal:** Record focused verification and repo-required checks.
**Success Criteria:** Targeted tests pass, Bandit is skipped with rationale for frontend-only changes, and `TASK-12171` has final notes.
**Tests:** Focused test command, lint/build only if touched scope warrants it, browser QA evidence.
**Status:** Complete

Verification:
- `npx playwright test e2e/smoke/dark-theme-visual-fidelity.spec.ts --reporter=line` passed.
- `git diff --check -- apps/packages/ui/src/assets/tailwind-shared.css apps/tldw-frontend/e2e/smoke/dark-theme-visual-fidelity.spec.ts Docs/superpowers/plans/2026-07-05-chat-notes-dark-theme-visual-fidelity-plan.md "backlog/tasks/task-12171 - Fix-dark-theme-visual-drift-in-Chat-extension-chat-character-chat-and-Notes.md"` passed.
- Bandit skipped: touched code is frontend CSS and Playwright TypeScript only.

Notes:
- Confirmed and fixed white Ant Select wrapper surfaces in Notes dark mode.
- Confirmed and fixed low-contrast Ant button/radio-button text in Notes dark mode.
