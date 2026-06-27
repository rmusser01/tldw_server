# Research Workspace Visible Undo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore visible, keyboard-accessible Undo controls for destructive Research Workspace message toasts.

**Architecture:** Ant Design `message.open` ignores the notification-only `btn` field, so undo actions must be rendered inside `content`. Keep the change local to Research Workspace toast configs and follow the already-live-passing source bulk remove pattern.

**Tech Stack:** React, Ant Design message API, Vitest, React Testing Library, Backlog.md.

---

## Stage 1: Reproduce With Failing Rendered-Content Tests

**Goal:** Prove the current message config loses the Undo action in rendered message content for both workspace-level and artifact-level destructive paths.

**Success Criteria:** Focused tests fail because rendered `message.open` content contains the status text but not an `Undo` button.

**Tests:** `WorkspaceHeader.test.tsx` archive path and `StudioPane.stage1.test.tsx` failed artifact delete path.

**Status:** Complete

- [x] Add a WorkspaceHeader regression that renders the archive message `content` and expects a visible `Undo` button.
- [x] Add a StudioPane regression that renders the artifact-delete message `content` and expects a visible `Undo` button.
- [x] Run both focused tests and confirm they fail for the missing rendered Undo control.

## Stage 2: Move Destructive Message Actions Into Content

**Goal:** Replace Research Workspace `message.open({ ..., content: string, btn: <Undo /> })` patterns with content that contains both text and action controls.

**Success Criteria:** No Research Workspace `message.open` destructive recovery path depends on the ignored `btn` field; focused tests pass.

**Tests:** Same focused tests from Stage 1, then broader Research Workspace destructive/undo suites.

**Status:** Complete

- [x] Update workspace archive/delete, artifact delete, chat clear/message delete, quick note clear, source annotation/single-source remove, source transfer, template start-over, duplicate open-original, and note shortcut undo message configs.
- [x] Keep the existing message keys, durations, and success feedback.
- [x] Run focused tests and confirm the new regressions pass.

## Stage 3: Live Recheck and Documentation

**Goal:** Recheck the fixed archive and artifact destructive paths in the in-app browser and record evidence.

**Success Criteria:** Archive success and artifact delete success show a visible `Undo` control; clicking Undo restores state and shows restored feedback.

**Tests:** In-app browser/CDP evidence, focused Vitest bundle, `git diff --check`, Bandit skip note for frontend/docs-only scope.

**Status:** Complete

- [x] Reload the local WebUI and repeat archive success plus Undo restore.

  Result: The locked WebUI on `127.0.0.1:8081` served a stale
  `WorkspaceHeader` chunk older than the source change and still reproduced the
  pre-fix missing Undo behavior. A clean temporary webpack server on
  `127.0.0.1:8083` loaded the current bundle. In the in-app browser, archiving a
  disposable duplicated workspace showed `Workspace archived.` with a visible
  `Undo` button, and clicking `Undo` restored `New Research (Copy)` and showed
  `Workspace restored`. Screenshots:
  `/private/tmp/task12020_31_archive_undo_visible.png` and
  `/private/tmp/task12020_31_archive_undo_restored.png`.

- [x] Seed or use an artifact and repeat artifact delete plus Undo restore.

  Result: After local network permission was restored, a shell-launched
  standalone Chromium using the cached browser binary reached the clean webpack
  WebUI on `127.0.0.1:8083`. It imported an attached disposable
  `.workspace.json` bundle containing one failed artifact, confirmed the failed
  output rendered, deleted it through the visible failed-output delete control,
  verified `Output deleted.` plus a visible `Undo` button, clicked `Undo`, and
  verified `Output restored` plus the restored `Failed output` card. Evidence:
  `/private/tmp/task12020_31_failed_artifact_imported.png`,
  `/private/tmp/task12020_31_failed_artifact_deleted_undo_visible.png`, and
  `/private/tmp/task12020_31_failed_artifact_restored.png`.

- [x] Update the live UAT matrix and Backlog with screenshots, verification output, and remaining scope.
