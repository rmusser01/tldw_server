# Last-reply actions implementation plan — TASK-13279

> For agentic workers: execute the approved bounded stages inline with test-first checkpoints; the parent coordinates all runners and commits.

**Goal:** Keep the last assistant reply's Copy/Regenerate controls visible, expose one working overflow for older assistant and user rows, and preserve 44px persistent targets.

**Architecture:** Reuse the shared Message/MessageActionsBar and existing user bubble controls with AntD Popover. Existing `isLastMessage`, busy/edit checks, render-bound callbacks and source authority stay intact. No new state store, transport, persistence or action framework.

**Tech stack:** React, AntD, Tailwind, Vitest/Testing Library.

**Spec:** `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md`; TASK-13279 acceptance criteria.

**ADR check:** ADR required: no. This is presentation and reachability within existing shared controls, with no durable rule or public contract change. Task tracking follows [ADR-059](../ADR/059-backlog-py-task-editor-cutover.md); ownership continues under [ADR-049](../ADR/049-chat-history-selection-ownership.md).

## Stage 1: Prove the current presentation failures
**Goal:** Record causal failures through actual Casual full/compact renderers and working AntD controls.
**Success criteria:** Last-reply Copy/Regenerate are discoverable without hover; older assistant/user overflow opens with pointer/keyboard; only one trigger exists; real generated utilities certify target minimums. Busy/edit positives remain.
**Tests:** Extend `Message.loaded-server-chat-knowledge.integration.test.tsx` with selectable actual AntD Popover and Casual mode; compile both current Tailwind configurations for emitted display/target declarations. Extend focused action/user tests only if required for a concrete gap.
**Status:** Complete

- [x] Freeze test-only changes and request the parent's RED runner window.
- [x] Confirm failures are missing visibility/menu/target behavior, preserving positive authority controls.

Qualified RED: the sixteen new actual full/compact cases produced twelve causal failures and four passing processing/Pro controls. Existing sixteen renderer/ownership controls passed. Failures covered hidden latest controls, duplicate/inert overflow and real 32px desktop/unsized user declarations; unrelated response-dispatch fixture errors in the combined log are excluded.

## Stage 2: Repair only the shared presentation
**Goal:** Make existing actions reachable without duplicating menus.
**Success criteria:** `Message.tsx` uses its existing last-row condition to expose the completed assistant actions; `MessageActionsBar.tsx` retains one working overflow outside older-row hover-only primary controls and removes the inert chip; `PlaygroundUserMessage.tsx` exposes its existing actions through one working Popover for Casual/touch/keyboard. Existing always-visible FeedbackButtons targets also remain at least 44px at all widths, preserving compact icons and exact feedback handlers. Busy/edit guards and existing callbacks remain unchanged.
**Tests:** Actual renderer and direct real FeedbackButtons geometry against both generated client utility cascades must fail before changing its size declarations. Same causal tests plus existing MessageActionsBar menu/feedback, user design-system and message ownership/error-recovery controls.
**Status:** In Progress

- [x] After qualified RED, implement only the three owned production files.
- [x] Freeze the three original production files for the parent's focused GREEN runner.
- [x] Record causal real feedback target RED, then repair only FeedbackButtons size declarations and freeze.
- [x] Record actual secondary menu-row geometry RED before its minimum-height repair.

Feedback qualified RED: twelve independent real-component/full/compact controls reproduced 20px compact and 24px normal targets. The shared size declaration is now 44px, keeping icons and handlers unchanged. Eight real menu visibility assertions found controls before AntD's motion completed; the test now waits for actual visible portal and preserves descendant visibility assertions, with no production motion change.

Secondary menu qualified RED: all six actual Notes/Flashcards/Delete controls reproduced 32px rows; the shared OverflowMenuItem now has 44px minimum height and width. Main menu visibility and all twelve feedback controls passed in that checkpoint. Two keyboard fixtures reached the actual preserved user Delete label but expected lowercase; only that exact expected accessible name was corrected, keeping keyboard/visibility assertions.

## Stage 3: Qualify and review the bounded unit
**Goal:** Verify the shared full/compact/Compare reach and document proof limits.
**Success criteria:** Focused and neighboring controls pass, both clients compile, owned formatting/lint introduce no diagnostics, independent source review finds no authority regression. Native geometry/touch/keyboard and packaged extension remain explicit acceptance gates until actually exercised.
**Tests:** Parent-run affected suites/types/static and native shared full/compact action visibility/menu/target checks. Python untouched; Bandit not applicable to this TS-only unit.
**Status:** Not Started

- [ ] Report exact changed paths and gate results to the parent for task notes.
- [ ] Complete parent-owned native and commit checkpoints without claiming whole-program closure.
