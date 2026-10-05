# Chat save-mode transition cancellation plan

**Task/spec:** TASK-13454; Stage 1 of `Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md`.
**Goal:** A rejected clear/new-chat action cannot change the conversation's temporary mode or announce a successful transition.
**Architecture:** Respect the existing synchronous `clearChat` acceptance result in both shared mode handlers. Keep compatibility with compact reset callbacks that return void. No new transition framework or storage authority.
**ADR assessment:** No new ADR. This preserves ADR-049 lifecycle and the existing settings navigation cancellation contract.
**Baseline:** `73eedc8101` on `codex/chat-ux-remediation-20261004`; origin/dev remains `75ab224081bf140ef52017c1a9b0a04f6878d488`.

## Stage 1: Reproduce cancellation and ordering
**Goal:** Show that both mode handlers mutate mode before the clear is accepted.
**Success Criteria:** Rejected clear leaves mode and feedback unchanged; accepted clear precedes mode change.
**Tests:** Existing persistence suite, with a small parameterized regression covering WebUI and compact handlers.
**Status:** Complete

- [x] Run the existing persistence suite as baseline: 13 passed.
- [x] Add accepted/rejected clear cases through both real handlers and the existing confirmation callback; run red: four failures on premature mode mutation/wrong ordering.

## Stage 2: Respect the accepted transition
**Goal:** Both handlers retain the source mode when clear returns false.
**Success Criteria:** Minimal production changes pass the regressions and existing persistence behavior.
**Tests:** Complete persistence suite plus navigation guard regression; WebUI typecheck and baseline-relative touched lint.
**Status:** Complete

- [x] Return before any mode mutation/notice when clear is rejected; perform accepted clear before changing mode.
- [x] Preserve no-message toggling, private-chat lock and confirmation behavior.
- [x] Verify tests, typecheck, lint and diff: complete persistence/navigation suites 19 passed; WebUI typecheck exit 0; touched ESLint zero errors/no new warnings (baseline persistence source 20/test 2); git diff --check clean. Bandit does not apply to this TS/docs-only unit.

## Stage 3: Review and record
**Goal:** Keep concrete progress reviewable without claiming wider Stage 1 completion.
**Success Criteria:** Review findings resolved, canonical Backlog notes and a working local commit.
**Tests:** Independent review, Backlog normalization, staged diff checks.
**Status:** In Progress

- [x] Review with the two authorized independent reviewers: both found no Critical/Important issues in this unit. Their separate preflight/draft recommendations remain the next work.
- [ ] Record results, archive the preceding creation setup plan in `73eedc8101`, remove only that completed plan, and commit this delivery.
- [ ] Continue character preflight and source-draft retention. Other clear callers that also replace identity belong to the shared character-transition contract and remain under review; deletion-only/clear-only callers do not alter the mode after a rejection.

Formatting check: all three touched source/test files are flagged by Prettier on both HEAD and delivery. Existing formatting retained without an unrelated rewrite.
