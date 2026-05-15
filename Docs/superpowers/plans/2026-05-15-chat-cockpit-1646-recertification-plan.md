# Chat Cockpit Issue 1646 Recertification Plan

Issue: https://github.com/rmusser01/tldw_server/issues/1646
Task: TASK-369
Branch: codex/chat-cockpit-1646-recertification

## Scope

This PR targets only strict evidence and implementation gaps from issue #1646 for the main WebUI `/chat` cockpit rails. It must not change the browser-extension sidepanel/sidebar, app sidebar, unrelated pages, backend architecture, or non-chat workflows except where a focused test fixture requires disposable data through an existing real API.

Real-server proof must use the running server with no mocked route payloads and no `page.route`.

## Stage 1: Baseline And Gap Confirmation
**Goal**: Reconfirm current implementation/test coverage against the live #1646 checklist from `origin/dev`.
**Success Criteria**:
- Identify exact components, hooks, and tests responsible for prompt, assistant, model settings, MCP, mobile, and focus workflows.
- Record whether each open item needs implementation, focused tests, real-server proof, or issue-only closeout evidence.
- Preserve the branch scope around main `/chat`.
**Tests**:
- Run existing focused component tests where feasible to establish baseline.
- Run or inspect the existing real-server spec preconditions before modifying it.
**Status**: Complete

## Stage 2: P0 Proof Gaps
**Goal**: Close the highest-risk P0 evidence gaps first: real prompt, real persona, model setting persist/restore, MCP populated/unavailable state distinction, and assistant transition matrix.
**Success Criteria**:
- Prompt rail real-server proof selects and clears a real prompt when available, otherwise asserts the real recoverable empty state and has populated component/integration coverage.
- Persona proof covers real persona selection/clear when available, otherwise real empty state plus populated component/integration coverage.
- Assistant transitions are covered for character to character, character to persona, persona to character, persona to none, and none to character/persona.
- Model & Chat rail proof persists one harmless scoped setting, verifies visible/default-vs-override state, and restores the original value.
- MCP rail distinguishes populated, chat-enabled/executable, user-disabled, unavailable/degraded states where state exists, and disabled controls explain why they cannot be used.
**Tests**:
- Focused Vitest for assistant transition matrix and MCP state display.
- Real-server Playwright expansion in `chat-cockpit.real-server.spec.ts` without route mocking.
**Status**: Complete

## Stage 3: P1 Mobile And Focus Proof
**Goal**: Prove mature cockpit quality for mobile workflows and keyboard focus return across dialogs/selectors.
**Success Criteria**:
- Mobile proof covers assistant selection/clear, prompt selection/clear, model settings, MCP settings, web search, and focus-mode return.
- Focus moves into and returns from prompt selector, assistant selector, MCP settings, and Model & Chat settings.
- Run controls are verified from the rail/status surface for enabled and disabled states where current implementation supports them.
**Tests**:
- Focused component tests for focus return.
- Real-server Playwright mobile flow with the same no-mock restriction.
**Status**: Complete

## Stage 4: P2 Visual QA And Copy/IA Closeout
**Goal**: Refresh final responsive evidence and tighten only directly observed `/chat` cockpit polish gaps.
**Success Criteria**:
- Desktop cockpit, desktop focus mode, mobile cockpit tabs, and mobile focus mode screenshots are captured from the real WebUI.
- Visible rail state includes prompt, context/web search, model/chat settings, MCP, and character/persona where available.
- Long labels, empty states, degraded states, and selected prompt/persona/model states do not overflow or overlap.
- Any copy or information architecture changes are limited to making existing rail controls clearer and more actionable.
**Tests**:
- Visual QA screenshots saved under `/private/tmp` and referenced in closeout notes.
- Focused regression tests if copy/IA changes alter behavior.
**Status**: Complete

## Stage 5: Verification, PR, And Issue Closeout
**Goal**: Package a recertification PR against `dev` and update issue #1646 checkbox-by-checkbox using evidence from this branch.
**Success Criteria**:
- Focused Vitest/Playwright suites pass or any non-chat baseline failures are clearly separated.
- Bandit is run for touched backend code if any backend code is touched; otherwise document a frontend-only skip.
- Draft PR links TASK-361 and #1646, summarizes completed P0/P1/P2 evidence, and requests the required human-written Change summary before merge readiness.
- Issue #1646 is updated checkbox-by-checkbox and closed only after every P0, P1, and P2 item has evidence.
**Tests**:
- Final focused component tests.
- Final real-server Playwright test.
- Git diff self-review and `git diff --check`.
**Status**: Complete
