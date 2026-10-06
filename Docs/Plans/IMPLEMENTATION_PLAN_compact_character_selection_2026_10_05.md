# Compact character selection — TASK-13503

Approved program: Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Baseline latest dev 587cd8e9fe, merged in f460db867c. Governing ADR-049/050 retain owner-validated history and private drafts; ADR-059 governs tracking. No new ADR: this uses the existing contracts at the compact entry point.

## Stage 1: Trace the existing transition
**Goal**: Trace Sheet, real AssistantSelect, compact form, clearChat, selected-assistant writes and native preflight.
**Success Criteria**: Opening is pure picker intent; the existing onTrackedSelection interception bypasses legacy mutation. Outer modal close, source/address revision, account verification and asynchronous preparation have explicit fences.
**Tests**: Baseline existing Sheet tests, then red assertions for opening and delegated acceptance/decline.
**Status**: Complete

## Stage 2: Retain and prepare before accepted selection
**Goal**: Use the real guarded picker callback; retain exact source draft and run applicable native preflight before identity write and destination reset. Preserve source overlay/resume marker on cancel. Honor navigation rejection; do not dispatch a second cancellable request after accepted preparation.
**Success Criteria**: Failed/retired work cannot replace source or publish metadata into another chat. Successful transition retains private source draft, binds fresh destination metadata and supported settings scope. Closing unmounts the nested picker. Unsupported tracked persona/native configuration fails before leaving source.
**Tests**: Actual production callback boundary adapter with real preflight/store; held retention, owner/source changes, declined navigation, identity failure and accepted character metadata; real picker decline/lifetime regressions.
**Status**: Complete

## Stage 3: Review and verify the bounded result
**Goal**: Relevant suites, WebUI typecheck, extension compile, no-new touched lint, two independent reviews and a local checkpoint.
**Success Criteria**: No unresolved Important/Critical findings; exact checks and source-vs-native-shell limits recorded. Library and Quick Chat remain separate units in the parent program.
**Tests**: Targeted tests and both client type gates; no Python source planned, Bandit N/A unless scope changes.
**Status**: In Progress
