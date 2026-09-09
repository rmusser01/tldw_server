# Buddy PR review and merge

Task: TASK-13226
Spec: Docs/superpowers/specs/2026-09-08-buddy-persona-ux-remediation.md
ADR required: no new ADR
ADR path: backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md
Reason: corrections within existing ownership, runtime and interaction contracts.

## Global Constraints

- Preserve independent artwork/Persona identity, exact explicit target, one visible Buddy, navigation continuity, workspace no-microphone and guarded speech.
- Verify external review claims; do not blindly follow their suggested implementation. Preserve central security and transaction/thread ownership, bounded resource behavior, attribution and public compatibility.
- No full local test sweep, live provider/microphone/audio or production profile changes. Run affected regressions and existing CI checks. Never bypass branch protection or ignore a required failing check.
- Work only in this isolated worktree. Implementer owns assigned production/tests; root owns Backlog/docs/derived artifacts, commits, pushes and merge. No nested agents.

## Task 1

Reproduce and resolve the three failing shared frontend CI shards using faithful provider/router/query mocks and correct current behavior. Preserve Buddy and all existing assertions; no skipped tests or broad unrelated refactors. Root owns prior Qodo disposition confirmation and publication.

1. Read AGENTS, the listed task/spec/ADR and relevant testing lessons. Reproduce each suspected defect or establish source evidence for a non-applicable suggestion.
2. Make the smallest corrections and add behavior-focused tests. Preserve UI-thread publication after background reads and revalidate authority after awaits.
3. Run affected tests/static checks, freeze source, and record commands/results and every review disposition.
4. Independent review before root commits/pushes; resolve Important/Critical findings with scoped rechecks.

## Root integration

Confirm latest dev and rebase, retaining independent conflict additions. Run required derived checks after source is frozen, verify new PR head/checks/Qodo state, then merge only once the requested gates are met. The explicit user request authorizes rebase, lease-protected feature-branch updates and merge; no merge to a different base or administrative bypass.

## Task 2 — diagnostics publication race in shard7

The new CI run reports one failing cap-rendering test in ChatPane.stage4.lorebook-activity.test.tsx. It observes request invocation then performs a synchronous DOM read; investigate a response/publication race with a deferred response and verify actual behavior before changing the test. Preserve the bounded eight-card contract and existing authorization tests; do not add sleeps/retries or raise timeouts. Own this test file unless evidence establishes a production defect. Run focused ChatPane cases and changed-scope static checks, freeze, and independently review. No new ADR for test synchronization; existing ADR-005 remains applicable to this PR. Root owns docs/task/git.

## Task 3 — cockpit mock and assistant-reload test contracts

The next CI impact partition exposed two failures: the Playground cockpit mock omits the currently consumed LEGACY_SERVICE_PROMPT_DEFAULTS (also fails exactbase); the existing ReviewTab assistant-reload test clicks the loading Reload control during automatic conflict recovery, so no manual reload starts (base replay passes). Investigate and reproduce deterministic boundaries, fix only faithful mock/synchronization contracts, preserve product behavior and exact assertions. Own these two test files; no Flashcards feature or production changes, sleeps, retries or timeout increases. Run affected suites and pinned static/formatter comparisons, freeze, then independent review. No new ADR for test harness corrections. Root owns tasks/docs/git and checks all published CI before another push.
