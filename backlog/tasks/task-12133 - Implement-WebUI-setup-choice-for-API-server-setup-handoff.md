---
id: TASK-12133
title: Implement WebUI setup choice for API server setup handoff
status: Done
assignee: []
created_date: 2026-07-03 22:59
updated_date: 2026-07-04 01:55
labels:
- webui
- setup
- onboarding
dependencies:
- TASK-12123
references:
- Docs/superpowers/specs/2026-07-03-webui-setup-choice-design.md
- Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved WebUI /setup pre-wizard choice screen that explains WebUI setup versus API server setup, resolves a browser-openable API setup URL when possible, handles blocked/recovery state safely, and preserves existing manual recovery UI when setup state or metadata cannot be loaded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SetupEntryChoice renders before the existing WebUI wizard on /setup when backend setup is incomplete.
- [x] #2 API server setup link/fallback behavior follows the approved URL-resolution and local/remote copy rules.
- [x] #3 Blocked first-run state cannot enter the normal WebUI wizard until refresh returns a mutable state.
- [x] #4 Manual connection and recovery UI remain available when first-run state or metadata cannot be loaded.
- [x] #5 Focused Vitest, Playwright, typecheck, and applicable security verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-03-webui-setup-choice-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation started in isolated worktree .worktrees/codex-webui-setup-choice-impl on branch codex/webui-setup-choice-impl. Baseline focused Vitest passed: setup-status and option-setup-readiness, 19 tests.

Task 1 complete: helper and resolver committed in f8adf59740. Spec-compliance and code-quality reviews approved. Focused helper Vitest passed: 12 tests.

Task 2 complete: SetupEntryChoice component committed in 5ed2d65140. Spec-compliance and code-quality reviews approved. Focused component/helper Vitest passed: 23 tests.

Task 3 complete: /setup route integration committed in c48d86f2aa, with review fixes 8e559b37d4 and f57ae2ba97. Spec-compliance and code-quality reviews approved. Focused route Vitest passed: option-setup-readiness 17 tests. Combined setup Vitest passed: setup-entry-choice-utils, SetupEntryChoice, option-setup-readiness, setup-status, 51 tests. Bandit skipped for Task 3 because only TypeScript/React files changed.

Task 4 complete: Playwright /setup desktop/mobile smoke coverage and /setup-to-first-chat handoff coverage committed in 14c143dee1. Added quickstart runtime-auth media-readiness regression and fix so first-source milestone works when the API key is provided by runtime config instead of persisted storage.

Verification for Task 4: Playwright unified-first-run-onboarding passed 6/6; focused Vitest passed 5 files / 53 tests; git diff --check passed for changed files. bun run typecheck still fails on pre-existing unrelated baseline files: AudioStudio/TimelineEditor.tsx, ScheduledTasks/ScheduledTaskAutomationDefinitionEditor.tsx, Skills/Manager.tsx, scheduled-tasks-control-plane.ts, tldw/mcp-hub.ts, voice-cloning.ts, e2e/fixtures/knowledge-qa-live.ts, and e2e/workflows/tier-2-features/flashcards.spec.ts. No errors referenced the changed files. Bandit skipped because only TypeScript/Playwright/Markdown task files changed.

Spec/code review note: the Task 4 review-agent attempt failed due the account usage limit, so the final Task 4 review was completed locally against the changed files and verification output; no task-specific follow-up issues were found.

User requested fixing the typecheck baseline as part of the current task. Reopening TASK-12133 to include the TypeScript errors reported by apps/tldw-frontend bun run typecheck while leaving unrelated generated dirty files unstaged.

Typecheck repair complete. Fixed the TypeScript baseline errors in AudioStudio, ScheduledTasks editor/control-plane, Skills manager, MCP hub readiness path typing, voice-cloning ArrayBuffer conversion, and two E2E fixture/spec narrowing sites.

Verification after repair: bun run typecheck passed from apps/tldw-frontend; focused setup Vitest passed 5 files / 53 tests; ScheduledTaskAutomationDefinitionEditor Vitest passed 9 tests; Playwright unified-first-run-onboarding passed 6/6; git diff --check passed. Extra broad ScheduledTasksPage Vitest run has 3 unrelated existing failures where tests expect raw endpoint strings but UI now shows sanitized diagnostics; left out of this typecheck repair.

User requested investigation of the remaining ScheduledTasksPage diagnostics-copy expectation failures. Root cause: ScheduledTasksPage tests still assert raw API paths, while shared buildCapabilityState intentionally sanitizes diagnostic values to [server-endpoint]/[server-url] and has unit coverage for that behavior.

ScheduledTasksPage diagnostics-copy failures fixed. The tests now assert the sanitized diagnostic copy produced by buildCapabilityState: [server-endpoint] and messages containing [server-endpoint], matching the existing capability-state sanitizer contract. Verification: ScheduledTasksPage Vitest passed 48/48; bun run typecheck passed; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2615 onto latest origin/dev and addressed the actionable PR review comments. Verification: setup/onboarding Vitest slice passed 5 files / 70 tests; ScheduledTaskAutomationDefinitionEditor Vitest passed 9 tests; ScheduledTasksPage Vitest passed 48 tests; bun run typecheck passed; Playwright unified-first-run-onboarding passed 6/6 after escalating local server binding; git diff --check passed. Bandit skipped because the follow-up touched only TypeScript/Playwright/Markdown files.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2615 follow-up complete after rebase onto latest origin/dev. Addressed still-valid review comments by preserving API setup base paths, adding optional metadata connection guards, covering public dotted hostnames, link-local IPv4, and IPv6 API origins, making the /setup wizard exclusive from manual recovery panels, surfacing setup refresh errors, replacing the direct DOM focus query by deleting the now-unneeded co-rendered recovery action, and switching the Playwright milestone helper to real waitFor visibility checks. The Gemini voice-cloning comment was already addressed by the rebased dev implementation using exact-byte copyBytes instead of Uint8Array.from. Also fixed the post-rebase scheduled-task editor type regression by keeping ScheduledTaskDefinitionResponse for the save result.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
