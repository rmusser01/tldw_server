---
id: TASK-12036
title: Adopt Agent Registry capability recovery state
status: Done
created_date: 2026-06-26 01:11
labels:
- webui
- agents
- ux
- accessibility
priority: medium
references:
- TASK-418.11
- TASK-214
documentation:
- Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
- Docs/superpowers/plans/2026-06-25-webui-stage7-agent-registry-capability-recovery-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-webui-stage7-agent-registry-capability-recovery-plan.md
- apps/packages/ui/src/components/Option/AgentRegistry/index.tsx
- apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx
updated_date: 2026-06-26 01:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the deferred WebUI capability/error-state follow-up for the standalone Agent Registry route. Route ACP health, admin execution-health, and agent-list load failures through shared user-language recovery states with retry and diagnostics where appropriate, while preserving the existing registry cards and execution-health summary when the backend is available.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Registry renders shared RecoveryCallout/StatePanel capability states for unavailable ACP health, admin execution-health, and agent-list failures instead of local raw alert copy.
- [x] #2 Recovery states include user-language title/message, retry or dismiss action as appropriate, and non-secret diagnostics for method/path/status/raw message/server URL.
- [x] #3 Existing registry cards, compatibility labels, execution-health summary, refresh action, and canonical connection behavior remain available when requests succeed or optional admin summary is permission-gated.
- [x] #4 Focused tests cover successful registry rendering plus unavailable health/summary/agent-list failure paths.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a focused Stage 7 plan document for Agent Registry capability recovery.
2. Add failing focused tests that expect shared recovery primitives and diagnostics for ACP health, admin execution-health, and agent-list load failures.
3. Preserve existing successful registry and execution-health behavior while replacing local unavailable/error alerts with shared state primitives.
4. Run focused Agent Registry tests, touched-file ESLint, whitespace checks, and record Bandit applicability.
5. Update Backlog and commit the Stage 7 slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added red/green Agent Registry coverage for ACP health failure, admin execution-health failure, and ACP agent-list load failure.
- ACP health and admin execution-health failures now retain structured request diagnostics and render non-blocking shared RecoveryCallout sections with retry actions.
- Agent-list load failure now renders a retryable shared RecoveryCallout with method/path/status/server URL/raw message diagnostics instead of a local alert with raw error text.
- Existing successful registry cards, compatibility labels, canonical connection behavior, execution-health summary, and refresh flow remain covered by the focused connection suite.
- Added local diagnostic redaction for secret-shaped raw messages before they enter capability diagnostics.
- Verification: initial focused Vitest red run failed on the three new shared-recovery assertions; after implementation, `bun run test:run ../packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx` passed with 11 tests. Touched-file ESLint exited 0 with only the existing repo-level Next pages-directory warning. `git diff --check` passed.
- Bandit: not applicable for this TS/TSX/docs-only slice.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Agent Registry capability recovery slice. ACP health, admin execution-health, and agent-list failures now use shared RecoveryCallout states with retry actions and non-secret diagnostics, while successful registry cards and execution-health summary behavior remain intact. Focused Agent Registry tests cover the successful and unavailable paths.
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
