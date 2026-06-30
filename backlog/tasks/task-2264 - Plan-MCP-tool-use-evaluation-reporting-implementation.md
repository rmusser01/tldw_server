---
id: TASK-2264
title: Plan MCP tool-use evaluation reporting implementation
status: Done
labels:
- mcp
- plan
- observability
- evals
- gateway
references:
- Docs/superpowers/specs/2026-06-06-mcp-tool-use-eval-reporting-design.md
- backlog/tasks/task-2263 - Design-MCP-tool-use-evaluation-reporting-surface.md
modified_files:
- Docs/superpowers/plans/2026-06-06-mcp-tool-use-eval-reporting-implementation-plan.md
- backlog/tasks/task-2264 - Plan-MCP-tool-use-evaluation-reporting-implementation.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for metadata-only MCP tool-use event capture, storage, export, and aggregate reporting across standalone gateway and in-process MCPProtocol paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is saved under `Docs/superpowers/plans` with exact files, TDD steps, verification commands, and execution handoff notes.
- [x] #2 Plan incorporates local design review passes and resolves the outstanding context-key, CLI-output, early-capture, compatibility, bridge-name, immutability, and bounded-report decisions.
- [x] #3 Backlog task records verification, known skips, and final summary.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created implementation plan at `Docs/superpowers/plans/2026-06-06-mcp-tool-use-eval-reporting-implementation-plan.md`.
- Local review pass tightened two issues before finalizing: package root must not re-export `SQLiteToolUseEventStore`, and standalone CLI report/export/cleanup require a persistent store instead of in-memory state.
- Subagent plan review was skipped because the current thread has not explicitly authorized subagent use for this planning step.
- Follow-up review pass incorporated five additional implementation hardening items: early `process_request()` tool-call failures, duck-typed dependency fallback to `NoopToolUseRecorder`, `profile.tools.call` effective-name side-channel, immutable `ToolUseEvent` tests, and bounded-report disclosure fields.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Saved the MCP tool-use evaluation reporting implementation plan with seven TDD stages covering event models, recorder contracts, store/report service, protocol capture, gateway runtime wrapping, CLI surfaces, docs, and verification. The plan incorporates review-pass decisions around lazy optional imports, persistent CLI reporting, UTC epoch ordering, metadata-only capture, recorder failure handling, idempotency replay capture, double-counting guards, early process_request capture, dependency compatibility, bridge effective-name reporting, event immutability, and bounded-report disclosure.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run skipped because this task only adds markdown/Backlog planning artifacts and no Python code
- [x] #5 Final summary added
- [x] #6 Known skips documented: subagent plan review skipped without explicit subagent authorization
<!-- DOD:END -->
