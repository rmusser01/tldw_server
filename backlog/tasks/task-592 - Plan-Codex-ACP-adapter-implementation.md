---
id: TASK-592
title: Plan Codex ACP adapter implementation
status: Done
labels:
- ACP
- Codex
- agents
- planning
priority: High
ordinal: 592
documentation:
- Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md
- Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the implementation plan for the first Codex ACP implementation slice from the approved orchestration design. Scope includes backend adapter registry/status semantics, Go runner launch rules, Codex external ACP adapter profile metadata, frontend readiness handling, documentation, and verification strategy. This task creates the implementation plan only; code implementation follows after plan approval.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A complete implementation plan is saved under Docs/superpowers/plans with task-by-task TDD steps and exact file references.
- [x] #2 The plan separates profile/runtime implementation from live certification evidence and keeps later app-server support out of the first slice.
- [x] #3 The plan includes regression coverage for legacy adapter_acp input and native ACP profiles without acp_command.
- [x] #4 The plan review loop is run and any issues are addressed before execution handoff.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan review loop completed. Iteration 3 status: Approved. Advisory accepted by adding runner fallback readiness fields for display_command, display_binary_found, adapter_found, and credential_state. Verification for this planning task: git diff --check passed; ASCII check passed; placeholder/TODO scan found no unresolved placeholders except legitimate Python field_validator(...) notation. Bandit skipped because this task only adds documentation/Backlog plan artifacts and changes no executable Python code. Live Codex ACP certification remains an explicit follow-up task and is not claimed by this plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the Codex ACP adapter implementation plan at Docs/superpowers/plans/2026-06-01-codex-acp-adapter-implementation-plan.md. The plan covers canonical external_acp_adapter strategy support, legacy adapter_acp import compatibility, static fallback normalization, DB adapter metadata, pinned codex-acp 0.15.0 profile docs, passive/no-spawn Go runner readiness, frontend structured readiness gating, certification-manifest compatibility, and final verification. The review loop found and fixed passive runner probing, stale frontend readiness fallback, and stale static Codex fallback risks before approval.
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
