---
id: TASK-2307
title: Plan MCP policy decision core implementation
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-07 16:29'
labels:
  - mcp
  - profiles
  - policy
  - planning
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-06-07-mcp-profile-policy-decision-model-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a focused implementation plan for the first MCP/profile policy decision-model slice: package-level deny/ask/allow decision core, compatibility compilation from existing profile policy fields, explain/simulation contract, and tests. This plan should defer catalog visibility, path matcher compiler, external MCP wildcards, shell alias hardening, and hooks enforcement to later slices unless a small seam is needed for the core contract.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan artifact created at Docs/superpowers/plans/2026-06-07-mcp-policy-decision-core-implementation-plan.md. The plan scopes the first implementation slice to package-owned policy decision primitives, compatibility rule compilation, optional EffectivePolicyResult decision metadata, redacted explain/simulation helpers, exports, targeted tests, and Bandit validation.

Local review findings addressed before closeout:
- Kept runtime catalog visibility, path matcher compilation, external MCP wildcard enforcement, shell alias hardening, hooks, and admin/CLI surfaces out of Slice 1.
- Corrected stale implementation-closeout references from TASK-2307 to TASK-2308.
- Retained TASK-2308 as the implementation task for executing the plan.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and locally reviewed the MCP policy decision core implementation plan. Added TASK-2308 as the follow-up implementation task and corrected closeout references so implementation validation is recorded against TASK-2308.

Verification: git diff --check passed for the planning changes. Bandit was not run because this task only adds planning/backlog Markdown and no executable code.

Known skips/blockers: plan-document-reviewer subagent was not dispatched because the available multi-agent tool requires explicit user delegation; local review was used instead.
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
