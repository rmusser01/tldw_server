---
id: TASK-2294
title: Add skill and workflow runner tools for reusable agentic routines
status: In Progress
labels:
- mcp
- skills
- workflows
- agentic-execution
- tools
references:
- https://code.claude.com/docs/en/tools-reference
updated_date: 2026-07-24 00:24
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement Skill/Workflow-style agentic execution tools for reusable prompt-driven routines. Cover skill catalog discovery, allowed-tools frontmatter/profile binding, workflow scripts that orchestrate subagents or tools, permission checks, output consolidation, hooks, telemetry/evaluations, and a clear boundary from external MCP tools and tldw_server workflow/jobs infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Execution work is decomposed into completed read-only catalog/render slices TASK-2294.1 and TASK-2294.2, active model-only runner TASK-2294.3, deferred canonical read-only nested execution TASK-2294.4, and deferred durable effectful execution TASK-2294.5. This sequencing prevents the existing core SkillExecutor from becoming an MCP security bypass.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
