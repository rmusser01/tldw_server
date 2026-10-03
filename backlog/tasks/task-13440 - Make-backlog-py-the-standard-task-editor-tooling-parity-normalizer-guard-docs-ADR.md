---
id: TASK-13440
title: Make backlog-py the standard task editor (tooling parity, normalizer, guard,
  docs, ADR)
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner decision (2026-10-03): tools/backlog-py replaces the Node Backlog.md CLI/MCP as the standard task editor. Today the two editors corrupt each other's files: the Node CLI writes SECTION:NOTES and, when it edits a backlog-py task, wraps the existing SECTION:IMPLEMENTATION_NOTES in a second NOTES section and duplicates FINAL_SUMMARY markers (even on a label-only edit). 331 task files on dev already mix the formats. The Node CLI also picks task ids from local branches only, so ids collide across sessions. This task covers the tooling PR: close the agent-used CLI/MCP gaps in backlog-py (labels on create/edit; acceptance criteria on create; --ac/--remove-ac and title edits; notes replace), add a normalize command that rewrites Node-style/nested/duplicated sections into backlog-py's canonical form, add a CI ratchet that every task file parses with backlog-py and the count of Node-style sections cannot grow, update AGENTS.md and the backlog-py README for the cutover, and record the decision in a new ADR (ADR-002's 'require a task' decision stands; this changes the tool).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 backlog-py CLI and MCP support labels on create/edit, acceptance criteria on create, adding/removing criteria and retitling on edit, with tests
- [ ] #2 A backlog-py normalize command rewrites SECTION:NOTES, nested NOTES/IMPLEMENTATION_NOTES and duplicated FINAL_SUMMARY markers into one canonical form, idempotently, with tests on real mixed fixtures
- [ ] #3 A CI-enforced test checks every backlog task file parses with backlog-py and the count of Node-style sections does not exceed a recorded baseline
- [ ] #4 AGENTS.md and tools/backlog-py/README.md direct agents to backlog-py (command name backlog-py; explicit-id creation guidance), and a new ADR records the cutover
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
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
