---
id: TASK-2270
title: Add MCP gateway tool-use reporting CLI commands
status: Done
labels:
- mcp
- gateway
- cli
- observability
- evals
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 6 from the MCP tool-use eval/reporting plan: gateway CLI tool-events report/export/cleanup commands, config validation payload, and focused CLI tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Gateway CLI parses nested tool-events report/export/cleanup subcommands with config, filters, limits, and output options.
- [ ] #2 CLI commands load reporting config and return deterministic JSON errors when reporting is disabled or lacks a persistent store.
- [ ] #3 CLI report/export/cleanup commands operate against the configured SQLite tool-use event store.
- [ ] #4 Gateway config validation includes the tool_use_reporting payload.
- [ ] #5 Focused gateway CLI tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added gateway CLI tool-events report, export, and cleanup commands backed by the configured SQLite tool-use reporting store. Validate-config now includes tool_use_reporting settings, report queries honor --since filters, and CLI commands fail clearly when reporting is disabled or only memory-backed.
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
