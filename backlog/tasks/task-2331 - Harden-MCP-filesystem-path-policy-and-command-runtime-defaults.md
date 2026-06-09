---
id: TASK-2331
title: Harden MCP filesystem path policy and command-runtime defaults
status: In Progress
labels:
- mcp
- filesystem
- security
- policy
priority: High
references:
- 'User-approved approach: policy hardening first'
- runtime routing second; diff parser expansion only where current gaps block real
  use.
documentation:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-06-09-mcp-filesystem-policy-hardening-design.md
- Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design and implement the next MCP filesystem slice: verify path/action constraints for fs.read, fs.write, fs.edit, and fs.patch first, then update command-runtime defaults to prefer structured primitives over legacy fs.read_text/fs.write_text where safe.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-06-09-mcp-filesystem-policy-hardening-implementation-plan.md. It sequences work as: adapter candidate forwarding; action-aware path-grant regression tests; virtual CLI structured fs.read/write-create routing; focused tests, Bandit, and Backlog closeout. Plan review correction included filtered visible command descriptors so adapters can reliably choose fs.read over fs.read_text only when fs.read is actually visible.
<!-- SECTION:PLAN:END -->

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
