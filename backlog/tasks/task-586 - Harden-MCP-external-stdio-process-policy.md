---
id: TASK-586
title: Harden MCP external stdio process policy
status: In Progress
labels:
- mcp
- security
- external-runtime
priority: High
documentation:
- Docs/superpowers/specs/2026-06-01-mcp-stdio-process-policy-design.md
- Docs/superpowers/plans/2026-06-01-mcp-stdio-process-policy-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add explicit process-execution policy for standalone MCP external stdio transports: executable allowlisting, bounded cwd validation, environment allowlist checks, deterministic denial/status payloads, and focused tests before real installer execution work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan saved at Docs/superpowers/plans/2026-06-01-mcp-stdio-process-policy-implementation-plan.md. Stages: add process_policy helper tests/module, wire stdio transport enforcement, add gateway config/CLI wiring, add runtime-manager redaction coverage, then run targeted pytest and Bandit.
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
