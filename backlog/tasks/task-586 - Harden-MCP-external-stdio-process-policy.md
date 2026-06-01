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
Design phase updated after review: the spec now requires a sibling process_policy module, explicit PATH trust semantics, JSON/TOML coercion validation, Windows/POSIX path normalization guidance, policy-aware env/PATH interaction, default factory identity preservation when no custom policy is configured, and runtime-manager redaction/status coverage for policy-denied starts.
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
