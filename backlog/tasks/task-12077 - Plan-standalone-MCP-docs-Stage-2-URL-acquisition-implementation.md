---
id: TASK-12077
title: Plan standalone MCP docs Stage 2 URL acquisition implementation
status: Done
priority: high
documentation:
- Docs/superpowers/specs/2026-06-30-standalone-mcp-docs-url-acquisition-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for Stage 2 optional single-page URL acquisition for the standalone MCP docs corpus. The plan must follow the hardened design spec, use TDD task slices, preserve the standalone import boundary, and avoid implementation code changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan covers settings, source policy, URL fetch safety, extraction, acquisition service, MCP provider/shim, tests, status reporting, and final verification.
- [x] #2 Plan explicitly includes DNS rebinding, structured domain/prefix matching, no-fetch-before-approval, fail-closed robots behavior, body-size limits, lazy optional extractor import boundaries, and no live-internet tests.
- [x] #3 Plan is committed and ready for execution choice.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan written at Docs/superpowers/plans/2026-06-30-standalone-mcp-docs-url-acquisition-implementation-plan.md. Self-review completed against the hardened Stage 2 spec: settings, source policy, structured domain/prefix matching, no-fetch-before-approval, DNS/IP and DNS rebinding defenses, redirect checks, transferred and decoded body limits, robots fail-closed behavior, lazy extraction, acquisition service/store integration, MCP provider/shim exposure, import-boundary tests, disabled default config, and final verification steps are covered. Placeholder scan passed with no TBD/TODO/fill-in markers.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Stage 2 implementation plan for optional standalone MCP docs URL acquisition. The plan is execution-ready, TDD-oriented, keeps web acquisition optional, preserves the standalone import boundary, and includes explicit verification/Bandit steps.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Implementation plan written under Docs/superpowers/plans/.
- [x] #2 Plan self-review completed for spec coverage, placeholders, and type consistency.
- [x] #3 Backlog task updated with final summary.
- [x] #4 Plan committed to git.
<!-- DOD:END -->
