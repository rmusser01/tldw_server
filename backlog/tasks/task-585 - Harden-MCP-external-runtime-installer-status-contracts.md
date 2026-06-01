---
id: TASK-585
title: Harden MCP external runtime installer status contracts
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 03:58'
labels:
  - mcp-unified
  - external-runtime
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-01-mcp-external-runtime-installer-status-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add stable, sanitized installer availability/status and install/update operation contracts for the standalone MCP gateway external runtime. Keep default installer side-effect-free and defer real package-manager execution.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime status rows expose sanitized installer capability/status metadata per configured external server.
- [x] #2 Install/update operation responses are normalized, deterministic, and do not leak credential, env, header, or command secret values.
- [x] #3 Unexpected installer adapter failures are logged for diagnostics and surfaced through stable external runtime reason codes.
- [x] #4 Focused tests cover default unsupported behavior, fake installer status/operation success, adapter failure handling, disabled/not-found paths, and secret redaction.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-mcp-external-runtime-installer-status-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
['Design approved by user on 2026-06-01.', 'Real third-party install/update execution, durable lifecycle CLI controls, frontend changes, and WebSocket upstream transport are out of scope for this slice.']
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented stable, sanitized installer status and install/update operation contracts for the standalone MCP gateway external runtime. Runtime status rows now include best-effort installer capability metadata, install/update adapter payloads are normalized and redacted, adapter failures are logged internally and surfaced through deterministic reason codes, and FastAPI response models document the new fields. Verification: focused gateway pytest suite passed (168 tests), Ruff passed on touched Python files, Bandit passed on touched gateway source with JSON output at /tmp/bandit_mcp_external_runtime_installer_status.json, and git diff --check passed. Out of scope remains real third-party package-manager execution, durable CLI lifecycle controls, frontend changes, and WebSocket upstream transport.
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
