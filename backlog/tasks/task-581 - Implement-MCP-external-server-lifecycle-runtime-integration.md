---
id: TASK-581
title: Implement MCP external server lifecycle runtime integration
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-01 00:31'
labels:
  - mcp-unified
  - external-servers
  - runtime
  - security
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-design.md
  - >-
    Docs/superpowers/plans/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next MCP Unified slice after external registry management: real upstream external server start/stop/refresh behavior, credential-secret handling, and install/update flow foundations for the standalone gateway while keeping changes reviewable and tested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 External server lifecycle operations expose deterministic start, stop, restart/refresh behavior through the package gateway/CLI or clearly scoped runtime layer.
- [x] #2 Credential-secret handling keeps secrets out of persisted catalog/plain responses/logs and resolves runtime environment values through explicit secret references or broker paths.
- [x] #3 Install/update flow support is defined and implemented to the agreed minimal slice with validation and safe failure modes.
- [x] #4 Focused unit/integration tests cover lifecycle state transitions, credential redaction/resolution, install/update success and failure paths, and package boundary behavior.
- [x] #5 Bandit on touched Python source, focused pytest suite, and git diff --check pass before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added GatewayExternalRuntimeManager with injected transport, credential broker, and installer contracts.
- Added FastAPI in-process lifecycle routes; durable CLI control remains deferred by design.
- Addressed PR review feedback after rebasing on origin/dev: runtime routes now use Pydantic response models; external execute enforces server/tool policy before transport calls; credential broker calls no longer deep-copy noncopyable policy/context objects; health-check failures produce unhealthy status rows instead of crashing; unknown virtual tools map to a 404-style reason; transport call failures are wrapped in structured runtime errors; lifecycle/install/broker/execute paths no longer hold the manager lock across external I/O; route-reserved external server ids are rejected.
- Updated verification: focused pytest suite passed (144 tests), expanded MCP pytest suite passed (228 tests), Ruff passed, Bandit JSON has zero results at /tmp/bandit_mcp_stage4n_external_runtime.json, and git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented package-owned external runtime lifecycle integration for MCP Unified Stage 4N and addressed all current PR review threads after rebasing on `origin/dev`. The follow-up fixes added validated FastAPI runtime response schemas, fail-closed policy enforcement for external execution, non-deepcopying broker context/policy handling, health-check fallback status rows, dedicated unknown-tool and call-failure reason codes, narrower lock scope around external I/O, and reserved-id validation for runtime management route names. Final verification: focused pytest suite passed (144 tests); expanded MCP pytest suite passed (228 tests); Ruff passed for touched package/test scope; Bandit passed for `mcp_unified/gateway` and `mcp_unified/federation` with zero results in `/tmp/bandit_mcp_stage4n_external_runtime.json`; `git diff --check` passed. Known deferrals by design remain durable lifecycle CLI control, package-owned real stdio process spawning, and third-party install/update execution.
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
