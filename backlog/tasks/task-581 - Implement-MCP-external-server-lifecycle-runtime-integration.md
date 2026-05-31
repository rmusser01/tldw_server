---
id: TASK-581
title: Implement MCP external server lifecycle runtime integration
status: Done
labels:
- mcp-unified
- external-servers
- runtime
- security
documentation:
- Docs/superpowers/specs/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-design.md
- Docs/superpowers/plans/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-implementation-plan.md
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `GatewayExternalRuntimeManager` with injected transport, credential broker, and installer contracts.
- Added FastAPI in-process lifecycle routes; durable CLI control remains deferred by design.
- Verification recorded: expanded pytest suite passed (214 tests), Ruff passed, Bandit JSON has zero results/errors/skips at `/tmp/bandit_mcp_stage4n_external_runtime.json`, and `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented package-owned external runtime lifecycle integration for MCP Unified Stage 4N. Added start/stop/restart/refresh/reconcile manager behavior over injected transports; per-call credential broker resolution with public metadata redaction; disabled-by-default install/update contracts; FastAPI lifecycle/runtime routes; and package boundary/export coverage. Final verification: expanded pytest suite passed (214 tests); Ruff passed for touched package/test scope; Bandit passed for mcp_unified/gateway and mcp_unified/federation with zero results/errors/skips in /tmp/bandit_mcp_stage4n_external_runtime.json; git diff --check passed. Known deferrals by design: durable lifecycle CLI control, package-owned real stdio process spawning, and third-party install/update execution.
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
