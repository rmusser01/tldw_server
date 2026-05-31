---
id: TASK-581
title: Implement MCP external server lifecycle runtime integration
status: In Progress
labels:
- mcp-unified
- external-servers
- runtime
- security
documentation:
- Docs/superpowers/specs/2026-05-31-mcp-unified-stage4n-external-lifecycle-runtime-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next MCP Unified slice after external registry management: real upstream external server start/stop/refresh behavior, credential-secret handling, and install/update flow foundations for the standalone gateway while keeping changes reviewable and tested.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 External server lifecycle operations expose deterministic start, stop, restart/refresh behavior through the package gateway/CLI or clearly scoped runtime layer.
- [ ] #2 Credential-secret handling keeps secrets out of persisted catalog/plain responses/logs and resolves runtime environment values through explicit secret references or broker paths.
- [ ] #3 Install/update flow support is defined and implemented to the agreed minimal slice with validation and safe failure modes.
- [ ] #4 Focused unit/integration tests cover lifecycle state transitions, credential redaction/resolution, install/update success and failure paths, and package boundary behavior.
- [ ] #5 Bandit on touched Python source, focused pytest suite, and git diff --check pass before PR.
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
