---
id: TASK-300
title: Design VN script operation catalog and guided draft editing
status: Done
assignee: []
created_date: '2026-05-12 14:25'
labels:
  - vn
  - vn-scripts
  - authoring
  - api
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1610'
  - 'https://github.com/rmusser01/tldw_server/issues/1391'
documentation:
  - Docs/API-related/VN_PLATFORM_API.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next API-first VN authoring sprint from GitHub issue #1610: backend-owned script operation/snippet catalog and server-validated guided draft editing for custom frontends and the bundled WebUI. Scope is design/spec first before implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design defines backend-owned authoring catalog scope, opcode metadata, snippet metadata, and custom frontend contract.
- [x] #2 Design defines server-side snippet preview/apply behavior and how it reuses existing draft validation/diagnostics.
- [x] #3 Design preserves backend validation, manifest, policy, generation-profile, and publish authority without adding a frontend rule engine.
- [x] #4 Design includes WebUI consumption model, error handling, tests, docs, and review risks.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
- Wrote design spec at `Docs/superpowers/specs/2026-05-12-vn-script-authoring-catalog-design.md`.
- Ran a focused spec review through subagent Raman. Addressed findings around non-mutating preview validation, generation/profile-owned limits, generated-choice patch shape, strict parameter schemas, concrete error details, and backend-owned invalid-draft behavior.
- Re-ran focused review through subagent Hegel; result was APPROVED.
- Ran an additional local critique at user request before implementation planning. Tightened capability-token ownership, supplied-draft preview semantics, transport status mapping, and nested snippet-parameter validation.
- Bandit skipped because this task only changes Markdown design/task documents.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed the VN script authoring catalog and guided draft editing API for issue #1610. The spec defines `vn-authoring-catalog`, snippet preview/apply endpoints, backend-owned validation and diagnostics boundaries, custom frontend behavior, WebUI consumption, error contracts, security controls, tests, rollout stages, and review risks.
<!-- SECTION:FINAL_SUMMARY:END -->
