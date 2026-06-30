---
id: TASK-247
title: ACP compatibility matrix contract for downstream agents
status: Done
assignee: []
created_date: '2026-05-10 21:21'
updated_date: '2026-05-10 21:24'
labels:
  - ACP
  - compatibility
  - docs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1539'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
documentation:
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Development/ACP_Production_Readiness.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement #1539 PR 1: document the ACP downstream-agent compatibility matrix contract, support states, verification levels, evidence fields, caveat taxonomy, and where status should surface. Keep scope to docs/contract and avoid installer or marketplace implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Compatibility matrix format is documented.
- [x] #2 Minimum certification checks and required evidence fields are listed.
- [x] #3 Support states distinguish protocol incompatibility from missing local runtime, API key, provider, or host prerequisites.
- [x] #4 Agent Registry/setup/docs have a planned place for compatibility status.
- [x] #5 The matrix can be updated without code changes.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented docs-only #1539 PR 1 contract in isolated worktree .worktrees/acp-compatibility-matrix-contract on branch codex/acp-compatibility-matrix-contract. Added Docs/Development/ACP_Compatibility_Matrix.md and linked it from Agent_Client_Protocol.md and ACP_Production_Readiness.md.

Verification: git diff --check passed. Targeted rg confirmed compatibility doc links, support-state language, current matrix, caveat taxonomy, and status surface plan. Bandit not run because this slice changes docs and Backlog metadata only; no Python code was touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the ACP downstream-agent compatibility matrix contract for #1539 PR 1. The new docs define support states, verification levels, capability checks, required evidence fields, current documented/unverified agent rows, minimum certification checklists, caveat taxonomy, and the planned setup/Agent Registry/admin reporting status surfaces. Linked the contract from the ACP operator guide and production readiness matrix. Verification was docs-focused: git diff --check and targeted rg review passed; Bandit was skipped because no Python code changed.
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
