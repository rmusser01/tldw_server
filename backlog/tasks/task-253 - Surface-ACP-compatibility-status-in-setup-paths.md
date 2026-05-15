---
id: TASK-253
title: Surface ACP compatibility status in setup paths
status: In Progress
assignee: []
created_date: '2026-05-11 00:35'
updated_date: '2026-05-11 04:32'
labels:
  - ACP
  - compatibility
  - frontend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1539'
  - 'https://github.com/rmusser01/tldw_server/pull/1555'
documentation:
  - Docs/Development/ACP_Compatibility_Matrix.md
  - Docs/Development/ACP_Certification_Checklist.md
  - Docs/Development/Agent_Client_Protocol.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement #1539 PR3: make downstream-agent compatibility status visible in Agent Registry/setup paths using the same support-state language as the ACP compatibility docs. Keep scope status-oriented: shared status/caveat contract, setup/registry surfacing, docs/release-note linkage. Do not build installer or marketplace behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Registry/setup surfaces can distinguish configured-but-unverified from verified/supported.
- [x] #2 Docs and UI use the same support-state language.
- [x] #3 Release notes can make accurate live-agent claims.
- [x] #4 Unsupported or unverified states are actionable without implying installer or marketplace support.
- [x] #5 Remaining per-agent live verification is split into follow-up issues or documented as out of scope for #1539.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:BEGIN -->

Opened PR #1562: https://github.com/rmusser01/tldw_server/pull/1562.

Created follow-up live-certification issues #1563 and #1564 for remaining commercial and OSS/custom agent verification work.

Added PR3 progress comment to #1539: https://github.com/rmusser01/tldw_server/issues/1539#issuecomment-4416884893.

PR #1562 review fixes: added Pydantic setup-guide response models, switched compatibility docs URLs to /docs-static/Development/ACP_Compatibility_Matrix.md, normalized invalid support_state/verification_level values to conservative defaults, and moved the static UI compatibility color map out of AgentCard.

Review-fix verification passing: python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py -q; ./node_modules/.bin/vitest run src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism; python -m py_compile touched ACP Python files; git diff --check; Bandit JSON at /tmp/bandit_acp_compatibility_status_surfaces.json with zero findings.
<!-- SECTION:NOTES:END -->

Implemented ACP compatibility status surfacing across registry metadata, /api/v1/acp/health, /api/v1/acp/setup-guide, /api/v1/acp/agents, and the WebUI Agent Registry.

Verification passing: python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py -q; ./node_modules/.bin/vitest run src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism; python -m py_compile touched ACP Python files; git diff --check; Bandit JSON at /tmp/bandit_acp_compatibility_status_surfaces.json with zero findings.

Repo-wide UI TypeScript check attempted with ./node_modules/.bin/tsc --noEmit -p tsconfig.json and failed on existing unrelated baseline errors outside this ACP/Agent Registry slice.
<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
- [x] #7 Focused UI/backend tests updated or added
- [x] #8 Bandit run for touched Python paths or documented skip
- [x] #9 PR3 progress comment added to #1539
<!-- DOD:END -->
