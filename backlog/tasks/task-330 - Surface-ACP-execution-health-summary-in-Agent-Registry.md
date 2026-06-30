---
id: TASK-330
title: Surface ACP execution health summary in Agent Registry
status: Done
assignee: []
created_date: '2026-05-14 02:18'
updated_date: '2026-05-14 02:31'
labels:
  - ACP
  - admin
  - frontend
  - reporting
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1537'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/pull/1648'
documentation:
  - Docs/Development/Agent_Client_Protocol.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the ACP execution-health summary contract from issue #1537 in the Agent Registry/admin setup surface so operators can see recent session totals, failures, setup-health dimensions, retention/redaction state, and compatibility warnings without digging through raw API responses.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Agent Registry fetches the ACP execution-health summary using the backend admin contract and supports the same direct/proxy transport behavior as existing ACP health calls.
- [x] #2 The UI summarizes session totals, completion/failure counts, recent failure categories, setup-health dimensions, retention/redaction metadata, and compatibility warnings without claiming unsupported agent certification.
- [x] #3 Unavailable, unauthorized, or failed summary fetches degrade gracefully without blocking the existing registry health view.
- [x] #4 Focused frontend tests cover representative summary rendering and failure handling.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD checkpoint: added failing Agent Registry coverage for the admin ACP execution-health summary transport/rendering and unavailable-summary behavior; implemented the summary fetch/UI; focused Vitest is now green.

Verification: ./node_modules/.bin/vitest run src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx --maxWorkers=1 --no-file-parallelism passed; bun run verify:design-system-state passed; git diff --check passed. Full package tsc --noEmit -p tsconfig.json still fails on existing repo-wide TypeScript baseline errors outside this slice; no Python touched, so Bandit is not applicable.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the ACP admin execution-health summary display in Agent Registry, backed by the existing browser transport/auth helpers and typed frontend models for the backend summary contract. Added focused tests for direct/proxy summary fetches, representative summary rendering, compatibility warnings, and graceful unavailable-summary handling. Also migrated the touched Agent Registry and ACP history error alerts to design-system primitives and removed stale product-state baseline entries.
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
