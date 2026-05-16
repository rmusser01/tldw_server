---
id: TASK-391
title: ACP artifact release verification and signoff for issue 1704
status: Done
assignee: []
created_date: '2026-05-16 00:11'
labels:
  - acp
  - artifacts
  - verification
dependencies:
  - TASK-381
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1704'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
  - 'https://github.com/rmusser01/tldw_server/issues/1705'
  - 'https://github.com/rmusser01/tldw_server/issues/1706'
  - 'https://github.com/rmusser01/tldw_server/issues/1707'
documentation:
  - Docs/Product/Traceable_Work_Product_Artifact_Contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
  - Docs/Development/ACP_Artifact_Release_Verification_2026_05_15.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1704: release-grade verification for the ACP traceable artifact stack after storage/API, promotion, UI, and export slices have landed. Cover ACP-to-artifact promotion/API invariants, UI artifact detail/export/provenance behavior, fixture evidence, docs/readiness links, and explicit deferral of non-golden-path artifact types where needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Contract tests cover ACP-to-artifact promotion and artifact API invariants.
- [x] #2 UI tests cover artifact detail, redacted views, provenance, and export controls or document any host/tooling blocker with a follow-up issue.
- [x] #3 Verification evidence is linked from #1532 and the relevant implementation issues.
- [x] #4 Remaining non-golden-path artifact types are tracked separately or explicitly deferred.
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

## Implementation Notes

- Added focused ACP artifact promotion contract tests for accepted promotion, update/version lineage, rejected and needs-revision skip paths, malformed candidate rejection, and accepted-version export identity.
- Added `Docs/Development/ACP_Artifact_Release_Verification_2026_05_15.md` as the #1704 evidence record and linked the artifact stack from the ACP readiness matrix and product contract.
- Updated the ACP PRD and traceable artifact contract so storage/API, promotion, UI detail, export identity, and verification reflect the shipped slices while preserving explicit deferrals.

## Verification

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py -q` -> `6 passed`, 5 warnings.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Agent_Orchestration/test_orchestration_api.py tldw_Server_API/tests/ChaChaNotesDB/test_workspace_sub_resources_db.py tldw_Server_API/tests/Workspaces/test_workspaces_api.py -q` -> `104 passed`, 5 warnings.
- `cd apps/packages/ui && bun run test src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx src/store/__tests__/workspace-api-first.test.ts` -> 2 files passed, 27 tests passed.
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/tests/Agent_Orchestration/test_artifact_promotion_contract.py -s B101 -f json -o /tmp/bandit_acp_artifact_signoff_1704.json` -> `results=[]`, `errors=[]`; `B101` skipped for pytest assertions.
- `git diff --check` -> clean.

## Final Summary

Release signoff for #1704 is ready for PR review: the branch adds combined backend contract coverage, records verification evidence, updates ACP readiness/product docs, and explicitly defers non-golden-path artifacts, rich exports, Chatbook packaging, file-artifact materialization, and live downstream-agent certification.
