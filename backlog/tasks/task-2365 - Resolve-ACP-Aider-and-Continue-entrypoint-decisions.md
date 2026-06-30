---
id: TASK-2365
title: Resolve ACP Aider and Continue entrypoint decisions
status: Done
references:
- https://github.com/rmusser01/tldw_server/issues/2050
- https://github.com/rmusser01/tldw_server/issues/2051
- https://github.com/rmusser01/tldw_server/issues/1563
- https://github.com/rmusser01/tldw_server/pull/2369
documentation:
- Docs/superpowers/specs/2026-06-16-acp-aider-continue-entrypoint-decisions.md
modified_files:
- Docs/Development/ACP_Certification_Checklist.md
- Docs/Development/ACP_Compatibility_Matrix.md
- Docs/Development/ACP_OSS_Custom_Certification_2026_05_11.md
- Docs/superpowers/plans/2026-06-16-acp-aider-continue-entrypoint-decisions.md
- Docs/superpowers/specs/2026-06-16-acp-aider-continue-entrypoint-decisions.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py
- tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track the narrow ACP follow-up to resolve Aider and Continue entrypoint/adapter decisions, keeping support claims conservative unless live ACP stdio evidence exists.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Aider entrypoint decision is resolved as an unverified external adapter candidate, not native ACP support.
- [x] #2 Continue entrypoint decision is resolved as documented-only `cn` CLI with no ACP stdio entrypoint.
- [x] #3 Compatibility matrix, certification checklist, and legacy evidence note reflect both decisions.
- [x] #4 Tests cover seeded registry rows and registry-backed smoke manifests.
- [x] #5 GitHub child issues #2050 and #2051 are updated after PR creation/merge; parent #1563 remains open until all closeout criteria are satisfied.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-16-acp-aider-continue-entrypoint-decisions.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Aider/Continue entrypoint decision metadata and docs. Aider is now represented as a documented-unverified external adapter candidate for aider-acp; Continue now uses the current cn display command and remains documented-only with no ACP stdio entrypoint.

Verification:
- `python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q` passed with 103 tests.
- `python -m bandit -r tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -s B101 -f json -o /tmp/bandit_acp_aider_continue_decisions.json` reported 0 results and 0 errors.
- Aider and Continue ACP smoke manifests render valid JSON and carry the expected adapter/documented-only blocker metadata.
- `git diff --check` passed.

Review pass:
- Rebasing PR #2369 onto `origin/dev` completed without conflicts.
- Addressed review feedback by aligning the Continue matrix strategy with `documented_candidate`, making the Aider manifest test independent of host PATH state, folding long YAML compatibility notes, and sharing seeded registry setup through a pytest fixture.
- Re-ran the focused pytest suite with 103 passing tests and Bandit with 0 results / 0 errors.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Opened PR #2369 to resolve the remaining Aider and Continue ACP decision work. Aider is recorded as an unverified `aider-acp` external adapter candidate that blocks on `adapter_missing`; Continue uses the current `cn` CLI command but remains documented-only with `entrypoint_strategy_missing` because no ACP stdio entrypoint or maintained adapter is identified. Posted status comments on #2050, #2051, and #1563; #1563 remains open pending post-merge parent closeout review. Rebased the PR onto latest `dev` and addressed review feedback on strategy consistency, PATH-hermetic tests, YAML readability, and registry-test fixture reuse.

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
