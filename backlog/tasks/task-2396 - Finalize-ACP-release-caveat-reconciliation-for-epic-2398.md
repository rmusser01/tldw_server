---
id: TASK-2396
title: Finalize ACP release caveat reconciliation for epic 2398
status: In Progress
labels:
- ACP
- release-closeout
- github-2398
references:
- https://github.com/rmusser01/tldw_server/issues/2398
modified_files:
- IMPLEMENTATION_PLAN_acp_final_reconciliation_2398.md
- Docs/Development/ACP_Release_Caveat_Closeout_2026_06_21.md
- Docs/Development/ACP_Production_Readiness.md
- tldw_Server_API/Config_Files/agents.yaml
- tldw_Server_API/tests/Agent_Client_Protocol/test_acp_agent_registry.py
documentation:
- 'Implementation notes - GraphQL confirmed #2398 open and all child issues closed;
  audit covered readiness docs plus compatibility docs plus setup guide plus seeded
  registry plus runner config plus Agent Registry UI; corrected stale Goose Hermes
  OpenCode registry caveats so June 20 workspace-live-e2e evidence is reflected while
  artifact sandbox reviewer-loop and failure-diagnostic caveats remain.'
- Verification - registry pytest passed; git diff check passed; targeted stale wording
  search passed with the old non-empty MCP caveat only on Codex as expected; Bandit
  was run on the touched pytest file and reported the existing B101 test assert baseline
  only with no new assert statements added.
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All #2398 child issues are confirmed closed or explicitly accounted for.
- [ ] #2 ACP docs and setup/registry surfaces agree with final evidence state.
- [ ] #3 Release support claims distinguish backend E2E, host stdio, sandbox, artifact, reviewer-loop, and failure-diagnostic evidence.
- [ ] #4 Any drift is corrected with minimal changes and validation is recorded.
- [ ] #5 #2398 receives a final reconciliation comment and can be closed when appropriate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Audit ACP readiness, compatibility, setup, sandbox, live-agent, and retention/redaction docs against the now-closed child issue evidence.
2. Search API/setup-guide and Agent Registry surfaces for stale release-caveat wording that conflicts with the final evidence state.
3. Apply only minimal docs/setup corrections if drift exists.
4. Record verification and final reconciliation evidence for #2398, then open a narrow PR or close #2398 directly if no repo changes are needed.
<!-- SECTION:PLAN:END -->

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
