---
id: TASK-13411
title: >-
  MCP gateway CLI approval-grant test flakes: CLI prints nothing
  (JSONDecodeError)
status: To Do
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-01 17:55'
labels:
  - bug
  - mcp
  - testing
  - flaky
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py::test_gateway_cli_approval_grant_lifecycle failed in CI (platform-mcp-inapp, #3063 run 36693239728, 2026-09-30) with json.decoder.JSONDecodeError at line 2808: the CLI subprocess produced empty stdout. It passes 3/3 locally on two trees.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Cause of the empty CLI output identified (subprocess timing, stderr-only error, or race) and the test passes reliably in CI
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
