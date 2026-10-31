---
id: TASK-13411
title: >-
  MCP gateway CLI approval-grant test flakes: CLI prints nothing
  (JSONDecodeError)
status: Done
assignee: []
created_date: '2026-10-01 17:53'
updated_date: '2026-10-02 22:08'
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
- [x] #1 Cause of the empty CLI output identified (subprocess timing, stderr-only error, or race) and the test passes reliably in CI
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Mechanism: secrets.token_urlsafe grant ids start with '-' about 1 time in 64. The failing CI commit d2aff91b86 predates 874139d2be and called 'revoke-approval-grant <id> --config <path>', so argparse read a dash-leading id as an option. The CLI wrote {"error": "the following arguments are required: grant_id"} to stderr and exited 2, and the test parsed the empty stdout before checking the exit code. CI evidence: the captured log has 4 'asyncio: Using selector' lines (create and list, two asyncio.run calls each; a full lifecycle logs 8 locally), so revoke never got past argument parsing. Rejected: an argparse change between 3.12.11 and 3.12.14 (argparse.py is identical), and sqlite or config races (3,000-run in-process lifecycle fuzz: 0 failures). The CLI is correct: it exits non-zero with a JSON error on stderr. Fix (test only): parametrize the approval lifecycle with plain, '-' and '--' leading ids, and add _run_cli_json, which asserts exit 0 with stderr and stdout in the message before parsing. Verification: RED with the old argv order gives 2 failed with 'CLI revoke-approval-grant exited 2; stderr=...required: grant_id'. GREEN: test_gateway_cli_package.py 99 passed, 30/30 loop. 50k random ids parsed after '--': 0 failures. Ruff clean; Bandit -ll: no Medium or High findings (test file only; no product files touched). No docs change needed. PR #3076.

Qodo follow-up on PR #3076 (commit 484d23af8e). Marked the approval and path grant lifecycle tests, both new or modified here, as @pytest.mark.unit, matching the per-test unit marks in sibling gateway tests. They run the CLI in-process against a temporary sqlite store. Wrapped the token_id parametrize decorators and the list helper calls. The project ruff and black limit is 120, which no line exceeded; the file keeps lines to 88 or fewer (max 92), and the changed lines now match. Verification: test_gateway_cli_package.py 99 passed and passed 30/30 in a loop; -m unit selects the 6 lifecycle cases; ruff clean; Bandit -ll: no Medium or High findings. Test-only change.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The CI flake was a 1-in-64 dash-leading opaque grant id passed before --config on a pre-874139d2be commit. argparse rejected it, and the test hid the stderr error behind a JSONDecodeError. dev already passes ids after '--'. The test now pins plain and dash-leading ids, so that case runs every time, and the new _run_cli_json helper reports exit code, stderr and stdout. Test-only change, 30/30 runs pass. PR #3076.
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
