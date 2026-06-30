---
id: TASK-2393
title: Deepen ACP live-agent caveat certification for Goose Hermes and OpenCode
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-20 18:41'
labels:
  - ACP
  - certification
  - Goose
  - Hermes
  - OpenCode
  - github-2402
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2402'
  - 'https://github.com/rmusser01/tldw_server/issues/2398'
  - 'https://github.com/rmusser01/tldw_server/pull/2416'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track GitHub issue #2402: verify or explicitly preserve deeper live-agent caveats for Goose, Hermes, and OpenCode across non-empty MCP server injection, artifact-producing workflows, reviewer-loop behavior, and failure diagnostic payloads. Keep evidence separated by agent and host/runtime and update ACP compatibility/readiness docs plus parent #2398 with final status.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each deeper scenario has pass evidence, fail evidence, or an explicit retained caveat per agent.
- [x] #2 Compatibility matrix and setup surfaces do not overclaim beyond evidence.
- [x] #3 Parent #2398 is updated with the evidence link and final status.
- [x] #4 Verification results and any accepted skips are recorded in Backlog and docs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

- Evidence captured in `Docs/Development/ACP_Live_Agent_Caveat_Verification_2026_06_20.md`.
- Goose, Hermes, and OpenCode each passed `workspace-live-e2e` with `workspace_env=pass`, `mcp_injection=pass`, `mcp_server_count=1`, structured completion, redacted support views, diagnostics endpoint, cancel, and close.
- All three retained `artifacts=skip` (`artifacts_total=0`), `sandbox=skip`, `review_loop=skip`, and no failure diagnostic payload evidence (`diagnostics_total=0` on success path).
- Docs updated: `ACP_Compatibility_Matrix.md` upgrades only workspace/MCP checks and keeps `supported_with_caveats`; `ACP_Production_Readiness.md` links #2402 and the evidence note.
- Verification: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q` -> 49 passed, 6 warnings; `git diff --check` -> passed.
- Bandit not run: documentation/Backlog-only branch, no Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added `Docs/Development/ACP_Live_Agent_Caveat_Verification_2026_06_20.md` with per-agent live `workspace-live-e2e` results for Goose, Hermes, and OpenCode.
- Updated `ACP_Compatibility_Matrix.md` to mark only `workspace_env=pass` and `mcp_injection=pass` for Goose/Hermes/OpenCode while retaining `supported_with_caveats`, `artifacts=skip`, `sandbox=skip`, `review_loop=skip`, and `diagnostics=limited`.
- Updated `ACP_Production_Readiness.md` to link #2402 and the evidence note.
- Opened PR #2416: https://github.com/rmusser01/tldw_server/pull/2416
- Updated #2402: https://github.com/rmusser01/tldw_server/issues/2402#issuecomment-4759612261
- Updated parent #2398: https://github.com/rmusser01/tldw_server/issues/2398#issuecomment-4759612213
- Verification: three live workspace E2E runs passed; `python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q` -> 49 passed, 6 warnings; `git diff --check` -> passed. Bandit skipped because no Python code changed.
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
