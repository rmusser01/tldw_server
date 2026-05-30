---
id: TASK-568
title: Implement MCP Unified Stage 4J preset details CLI
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 21:45'
labels:
  - mcp-unified
  - stage-4
  - cli
  - profiles
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow standalone gateway CLI command for deterministic full built-in preset inspection so front-ends can inspect role/profile policy documents before applying a profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 mcp-unified-gateway exposes a command to show one built-in preset as deterministic JSON.
- [x] #2 Unknown preset ids return machine-readable JSON errors without tracebacks.
- [x] #3 The command output includes the preset id/version and full profile policy data needed by front-ends.
- [x] #4 Focused CLI tests, extraction contract tests, lint, diff check, and Bandit touched-scope validation are run.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-30-mcp-unified-stage4j-preset-details-cli-plan.md

Review follow-up:
- Rebased onto origin/dev at 8b5fca0e68.
- Renumbered this task from TASK-567 to TASK-568 after the rebase because origin/dev now contains a different TASK-567.
- Verified Qodo's version/date drift finding against current code and fixed it by deriving PRESET_VERSION and PRESET_CREATED_AT from PRESET_RELEASE_DATE.
- Verified CodeRabbit's timestamp string-literal test finding against current code and fixed it by parsing CLI timestamps into aware datetimes before asserting equality. Gemini was quota-limited.

Verification:
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -q (baseline: 6 passed; initial show-preset red path: 2 failed before implementation; after implementation: 8 passed)
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py -q (timestamp determinism red path: 2 failed before fixed template timestamps; after implementation: 19 passed)
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py::test_builtin_preset_template_timestamps_are_version_stable -q (red: failed before PRESET_RELEASE_DATE; green: 1 passed)
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py::test_gateway_cli_show_preset_reports_full_builtin_profile -q (passes with timestamp parsing)
- source .venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_mapping.py -q (66 passed)
- source .venv/bin/activate && python -m ruff check mcp_unified/gateway/cli.py mcp_unified/profiles/presets.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py tldw_Server_API/app/core/MCP_unified/tests/test_profile_presets.py (passed)
- source .venv/bin/activate && python -m bandit -r mcp_unified/gateway mcp_unified/profiles -f json -o /tmp/bandit_mcp_stage4j_preset_details.json (0 findings)
- git diff --check (passed)

Skipped/blocked: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Stage 4J standalone gateway CLI preset detail slice and PR review follow-up. The CLI exposes mcp-unified-gateway show-preset <preset_id>, emits deterministic full preset/profile policy JSON, returns machine-readable unknown-preset errors, and derives preset version and template timestamps from one PRESET_RELEASE_DATE source so provenance cannot drift silently.
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
