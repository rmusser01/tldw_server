---
id: TASK-566
title: Implement MCP Unified Stage 4I gateway CLI
status: Done
labels:
- mcp-unified
- stage-4
- gateway
- cli
priority: medium
modified_files:
- mcp_unified/gateway/cli.py
- pyproject.toml
- tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4i-gateway-cli-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a narrow package-owned CLI for standalone gateway configuration workflows: validate JSON/TOML gateway profile bootstrap config files and list built-in profile presets without starting transports or managing external MCP service lifecycle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-30-mcp-unified-stage4i-gateway-cli-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 4I gateway CLI with deterministic JSON commands for config validation and built-in preset listing. Review pass rebased PR #2163 onto origin/dev and addressed all still-valid comments: typed pytest capsys fixtures, added JSON handling for argparse failures, converted loader failures at the CLI boundary to JSON stderr without tracebacks, and replaced brittle script-entry substring testing with semantic TOML parsing. Verified focused gateway, extraction/http compatibility, ruff, Bandit, and diff checks.
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
