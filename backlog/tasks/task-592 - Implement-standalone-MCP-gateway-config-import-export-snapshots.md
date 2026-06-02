---
id: TASK-592
title: Implement standalone MCP gateway config import export snapshots
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 03:12'
labels:
  - mcp-unified
  - standalone-gateway
  - config
dependencies:
  - TASK-591
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add versioned import/export snapshot workflows for standalone gateway profiles, default assignment, external servers, and credential grant metadata, including dry-run validation and secret-safe output.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Exported snapshots include schema version, profiles, default assignment, external servers, and credential grants.
- [x] #2 Snapshot output contains no plaintext secrets and rejects secret-looking metadata/provenance.
- [x] #3 Import dry-run validates references and reports planned mutations without writing.
- [x] #4 Import applies in safe order: profiles, default assignment, external servers, credential grants.
- [x] #5 Import defaults to upsert semantics and does not delete missing local records.
- [x] #6 A snapshot exported from one SQLite store can be imported into a fresh SQLite store and exported again with equivalent semantic content.
- [x] #7 Snapshot validation rejects secret-looking external server command args, URL userinfo, and sensitive URL query keys.
- [x] #8 Import validates the full snapshot before the first write and reports partial write failures explicitly for non-transactional stores.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-config-snapshots.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented package-owned gateway config snapshots in mcp_unified/gateway/snapshots.py with versioned Pydantic models, deterministic ordering, secret-key validation, external-server inline secret validation, validate-first import reference checks, dry-run mutation plans, non-destructive upsert import order, best-effort audit events, and explicit partial-write failure reporting. Added config builder wiring and CLI export-config/import-config workflows. Touched files: mcp_unified/gateway/snapshots.py, mcp_unified/gateway/config.py, mcp_unified/gateway/cli.py, tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py, tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py, Docs/superpowers/plans/2026-06-02-mcp-gateway-config-snapshots.md. Verification: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_config_snapshots.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_cli_package.py -v -> 100 passed, 5 warnings. Bandit: source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r mcp_unified/gateway/snapshots.py mcp_unified/gateway/config.py mcp_unified/gateway/cli.py -f json -o /tmp/bandit_mcp_gateway_config_snapshots.json -> 0 findings. git diff --check -> clean. Known skips/blockers: none.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added standalone gateway config import/export snapshots for profiles, the gateway default assignment, external servers, and credential grants. Snapshot export is deterministic and secret-safe; import supports dry-run planning, validates references before writes, applies non-destructive upserts in dependency order, reports partial write failures by action id, and is exposed through export-config/import-config CLI commands. Focused snapshot and CLI tests pass, Bandit has zero findings, and whitespace validation is clean.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
