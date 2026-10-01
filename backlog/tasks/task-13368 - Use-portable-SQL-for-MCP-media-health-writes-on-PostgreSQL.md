---
id: TASK-13368
title: Use portable SQL for MCP media health writes on PostgreSQL
status: Done
assignee: []
created_date: '2026-09-25 19:49'
updated_date: '2026-09-25 19:58'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Full PostgreSQL startup logs MCP media module health failure because check_health issues SQLite-only INSERT OR REPLACE against the PostgreSQL content backend. Change the health write to portable conflict handling and keep the health check meaningful on both backends.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 MCP media database_writable health check passes on PostgreSQL without SQLite-only INSERT OR REPLACE
- [x] #2 SQLite behavior and cleanup remain intact
- [x] #3 Full synthetic PostgreSQL startup no longer logs the MCP media health SQL failure
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Diagnostic full-app PostgreSQL probe recorded INSERT OR REPLACE INTO _mcp_healthcheck during optional MCP media startup; PostgreSQL backend rejects this SQLite-only syntax. A targeted check_health regression will reject that statement and require database_writable to stay true with portable SQL.

Portable ON CONFLICT health upsert passed focused regression and final PostgreSQL and SQLite live probes; PostgreSQL log had no MCP media health SQL failure. Bandit 0 findings; fatal Ruff clean. Skip: no production deployment tested.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Portable ON CONFLICT health upsert passed focused regression and final PostgreSQL and SQLite live probes; PostgreSQL log had no MCP media health SQL failure. Bandit 0 findings; fatal Ruff clean. Skip: no production deployment tested.
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
