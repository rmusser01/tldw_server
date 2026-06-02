---
id: TASK-592
title: Implement standalone MCP gateway config import export snapshots
status: To Do
assignee: []
created_date: ''
updated_date: '2026-06-02 02:20'
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
- [ ] #1 Exported snapshots include schema version, profiles, default assignment, external servers, and credential grants.
- [ ] #2 Snapshot output contains no plaintext secrets and rejects secret-looking metadata/provenance.
- [ ] #3 Import dry-run validates references and reports planned mutations without writing.
- [ ] #4 Import applies in safe order: profiles, default assignment, external servers, credential grants.
- [ ] #5 Import defaults to upsert semantics and does not delete missing local records.
- [ ] #6 A snapshot exported from one SQLite store can be imported into a fresh SQLite store and exported again with equivalent semantic content.
- [ ] #7 Snapshot validation rejects secret-looking external server command args, URL userinfo, and sensitive URL query keys.
- [ ] #8 Import validates the full snapshot before the first write and reports partial write failures explicitly for non-transactional stores.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-02-mcp-gateway-config-snapshots.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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
