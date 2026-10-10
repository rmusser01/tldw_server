---
id: TASK-13513
title: Eliminate per-request fixed costs in auth, ChaCha deps, MCP tools/list, config
  loading
status: To Do
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1. Plan: Docs/Plans/2026-10-06-perf-batch-1-request-fixed-costs-implementation-plan.md. PBKDF2 verify cache + to_thread (api_key_manager.py:526); ChaCha dep default-character ensure-once + TTL health + log demote (API_Deps/ChaCha_Notes_DB_Deps.py:802); MCP tools/list batched RBAC resolution (MCP_unified/protocol.py:2004); cache load_and_log_configs (core/config.py:3743).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

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
