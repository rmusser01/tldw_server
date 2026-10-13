---
id: TASK-13585
title: Record LLM usage for MCP tool calls and the in-process claims rebuild
status: To Do
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two roots still make chat LLM calls with no known user after the LLM usage accounting PR 1b. MCP: tool calls over /ws and HTTP reach perform_chat_api_call_async (core/MCP_unified quizzes module), and MCP authentication sets no content scope; the tool-execution entry has the user in its RequestContext. Claims rebuild: core/Claims_Extraction/claims_rebuild_service.py runs daemon threads whose ClaimsRebuildTask carries only media_id and db_path, so there is no owner to attribute to. Parent: TASK-13437.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An MCP tool call that makes an LLM call records one row for the calling user
- [ ] #2 A claims rebuild task carries its owner and its LLM calls are recorded for that user
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
