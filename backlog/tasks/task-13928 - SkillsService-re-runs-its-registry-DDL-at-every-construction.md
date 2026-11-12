---
id: TASK-13928
title: SkillsService re-runs its registry DDL at every construction
status: To Do
labels:
- skills
- postgres
- performance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13896's planning and its review. SkillsService.__init__ calls CharactersRAGDB._ensure_skill_registry_table (ChaChaNotes_DB.py:28413), which on PostgreSQL runs 4 CREATE TABLE/INDEX IF NOT EXISTS statements through backend.execute, each with its own pool checkout, at every construction: 4 checkouts measured per skill tool call (TASK-13896 plan F3). Production constructs the service per request or call at: Skills/context_integration.py get_skills_context_text_async (:59) and add_skill_tool_to_tools_list_async (:242), which /chat/completions calls on the event loop for every chat request with skills (chat.py:3634, :5011), the hottest path; the sync helpers at context_integration.py :43 and :210; api/v1/endpoints/skills.py:136; Chat/command_router.py:688 and :759; MCP skills_module.py:559. (handle_skill_tool_call builds it off the loop since TASK-13896.) Remember per handle and backend target that the table is ready, as _bootstrapped_backend_targets does for the schema; that removes the statements from every site at once. Lock behaviour of the repeated IF NOT EXISTS DDL under concurrent writes is not measured.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A second SkillsService for the same handle runs no table statement (test counts backend.execute calls)
- [ ] #2 The first construction still creates the table on SQLite and PostgreSQL
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
