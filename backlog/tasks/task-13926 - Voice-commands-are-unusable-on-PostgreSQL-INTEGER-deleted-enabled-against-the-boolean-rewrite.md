---
id: TASK-13926
title: 'Voice commands are unusable on PostgreSQL: INTEGER deleted/enabled against
  the boolean rewrite'
status: To Do
labels:
- voice
- postgres
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13896's planning and its review. voice_commands.deleted and .enabled are INTEGER on PostgreSQL (information_schema on a fresh ChaCha schema: ('deleted', 'integer'), ('enabled', 'integer')), but the SQLite-to-PostgreSQL transform rewrites '= 0' and '= 1' to booleans. Measured on origin/dev e06eeec68c, raw psycopg errors: get_user_voice_commands, get_voice_command (VoiceAssistant/db_helpers.py:77), get_voice_command_counts (:925) and delete_voice_command raise UndefinedFunction: operator does not exist: integer = boolean; save_voice_command -> CharactersRAGDB.upsert_voice_command (ChaChaNotes_DB.py:9586) raises DatatypeMismatch: column "deleted" is of type integer but expression is of type boolean (its ON CONFLICT ... SET deleted = 0). So on a PostgreSQL deployment the whole user voice-command feature fails: create, list, get, count and delete, over HTTP and the voice-assistant WebSocket. On dev, VoiceCommandRegistry.refresh_user_commands' CharactersRAGDBError also escaped websocket_voice_assistant at every session start and killed the socket; since TASK-13896 the session degrades with 'Voice assistant DB init failed', and the router logs 'Failed to refresh voice commands from DB' per command. No data loss, no security impact. The fix needs a choice: a PostgreSQL migration of the two columns to BOOLEAN, or per-query casts; check the other voice_* tables.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A voice command can be saved, listed, read, counted and deleted on PostgreSQL, pinned by a PostgreSQL test
- [ ] #2 A voice-assistant session start on PostgreSQL with the real registry loads the user's commands
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
