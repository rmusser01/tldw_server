---
id: TASK-13927
title: 'ChaCha on PostgreSQL: a failed statement outside a managed transaction leaves
  the operation''s connection aborted'
status: To Do
labels:
- chacha
- postgres
- bug
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-13896's planning. PostgreSQLBackend.execute rolls back a failed statement only on a connection it took itself ('if not external_conn: conn.rollback()', postgresql_backend.py near :1141). A CharactersRAGDB handle passes its own connection (the operation's, or the thread-local one), so a failed statement outside BackendManagedTransaction leaves that connection in an aborted transaction and every later statement on it fails with 'current transaction is aborted' until the operation ends or the thread closes it. Measured on origin/dev e06eeec68c (TASK-13896 plan F4): inside one voice command operation, the command refresh failed (TASK-13926), and the same command's analytics insert then failed ('Failed to record voice analytics: Query execution failed'; voice_command_events count 0). An HTTP request has the same shape: one failed read poisons the rest of its writes. Direction: roll back after a failure when no managed transaction is open on the state (tx_depth == 0), taking care of execute_query(commit=False) writes a caller means to commit later.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A failed statement outside a managed transaction inside a chacha_operation does not make the next statement in the operation fail (PostgreSQL test)
- [ ] #2 A failure inside a caller's managed transaction still leaves the rollback to that transaction
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
