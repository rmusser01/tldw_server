---
id: TASK-15417
title: 'Deferred minors of #3391 and #3392: vendor-host labels, stream and PostgreSQL
  tests, task-id check hardening'
status: Done
labels:
- quotas
- backend
- byok
- backlog
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request 2026-11-02: 'address the deferred minor findings in a follow up pr'. The minors deferred by the reviews of PR #3391 (TASK-13959, BYOK quota exemption minors), PR #3392 (TASK-13960, backlog task-id renumbering) and TASK-13955's leftovers. (B1) A '*' in a VENDOR_HOSTS pattern matches an empty DNS label, and a trailing-dot FQDN counts. (B3) Three long test lines and an unannotated parameter. (B4) No test of an awaitable own-key astream, or of a checked server-key stream's nested call. (B6) No PostgreSQL test of the missing credential_source retry, and the call-layer test asserts no row. (R1) The CI id check reads id: lines while backlog-py reads file names. (R2) No renumber tests for an invalid or a subtask id. (R4) The CI check gives no annotation for an unreadable file, does not escape annotations, and lets a trailing dot into an id. (R7) task renumber joins several --append-notes into one block. (R8) The wiring test does not pin the PR trigger's paths filter. (R9) The docs do not say new ids continue above TASK-15281. No change: B2 (the gateway checks the raw base_url first; moving the check below the request build puts every build error before the 402 and broke two TASK-13955 tests), B5 (environment), B6b (one AuthNZ database per process), B6d (none left from TASK-13955), R3 (one write), R5 (one-off generator), R6 (moot), R10 (#3323's). Plan: Docs/superpowers/plans/2026-11-02-deferred-minors-3391-3392.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A VENDOR_HOSTS pattern matches only a host whose labels are all non-empty, one label per '*', after one trailing dot is stripped, with tests
- [x] #2 Tests cover a nested call in an awaitable astream and in a server-key stream checked at its start, and the call-layer test asserts the own-key row's source; the three long test lines are wrapped and lost is annotated
- [x] #3 A PostgreSQL test drives the insert and sum retries on a usage table without credential_source, through the repo fixture
- [x] #4 check_backlog_task_ids.py holds each file under its file-name id and its id: id, stops an id at a trailing dot, escapes annotations, and annotates a file it cannot read, with tests (incl. the file-name fallback and case folding); the wiring test pins no paths filter on pull_request
- [x] #5 task renumber appends each --append-notes as its own block, as task edit does, and tests cover an invalid and a subtask new id
- [x] #6 AGENTS.md and the backlog-py README say new ids continue above TASK-15281
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-02-deferred-minors-3391-3392.md. Owner answer: Decision 1 (Change summary) waived: on 2026-11-02 the owner answered 'Yes, waiver covers it' for this PR.
RED/GREEN (record the real counts): Task 1 3 failed, 33 passed -> 36 passed; PostgreSQL file 3 -> 4 passed (mutations: insert retry -> raise 'assert None == 3', sum retry -> raise UndefinedColumnError); Task 1 mutation probes: row source None 2 failed, _resolve unwrapped 2 failed, sync stream gate without user_id 1 failed, async stream gate without user_id 2 failed; Task 2 5 failed, 51 passed -> 56 passed (mutations: no case fold, raw new id, undotted id regex, paths on the trigger: 1 failed each). Focused set 253 -> 269 passed. Wider set: SQLite 1 failed, 9965 passed, 35 skipped, 1 error -> 1 failed, 9981 passed, 35 skipped, 1 error (both pre-existing: test_document_factory_starts_without_default_executor_and_drains_cancel; teardown of test_extractor_never_opens_files); serial 88 -> 88 passed; PostgreSQL 6 -> 7 passed. tests/Docs + lint + CI + backlog-py 1604 passed, 5 skipped. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth ratchet exact; OpenAPI unchanged.
Decisions: B2 no change (moving the gateway check below the request build cannot fail open, but it puts every build error before the 402; prototyped, it turned test_messages_native_usage_recording's two 402 cases into 502; today's order fails closed and no public endpoint passes a base_url); R1 indexes each file under its file-name id and its id: id (0 mismatches of 4435 files); no change for B5 (environment), B6b (one AuthNZ database per process; same pattern as the SQLite corruption warning), B6d (the two E30x hits in test_messages_endpoint.py predate TASK-13955), R3 (one write), R5 (one-off generator, already run), R6 (moot: #3392 merged), R10 (#3323's).
Docs: AGENTS.md and tools/backlog-py/README.md (ids continue above TASK-15281); nothing mirrored; no ADR change (ADR-083's 'one DNS label' already describes B1). Bandit: No issues identified (Medium 0, High 0).
This PR also files TASK-15418 (owner request 2026-11-02: the ACP runner writes a read-only Go module cache into the repo tree).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settles the minors deferred by the reviews of #3391 and #3392 and TASK-13955's leftovers. A VENDOR_HOSTS pattern matches only non-empty labels, one per '*', after one trailing dot is stripped. New tests: an awaitable astream and a checked server-key stream are not re-checked mid-stream; the call-layer test asserts the own-key row; a PostgreSQL table without credential_source keeps and counts the row. check_backlog_task_ids.py holds each file under its file-name and id: ids, stops an id at a trailing dot, escapes annotations and annotates an unreadable file; the wiring test pins the PR trigger. task renumber appends each --append-notes like task edit; tests cover an invalid and a subtask id. AGENTS.md and the backlog-py README say new ids continue above TASK-15281. No change: B2, B5, B6b, B6d, R3, R5, R6, R10, each with its reason in the notes.
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
