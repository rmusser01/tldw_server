---
id: TASK-15417
title: 'Deferred minors of #3391 and #3392: vendor-host labels, stream and PostgreSQL
  tests, task-id check hardening'
status: In Progress
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
- [ ] #1 A VENDOR_HOSTS pattern matches only a host whose labels are all non-empty, one label per '*', after one trailing dot is stripped, with tests
- [ ] #2 Tests cover a nested call in an awaitable astream and in a server-key stream checked at its start, and the call-layer test asserts the own-key row's source; the three long test lines are wrapped and lost is annotated
- [ ] #3 A PostgreSQL test drives the insert and sum retries on a usage table without credential_source, through the repo fixture
- [ ] #4 check_backlog_task_ids.py holds each file under its file-name id and its id: id, stops an id at a trailing dot, escapes annotations, and annotates a file it cannot read, with tests (incl. the file-name fallback and case folding); the wiring test pins no paths filter on pull_request
- [ ] #5 task renumber appends each --append-notes as its own block, as task edit does, and tests cover an invalid and a subtask new id
- [ ] #6 AGENTS.md and the backlog-py README say new ids continue above TASK-15281
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-11-02-deferred-minors-3391-3392.md. Owner answer: Decision 1 (Change summary) waived: on 2026-11-02 the owner answered 'Yes, waiver covers it' for this PR.
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
