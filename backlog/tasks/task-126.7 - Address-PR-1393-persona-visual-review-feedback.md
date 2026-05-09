---
id: TASK-126.7
title: Address PR 1393 persona visual review feedback
status: In Progress
assignee: []
created_date: '2026-05-09 03:46'
updated_date: '2026-05-09 04:00'
labels:
  - persona
  - visual-packs
  - pr-review
  - api
  - jobs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1393'
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review and address all actionable GitHub review comments on PR #1393 for persona visual packs. Scope includes Gemini and Qodo review threads: portability worker startup registration, async worker blocking calls, authored trigger duplicate handling, export/import-preview cancel/delete lifecycle routes, PersonaVisualService dependency injection, bounded upload reads, generation job idempotency, ZIP directory entries, and a technical disposition for the archive-ingestion-pipeline comment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All open PR #1393 review threads are evaluated against codebase reality and either fixed or answered with technical reasoning.
- [x] #2 Portability export/import-preview job lifecycle supports scoped cancel/delete where appropriate, with tests.
- [x] #3 Persona visual upload reads enforce caps before buffering unbounded content, with tests.
- [x] #4 Persona visual generation idempotency distinguishes distinct prompt/backend requests, with tests.
- [x] #5 Persona visual archive validation tolerates explicit ZIP directory entries while preserving unsafe member rejection, with tests.
- [x] #6 Persona visual worker async handlers offload blocking DB/file operations and portability worker startup is registered, with tests.
- [x] #7 Authored trigger patch merges prevent duplicate trigger IDs, with tests.
- [ ] #8 Focused backend tests and Bandit run successfully, and PR branch is pushed with review-fix commit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused red tests for each accepted review finding: portability worker startup registration, worker blocking-call offload hooks, duplicate authored trigger handling, export/import-preview cancel/delete endpoints, bounded upload reads, generation job idempotency digesting prompt/backend, and ZIP directory entries.
2. Implement small backend fixes using existing local patterns: mirror VN asset portability cancel/delete routes, inject PersonaVisualService through FastAPI dependency helper, add bounded upload read helper, update generation idempotency keys, skip ZIP directory entries during archive validation, offload generation worker DB/file operations, and register a portability worker behind an explicit env flag.
3. Evaluate the archive-ingestion-pipeline comment against existing VN portability and persona archive behavior. Do not route persona visual archives through media ingestion; instead ensure the archive upload path has bounded size/type staging plus import-preview validation, and document/reply with that technical disposition.
4. Run focused pytest for touched persona visual service/API/jobs/portability/startup tests, run Bandit on touched backend scope, run git diff checks, update Backlog, commit, push, and reply/resolve PR threads where possible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented review fixes: export cancel endpoint; import-preview cancel and delete endpoints; PersonaVisualService FastAPI dependency helper; chunked persona visual asset upload cap enforcement; generation idempotency digest including prompt/backend/target state; ZIP directory member tolerance while preserving unsafe member checks; generation worker DB/persistence offload; portability worker runner and optional startup registration; authored trigger ID replacement on candidate patch merge.

Verification passed before PR replies: focused pytest for persona visual jobs, portability, service, API, and startup worker tests reported 45 passed; py_compile passed for touched production files; Bandit reported zero findings on touched production files and touched tests with B101 skipped; git diff --check passed.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
