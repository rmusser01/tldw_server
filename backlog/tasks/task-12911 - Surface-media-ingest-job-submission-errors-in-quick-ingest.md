---
id: TASK-12911
title: Surface media ingest job submission errors in quick ingest
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-07 23:17'
labels:
  - bug
  - frontend
  - media-ingest
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate reported PDF ingest failure where quick ingest reports a generic invalid/no-job message when async media ingest job submission returns no usable jobs. Preserve backend submit errors in the UI/client error path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Focused frontend and backend ingest-job tests pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the reported no-jobs submit path with a focused quick-ingest regression test.
2. Preserve backend submit errors when ingest job creation returns no usable jobs.
3. Reuse the same validation in the direct quick-ingest and extension background submit paths.
4. Verify focused frontend and backend ingest-job tests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Red test confirmed prior behavior: backend errors were present in the mocked submit response but the quick-ingest result showed only "Ingest job submission returned no job IDs."
- Verification: bunx vitest run src/services/__tests__/quick-ingest-batch.test.ts -t "surfaces backend ingest job submit errors when no jobs are created" passed after the fix.
- Verification: bunx vitest run src/services/__tests__/ingest-jobs-orchestrator.test.ts passed.
- Verification: bunx vitest run src/services/__tests__/quick-ingest-batch.test.ts passed.
- Verification: source .venv/bin/activate && python -m pytest tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_jobs_endpoint.py -q passed.
- Attempted package typecheck with NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit; it failed on pre-existing unrelated baseline errors across tests/background typing. Initial no-heap run hit Node OOM.
- git diff --check scoped to touched frontend files passed. Full git diff --check is blocked by an unrelated existing whitespace issue in Docs/Design/Tool-Calling.md.
- Bandit not run because no Python source was changed; frontend TypeScript and tests only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Investigation found no evidence that PDF validation was broadly rejecting PDFs. The broken user-visible behavior was that /api/v1/media/ingest/jobs can return HTTP 207 with jobs: [] and errors: [...] when upload staging/validation fails, but the frontend treated the 2xx response as success and replaced the useful backend error with the generic "Ingest job submission returned no job IDs." Added a shared submit-response validator that extracts backend errors, wired it into quick ingest URL/file submissions and the extension background ingest path, and added regression coverage.
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
