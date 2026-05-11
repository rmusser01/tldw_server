---
id: TASK-265
title: Implement OpenWebUI attachment hydration API slice
status: Done
assignee: []
created_date: '2026-05-11 16:30'
updated_date: '2026-05-11 16:48'
labels:
  - chatbooks
  - openwebui
  - implementation
dependencies:
  - TASK-264
references:
  - >-
    Docs/superpowers/plans/2026-05-11-openwebui-attachment-hydration-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 5 of the OpenWebUI attachment hydration implementation plan: expose preview, job creation, and job status API contracts for attachment hydration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pydantic request/response schemas cover preview/job requests and redact raw source paths
- [x] #2 Preview endpoint authorizes access and returns hydration summary/counts
- [x] #3 Job creation endpoint enqueues a chatbooks openwebui_attachment_hydration Jobs row with scoped payload
- [x] #4 Job status endpoint returns only owned/admin-visible hydration jobs
- [x] #5 Focused pytest, diff checks, and Bandit for touched backend code are recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Stage 5 API slice: added OpenWebUI hydration request/response schemas, preview endpoint, job create/status endpoints, chatbooks Jobs helper, safe preview normalization, recursive path redaction for job result/error values, and focused API tests. Multi-user non-admin callers are denied for server-local hydration; single-user and admin principals are allowed. Worker-side root/authorization revalidation remains in Stage 6. Verification: focused API tests 7 passed; broader OpenWebUI hydration/import suite 47 passed; chatbooks path traversal suite 3 passed; git diff --check clean; Bandit report /tmp/bandit_openwebui_hydration_api.json has 0 findings and 0 errors. Known non-gating check: combining test_chatbooks_api_path_guard.py with path traversal timed out in full app.main TestClient teardown after unrelated lifespan workers started.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 5 exposes the OpenWebUI attachment hydration API contract: schemas for preview/jobs, an admin/single-user gated preview route, core Jobs enqueue/status routes for openwebui_attachment_hydration, raw-path-free preview items, and recursive redaction for job result/error strings. Verification recorded: API tests 7 passed, OpenWebUI hydration/import suite 47 passed, chatbooks path traversal suite 3 passed, diff check clean, Bandit 0 findings. The remaining worker-side execution and revalidation is intentionally deferred to Stage 6.
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
