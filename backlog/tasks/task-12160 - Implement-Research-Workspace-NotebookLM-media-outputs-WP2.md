---
id: TASK-12160
title: Implement Research Workspace NotebookLM media outputs WP2
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-05 05:57
labels: []
dependencies: []
references:
- Docs/superpowers/specs/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-design.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py
- tldw_Server_API/app/core/Research_Workspace/capabilities.py
- tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py
- tldw_Server_API/app/api/v1/schemas/research_workspace_outputs.py
- tldw_Server_API/app/core/Research_Workspace/output_jobs.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track WP2 work for real NotebookLM-style media outputs in Research Workspace: backend jobs for narrated slideshow Video Overview and image-backend generated Infographic artifacts. Supersedes the earlier cheap text/storyboard WP2 idea; outputs must be real media artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Video Overview generates a real backend-rendered narrated slideshow artifact with durable preview/download.
- [ ] #2 Infographic generates a real image-backend PNG artifact with durable preview/download.
- [ ] #3 Research Workspace output jobs are submitted, drained by a registered worker, and expose status/progress/errors.
- [ ] #4 Final media and per-slide narration assets use durable output artifacts, not TTL file-artifact export URLs or generated-file ids alone.
- [ ] #5 UI capability gates, pending states, completed previews, and unavailable states are covered by tests.
- [ ] #6 Backend validation, worker, capability, and persistence paths are covered by tests.
- [ ] #7 Bandit and targeted frontend/backend verification are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implementation plan approved: Docs/superpowers/plans/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-plan.md. Plan review status: Approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 started: adding Research Workspace output job schemas/API skeleton only; worker/media generation/persistence remain for later plan tasks.
Task 2 complete: added output submit/status schemas, minimal Research Workspace output job helpers, POST/GET workspace output routes, and focused tests for validation, pending artifact/job creation, status projection, and status ownership isolation. Verification: red pytest failed on missing research_workspace_outputs module; green pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v passed 7 tests; git diff --check passed; Bandit on touched backend implementation files passed with 0 results.
Task 2 follow-up complete: hardened output job enqueue rollback/error mapping and fail-closed status domain scoping. Verification: red focused pytest failed on raw enqueue RuntimeError/500 and missing-domain acceptance; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v passed 11 tests; git diff --check passed; Bandit on touched backend implementation files passed with 0 results.
Task 2 status-lookup follow-up complete: mapped Jobs backend get_job failures to stable output_job_status_unavailable/503 before route handling. Verification: red focused pytest failed with 500 instead of 503; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v passed 12 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 3 started: adding Research Workspace output worker registration and fail-fast worker skeleton only; media generation/artifact persistence remain for later plan tasks.
Task 3 complete: added startup registration plus Research Workspace output Jobs worker skeleton with payload normalization, owner resolution, DB opening/closing, WorkerSDK runner/progress callback, and fail-fast processing placeholder. Verification: red focused pytest failed on missing worker/startup registration; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 2 tests; regression pytest including test_output_jobs_api.py passed 14 tests; git diff --check passed; Bandit on touched backend implementation files passed with 0 results.
Task 3 follow-up complete: updated service lifecycle/catalog contract tests for research_workspace_output_jobs_task, added worker delegation and WorkerSDK job_type coverage, and replaced duplicate worker helper code with shared Jobs worker_utils. Verification: red required suite reproduced 4 stale catalog/delegation failures; green required pytest suite passed 62 tests; git diff --check passed; Bandit on research_workspace_output_jobs_worker.py passed with 0 results.
Task 3 owner-isolation follow-up complete: made the Research Workspace output worker derive user_id from canonical job owner_user_id and reject mismatched payload user_id as non-retryable owner_user_id_mismatch; also awaited stop watcher cancellation. Verification: red worker test failed on missing mismatch rejection; green required pytest suite passed 50 tests; git diff --check passed; Bandit on research_workspace_output_jobs_worker.py passed with 0 results.
Task 4 complete: added bounded shared Research Workspace output source context assembly and durable byte persistence through output artifacts only. Verification: red focused pytest failed on missing build_research_workspace_output_source_context/persist_research_workspace_output_bytes helpers; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 6 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 spec-review follow-up complete: bounded source context now counts emitted headers/titles/separators, and output persistence filters caller metadata before required research_workspace fields are set last. Verification: red focused pytest failed on oversized-title context length and metadata override/path cases; green focused pytest test_output_jobs_worker.py -v passed 8 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 spec re-review follow-up complete: source context now matches source preview by treating empty media content as unavailable without document/transcript fallbacks, and output persistence recursively filters nested caller metadata keys/values that could leak path-like data while required research_workspace metadata remains immutable. Verification: red focused pytest failed on empty-content fallback and nested metadata path leakage; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 10 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 final spec-review follow-up complete: output metadata sanitization now rejects nested string values containing embedded POSIX or Windows absolute path-like substrings, not only values that are entirely paths. Verification: red single regression pytest failed on embedded /private/tmp and C:\\Users metadata strings leaking; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 10 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 metadata delimiter follow-up complete: output metadata sanitization now rejects embedded Unix absolute path-like tokens after non-word delimiters, covering rendered_from=/private/tmp/source.png and JSON-looking strings such as {"path":"/private/tmp/source.png"}, while preserving safe nested metadata. Verification: red single regression pytest failed on delimited/jsonish metadata paths leaking; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 10 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 source-context shape follow-up complete: source context bounding now budgets title and body separately so usable blocks preserve '# {title}\n\n{excerpt}' shape with a non-empty excerpt, and tiny limits skip the source rather than returning header-only context. Verification: red single regression pytest failed on missing blank-line separator/body for long title with max_chars=40; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 10 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results.
Task 4 persistence hardening follow-up complete: persist_research_workspace_output_bytes now fails closed on collections_db/user_id mismatch, removes written bytes if output artifact row creation fails, and recursively drops caller metadata keys/values that look file/path-like including relative local paths. Verification: red focused pytest failed on user mismatch acceptance, row-failure file leak, and relative metadata path leakage; green focused pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 12 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 results. Real CollectionsDatabase integration-style test skipped to avoid broader DB backend/schema setup churn in this focused follow-up.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
