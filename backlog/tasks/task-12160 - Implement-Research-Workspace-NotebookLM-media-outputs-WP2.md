---
id: TASK-12160
title: Implement Research Workspace NotebookLM media outputs WP2
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-05 09:45
labels: []
dependencies: []
references:
- Docs/superpowers/specs/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-design.md
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

Task 5 complete: implemented infographic output processing through ImageAdapter.normalize/validate/export, durable PNG output persistence, optimistic-lock workspace artifact completion updates, and sanitized failed-artifact updates. Verification: red focused pytest failed on research_workspace_output_processing_not_implemented; green focused infographic pytest passed 2 tests; full pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 14 tests; Bandit on output_jobs.py passed with 0 findings.

Task 5 review fixes: preserved original worker errors when failed-artifact marking fails, mapped FileArtifactsError public codes/retryability through the Research Workspace job error contract, rejected malformed job ids before persistence, and rejected non-PNG image adapter exports before writing output artifacts. Verification: red regression pytest failed 4 focused tests; green focused regression pytest passed 4 tests; full pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 18 tests; git diff --check passed; Bandit output_jobs.py passed with 0 findings (/tmp/bandit_task12160_task5_review_fix.json).

Task 5 final hardening: required infographic exports to have PNG magic bytes in addition to acceptable MIME, covering adapters that omit content_type before durable persistence. Verification: red pytest failed test_infographic_worker_rejects_non_png_bytes_when_image_export_omits_content_type; green focused pytest passed 2 tests; full pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 19 tests; git diff --check passed; Bandit output_jobs.py passed with 0 findings (/tmp/bandit_task12160_task5_png_signature.json).

Task 6 complete: routed video_overview jobs to real backend narrated slideshow generation using SlidesGenerator, SlidesDatabase, per-slide TTS MP3 output artifacts, direct render_presentation_video MP4 rendering, and final Collections output artifact persistence from the renderer storage_path. Added complete/failed workspace artifact updates with durable export refs and sanitized producer metadata. Verification: red focused pytest failed with research_workspace_video_overview_not_implemented; green focused video pytest passed; focused render-failure pytest passed; full pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_worker.py -v passed 21 tests; git diff --check passed; Bandit output_jobs.py passed with 0 findings (/tmp/bandit_task12160_task6_video_overview.json).
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
Task 6 quality-review fixes starting: adding red regression coverage for provider-aware TTS defaults, best-effort progress updates, slide metadata sanitization before render, TTS failure error mapping/retryability, and cleanup of durable output rows/files when finalization fails.
Task 6 quality-review fixes complete: video_overview now resolves TTS provider/model/voice via the shared TTS default resolver, treats progress persistence as best effort, strips generated slide metadata before render, maps slide/TTS failures to bounded public error codes with retryability, and cleans durable narration/final output rows plus files when rendering/final workspace updates fail. Verification: red focused pytest failed 5 review-regression tests for the expected issues; green focused pytest passed the same 5 tests; full worker suite passed 26 tests; combined output API/startup/worker suite passed 39 tests; git diff --check passed; Bandit on output_jobs.py passed with 0 findings (/tmp/bandit_task12160_task6_review_fix.json).
Task 7 starting: frontend contract slice for media-output artifact types, capability mapping, and workspace output submit/status API client methods. Scope excludes Studio submission/polling UI, which remains Task 8.
Task 7 complete: added frontend media artifact types/output config, Research Workspace media capability IDs/mapping, and workspace output submit/status API client types and methods. Also corrected stale path-encoding tests to use valid space-containing IDs and added explicit slash-rejection coverage for the current path-safety contract. Verification: red focused Vitest failed for missing media capability mapping and missing submitWorkspaceOutput method; package dependencies were restored with bun install from apps because the worktree UI node_modules links were incomplete; green focused Vitest passed 2 files / 17 tests; git diff --check passed. Bandit not applicable for this frontend-only slice.
Task 8 started: implementing Studio pane submission, polling, media preview, and download wiring for video overviews and infographics. Plan adjustment: the existing Stage 3 test mock intentionally replaces `OUTPUT_TYPES` with `[]`, so media button assertions will target labels and behavior from the Studio pane config rather than shared description strings.
Task 8 complete: Studio pane now exposes Video Overview and Infographic output actions, capability-gates media generation, submits/polls backend workspace output jobs for media outputs, maps completed workspace artifacts into local generated artifacts, previews MP4/PNG export refs, and downloads media outputs via durable output artifact refs. Verification: red Stage 3 Vitest failed 4 intended media-output tests; green focused Vitest passed Stage 3 (24 tests); planned frontend verification passed 3 files / 41 tests; git diff --check passed. UI no-emit type check required NODE_OPTIONS=--max-old-space-size=8192 and now reports only pre-existing unrelated test type errors in ChatGreetingPicker, background-session-store, TldwChat.abort, and character-export.ssrf; no touched-file errors remain. Bandit not applicable for this frontend-only slice.
WP2 verification complete. Backend focused pytest passed 68 tests across capability derivation, capability endpoint, output jobs API, output jobs worker, and startup registration. Frontend focused Vitest passed 3 files / 41 tests across media capabilities, workspace output API client, and Studio pane behavior. Bandit passed with 0 findings in the touched backend scope; report: /tmp/bandit_research_workspace_wp2.json. Optional local smoke was not run because local image generation, TTS, and ffmpeg readiness were not verified in this session; only the local llama.cpp API availability was known.
Post-review fix pass starting: addressing Important reviewer findings for infographic finalization cleanup, authenticated media previews, Slides visual style propagation, and source lineage normalization with focused red/green regression tests.
Post-review fixes complete: infographic final workspace-update failures now clean durable output rows/files and mark the workspace artifact failed; video_overview resolves builtin Slides visual styles, passes the snapshot into SlidesGenerator, and persists style fields on the presentation; Studio media previews now fetch authenticated output blobs through tldwClient.downloadOutput and use object URLs; frontend workspace-artifact lineage normalization now handles backend source_lineage records with usable/selected source ids. Verification: red focused backend pytest failed the new cleanup/style regressions, then passed after fixes; red focused Vitest failed the authenticated preview/lineage regressions, then passed after fixes. Full worker suite passed 28 tests; focused backend Research Workspace suite passed 70 tests; focused frontend Vitest passed 3 files / 48 tests; TypeScript no-emit still reports only the pre-existing unrelated baseline test errors; git diff --check passed; Bandit output_jobs.py passed with 0 findings (/tmp/bandit_research_workspace_wp2_review_fix.json).
Post-rebase PR review fixes complete: rebased onto latest origin/dev, preserved original enqueue failures when rollback cleanup also fails, centralized ResearchWorkspaceOutputJobError in core exceptions, added module/class/function docstrings for the new output job/schema modules, marked output job pytest suites, moved Slides generation to asyncio.to_thread, and switched TTS byte accumulation to bytearray. Verification: the two new red regressions failed before implementation and passed after fixes; focused backend Research Workspace suite passed 72 tests; git diff --check passed; Bandit touched backend scope passed with 0 findings (/tmp/bandit_research_workspace_wp2_rebase_pr_comments.json).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
