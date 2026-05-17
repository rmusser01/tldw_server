---
id: TASK-416
title: Plan llama.cpp model acquisition and import workflows
status: Done
labels:
- llamacpp
- planning
- webui
- local-llm
priority: high
documentation:
- Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
modified_files:
- Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md
- backlog/tasks/task-416 - Plan-llama.cpp-model-acquisition-and-import-workflows.md
references:
- https://github.com/rmusser01/tldw_server/pull/1810
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the next repo-grounded plan for llama.cpp acquisition/import workflows after the merged managed runtime, asset inventory, mmproj, metadata, capability visibility, and saved profile editor slices. The plan should cover local import/register hardening first and reserve remote downloads as managed Jobs-backed acquisition work rather than implementing runtime code immediately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan file is added under Docs/superpowers/plans and references the approved managed runtime roadmap.
- [x] #2 Plan separates local import/register workflows from remote download/acquisition jobs and keeps both feeding the existing LlamaCppAsset inventory contract.
- [x] #3 Plan covers backend APIs, Jobs/Scheduler choice, safety/allowlist/quota/partial-file cleanup, WebUI workflow, tests, verification commands, and commit checkpoints.
- [x] #4 Plan preserves explicit profile creation/start/use-in-chat behavior and avoids automatic provider rewiring or executable trust.
- [x] #5 Task notes record verification and any deferred work.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created `Docs/superpowers/plans/2026-05-16-llamacpp-model-acquisition-import-workflows-plan.md`.

Manual critique pass checked the current merged llama.cpp inventory/runtime/API/WebUI surfaces and nearby Jobs worker patterns. The plan separates synchronous local import preview/result hardening from remote download acquisition jobs, uses Jobs rather than Scheduler because downloads are user-visible and need status/cancel/retry/admin controls, and keeps completed downloads feeding the existing `LlamaCppAsset` inventory contract.

Review fixes made during the planning pass: removed ambiguous deferral wording, tightened worker progress guidance to use existing `JobManager.update_job_progress()` or WorkerSDK `progress_cb`, and kept acquisition actions from creating, starting, or wiring profiles automatically.

Verification: `git diff --check` passed. Placeholder and ambiguity scan returned no matches after review fixes. ASCII scan returned no matches. Bandit was skipped because this task only adds planning/task documentation and no Python code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the next llama.cpp managed-runtime implementation plan for model acquisition/import workflows. The plan starts with local import preview/result hardening, then introduces an admin-only Jobs-backed download API and worker with destination allowlists, private-network source policy, checksum/size validation, partial-file cleanup, cancellation, and atomic registration into the existing LlamaCppAsset inventory. WebUI work is scoped to the existing Admin assets panel and preserves explicit profile creation/start/use-in-chat behavior.
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
