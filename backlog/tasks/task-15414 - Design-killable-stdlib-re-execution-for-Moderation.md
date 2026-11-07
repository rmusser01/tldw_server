---
id: TASK-15414
title: Design killable stdlib re execution for Moderation
status: Done
labels:
- moderation
- design
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design parent-supervised child-process execution using standard-library re, preserving normal evaluator semantics while blocking on execution failures and surfacing sanitized admin incidents. User explicitly rejected replacing re with regex.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Specify stdlib-only execution, parent deadlines, bounded admission and IPC, cancellation, kill/reap ownership, and platform limitations.
- [x] #2 Specify fail-closed service/caller and streaming behavior plus sanitized deduplicated admin visibility independent of optional content capture.
- [x] #3 Document compatibility boundaries, validation strategy, ADR assessment, and fresh builder-check disposition; obtain user approval before implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task allocation checked origin/dev maximum 15412, local TASK-15413, all open PR filename inventories and paginated >100-file PRs (maximum 13950); allocated 15414. Follows TASK-15413 CI triage. ADR required: yes; proposed Docs/ADR/086-moderation-killable-stdlib-regex-execution.md because process ownership and fail-closed moderation are durable security rules. Design target: Docs/superpowers/specs/2026-11-02-moderation-killable-stdlib-re-design.md. No implementation authorized by this design task.
Draft spec and proposed ADR-086 written. Chosen architecture: standard-library re only in killable per-batch subprocesses; one parent-owned service deadline across decision/redaction; explicit fail-closed caller integration and independent metadata-only admin incidents. Numerical limits are proposals, not user commitments. Fresh builder review dispatched; design approval pending. Documentation/task-only changes; Bandit is not applicable because no Python/code changed. Canonical task normalization, duplicate task-ID check, and git diff whitespace check passed.
Fresh builder check completed (Copernicus, 01a19d4a-4a37-7133-a058-aa4121fc8e66): nothing important left; no blocking questions. Metadata CI suite: 4 passed, 20 warnings. No runtime, tests, dependencies, or workflow files changed. Written design and proposed ADR await requester approval; AC3 intentionally remains unchecked until approval. No PR opened or auto-merge enabled.
Requester approved the complete written design on 2026-11-02 (yes). Spec approval metadata and ADR-086/index updated to Accepted; no accepted ADR rationale changed. Planning handed to TASK-15415; implementation has not started. Documentation-only validation remains canonical task/ID checks, metadata CI 4 passed with 20 warnings, and whitespace checks; Bandit not applicable. No pending design blockers; plan approval still required before implementation.
Historical CI triage reference TASK-15413 was renumbered to TASK-15416 after open PR 3404 independently allocated15413; original allocation notes above remain historical. Approved implementation now tracked by TASK-15417.
2026-11-07 integration identity correction: implementation task is now TASK-15466 (previously TASK-15417), and accepted Moderation decision is Docs/ADR/096-moderation-killable-stdlib-regex-execution.md (previously ADR-086). Only this branch records were renumbered to avoid independently assigned dev/open-PR IDs; historical allocation notes/commit IDs retained.
2026-11-07 final publication identities: TASK-15477 and ADR-097 supersede interim TASK-15466/ADR-096, which concurrently opened PR3436 claimed during verification. Current governing path Docs/ADR/097-moderation-killable-stdlib-regex-execution.md. Historical notes/commits preserved.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved stdlib-re-only child-process isolation design with parent deadlines, bounded admission/IPC/resources, fail-closed callers and streaming, sanitized admin incidents independent of optional capture, compatibility tests and documented platform limits. Independent builder review found no blockers. ADR-086 accepted; implementation plan tracked by TASK-15415.
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
