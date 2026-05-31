---
id: TASK-397.7
title: Implement llama.cpp WebUI capability visibility
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-16 21:45
labels:
- llamacpp
- webui
- admin
dependencies: []
documentation:
- Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
- Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md
parent_task_id: TASK-397
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 4 from the llama.cpp model-family/mmproj profile wiring plan: surface managed profile capabilities, multimodal projector state, and capability warnings in the llama.cpp Admin WebUI while keeping the slice display-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Runtime panel shows managed profile capability/mode tags, mmproj display details, and capability warnings for profile/runtime payloads without adding editor controls.
- [x] #2 Admin llama.cpp TypeScript types accept optional capability, modality, warning, and mmproj display fields while staying compatible with older servers.
- [x] #3 Focused runtime/assets/admin page tests cover capability visibility and continue to pass.
- [x] #4 Frontend verification and diff checks are recorded; Bandit is recorded as not applicable if no Python files are touched.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the display-only WebUI capability visibility slice. Added optional response-only profile/runtime capability, modality, capability warning, and mmproj display fields to the llama.cpp admin TS types while excluding those response-only fields from create/update request types. Updated LlamacppRuntimePanel to render compact Vision input, Embeddings, Rerank, and mmproj tags from profile/runtime capability metadata, mmproj display/path/model IDs, and merged profile/runtime capability warnings with existing runtime warnings. Added regression coverage for managed profile capability and projector state visibility.

PR review follow-up: Qodo reported two actionable findings on PR #1799: narrow the capability/modalities map types to known optional keys, and merge runtime/profile modalities so an empty runtime modalities object does not suppress profile-derived capability tags. Reopening TASK-397.7 for the minimal review-fix patch.

PR review fix complete: narrowed llama.cpp capability/modalities metadata to known optional keys and merged profile/runtime modalities so an empty runtime modalities object no longer suppresses profile-derived capability tags. Added focused regression coverage for the fallback case.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Llama.cpp Admin runtime rows now surface managed-profile capabilities and multimodal projector state without adding editor controls. Verification: watched the new runtime-panel test fail on missing Vision input display, then pass after implementation; final focused Vitest suite passed with 23 tests across runtime/assets/admin page specs; git diff --check passed. Package-wide tsc --noEmit still fails on existing unrelated baseline type debt and did not report touched llama.cpp runtime/type files. Bandit was not applicable because this slice only touched frontend TypeScript and Backlog task files. Attempted a temporary Vite/Playwright visual preview, but the preview harness hung before producing a useful screenshot; temporary files/processes were cleaned up.
PR review update: addressed both Qodo findings on PR #1799. Verified the new runtime modality fallback test fails before the merge fix and passes after; focused Admin llama.cpp Vitest suite passes with 24 tests. Full UI tsc --noEmit still fails on existing baseline errors outside the touched llama.cpp files. PR-style git diff --check origin/dev...HEAD passes; two-dot origin/dev..HEAD reports unrelated base-drift whitespace in Watchlists history and is not representative of the PR diff. Bandit remains not applicable because this follow-up only touched frontend TypeScript and Backlog task metadata.
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
