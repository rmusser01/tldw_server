---
id: TASK-250
title: Add strict VN generation output parsing
status: Done
assignee: []
created_date: '2026-05-10 22:04'
updated_date: '2026-05-10 22:13'
labels:
  - vn
  - scripted-generation
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
documentation:
  - Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md
  - Docs/superpowers/specs/2026-05-10-vn-scripted-model-generation-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Task 3 from Docs/superpowers/plans/2026-05-10-vn-scripted-generation-backend-runtime.md: strict parser models and provider/moderation adapter seams for VN scripted model generation. Scope is parser and adapter contracts, not full runtime orchestration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Strict parser rejects unknown fields at root and nested levels
- [x] #2 narrative_dialogue choice_set and scene_update outputs enforce required content and bounded sizes
- [x] #3 Choice IDs are valid and unique and visual directive labels/metadata are capped
- [x] #4 Generation adapter maps provider/rate-limit failures to stable public codes and preserves usage metadata
- [x] #5 Moderation adapter can fail closed for hosted/public profiles and record local opt-out skips
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan:
1. Add focused pytest coverage in test_vn_play_generated_outputs.py for strict schema rejection, caps, choice ID uniqueness, adapter error mapping, usage extraction, and moderation policy behavior.
2. Implement generated_outputs.py with Pydantic v2 extra=forbid models, bounded JSON metadata/labels, attached-character validation hook, and a parse_vn_generation_output public normalizer.
3. Extend adapters.py with generation-specific provider request/result/error types, stable public error mapping, usage metadata extraction, pinned profile snapshot routing, late-bound chat call usage, and moderation seam behavior.
4. Run focused pytest for the new test file, compileall for touched backend modules, and Bandit on touched backend files.
5. Update TASK-250 acceptance criteria/notes/final summary with verification results and any blockers.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented strict Pydantic parser models in VN_Play/generated_outputs.py and generation-specific provider/moderation adapter contracts in VN_Play/adapters.py. Kept the scope out of service.py; usage metadata is exposed on VNGenerationCallResult for the later runtime persistence step, and VNPlayRepository already supports usage_metadata on revisions from TASK-249. Addressed review findings by making moderation decisions fail closed on malformed results, honoring moderation_required plus provider_class/deployment_class, mapping 408/504 and timeout-like provider failures to model_timeout, and rejecting narrative/dialogue line metadata.

Verification: pytest tldw_Server_API/tests/VN_Play/test_vn_play_generated_outputs.py -q --tb=short --disable-warnings -> 12 passed; pytest tldw_Server_API/tests/VN_Play/test_vn_play_turns.py -q --tb=short --disable-warnings -> 48 passed; compileall touched backend modules -> exit 0; Bandit touched backend modules -> 0 findings in /tmp/bandit_vn_generated_outputs.json; git diff --check -> exit 0.

No blockers. Documentation-only updates were not needed because this implements the documented Task 3 contracts without changing public API docs.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Change summary:
- Added strict generated-output parsing for narrative_dialogue, choice_set, and scene_update with extra=forbid Pydantic models, string/array bounds, metadata and visual-label size caps, choice ID regex/uniqueness validation, and an attached-character validation hook.
- Added scripted generation adapter contracts that resolve provider/model/max tokens/temperature from pinned profile snapshots, call the existing late-bound chat seam, pass VN usage/accounting context, extract usage metadata, and normalize provider failures to stable public error codes.
- Added a moderation adapter seam that fails closed for hosted/public/moderation_required profiles, rejects malformed moderation decisions, and records moderation_skipped_by_policy for local opt-out.
- Added focused tests covering parser rejection paths, adapter error/usage handling, and moderation policy behavior.

Verification:
- 12 generated-output tests passed.
- 48 existing VN Play turn tests passed.
- compileall, Bandit on touched backend files, and git diff --check passed.

No known blockers; service orchestration/persistence wiring remains for the later runtime tasks by design.
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
