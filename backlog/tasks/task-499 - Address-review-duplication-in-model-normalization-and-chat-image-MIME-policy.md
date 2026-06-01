---
id: TASK-499
title: Address review duplication in model normalization and chat image MIME policy
status: Done
labels:
- review
- frontend
- maintainability
priority: medium
modified_files:
- Docs/superpowers/plans/2026-06-01-review-duplication-normalization-image-policy-plan.md
- apps/packages/ui/src/services/tldw/model-normalization.ts
- apps/packages/ui/src/services/tldw/__tests__/model-normalization.test.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/services/tldw/domains/models-audio.ts
- apps/packages/ui/src/utils/image-utils.ts
- apps/packages/ui/src/utils/__tests__/image-utils.test.ts
- apps/packages/ui/src/components/Chat/composer/hooks/useComposerAttachments.ts
- apps/packages/ui/src/components/Option/Knowledge/utils/unsupported-types.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve two maintainability review findings: centralize Tldw model normalization/provider availability enrichment and move chat attachment image MIME/extension policy behind shared image utilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-01-review-duplication-normalization-image-policy-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Acceptance criteria completed: shared model normalizer used by both getModels paths; shared image utility policy used by composer and unsupported-types; regression tests added. Verification recorded in Final Summary. Bandit skipped because no Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the two review duplication findings by centralizing Tldw model normalization/provider availability enrichment and chat attachment image MIME policy. Verification: initial red tests failed for missing helper exports; targeted Vitest from apps/packages/ui passed 4 files / 25 tests in the original workspace and 4 files / 31 tests after cherry-picking onto the clean dev-based PR branch. Touched-file whitespace check passed. Package-wide TypeScript still has unrelated baseline failures; exact touched-source type-error filter returned no matches. Bandit skipped because this slice touched no Python code.
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
