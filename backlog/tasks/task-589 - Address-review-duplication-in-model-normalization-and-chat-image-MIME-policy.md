---
id: TASK-589
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
Acceptance criteria completed: shared model normalizer used by both getModels paths; shared image utility policy used by composer and unsupported-types; regression tests added. PR #2218 was rebased onto the latest `origin/dev` on 2026-06-01; the rebase was a no-op because the branch was already current. Addressed Gemini review threads for nullish image MIME handling, array-valued model metadata records, and raw metadata normalization. Addressed later Qodo/CodeRabbit/Cubic threads for duplicate `TldwModel` fields, image/ico MIME parity, array-shaped capability flags, and explicit false capability precedence. Added concise JSDoc to the shared helper contracts after the PR-level docstring coverage warning. Replaced the duplicate `TASK-499` branch task record with unique `TASK-589`. Bandit skipped because no Python code changed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved the two review duplication findings by centralizing Tldw model normalization/provider availability enrichment and chat attachment image MIME policy. PR #2218 was rebased onto the latest `origin/dev`; no commits were replayed because it was already current. Addressed all current inline review comments by accepting nullish MIME values in `normalizeImageDataUrlMime`, making model-normalization records exclude arrays, passing the raw metadata payload into `normalizeTldwModels`, removing duplicate `TldwModel` readiness fields, restoring image/ico MIME parity for the shared unsupported image policy, deriving capability flags from array-shaped metadata, and preserving explicit false capability values. Added JSDoc to the shared helper contracts in response to the PR-level docstring warning. Also replaced the duplicate branch-local `TASK-499` tracking file with unique `TASK-589`. Verification: targeted Vitest from apps/packages/ui passed 4 files / 36 tests after the review-comment fixes. Touched-file whitespace check passed. Package-wide TypeScript still has unrelated baseline failures; exact touched-source type-error filter returned no matches. Bandit skipped because this slice touched no Python code. PR: https://github.com/rmusser01/tldw_server/pull/2218
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
