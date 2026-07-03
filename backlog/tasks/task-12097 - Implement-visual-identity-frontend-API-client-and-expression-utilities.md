---
id: TASK-12097
title: Implement visual identity frontend API client and expression utilities
status: Done
assignee: []
created_date: ''
updated_date: 2026-07-02 09:16
labels:
- visual-identities
- expression-packs
- frontend
- api-client
dependencies: []
references:
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
- Docs/Design/Visual_Identity_Expression_Packs.md
priority: high
modified_files:
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
- apps/packages/ui/src/types/visual-identities.ts
- apps/packages/ui/src/services/tldw/domains/visual-identities.ts
- apps/packages/ui/src/services/tldw/domains/index.ts
- apps/packages/ui/src/services/tldw/TldwApiClient.ts
- apps/packages/ui/src/utils/visual-identity-expressions.ts
- apps/packages/ui/src/utils/visual-identity-emote.ts
- apps/packages/ui/src/utils/__tests__/visual-identity-expressions.test.ts
- apps/packages/ui/src/utils/__tests__/visual-identity-emote.test.ts
- apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 8 frontend support for visual identity expression packs: typed API contracts, Tldw API domain methods, expression normalization, and client-side /emote parsing utilities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 8: add frontend Visual Identity types, Tldw API client domain methods, expression alias normalization, slash-emote parsing utilities, and Vitest coverage.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-02: Implemented Stage 8 frontend API/types/utilities. Added Visual Identity TypeScript request/response contracts matching backend schemas. Added visualIdentityMethods for capabilities, expression slots, pack CRUD, asset upload, generated-file asset import, asset content path building, ZIP import, draft read/update/activate, binding upsert/delete, and binding resolve. Wired the domain into TldwApiClient declaration merging and Object.assign. Added expression normalization with the eight canonical V1 slots and backend-compatible aliases/custom labels. Added /emote parsing that handles slash commands client-side and returns null for regular messages. Spec review found optional multipart fields were leaking undefined/null into bgUpload fields; fixed with compactMultipartFields and regression coverage. Quality review found no Critical or Important issues. Verification: git diff --check passed; bunx vitest run apps/packages/ui/src/utils/__tests__/visual-identity-expressions.test.ts apps/packages/ui/src/utils/__tests__/visual-identity-emote.test.ts apps/packages/ui/src/services/__tests__/tldw-api-client.visual-identities.test.ts passed with 15 tests. Bandit skipped: no Python/backend files touched in this Stage 8 task.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 8 frontend Visual Identity API support is complete. The UI package now has typed contracts and a Tldw API domain for expression-pack operations, client-side expression normalization and /emote parsing utilities, regression coverage for optional multipart fields, and focused Vitest coverage for the new utilities/domain contract.
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
