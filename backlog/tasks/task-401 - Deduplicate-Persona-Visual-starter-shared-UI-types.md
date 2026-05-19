---
id: TASK-401
title: Deduplicate Persona Visual starter shared UI types
status: Done
assignee: []
created_date: '2026-05-16 02:33'
updated_date: '2026-05-16 02:39'
labels:
  - persona
  - visual-packs
  - webui
  - type-cleanup
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1740'
  - 'https://github.com/rmusser01/tldw_server/pull/1743'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1740 by removing duplicate Persona Visual starter-pack shared UI type declarations while preserving exported names and existing starter catalog API behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 persona-visuals.ts declares each starter-pack request/response type once
- [x] #2 Existing starter catalog service/component coverage continues to pass
- [x] #3 Touched-path TypeScript or focused tests validate the cleanup
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the duplicate starter-pack type declarations from apps/packages/ui/src/types/persona-visuals.ts, leaving the canonical declaration near PersonaVisualStarterPackAssetSummary/Detail. Validation: rg confirms one export each for PersonaVisualStarterPackSummary/ListResponse/CopyRequest; focused Vitest passed with 63 tests across persona-visuals service and VisualPackEditor; git diff --check passed. Package tsc still exits nonzero on existing repo-wide baseline errors; /tmp/persona_visual_starter_type_cleanup_tsc.log has no errors for apps/packages/ui/src/types/persona-visuals.ts. Bandit is not applicable because this is TypeScript-only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Deduplicated the shared UI Persona Visual starter-pack request/response type declarations without changing exported names or runtime behavior. Focused Persona Visual service/editor tests pass; repo-wide TypeScript remains blocked by pre-existing unrelated baseline errors.
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
