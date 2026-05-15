---
id: TASK-357
title: Implement Workspace UI detail surface for ACP issue 1707
status: Done
assignee:
  - '@codex'
created_date: '2026-05-15 02:33'
updated_date: '2026-05-15 02:46'
labels:
  - acp
  - artifacts
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1707'
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
documentation:
  - Docs/Product/Traceable_Work_Product_Artifact_Contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the Workspace UI detail surface for traceable work-product artifacts. Start from the merged storage/API foundation in #1703 and expose one golden-path generated workspace brief with review state, source lineage, ACP provenance, version metadata, redaction posture, export affordances, and drill-through links. Keep this slice UI-focused and do not implement ACP promotion or export adapter behavior from #1706/#1705.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Workspace UI can display one traceable generated artifact end to end.
- [x] #2 UI differentiates accepted, needs_revision, rejected, assigned, and archived states.
- [x] #3 Source lineage and ACP provenance are visible without exposing support-redacted data.
- [x] #4 Focused UI tests cover rendering, error/unavailable states, and review-state controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Survey existing Workspace/Agent Tasks frontend patterns and the #1703 workspace artifact API contract. - Complete
2. Add typed frontend client/model support for traceable workspace artifacts where the existing API helpers expect it. - Complete
3. Implement a focused artifact detail panel/list surface using existing UI primitives and state styles. - Complete
4. Add tests for golden-path rendering, state badges/controls, provenance/lineage, and unavailable/error states. - Complete
5. Verify with focused frontend tests plus diff hygiene. - Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented traceable artifact response typing and store hydration for review_state, lineage, ACP producer metadata, version metadata, export refs, redaction posture, owner placement, and content envelope fields. Added a Workspace Studio artifact summary/detail surface with review-state display, ACP session/diagnostics drill-through links, source lineage, version details, redaction posture, and export refs. Focused tests cover API mapping, rendering, unavailable metadata states, and review-state controls. Verification: vitest workspace-api-first + TraceableArtifactDetail passed; StudioPane.stage2 passed; git diff --check passed; design-system verifier passed with existing allowed baseline exceptions. Full UI tsc was attempted and failed on pre-existing unrelated repo-wide type errors outside touched files. Bandit not applicable because no Python code was changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workspace Studio now understands traceable work-product artifact metadata from the #1703 storage/API foundation and can display it in the generated-output cards and detail modal. The UI shows review state, ACP provenance, source lineage, version chain, redaction posture, export refs, and ACP session/diagnostics drill-through links, with focused coverage for contract mapping and detail rendering.
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
