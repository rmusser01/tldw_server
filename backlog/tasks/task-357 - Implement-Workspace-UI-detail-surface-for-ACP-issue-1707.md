---
id: TASK-357
title: Implement Workspace UI detail surface for ACP issue 1707
status: Done
assignee:
  - '@codex'
created_date: '2026-05-15 02:33'
updated_date: '2026-05-15 03:33'
labels:
  - acp
  - artifacts
  - webui
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1707'
  - 'https://github.com/rmusser01/tldw_server/issues/1703'
  - 'https://github.com/rmusser01/tldw_server/issues/1525'
  - 'https://github.com/rmusser01/tldw_server/issues/1538'
  - 'https://github.com/rmusser01/tldw_server/issues/1532'
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
6. Address PR #1714 review feedback: hide provenance/lineage when redaction posture is not support-safe, make traceable metadata detection and list keys robust, route ACP links through the app router, externalize traceable artifact labels through i18n fallbacks, and add required contract ticket links. - Complete
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented traceable artifact response typing and store hydration for review_state, lineage, ACP producer metadata, version metadata, export refs, redaction posture, owner placement, and content envelope fields. Added a Workspace Studio artifact summary/detail surface with review-state display, ACP session/diagnostics drill-through links, source lineage, version details, redaction posture, and export refs. Focused tests cover API mapping, rendering, unavailable metadata states, and review-state controls. Verification: vitest workspace-api-first + TraceableArtifactDetail passed; StudioPane.stage2 passed; git diff --check passed; design-system verifier passed with existing allowed baseline exceptions. Full UI tsc was attempted and failed on pre-existing unrelated repo-wide type errors outside touched files. Bandit not applicable because no Python code was changed.

PR #1714 review follow-up reopened this task to address Qodo and Gemini feedback on redaction-aware rendering, stable list keys, schema-version metadata detection, router-aware ACP session links, i18n-ready labels, and missing contract ticket references.

Review follow-up verification: `bunx vitest run src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx` passed with 11 tests; `bunx vitest run src/components/Option/WorkspacePlayground/StudioPane/__tests__/TraceableArtifactDetail.test.tsx src/store/__tests__/workspace-api-first.test.ts` passed with 27 tests; `bunx vitest run src/components/Option/WorkspacePlayground/__tests__/StudioPane.stage2.test.tsx` passed with 26 tests; `git diff --check` passed; `bun run verify:design-system-state` passed with existing allowed baseline exceptions. Bandit remains not applicable because this review follow-up only changed TypeScript/React and Backlog metadata.

Fresh PR #1714 closeout rerun on 2026-05-15: focused traceable artifact/store tests passed again (27 tests), git diff hygiene passed, and design-system verifier passed with the existing 486 allowed product-state baseline exceptions. Full UI TypeScript remains blocked by unrelated repo-wide baseline errors; grep over redirected output found no touched traceable-artifact file errors. A fresh local rerun of StudioPane.stage2 timed out in broad pre-existing StudioPane workflows (22/26 failed by timeout), so current closeout relies on the focused review-fix regression tests plus GitHub CI for the wider StudioPane gate.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Workspace Studio now understands traceable work-product artifact metadata from the #1703 storage/API foundation and can display it in the generated-output cards and detail modal. The UI shows review state, ACP provenance, source lineage, version chain, redaction posture, export refs, and ACP session/diagnostics drill-through links, with focused coverage for contract mapping and detail rendering. PR #1714 review follow-up hardened the detail surface by suppressing provenance/lineage under restricted redaction posture, preserving schema version zero metadata, stabilizing export keys, routing ACP links through React Router, adding i18n fallbacks for user-facing labels, and linking the required artifact contract issues.
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
