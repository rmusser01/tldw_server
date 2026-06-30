---
id: TASK-478.6
title: 'Gate C: fix Studio enablement and generation from selected workspace sources'
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-25 04:36
labels:
- research-workspace
- uat
- gate-c
- studio
- frontend
milestone: Research Workspace UAT Remediation
dependencies: []
parent_task_id: TASK-478
priority: high
modified_files:
- apps/packages/ui/src/types/workspace.ts
- apps/packages/ui/src/store/workspace.ts
- apps/packages/ui/src/store/workspace-slices/sources-slice.ts
- apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/hooks/useArtifactGeneration.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/WorkProductTemplateChooser.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage3.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage5.folder-context.test.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage3.test.tsx
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
User-visible failure: Studio remained disabled with copy saying "Select sources from the Sources pane to enable generation" even after sources were selected and status APIs reported selected sources.

User goal: generate workspace outputs such as summaries, briefs, comparisons, or reports from the selected research sources without guessing why the controls are disabled.

Scope:
- Connect Studio enablement to the canonical selected-source and readiness contracts.
- Clarify which Studio actions require FTS-ready, vector-ready, citation-ready, or summary-ready sources.
- Show precise disabled reasons: no selected sources, selected sources still indexing, no model selected, provider unavailable, unsupported source state, etc.
- Validate output generation with a configured provider and source evidence where applicable.
- Add tests for enabled, disabled, partially queryable, failed source, and missing model states.

Acceptance criteria:
- Studio enables when selected sources meet the action's readiness requirements.
- Disabled controls explain the exact missing prerequisite.
- Generated outputs are saved/rendered in the expected workspace location and survive reload if that is the product contract.
- Live CDP/Playwright validation covers at least one successful generation path and one disabled prerequisite path.

Depends on: TASK-478.1, TASK-478.3, TASK-478.4.
Blocks: final acceptance matrix.
Parallelization: can run in parallel with grounded RAG Q&A once Gate A/B blockers are resolved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Gate C implementation completed with TDD regression coverage. Studio now derives selected source intent independently from the RAG-ready effective selection helper, uses backend readiness metadata to allow text-ready processing sources, blocks artifact creation when sources/model are not usable, and canonicalizes custom OpenAI provider ids before chat completion requests.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verification recorded: targeted RED tests reproduced the text-ready processing-source and provider alias failures; Vitest cluster passed 8 files / 166 tests; live Playwright/CDP validation passed against backend 127.0.0.1:8000 and WebUI localhost:3000 with local provider response HTTP 200, model gpt-4.1-2025-04-14, provider custom-openai-api, completed summary artifact length 198, screenshot /private/tmp/task4786-studio-summary-live.png. git diff --check passed. TypeScript check remains blocked by pre-existing WatchlistsPlaygroundPage.tsx syntax errors at lines 2355, 2416, 2521, 2763-2766. Bandit skipped because touched production files are frontend TypeScript only.
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
