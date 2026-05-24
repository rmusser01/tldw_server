---
id: TASK-464
title: Implement research workspace telemetry and route rename cleanup
status: Done
labels:
- frontend
- research-workspace
- telemetry
priority: High
references:
- Docs/superpowers/specs/2026-05-23-research-workspace-hard-replacement-roadmap-design.md
modified_files:
- apps/packages/ui/src/utils/research-workspace-telemetry.ts
- apps/packages/ui/src/utils/research-workspace-prefill.ts
- apps/packages/ui/src/routes/route-paths.ts
- apps/packages/ui/src/routes/route-metadata.ts
- apps/packages/ui/src/routes/route-registry.tsx
- apps/packages/ui/src/routes/option-research-workspace.tsx
- apps/packages/ui/src/components/Option/ResearchWorkspace/**
- apps/packages/ui/src/store/workspace-bundle.ts
- apps/packages/ui/src/store/workspace.ts
- apps/tldw-frontend/pages/research-workspace.tsx
- apps/tldw-frontend/extension/routes/route-registry.tsx
- apps/tldw-frontend/extension/routes/option-research-workspace.tsx
- apps/tldw-frontend/e2e/workflows/research-workspace*.spec.ts
- apps/extension/tests/e2e/research-workspace*.spec.ts
- apps/test-utils/research-workspace/**
- Docs/Operations/Research_Workspace_Trust_Status_Telemetry_Runbook.md
- Docs/Prompts/UX_RESEARCH_WORKSPACE_REVIEW_PROMPT.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rename active workspace-playground telemetry and current route metadata to research-workspace, preserve one-time legacy telemetry import, and ensure /workspace-playground is not kept as an alias or redirect.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the approved research-workspace hard rename across active WebUI, extension route registration, telemetry storage/API, prefill handoff, tests, and active docs. /workspace-playground is now only present in negative assertions and historical/generated legacy compatibility strings, not as a registered active route.

Verification: bunx vitest run focused route/telemetry/handoff bundle (16 files, 121 tests); bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__ --testTimeout=10000 (45 files, 320 tests); bun run lint (0 errors, existing warnings only). Bandit skipped because no Python backend code was touched.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
