---
id: TASK-97
title: Create aligned tldw_server product roadmap spec
status: Done
assignee: []
created_date: '2026-05-06 16:45'
updated_date: '2026-05-06 16:56'
labels:
  - product
  - roadmap
  - webui
dependencies: []
documentation:
  - Docs/Product/WebUI/WebUI_UX_Strategic_Roadmap_2026_02.md
  - Docs/Product/WebUI/Workspace_Playground_Redesign.md
  - Docs/Design/Workspace_Persistence_Architecture.md
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Product/ACP_Agent_Orchestration_PRD.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved product roadmap spec for tldw_server, WebUI, browser extension, SaaS, enterprise, and OSS packaging. The roadmap must use the existing workspace UI paradigm, align 6-8 week, 6-month, and 12-month horizons, and preserve the commercial framing: general workplace productivity for white-collar work with enterprise seat licensing, SaaS individual/team setup, and OSS self-hosting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures the approved workspace-first roadmap frame and three nested horizons.
- [x] #2 Spec defines stable product pillars aligned across OSS, SaaS, and enterprise packaging.
- [x] #3 Spec maps milestones to existing WebUI/backend surfaces, including WorkspacePlayground, ChatWorkspace, DocumentWorkspace, workspace persistence, source-grounded RAG, outputs, ACP/MCP, jobs/workflows, onboarding, admin, and design-system state language.
- [x] #4 Spec records workspace consolidation as the first roadmap milestone, unresolved but leaning toward WorkspacePlayground as canonical pending discovery.
- [x] #5 Spec is saved under Docs/superpowers/specs with a date-stamped roadmap filename and committed on an isolated branch.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/specs/2026-05-06-tldw-product-roadmap-design.md in isolated branch codex/product-roadmap-all-horizons.

Spec review iteration 1 found gaps in extension scope, SaaS setup boundaries, and telemetry privacy; patched all three. Spec review iteration 2 approved.

Verification: git diff --check passed; non-ASCII punctuation scan passed. Bandit skipped because touched files are documentation/task markdown only.

User requested an additional pre-implementation review pass. Reopened task to patch roadmap risks found during review: 6-8 week scope cutline, server-backed workspace persistence gap, connector candidate scope, and SaaS team setup minimum.

Additional pre-implementation review patch added scope and architecture guardrails: first-horizon cutline, one golden-path template, server-backed workspace persistence gap, SaaS team cutline, connector candidate scope, and template artifact contract. Follow-up reviewer approved with no remaining blocking or important issues.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the aligned tldw product roadmap spec covering workspace-first product strategy, OSS/SaaS/enterprise packaging, 6-8 week SaaS first-value milestones, 6-month enterprise pilot readiness, 12-month category platform direction, and repo-surface mapping. The spec records workspace consolidation as the first unresolved milestone with a bias toward WorkspacePlayground pending discovery, and adds explicit browser extension and telemetry/privacy boundaries.

Additional review hardening added explicit implementation-planning cutlines for scope, persistence, SaaS team setup, connectors, and template artifacts before proceeding to plans.
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
