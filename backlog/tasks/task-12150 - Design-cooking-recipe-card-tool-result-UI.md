---
id: TASK-12150
title: Design cooking recipe card tool-result UI
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-05 14:49'
labels:
  - design
  - mcp
  - frontend
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review a design spec for a read-only MCP tool that emits a typed recipe card UI payload and a shared frontend renderer for WebUI/browser extension chat tool results.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan written at Docs/superpowers/plans/2026-07-05-cooking-recipe-card-tool-result-ui.md. Plan slices: backend CookingModule contract tests and implementation; mcp_modules.yaml registration and config tests; frontend recipe-card payload parser; shared RecipeCard component; ToolCallBlock integration; tool-result replay guard; targeted pytest/Vitest/Bandit/typecheck/visual verification. Design spec remains at Docs/superpowers/specs/2026-07-05-cooking-recipe-card-tool-result-ui-design.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on branch codex/cooking-recipe-card-tool-ui.

Files changed:
- Backend MCP tool: tldw_Server_API/app/core/MCP_unified/modules/implementations/cooking_module.py
- MCP registration/surface: tldw_Server_API/Config_Files/mcp_modules.yaml; tldw_Server_API/app/core/MCP_unified/module_surface.py
- Backend tests: tldw_Server_API/app/core/MCP_unified/tests/test_cooking_module.py; tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
- Frontend parser/types: apps/packages/ui/src/types/recipe-card.ts; apps/packages/ui/src/utils/recipe-card-ui.ts; apps/packages/ui/src/utils/__tests__/recipe-card-ui.test.ts
- Shared component: apps/packages/ui/src/components/Common/RecipeCard/RecipeCard.tsx; apps/packages/ui/src/components/Common/RecipeCard/__tests__/RecipeCard.test.tsx
- Tool rendering/replay guard: apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx; apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ToolCallBlock.recipe-card.test.tsx; apps/packages/ui/src/components/Common/Playground/__tests__/tool-results-replay.guard.test.ts

Key decisions kept from the approved spec: domain-specific read-only MCP tool only; no OpenUI changes; no frontend prose detection; no persistent recipe database; no timers/notifications beyond inline cooking-mode display.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented a read-only cooking.recipe_card.render MCP tool that validates recipe data and returns a namespaced tldw_ui recipe_card payload. Registered it in the default MCP module config and classified it as read-only in module surface reporting. Added a shared frontend parser and RecipeCard component, then integrated rendering into ToolCallBlock for cooking tool results only, with generic fallback for errors, malformed JSON, unsupported payloads, and unknown tools. Added replay guards for WebUI, extension sidepanel, compare clusters, and Message-to-ToolCallBlock handoff.

Verification recorded:
- Backend targeted pytest: 17 passed.
- Frontend targeted Vitest suite: 24 passed.
- Bandit on cooking_module.py: zero findings, output /tmp/bandit_cooking_recipe_card.json.
- Typecheck attempted after temporarily repairing local antd symlink; our recipe parser TS issue was fixed, remaining failures are existing unrelated errors in AudioStudio, ScheduledTasks, Skills, service clients, and e2e fixtures.

Visual browser QA was not run; coverage is via focused component/integration tests and replay source guards.
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
