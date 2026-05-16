---
id: TASK-407
title: Plan llama.cpp model-family and mmproj profile wiring
status: Done
assignee: []
created_date: '2026-05-16 15:06'
updated_date: '2026-05-16 15:20'
labels:
  - llamacpp
  - planning
  - backend
  - webui
dependencies: []
documentation:
  - Docs/superpowers/specs/2026-05-16-llamacpp-managed-runtime-roadmap-design.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-managed-runtime-stage1-implementation-plan.md
  - >-
    Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the next implementation plan for the llama.cpp managed runtime roadmap after Stage 1 and Asset Inventory V2: model-family modes, mmproj/base asset profile wiring, managed profile metadata exposure, and explicit local provider capability routing. Keep remote downloads/catalogs and the full Admin Console V2 as follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan file is added under Docs/superpowers/plans and references the approved managed runtime roadmap.
- [x] #2 Plan scopes implementation to model-family modes, mmproj profile wiring, validation, metadata exposure, and minimal WebUI/client follow-through.
- [x] #3 Plan preserves V1 default-profile compatibility and avoids remote download/catalog behavior.
- [x] #4 Plan includes TDD steps, exact touched files, verification commands, Bandit expectations, and PR-ready checkpoints.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created Docs/superpowers/plans/2026-05-16-llamacpp-model-family-mmproj-profile-wiring-plan.md as the next implementation plan for the approved llama.cpp managed runtime roadmap. The plan scopes the next slice to model-family modes, base GGUF/mmproj profile launch resolution, managed profile metadata, and minimal WebUI capability visibility. It deliberately leaves remote downloads/catalogs, full profile editing, and advanced Chat/Knowledge routing for follow-up tasks. Verification for this planning-only slice: inspected current runtime/inventory/provider/UI code on origin/dev, reviewed the plan file, and ran git diff --check successfully. Bandit skipped because this task changes only planning/task documentation. Note: origin/dev currently has a duplicate TASK-397 ID collision, so this was tracked as standalone TASK-407 instead of being linked as a child of TASK-397.

PR: https://github.com/rmusser01/tldw_server/pull/1772

Review follow-up: PR #1772 has three unresolved Gemini inline threads on the plan. Verified as still valid before editing: repeated scan_assets() guidance, undefined mmproj_path in a test snippet, and direct JsonLlamaCppProfileStore use in metadata helper guidance.

Review follow-up fixed in plan: resolve_asset_id now accepts an optional pre-scanned asset list; profile capability helpers and managed profile metadata pass that asset list through to avoid repeated full scans; the supervisor test snippet now defines mmproj_path through a fixture helper; and metadata planning now reuses the existing llm_manager/supervisor path instead of constructing JsonLlamaCppProfileStore directly in llm_providers.py. Verification: reviewed the patched plan snippets and ran git diff --check successfully. Bandit still skipped because only docs/task files changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the llama.cpp model-family/mmproj implementation plan and addressed PR #1772 review feedback. The plan now avoids repeated asset scans, fixes the supervisor test snippet variable, and routes metadata planning through the existing supervisor/manager path. No runtime code changed in this planning PR.
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
