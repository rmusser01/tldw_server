---
id: TASK-126
title: Write persona visual packs implementation plan
status: Done
assignee: []
created_date: '2026-05-08 21:37'
updated_date: '2026-05-09 03:30'
labels:
  - persona
  - webui
  - planning
  - mcp
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1393'
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for the approved Persona Visual Packs design spec. Scope is planning only: produce a staged, test-driven plan that future implementation agents can follow for backend storage/API, animated Buddy shell runtime, pack editor, Jobs-backed generation review, and internal persona_visuals MCP tools.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is saved under the repository's superpowers plans directory using the current date and a descriptive filename
- [x] #2 Plan references the approved persona visual packs design spec and preserves v1 scope boundaries
- [x] #3 Plan decomposes the feature into staged, reviewable tasks with exact file paths, tests, commands, and expected outcomes
- [x] #4 Plan addresses the spec review advisory items: storage adapter, revision model, upload limits, and first generation-provider path
- [x] #5 Plan review is run and recorded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
Scope: V1 staged implementation for sprite/frame visual packs, Buddy shell runtime, pack editor, Jobs-backed generated candidates, and internal persona_visuals MCP tools.
Implementation decisions resolved for planning: per-user persona_visuals filesystem storage plus ChaChaNotes metadata, mutable draft packs with explicit activation, conservative upload limits, and Image_Generation adapter registry as the first generation-provider path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan at Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md.

Verification: checked the plan for TODO/TBD/FIXME/placeholder markers and ellipsis placeholders with rg; no matches. Ran git diff --check on the plan and TASK-126; no whitespace errors.

Bandit skipped for this planning task because only Markdown documentation and Backlog task metadata were changed.

Plan review not yet run because current runtime policy requires explicit user approval before dispatching a plan-document-reviewer subagent.

Plan review iteration 1 result: Issues Found. Blocking gaps were asset URL projection, generated-candidate review detail, MCP trigger_state runtime propagation, and authored trigger support.

Patched Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md to add authenticated asset content URLs, candidate list/detail review contracts and manifest patch merge rules, runtime visual override WebSocket propagation plus frontend store, and authored trigger schema/editor/resolver behavior. Re-ran placeholder scan and git diff --check; both clean.

Plan review iteration 2 result: Issues Found. Blocking gaps were runtime override priority masking errors/recovery, missing active-pack asset URL map contract, and missing deactivate/revert-to-derived-Buddy support. Advisory items were explicit owner check before generation-job creation and correcting a Task 8 note that candidate review belongs to Task 9.

Patched the plan to prioritize errors/recovery above MCP/authored overrides, add PersonaVisualPackDetailResponse with assets_by_id and active_pack projection, require shell/detail loading of that asset map, add service/API/editor/test coverage for deactivate/revert, add explicit generation-job ownership precheck, and correct the candidate-review task note. Re-ran placeholder scan and git diff --check; both clean.

Plan review iteration 3 result: Issues Found. Blocking gaps were missing sprite-sheet region/preview/frame-order modeling and missing optional-worker startup/lifecycle registration for the generation Jobs worker. Advisory item was live-session feedback for visual state source/override/fallback state.

Patched Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md to add ordered frame and sprite-sheet region manifest rules, preview_frame/preview_asset_id validation, core/frontend/editor tests for frame order and sprite-sheet crops, live visual-state feedback in AssistantVoiceCard/LiveSessionPanel tests, and explicit startup_optional_workers registration plus test coverage for PERSONA_VISUAL_GENERATION_WORKER_ENABLED. Re-ran placeholder scan and git diff --check; both clean.

Review loop has reached three plan-document-reviewer iterations. No fourth reviewer was dispatched; next step needs user direction to either accept the patched plan or explicitly run another review despite the loop cap.

Implementation branch published as draft PR #1393: https://github.com/rmusser01/tldw_server/pull/1393. The PR contains the staged persona visual packs implementation and PR #1135-aligned portability/import commit flow.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Accepted patched implementation plan for Persona Visual Packs. The plan covers the approved V1 scope: persona-owned sprite/frame visual packs, manifest validation, storage/API, Buddy shell rendering, pack editor, Jobs-backed generation review, optional worker startup, internal persona_visuals MCP tools, and final verification. It was reviewed by plan-document-reviewer three times; each review found issues that were patched and recorded. Mechanical verification passed: placeholder scan clean and git diff --check clean. Bandit was skipped because this task only changed Markdown documentation and Backlog metadata. Known blocker: repo still has unrelated unresolved conflicts, so plan/spec task files remain uncommitted in this checkout.
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
