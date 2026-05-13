---
id: TASK-320
title: Design WebUI and extension demo video series
status: Done
assignee:
  - Codex
created_date: '2026-05-13 15:55'
updated_date: '2026-05-13 15:58'
labels:
  - docs
  - webui
  - extension
  - marketing
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the strategic design spec for a repeatable demo-video campaign covering the tldw WebUI and browser extension. The approved direction is a real-app 20-30 minute copyparty-style master walkthrough, cut down into persona-based marketing videos for self-hoster/privacy, student/researcher, journalist/OSINT, team knowledge manager, and writer/knowledge-worker audiences. This task is limited to the design/spec artifact; detailed scripts, feature inventory, recording runbooks, and implementation plans are follow-up work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures the approved master-walkthrough-first campaign architecture and derived persona cuts.
- [x] #2 Spec documents the chapter-level full walkthrough structure without attempting full feature-by-feature script coverage.
- [x] #3 Spec documents persona-series structure and persona-to-value-proposition mapping.
- [x] #4 Spec documents the real local server recording approach, seeded demo environment requirement, and repeatable production workflow.
- [x] #5 Spec identifies follow-up artifacts needed for runbooks, scripts, asset checklists, and persona cuts without implementing them.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Create `Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md` as the strategic design spec for the approved demo campaign.
2. Capture the approved master-walkthrough-first architecture: one real-app 20-30 minute full walkthrough, then persona-based cuts and short clips.
3. Document the chapter-level walkthrough model, persona-series structure, real local recording approach, seeded demo environment requirements, and repeatable production workflow.
4. Explicitly mark detailed feature inventory, scripts, recording runbooks, and asset checklists as follow-up work, not part of this spec.
5. Verify the spec is present and scoped correctly, update TASK-320 acceptance criteria and final notes, then commit the spec/task changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the strategic design spec at `Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md`. The spec is intentionally scoped to campaign architecture, walkthrough chapter structure, persona cuts, production workflow, risks, and follow-up artifacts. Detailed feature inventory and scripts remain deferred to the next planning phase.

Verification: `git diff --check -- Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md backlog/tasks/task-320\ -\ Design-WebUI-and-extension-demo-video-series.md` passed with no whitespace errors. Independent bounded spec review returned APPROVED. Bandit is not applicable because this task changes only markdown documentation/backlog records.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the WebUI and extension demo video series design spec for the approved master-walkthrough-first campaign. The spec documents a real-app 20-30 minute full walkthrough, derived persona cuts, the chapter-level walkthrough structure, persona-to-value mappings, seeded local demo environment requirements, repeatable production workflow, risks, and follow-up runbook/script artifacts. It deliberately defers full feature coverage, narration scripts, shot lists, seed data, and recording runbooks to the next planning phase so this design remains strategic and reviewable.

Verification: whitespace check passed for the touched spec and task file; independent spec review returned APPROVED. Bandit skipped as not applicable to markdown-only documentation changes.
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
