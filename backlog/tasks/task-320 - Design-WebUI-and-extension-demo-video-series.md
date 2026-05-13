---
id: TASK-320
title: Design WebUI and extension demo video series
status: Done
assignee:
  - Codex
created_date: '2026-05-13 15:55'
updated_date: '2026-05-13 16:44'
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

Revision pass: patch the design spec with the critique findings above, verify markdown whitespace, update TASK-320 completion evidence, and commit the spec revision separately.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created the strategic design spec at `Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md`. The spec is intentionally scoped to campaign architecture, walkthrough chapter structure, persona cuts, production workflow, risks, and follow-up artifacts. Detailed feature inventory and scripts remain deferred to the next planning phase.

Verification: `git diff --check -- Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md backlog/tasks/task-320\ -\ Design-WebUI-and-extension-demo-video-series.md` passed with no whitespace errors. Independent bounded spec review returned APPROVED. Bandit is not applicable because this task changes only markdown documentation/backlog records.

User reviewed the committed design and asked for a critique pass before continuing. Review found concrete improvement areas to patch into the spec: add claim/evidence guardrails, WebUI-vs-extension surface mapping, recording readiness gates, stale-asset/version metadata, and clearer script-planning inputs before moving into implementation/script planning.

Design critique patch added guardrails that were missing from the first spec revision: claim/evidence ledger, WebUI-vs-extension surface map, script-planning feature matrix, recording readiness gate, asset versioning/staleness control, and additional follow-up artifacts under `Docs/Product/DemoVideos/`. Verification: `git diff --check -- Docs/superpowers/specs/2026-05-13-webui-extension-demo-video-series-design.md backlog/tasks/task-320\ -\ Design-WebUI-and-extension-demo-video-series.md` passed. Bandit remains not applicable because this revision only changes markdown documentation/backlog records.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Revised the WebUI and extension demo video series design after a critique pass. The updated spec keeps the approved master-walkthrough-first strategy, but strengthens the next phase with a claim/evidence ledger, WebUI-vs-extension surface map, script-planning feature matrix, recording readiness gate, asset versioning/staleness controls, and expanded follow-up artifacts for script/runbook planning. These changes reduce the risk of overclaiming, confusing sidepanel vs WebUI roles, leaking private demo data, or publishing clips that become stale.

Verification: markdown whitespace check passed for the touched spec and task file. Bandit remains skipped as not applicable to docs-only changes.
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
