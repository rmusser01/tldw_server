---
id: TASK-192
title: Design persona visual pack duplicate-to-persona workflow
status: In Progress
assignee: []
created_date: '2026-05-09 21:18'
updated_date: '2026-05-09 21:26'
labels:
  - persona
  - buddy
  - webui
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1449'
  - 'https://github.com/rmusser01/tldw_server/issues/1450'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for GitHub issue #1450: duplicate a Persona Visual pack from one persona to another same-user persona as a draft. The approved design direction is to physically copy asset bytes into the target persona/pack storage path, remap manifest asset references to newly created asset rows, leave source and active packs unchanged, and preserve same-user cross-persona lineage via parent_pack_id or an equivalent lineage field if implementation review finds parent_pack_id must remain same-persona only. This is Buddy/persona visual-pack work, not VN/CYOA runtime work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the same-user duplicate workflow and explicitly excludes cross-user sharing, marketplaces, VN/CYOA, Live2D renderer work, and auto-activation.
- [x] #2 Spec covers backend API, service, persistence/lineage, manifest remapping, frontend workflow, error handling, and tests for a future implementation plan.
- [x] #3 Spec aligns duplicate behavior with existing Persona Visual import/export asset-copy and manifest-remap patterns introduced for visual-pack portability.
- [x] #4 Spec is reviewed before implementation planning begins.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created draft spec at Docs/superpowers/specs/2026-05-09-persona-visual-duplicate-to-persona-design.md in worktree .worktrees/persona-visual-duplicate-spec on branch codex/persona-visual-duplicate-spec. Spec review subagent dispatched after whitespace check passed.

Spec review iteration 1 found one blocking ambiguity: same-persona duplication was both allowed and left open. Resolved V1 to require a different target persona and reject same-persona duplicate with a stable error. Spec review iteration 2 approved. Verification: git diff --check passed. Bandit not run because this task only adds design/backlog markdown.

Design self-review before implementation planning found and patched three issues: public API response is now the existing PersonaVisualPackResponse with asset_id_map kept internal only; idempotency keys are explicitly out of V1; duplicate now copies only manifest-referenced assets so unaccepted generated candidates and stale uploads are not copied. Also tightened preflight/cleanup guidance for source asset membership, checksum, file existence, and partial target draft failures.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
