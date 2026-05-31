---
id: TASK-419
title: Reconcile Buddy default Codex-compatible catalog docs
status: Done
updated_date: '2026-05-23'
labels:
- persona
- buddy
- visual-packs
- docs
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1510
- https://github.com/rmusser01/tldw_server/issues/1787
- https://github.com/rmusser01/tldw_server/issues/1803
- https://github.com/rmusser01/tldw_server/issues/1807
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconcile Persona/Buddy visual pack documentation and Backlog tracking so the current source of truth clearly states that the basic tier uses the six bundled defaults, Codex/Petdex import is first-class, additional packs are optional, and legacy simple-creator/3x4/96x96 artifacts are treated as source or interim review evidence rather than contradicting the final Codex-compatible direction.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Docs remove stale three-basic default wording where this task touches the Buddy default contract.
- [x] #2 Docs clarify that the six basic defaults are the basic tier and that final Codex-compatible default packs target the Codex/Petdex atlas/import contract, while currently bundled 96x96 frame packets remain accepted art-ready runtime assets until upgraded.
- [x] #3 Backlog tracker notes the stale GitHub issue bodies and records the resolved local source-of-truth clarification.
- [x] #4 No code behavior changes are made.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started in clean worktree `.worktrees/buddy-codex-contract-docs` on branch
`codex/buddy-codex-contract-docs` from `origin/dev`; the dirty main checkout was
not edited.

Patched Persona Visual docs and Buddy pipeline specs so the current source of
truth says the basic tier is the six Codex Buddy defaults:
`search-lens-basic`, `index-card-basic`, `archive-cube-basic`,
`paperclip-basic`, `terminal-tile-basic`, and `migu-marker-basic`. The docs now
state that the current 96x96 frame packets are accepted art-ready tldw runtime
assets, while Codex/Petdex `pet.json` plus 8x9 atlas packaging remains the
cross-app compatibility target.

Updated historical Backlog task notes for TASK-347, TASK-394, TASK-405,
TASK-410, and TASK-415 to avoid treating older three-basic/nine-pack wording as
the current catalog contract. Updated GitHub issue bodies for #1787, #1803, and
#1807 so the external tracker matches the six-basic/twelve-starter catalog.

Verification:
- `git diff --check` passed.
- `git diff --name-only` plus `git ls-files --others --exclude-standard` confirmed touched files are docs and Backlog task files only.
- `rg -n "research-buddy-basic|minimal-helper-basic|three basic|3 basic|nine bundled|nine scaffold|nine default|nine immutable|Simple Buddy|simple tldw draft-pack" Docs backlog -S` now reports only explicit supersession notes and current compatibility wording.
- Bandit skipped because this is docs/Backlog/GitHub tracker-only.

Closeout 2026-05-23: PR #1818 is merged into `dev` at `89e17c12d37a55ac202a5cf521f746ea2c5ffbbf`; no active PR or review blocker remains for this task. No additional code changes were made in this closeout.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled the Persona/Buddy default catalog docs and trackers around the
current six-basic Codex Buddy tier. The repo docs now distinguish current
art-ready 96x96 Persona Visual runtime packets from the Codex/Petdex 8x9 atlas
compatibility target, and the external GitHub tracker issues no longer present
the older three-basic/nine-default catalog as current.
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
