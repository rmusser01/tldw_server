---
id: TASK-13196
title: Bundle pixel-migu Buddy and expression character
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 23:03'
updated_date: '2026-09-05 23:17'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the supplied pixel-migu art usable immediately as an optional Buddy starter and independent emoting character, without manual imports or overwriting user choices.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh per-user SQLite databases expose pixel-migu with all 18 expression slots and an active character binding.
- [x] #2 Buddy catalog offers pixel-migu separately from Migu Marker and copying it leaves Buddy activation unchanged.
- [x] #3 Repeated startup preserves edits, deleted characters, user bindings and ownership; shipped assets work from wheel and sdist.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/003-pixel-migu-bundled-character-seeding.md. Reason: define first-run ownership and permanent opt-out semantics using the existing Visual Identity idempotency ledger. Read existing Visual Identity and Persona ownership designs. 1. Add failing real DB and catalog tests. 2. Bundle approved source PNGs and manifests as package resources. 3. Seed character, expression version, binding and completion receipt in one transaction before publishing a per-user DB instance; preserve existing state on replay. 4. Verify focused regressions, wheel/sdist contents, formatter, Ruff, Bandit, and review; open PR to dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bundled the supplied 64-frame pixel-migu Buddy starter and 18-slot Shared Visual Identity character. The canonical per-user SQLite DB factory publishes the character, expression version and binding atomically with a permanent owner-scoped seed receipt; subsequent starts preserve edits, unbinding, deletion, and existing live/deleted same-name cards. Existing Buddy defaults and activation remain unchanged. Added a backwards-compatible include_deleted name lookup for tombstone collision checks. Architecture: backlog/decisions/003-pixel-migu-bundled-character-seeding.md. Documentation: Docs/Development/BUNDLED_PIXEL_MIGU.md; asset provenance included in both resource directories. Verification: 156 focused catalog/seed/service/bootstrap tests passed; after review found a deleted-name collision, its factory regression failed then 35 seed/CharacterStore tests passed with the fix. Wheel/sdist builds contain all 88 bundled files byte-for-byte; isolated extracted-wheel canonical bootstrap, happy expression resolution and 64-asset catalog smoke passed. Ruff and Black pass on six feature files; CharacterStore changed method is Black-clean with its 3 unchanged preexisting Ruff findings recorded. Bandit reports zero findings across all touched production code. No full suite or PostgreSQL expansion; Shared Visual Identity remains SQLite-only. Review: parent and independent character agent, tombstone finding fixed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
pixel-migu is bundled as an optional Buddy starter and separate emoting character. Seed ownership and deletion behavior follow ADR003. Implementation ready for PR review; human requester Change summary remains a merge gate under repository policy.
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
