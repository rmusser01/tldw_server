---
id: TASK-13196
title: Bundle pixel-migu Buddy and expression character
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 23:03'
updated_date: '2026-09-05 23:34'
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
ADR required: no new ADR. Existing ADR: backlog/decisions/003-pixel-migu-bundled-character-seeding.md. Reason: review fixes preserve seed ownership and persistence boundaries. 1. Fetch and rebase latest dev. 2. Verify all Qodo review comments. 3. Add centralized seed error and regression coverage; document/type new test functions and split preservation scenarios. 4. Run focused tests, Ruff/Black/Bandit, and packaging checks. 5. Reply to and resolve review threads, wait final-head Qodo review and required CI, then merge the exact verified head without bypasses.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bundled the supplied 64-frame pixel-migu Buddy starter and 18-slot Shared Visual Identity character. The canonical per-user SQLite DB factory publishes the character, expression version and binding atomically with a permanent owner-scoped seed receipt; subsequent starts preserve edits, unbinding, deletion, and existing live/deleted same-name cards. Existing Buddy defaults and activation remain unchanged. Added a backwards-compatible include_deleted name lookup for tombstone collision checks. Architecture: backlog/decisions/003-pixel-migu-bundled-character-seeding.md. Documentation: Docs/Development/BUNDLED_PIXEL_MIGU.md; asset provenance included in both resource directories. Verification: 156 focused catalog/seed/service/bootstrap tests passed; after review found a deleted-name collision, its factory regression failed then 35 seed/CharacterStore tests passed with the fix. Wheel/sdist builds contain all 88 bundled files byte-for-byte; isolated extracted-wheel canonical bootstrap, happy expression resolution and 64-asset catalog smoke passed. Ruff and Black pass on six feature files; CharacterStore changed method is Black-clean with its 3 unchanged preexisting Ruff findings recorded. Bandit reports zero findings across all touched production code. No full suite or PostgreSQL expansion; Shared Visual Identity remains SQLite-only. Review: parent and independent character agent, tombstone finding fixed.

PR #2906 review follow-up: user authorized rebase on latest dev, address all Qodo findings, and merge after checks. Reopen for typed/documented test helpers, separate replay scenarios, and a centralized seed exception with bootstrap cleanup coverage.

Qodo review remediation: all new pixel-migu fixtures/helpers/tests now have non-empty docstrings and explicit annotations; customization and deletion replay are independent tests. Added BuiltinCharacterSeedError centrally and handled it at bootstrap cleanup and async error publication. Hash mismatch, failed creation and connection cleanup regressions failed before the fix. Validation: 184 targeted seed/catalog/service/bootstrap/CharacterStore tests passed; six touched files Ruff and Black clean; Bandit zero findings; rebuilt wheel bootstrap/expression/catalog smoke passed. Latest dev fetched and rebase confirmed up to date at f6d6a673b628c77a7e262d7638c658782906aef0. User subsequently explicitly authorized merge after review remediation and normal checks.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation and Qodo remediation verified. Awaiting final-head review and CI before the explicitly authorized merge of PR #2906.
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
