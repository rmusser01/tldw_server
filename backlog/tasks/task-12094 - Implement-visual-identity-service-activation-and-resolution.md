---
id: TASK-12094
title: Implement visual identity service activation and resolution
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 07:45'
labels:
  - visual-identities
  - expression-packs
  - service
dependencies: []
references:
  - Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
  - >-
    Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
documentation:
  - >-
    Implementation started with TDD for Stage 4 service activation/resolution.
    Focus remains limited to the core service
  - repository atomic helper if needed
  - exports
  - and service tests.
  - >-
    Follow-up spec review fixes: add legacy_character_mood resolution signal and
    raw default/normal neutral alias fallback coverage.
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 4 service activation and resolution for visual identity expression packs: pack shell creation, atomic draft activation into immutable versions, optional character/persona binding, actor ownership validation, deterministic expression resolution, legacy mood fallback, and service/repository regression coverage.
<!-- SECTION:DESCRIPTION:END -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 4: add VisualIdentityService for pack creation, draft activation into immutable versions, optional actor binding, actor ownership validation, and deterministic expression resolution with fallback reasons. Keep API/frontend out of scope for this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD RED captured before production implementation:
  - `test_visual_identity_service.py` failed collection with `ModuleNotFoundError: No module named 'tldw_Server_API.app.core.Visual_Identities.service'`.
  - Repository helper coverage failed with `AttributeError: 'VisualIdentityRepository' object has no attribute 'activate_draft_as_version'`.
- Added `VisualIdentityService` for active pack shell creation, ready-draft activation, optional actor binding, ownership validation, and expression asset resolution.
- Added `VisualIdentityRepository.activate_draft_as_version(...)` to create/update the pack, create a new immutable version, copy draft assets into version assets, set the active version, mark the draft activated, and optionally upsert a binding in one transaction.
- Changed visual identity binding actor IDs to string-safe storage/signatures so persona UUID bindings work while integer character callers remain supported.
- Resolution now returns explicit dataclass results with fallback reasons for manual override, requested expression, mood, pack default, neutral alias, and placeholder.
- Verification recorded: focused service pytest, full Visual_Identities pytest, Bandit JSON report with no findings, and diff whitespace checks.
- Follow-up TDD RED captured after spec review:
  - Legacy character mood fallback tests returned `placeholder` before the fix.
  - Raw `default`/`normal` version asset tests missed neutral-alias resolution before the fix.
- Added legacy character mood fallback before placeholder for character cards using supported mood image maps in `extensions.tldw.mood_images`, `extensions.tldw.moodImages`, extension-root `mood_images`/`moodImages`, and top-level card `mood_images`/`moodImages`.
- Updated neutral alias fallback to check raw stored `default` and `normal` asset keys as well as normalized `neutral`.
- Follow-up verification recorded: focused service pytest, full Visual_Identities pytest, Bandit JSON report with no findings, and range diff whitespace check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Quality review follow-up: preserved existing pack title, description, source kind, and source context during existing-pack draft activation unless replacement values are explicitly supplied; new-pack activation still derives metadata from the draft.

Quality review follow-up: legacy character mood fallback now checks manual override before requested, mood, and default expressions.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4 service activation and resolution is implemented. Added service and repository coverage for character bindings, persona UUID bindings, manual override priority, immutable draft-to-version asset copying, deleted-pack placeholder resolution, and active pack creation without an active version until activation. Verification passed: focused service pytest, full Visual_Identities pytest, Bandit JSON report with no findings, and diff whitespace checks.

Follow-up spec review fixes added legacy character mood-image fallback before placeholder, including extension-root alias coverage, and raw `default`/`normal` stored asset fallback under `neutral_alias`. Verification passed for the focused service file, the full Visual_Identities suite, Bandit, and `git diff --check 7eee48dc66..HEAD`.

Quality follow-up fixes preserve existing pack metadata during targeted draft activation and honor manual override priority for legacy character mood fallback. Verification passed: service+DB pytest, full Visual_Identities pytest, Bandit JSON with no findings, and diff whitespace checks.
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
