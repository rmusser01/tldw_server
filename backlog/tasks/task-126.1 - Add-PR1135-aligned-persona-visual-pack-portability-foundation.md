---
id: TASK-126.1
title: Add PR1135-aligned persona visual pack portability foundation
status: Done
assignee: []
created_date: '2026-05-09 02:10'
updated_date: '2026-05-09 02:17'
labels:
  - persona
  - webui
  - vn-assets
  - portability
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1135'
  - 'https://github.com/rmusser01/tldw_server/issues/1388'
  - 'https://github.com/rmusser01/tldw_server/issues/1389'
documentation:
  - Docs/superpowers/specs/2026-05-08-persona-visual-packs-design.md
  - >-
    Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md
parent_task_id: TASK-126
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first reviewable implementation slice for persona visual pack portability after rebasing onto PR #1135. The outcome is a persona visual pack archive/import contract that mirrors the VN asset portability shape closely enough that later API, Jobs, and UI work can reuse the same user-facing model instead of inventing a separate import/export system. This task should keep assets user-owned and persona-attached by default, while storing packs with a manifest/checksum/fingerprint format that can later support duplicate-to-persona, import/export, and shared libraries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Persona visual pack portability constants, models, archive validation, canonical fingerprints, and export assembly are added with tests
- [x] #2 Persona visual pack import preview validation can inspect an archive and report schema, checksum, manifest, asset, and persona-target issues without mutating existing packs
- [x] #3 The portability contract is aligned with PR #1135 VN pack concepts including schema versioning, checksums, archive fingerprinting, strict missing-byte handling, and review-before-commit semantics
- [x] #4 Existing persona visual pack create/upload/activate behavior remains unchanged and focused regression tests still pass
- [x] #5 Plan or task notes identify the next slice for Jobs/API/frontend review UX after the foundation is in place
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added persona visual pack portability foundation in tldw_Server_API/app/core/Persona/visual_portability/: archive member validation, constants, checksum/fingerprint helpers, export options/result models, PersonaVisualPackExporter, and PersonaVisualPackImportPreviewer. The archive contract mirrors PR #1135's VN portability shape with schema versioning, manifest.json, metadata section files, checksums/sha256.json, reserved signatures, canonical payload fingerprints, strict missing-byte mode, trust hints, and review-before-commit preview plans.

Added tests in tldw_Server_API/tests/Persona/test_persona_visual_portability.py covering safe archive member rejection, export archive contents and checksums, strict missing-byte failure, import preview without DB mutation, and missing-byte warnings. Existing upload/activation regression coverage in test_persona_visual_service.py still passes.

Updated Docs/superpowers/plans/2026-05-08-persona-visual-packs-implementation-plan.md with Task 12 for the PR #1135-aligned portability foundation. Next slice is Jobs-backed API endpoints and frontend review UX mirroring PR #1135 export job, import preview, import commit, status polling, cleanup, and portability panel flows.

Verification passed: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Persona/test_persona_visual_service.py tldw_Server_API/tests/Persona/test_persona_visual_portability.py -q (11 passed). Bandit passed with 0 findings: /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/core/Persona/visual_portability tldw_Server_API/tests/Persona/test_persona_visual_portability.py -f json -o /tmp/bandit_persona_visual_portability.json. git diff --check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the PR #1135-aligned persona visual pack portability foundation: safe archive validation, schema/constants, canonical checksums/fingerprints, exporter, import preview validator, focused tests, and a plan addendum. The foundation does not expose API/Jobs/frontend flows yet; those are explicitly identified as the next slice.
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
