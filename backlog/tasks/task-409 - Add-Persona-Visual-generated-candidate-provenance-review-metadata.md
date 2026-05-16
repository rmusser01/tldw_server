---
id: TASK-409
title: Add Persona Visual generated-candidate provenance review metadata
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-16 17:18'
labels:
  - persona
  - buddy
  - visual-packs
  - backend
  - issue-1782
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1782'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/pull/1784'
documentation:
  - Docs/superpowers/plans/2026-05-16-persona-visual-candidate-provenance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement GitHub issue #1782 as a narrow backend/API Persona Visual review metadata slice. Persist and expose trace-safe generated-candidate provenance for recipe-backed generation jobs, keep prompt-only behavior backward-compatible, add focused tests for bounded/unsafe metadata handling, update docs, and preserve review-gated/no-auto-activation semantics. Scope excludes WebUI redesign, final art generation, MCP provider execution/resource download, renderer expansion, marketplace/shared-library behavior, VN/CYOA behavior, and automatic activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Candidate rows persist and return bounded generation_provenance metadata without exposing raw unsafe strings.
- [x] #2 Generation workers store recipe-backed provenance fields from request, job, backend, target state, and recipe summary context while omitting raw prompts.
- [x] #3 Generated-candidate list and detail API responses include generation_provenance for review use.
- [x] #4 Persona Visual documentation describes the trace-safe provenance contract and review-gated boundary.
- [x] #5 Focused DB, worker, API, syntax, diff, and Bandit verification are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan saved at Docs/superpowers/plans/2026-05-16-persona-visual-candidate-provenance.md. Acceptance checks: persist bounded generation_provenance on persona_visual_candidates; worker stores recipe-backed provenance without raw prompts; API list/detail responses include provenance; docs describe trace-safe review metadata; focused tests, syntax, diff, and Bandit verification recorded.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verification recorded: focused Persona Visual DB/worker/API suite passed (78 passed); py_compile passed; git diff --check passed; Bandit passed with empty results array in /tmp/bandit_persona_visual_candidate_provenance.json. Draft PR: https://github.com/rmusser01/tldw_server/pull/1784
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added trace-safe generated-candidate provenance for Persona Visual review workflows. Candidates now persist bounded provenance metadata, recipe-backed generation jobs store review-relevant context without raw prompts, and API list/detail responses expose the normalized metadata for review screens. Documentation describes the V1 provenance shape and explicitly keeps it review-gated.
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
