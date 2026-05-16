---
id: TASK-411
title: Implement Persona Visual provider archive handoff
status: Done
labels:
- persona
- persona-visual
- mcp
- backend
priority: medium
references:
- https://github.com/rmusser01/tldw_server/issues/1510
documentation:
- Docs/Design/2026-05-13-persona-visual-external-mcp-provider-contract.md
- Docs/Code_Documentation/Persona_Visual_Packs.md
modified_files:
- tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py
- tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py
- Docs/Design/2026-05-13-persona-visual-external-mcp-provider-contract.md
- Docs/Code_Documentation/Persona_Visual_Packs.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next backend-only Persona Visual external-provider slice: convert a normalized portable_archive provider envelope into the existing import-preview Jobs handoff contract without executing providers, retrieving MCP resources, writing assets, committing imports, activating packs, changing renderer support, adding Persona Garden UI, or touching Buddy animation/VN behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider portable_archive envelopes can be validated into a deterministic import-preview handoff request using the existing provider envelope normalizer.
- [x] #2 The handoff fails closed for blocked/non-eligible envelopes, missing or unsafe mcp_resource_uri, unsupported result types, activation attempts, and missing archive payload metadata.
- [x] #3 The helper reuses existing Persona Visual import-preview job payload conventions where practical and does not retrieve provider resources, write DB rows/assets, enqueue Jobs directly, commit imports, or activate packs.
- [x] #4 Focused backend tests cover valid portable archive handoff and fail-closed invalid/blocked cases.
- [x] #5 Docs or task notes record the boundary and follow-up slices enabled by this handoff.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused RED tests for provider portable-archive handoff construction and fail-closed invalid inputs.
2. Add a pure provider-envelope helper that validates normalized portable_archive metadata into an MCP resource retrieval descriptor without persistence, Jobs enqueue, archive writes, imports, activation, runtime renderer changes, UI, Buddy animation, or VN behavior.
3. Update provider contract/code docs and Backlog notes with the handoff boundary and follow-up retrieval slice.
4. Run focused pytest, py_compile, diff check, and Bandit on touched Python scope.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented `build_provider_archive_import_preview_handoff()` in `provider_envelope.py`. The helper normalizes raw provider output, validates portable_archive MCP resource metadata, returns a deterministic import-preview handoff descriptor, and fails closed with machine-readable blockers. It intentionally does not retrieve MCP resources, create preview rows, write archive files/assets, enqueue Jobs, commit imports, activate packs, change renderer support, add UI, or touch Buddy animation/VN behavior.

RED verification: focused provider-envelope pytest failed on missing `build_provider_archive_import_preview_handoff` import before implementation.

GREEN verification: focused provider-envelope pytest passed with 33 tests; Persona Visual jobs pytest passed with 8 tests; py_compile passed for the touched helper/test; git diff --check passed; Bandit on `provider_envelope.py` wrote `/tmp/bandit_persona_provider_archive_handoff.json` with zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the backend-only Persona Visual provider archive handoff helper. Normalized portable_archive provider envelopes now produce a review-only MCP resource retrieval descriptor for a future import-preview retrieval adapter, with fail-closed blockers for invalid result types, activation attempts, missing MCP resource handles, invalid checksums, and unsafe normalized envelopes. Documentation now records that this remains pre-persistence and does not enqueue import-preview Jobs until a local archive path exists.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched Python code or documented non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
