---
id: TASK-340
title: Implement Persona Visual external MCP provider intake normalization
status: Done
assignee: []
created_date: '2026-05-14 07:17'
updated_date: '2026-05-14 07:24'
labels:
  - persona
  - buddy
  - mcp
  - visual-packs
  - backend
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1690'
  - 'https://github.com/rmusser01/tldw_server/issues/1689'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
  - 'https://github.com/rmusser01/tldw_server/issues/1682'
documentation:
  - Docs/Design/2026-05-13-persona-visual-external-mcp-provider-contract.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the first implementation seam for external MCP-compatible Persona Visual pack providers. The helper should validate and normalize untrusted provider result envelopes from the contract in issue #1682 and the design docs, returning deterministic diagnostics without resource downloads, MCP provider execution, database writes, Jobs enqueue, import commit, draft/asset creation, activation, runtime renderer loading, marketplace behavior, VN/CYOA behavior, or live response mutation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provider intake normalization accepts and returns a deterministic normalized summary for valid portable archive, generated candidate, manifest patch, and draft-pack request result envelopes.
- [x] #2 Provider intake diagnostics reject or block unsafe result envelopes that try to allow activation, claim runtime provider support, use unsupported result types, include provider-selected database IDs, include secrets or unsanitized provenance, or use remote manifest asset URLs.
- [x] #3 Portable archive media-type handling recognizes the Persona Visual vendor zip type and current application/zip compatibility without treating media type alone as proof of validity.
- [x] #4 Diagnostics use machine-readable blocker and warning codes and preserve review_required=true and activation_allowed=false invariants.
- [x] #5 Focused tests cover the valid and blocked paths without MCP network calls, resource downloads, DB writes, Jobs enqueue, import commit, draft/asset creation, or runtime renderer behavior.
- [x] #6 Documentation records the helper boundary and the follow-up implementation slices it enables.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a pure Persona Visual provider envelope normalizer under tldw_Server_API/app/core/Persona/visual_portability/provider_envelope.py. The helper normalizes bounded review metadata, preserves structured provider blockers and warnings, and fails closed with machine-readable blockers for invalid contract version, unsupported result type, missing review_required, activation_allowed=true, missing portable-archive import preview, malformed diagnostics, unsupported archive media type, and unsafe metadata or payload strings.

Confirmed this slice has no provider execution, no MCP resource retrieval, no asset writes, no job enqueueing, no persistence, no runtime activation, and no Persona Garden UI changes.

Focused validation completed: pytest tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py -q passed with 19 tests; py_compile passed for the helper and test; git diff --check passed; Bandit on the helper wrote /tmp/bandit_persona_visual_provider_envelope.json with no results and no errors.

Review fixes addressed the Gemini sanitizer hardening comments on PR #1691: oversized metadata strings now fail closed before regex scanning, mapping metadata is bounded with lazy iteration instead of materializing the whole mapping, and oversized contract_version strings are rejected before integer coercion. Added regression tests for all three paths. Updated validation passed: pytest tldw_Server_API/tests/Persona/test_persona_visual_provider_envelope.py -q passed with 22 tests; py_compile passed; git diff --check passed; Bandit wrote /tmp/bandit_persona_visual_provider_envelope_review.json with no results and no errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the review-only Persona Visual external provider envelope intake helper, focused regression coverage, and a provider-contract documentation pointer. The helper normalizes bounded provider, pack, provenance, diagnostics, and payload metadata; preserves structured blockers and warnings; and fails closed without provider execution, MCP resource retrieval, persistence, import-preview enqueueing, draft or asset writes, runtime activation, Persona Garden UI, VN/CYOA behavior, or live response mutation. PR review follow-up additionally bounds untrusted text and mapping inputs before expensive sanitizer operations.
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
