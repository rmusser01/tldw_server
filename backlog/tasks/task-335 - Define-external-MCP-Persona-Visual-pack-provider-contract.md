---
id: TASK-335
title: Define external MCP Persona Visual pack-provider contract
status: Done
assignee:
  - codex
created_date: '2026-05-14 04:37'
updated_date: '2026-05-14 05:46'
labels:
  - persona
  - buddy
  - mcp
  - visual-packs
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1682'
  - 'https://github.com/rmusser01/tldw_server/issues/1510'
documentation:
  - Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md
  - Docs/Code_Documentation/Persona_Visual_Packs.md
  - >-
    tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the contract/design slice for external MCP-compatible Persona Visual pack providers. The goal is to let local or third-party tools propose Persona/Buddy visual packs with poses, animations, renderer metadata, diagnostics, and provenance into the existing review-first Persona Visual import flow. This task is contract-first: provider discovery and handoff must be specified without adding runtime renderer activation, Live2D support, VN/CYOA behavior, or silent MCP activation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD/design documentation explains the external MCP pack-provider flow and how it connects to existing Persona Visual import preview and reviewed draft commit behavior.
- [x] #2 Contract separates provider discovery/listing from import-preview and commit; provider output cannot auto-activate a pack.
- [x] #3 Provider examples cover a valid pack, blocked diagnostics, renderer capability metadata, provenance, and review handoff fields.
- [x] #4 Safety rules document no executable scripts, bounded asset metadata, MIME and size constraints, sanitized provenance, no secrets, no cross-user access, and no auto-trust.
- [x] #5 Documentation preserves the reference-backed user-owned personal library model with no snapshot reintroduction.
- [x] #6 Verification records that this slice is contract-only and introduces no runtime renderer or activation behavior.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implementation plan created: Docs/superpowers/plans/2026-05-13-persona-visual-external-mcp-provider-contract.md.

Added a design-only external MCP provider contract. The contract keeps providers as review-input sources and defines discovery metadata, result envelopes, portable archive handoff, generated-candidate handoff, manifest patch handoff, draft-pack requests, blocked diagnostics, safety rules, and the relationship to the internal persona_visuals MCP module.

Updated the Persona Live Visual Packs PRD and Persona Visual Packs code documentation to point at the provider contract, preserve reference-backed personal-library semantics, and keep the slice out of runtime renderer activation, provider execution, Live2D support, marketplace behavior, VN/CYOA behavior, and silent activation.

Verification: git diff --check passed. Targeted scans found no activation_allowed=true, runtime_supported_by_provider=true, snapshot-field reintroduction, or provider-output activation claims. Positive scans confirmed review-required, activation_allowed=false, import-preview-required, review-input, reference-backed, and no-snapshot boundary language.

Bandit was not run because this slice touched only Markdown documentation and Backlog task text.

PR #1685 review sweep addressed still-valid Qodo, Gemini, and CodeRabbit findings: provider diagnostics examples now use machine-readable warning objects, portable archive examples use the existing Persona Visual vendor zip media type while documenting `application/zip` compatibility, safety rules explicitly reject secrets and require sanitized provenance, and TASK-335 timestamp metadata now satisfies updated_date >= created_date.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the external MCP-compatible Persona Visual pack-provider contract for review-first provider output. The docs now describe provider discovery, result envelopes, examples for archive/candidate/patch/draft request flows, blocked diagnostics, safety rules, sanitized provenance/no-secret requirements, canonical archive media-type guidance, and PRD/code-doc alignment while preserving user-owned reference-backed packs and explicit activation.
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
