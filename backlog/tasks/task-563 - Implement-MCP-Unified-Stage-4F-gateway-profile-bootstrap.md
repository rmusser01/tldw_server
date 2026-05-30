---
id: TASK-563
title: Implement MCP Unified Stage 4F gateway profile bootstrap
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
- Docs/superpowers/plans/2026-05-30-mcp-unified-stage4f-gateway-profile-bootstrap-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow Stage 4F standalone gateway slice after Stage 4E: provide package-owned gateway profile bootstrap helpers that seed/select a default profile from built-in presets or caller-supplied profiles and return a ProfileAwareGatewayRuntime around an injected backend runtime. This slice intentionally avoids SQLite CLI/config commands, external MCP lifecycle, upstream process spawning, and tldw_server host route integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway package exposes profile-bootstrap helpers without importing `tldw_Server_API`.
- [x] #2 Bootstrap can seed a deterministic default profile from a built-in preset and return a `ProfileAwareGatewayRuntime`.
- [x] #3 Bootstrap preserves caller-supplied profiles and supports selecting one as the default without duplicating presets.
- [x] #4 Unknown default preset ids fail fast with a clear `ValueError`.
- [x] #5 Focused gateway tests cover preset seeding, caller profile defaults, and unknown preset failure.
- [x] #6 Host extraction/http compatibility tests, Ruff, Bandit, and whitespace checks are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started from merged Stage 4E baseline and rebased final work onto `origin/dev` `53d224c4fb`.
- Added package-owned `mcp_unified.gateway.bootstrap` with `GatewayProfileBootstrap`, `bootstrap_profile_gateway`, and `build_profile_gateway_runtime`.
- Bootstrap accepts an injected backend runtime plus optional profile store, caller profiles, default profile id, and default preset id.
- Built-in default presets are duplicated into deterministic profile ids when no explicit default id is provided; caller profiles are stored before preset defaults and remain addressable.
- Unknown preset ids fail fast through the existing preset duplicate path with `ValueError`.
- RED evidence: `3 failed, 38 passed, 4 warnings` for the focused gateway package tests; all new failures were the expected missing bootstrap module.
- GREEN evidence after rebase: `41 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- PR review follow-up verified Qodo and Gemini's preset/profile collision finding as still valid.
- Review RED evidence after rebasing onto `origin/dev` `7ef87742ac`: `2 failed, 41 passed, 4 warnings`; the failures showed the caller default being overwritten and no collision error.
- Preset seeding now uses the built-in preset id for the seeded profile and rejects an existing seeded profile id with `ValueError` instead of silently overwriting.
- Review GREEN evidence: `43 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility evidence: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Quality evidence: Ruff passed, Bandit reported `0` results and no errors for `mcp_unified/gateway`, and `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4F adds the standalone gateway profile bootstrap seam. Callers can now build a `ProfileAwareGatewayRuntime` from an injected `GatewayRuntime`, optional caller profiles, and an optional built-in preset default without importing host `tldw_Server_API` code or introducing SQLite/config/lifecycle concerns. PR review follow-up also prevents built-in preset seeding from silently overwriting an existing profile id. No known blockers remain; external MCP lifecycle, persistent config, and host route integration stay intentionally out of scope for later stages.
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
