---
id: TASK-565
title: Implement MCP Unified Stage 4H gateway config file loader
status: Done
labels:
- mcp-unified
- standalone-extraction
- gateway
priority: medium
references:
- Docs/superpowers/specs/2026-05-26-mcp-unified-standalone-library-gateway-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add the next narrow Stage 4 standalone gateway slice after Stage 4G: package-owned config file loading helpers that parse explicit JSON/TOML config files into GatewayProfileBootstrapConfig while preserving the existing dataclass bootstrap seam. This slice intentionally avoids CLI commands, process entrypoints, external MCP spawning/lifecycle, host route integration, and broad settings frameworks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway package exposes a config-file loader without importing `tldw_Server_API`.
- [x] #2 Loader accepts explicit JSON and TOML files and returns `GatewayProfileBootstrapConfig`.
- [x] #3 Loader can feed `bootstrap_profile_gateway_from_config()` for a default preset flow.
- [x] #4 Unsupported formats, malformed files, and non-object top-level payloads fail fast with clear `ValueError`s.
- [x] #5 Focused gateway tests, host extraction/http compatibility tests, Ruff, Bandit, and whitespace checks are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started from merged Stage 4G baseline on `origin/dev` `f3a3f3a64f1c44e4325002895633a7c812a800b0`.
- Plan: `Docs/superpowers/plans/2026-05-30-mcp-unified-stage4h-gateway-config-loader-plan.md`.
- Added `load_gateway_profile_bootstrap_config()` with JSON/TOML parsing, suffix or explicit-format detection, non-object payload validation, and parser-specific `ValueError` messages.
- Exported `GatewayConfigFormat` and the loader through `mcp_unified.gateway` without adding host imports or new third-party dependencies.
- RED evidence: `7 failed, 51 passed, 4 warnings` for the focused gateway package tests; all new failures were expected missing loader/export imports.
- GREEN evidence: `58 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Review follow-up verified Gemini's schema/type error comment as valid. RED evidence: `2 failed, 58 passed, 4 warnings`; invalid config schema/type cases escaped as raw `TypeError`s.
- Review follow-up wraps config dataclass construction `TypeError`s as clear `ValueError`s. GREEN evidence: `60 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Qodo follow-up verified the JSON error location and loader docstring comments as valid. RED evidence: `1 failed, 60 passed, 4 warnings`; malformed JSON messages omitted line/column context.
- Qodo follow-up includes full JSON parser context in the `ValueError` message and expands the public loader docstring. GREEN evidence: `61 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility evidence: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Quality evidence: Ruff passed, Bandit reported `0` results and no errors for `mcp_unified/gateway`, and `git diff --check` passed.
- PR: https://github.com/rmusser01/tldw_server/pull/2161
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4H adds a package-owned gateway config file loader. Standalone callers can now load explicit JSON or TOML config files into the Stage 4G bootstrap dataclasses, then pass them to `bootstrap_profile_gateway_from_config()` for profile-aware runtime construction. Review follow-up also wraps invalid config schema/type failures as clear `ValueError`s, preserves JSON parser location details, and documents the public loader contract. The slice keeps config validation deterministic and remains intentionally below CLI commands, process entrypoints, external MCP lifecycle, and host route integration.
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
