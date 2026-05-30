---
id: TASK-564
title: Implement MCP Unified Stage 4G gateway config bootstrap
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
Add the next narrow Stage 4 standalone gateway slice after Stage 4F: package-owned gateway config models and bootstrap helpers that construct the profile store/default profile settings from explicit configuration while preserving caller injection. This slice intentionally avoids CLI commands, host route integration, real external MCP process lifecycle, and broad settings frameworks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Gateway package exposes config/bootstrap helpers without importing `tldw_Server_API`.
- [x] #2 Config can select an in-memory or SQLite profile store and build a `ProfileAwareGatewayRuntime` from an injected backend.
- [x] #3 Config can seed a default built-in preset with deterministic default profile behavior.
- [x] #4 Caller-supplied/injected profile stores remain supported and cannot be silently replaced by config defaults.
- [x] #5 Invalid store kinds and missing/blank SQLite paths fail fast with clear `ValueError`.
- [x] #6 Focused gateway tests, host extraction/http compatibility tests, Ruff, Bandit, and whitespace checks are recorded before PR handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Started from merged Stage 4F baseline on `origin/dev` `c6f192ce8a1c2e3ff822477a92d577d3d381f1eb`.
- Added package-owned `mcp_unified.gateway.config` with `GatewayProfileStoreConfig`, `GatewayProfileBootstrapConfig`, and `bootstrap_profile_gateway_from_config`.
- Config bootstrap supports memory and SQLite profile stores, validates unknown store kinds and missing SQLite paths, preserves an injected profile store when supplied, and delegates profile seeding/runtime creation to the Stage 4F bootstrap helper.
- SQLite store import is deferred until a SQLite config is used so gateway imports do not eagerly require the SQLite backend path.
- RED evidence: `5 failed, 43 passed, 4 warnings` for the focused gateway package tests; all new failures were the expected missing config module.
- GREEN evidence: `48 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Compatibility evidence: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Quality evidence: Ruff passed, Bandit reported `0` results and no errors for `mcp_unified/gateway`, and `git diff --check` passed.
- Review follow-up started from rebased `origin/dev` `1c91138f5320a623002e4da160966cbfbeab9ead`.
- Verified Gemini/Qodo blank `sqlite_path` comments as valid and now reject empty or whitespace-only SQLite paths before store construction.
- Verified the Qodo immutability advisory as valid and now copy-isolate profile mapping/model inputs during config construction.
- Verified the Qodo test-docstring comment as valid and added intent docstrings to the Stage 4G config bootstrap tests.
- Review RED evidence: `3 failed, 48 passed, 4 warnings` before the review fixes; failures covered blank path validation and copy isolation.
- Review GREEN evidence: `51 passed, 4 warnings` for `test_gateway_fastapi_package.py`.
- Review compatibility evidence: `47 passed, 4 warnings` for `test_extraction_contracts.py` and `test_http_mapping.py`.
- Review quality evidence: Ruff passed, Bandit reported `0` results and no errors for `mcp_unified/gateway`, and `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 4G adds an explicit gateway config bootstrap seam. Standalone callers can now choose in-memory or SQLite profile storage through package-owned dataclasses, seed default presets, or inject a caller-owned store without the config layer replacing it. Review follow-up tightened SQLite path validation, copy-isolated profile config inputs, and documented the focused config tests. This remains intentionally below full CLI/config commands, external MCP lifecycle, and host route integration.
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
