---
id: TASK-2338
title: Implement Phase 1 research discovery chokepoint
status: In Progress
labels:
- research
- implementation
- discovery
documentation:
- Docs/superpowers/specs/2026-06-20-research-source-discovery-chokepoint-design.md
- Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
- tldw_Server_API/app/core/Research/discovery/__init__.py
- tldw_Server_API/app/core/Research/discovery/catalog.py
- tldw_Server_API/app/core/Research/discovery/identity.py
- tldw_Server_API/app/core/Research/discovery/models.py
- tldw_Server_API/app/core/Research/discovery/oa.py
- tldw_Server_API/app/core/Research/discovery/router.py
- tldw_Server_API/app/core/Research/discovery/adapters.py
- tldw_Server_API/tests/Research/test_research_discovery_catalog.py
- tldw_Server_API/tests/Research/test_research_discovery_identity.py
- tldw_Server_API/tests/Research/test_research_discovery_router.py
- tldw_Server_API/tests/Research/test_research_discovery_adapters.py
- backlog/tasks/task-2338 - Implement-Phase-1-research-discovery-chokepoint.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the approved Phase 1 research discovery chokepoint plan: source catalog, source router, standalone discovery API, normalized metadata/OA candidates, persisted discovery snapshots, and focused tests. Scope excludes standalone ingest, Deep Research migration, compatibility delegation, and fallback site-search rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Phase 1 endpoints are available at GET /api/v1/research/sources and POST /api/v1/research/discovery/search.
- [ ] #2 Discovery snapshots are persisted owner-scoped, sanitized, and short-lived in ResearchSessionsDB.
- [ ] #3 Over-cap category/source selections return validation errors instead of silent truncation.
- [ ] #4 Fallback site search remains disabled by default.
- [ ] #5 Raw signed/token-bearing URLs do not appear in API responses, snapshots, logs, or candidate ids.
- [ ] #6 Existing provider-specific paper-search endpoints are unchanged.
- [ ] #7 Focused pytest commands and adjacent research tests pass.
- [ ] #8 Bandit touched-scope scan has no new actionable findings.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-06-20-research-discovery-chokepoint-phase1-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 1 complete. Implemented catalog models/default catalog in commits 03d6cd88a628 and a47defcd9919. Targeted verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_catalog.py -v` -> 7 passed, 7 warnings. Spec review passed. Code-quality review passed with only non-blocking minor test-hardening suggestions.

Task 2 complete. Implemented normalized discovery identity, dedupe/ranking, OA candidate sanitization, safe metadata filtering, and provider-id URL safety through commits 198f5e81a3b6c, 0961eea8d3f0, 0ac4fb90ec23, 7a69a60de69, f17ddb542bf4, and ca57548db10. Review-driven fixes hardened metadata key variants, URL-like values under unknown/nested keys, credentialed/tokenized OA URLs, unsafe provider-id values, encoded URL/path-param secret material, and false positives for safe filenames such as tokenization papers. Targeted verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_identity.py tldw_Server_API/tests/Research/test_research_discovery_catalog.py -v` -> 26 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Research/discovery/identity.py tldw_Server_API/app/core/Research/discovery/oa.py -f json -o /tmp/bandit_research_discovery_task2_controller_path_false_positive.json` -> 0 findings. Spec review and final code-quality re-review passed.

Task 3 complete. Implemented the provider router and first-slice adapters through commits 0fe89108da, ad12d13b31, ff460882ac, 9e0c969120, 9c411abad0, 8413b2f607, 93bafd4180, and cb99b27808. Review-driven fixes added sanitized `DiscoveryProviderError` handling, `internal_error` separation for adapter bugs, timeout continuation warnings, stricter malformed provider payload rejection, top-level error-payload precedence, and router validation for malformed direct adapter return values. Targeted verification: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Research/test_research_discovery_router.py tldw_Server_API/tests/Research/test_research_discovery_adapters.py tldw_Server_API/tests/Research/test_research_provider_adapters.py -v` -> 34 passed, 7 warnings. Bandit: `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Research/discovery/router.py tldw_Server_API/app/core/Research/discovery/adapters.py -f json -o /tmp/bandit_research_discovery_task3_controller_none_payload.json` -> 0 findings. Spec and quality reviews passed through the review-fix loop until the final subagent review attempt hit the Codex usage limit; controller-side final review verified the latest `None` payload fix and scoped checks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
