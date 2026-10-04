---
id: TASK-13450
title: Honor encrypted server provider overrides in catalog readiness
status: Done
labels:
- bug
- llm
- security
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore catalog readiness for enabled server-wide encrypted provider API-key overrides using the existing credential fallback and provider_readiness semantics. Keep principal-scoped BYOK out of the server catalog, preserve unsupported/unhealthy/local endpoint behavior, and never expose credentials or issue external provider calls during verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A regression test first reproduces enabled encrypted DeepSeek override remaining not configured, then passes with deepseek-chat visible.
- [x] #2 Common server fallback resolution and provider_readiness preserve unsupported, unhealthy, disabled, invalid-credential, and local endpoint behavior without keys in responses.
- [x] #3 Focused offline tests, lint, Bandit, ADR assessment, and reviewed commit/diff are recorded; no PR or merge.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Base: latest fetched origin/dev c95e41fc62a07e55fd74052023826c71b2a2c789 in /private/tmp/tldw-upstream-provider-catalog-20261004. Bounded defect restoration under the existing approved provider setup. Parent coordinates integration/review and will publish/merge under applicable upstream authorization; this worktree delivers a tested reviewed commit and does not open a PR or merge independently. ADR required: no. Governing ADR: Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md; this restores catalog consistency with existing server-wide override fallback without changing endpoint trust, adapter routing, public credential exposure, or principal ownership. Plan: trace common fallback and readiness boundaries; add and run offline red regression tests; apply minimal fix; run adjacent coverage/lint/Bandit and review; commit code/tests/task together. Anticipated production path: tldw_Server_API/app/api/v1/endpoints/llm_providers.py. Regression coverage: tldw_Server_API/tests/AuthNZ_Unit/test_llm_provider_catalog_overrides.py.
Root cause: get_configured_providers computed commercial readiness only from static keys, before the listing policy merge. Fix captures a server-wide ProviderOverrideCallSnapshot and uses its existing server_fallback merge before provider_readiness; disabled policy is reduced via current availability. No principal-scoped BYOK resolution is invoked, and local endpoint configuration/discovery semantics are unchanged. Invalid credentials and unhealthy override storage fail closed rather than falling back to static keys.
TDD: initial offline regression run reproduced 8 behavioral failures (3 provider/catalog routes, 3 readiness barriers, disabled readiness, invalid override/static fallback) with 3 passing cases before the fix. A collection-only attempt with /dev/null as the exclusive env file failed because the loader requires a regular file; reran with a dedicated empty temporary environment file. Final focused suite: 16 passed. Adjacent initial run: 101 passed, 3 failed due to lightweight readiness fixtures bypassing app-startup override initialization. Added a scoped healthy-empty cache fixture with original state restoration in test_llm_providers_readiness.py; preserved the explicit unhealthy-store regression. Final adjacent run: 104 passed, 91 warnings, no failures (24.07s), including provider details/health and adapter capability/error/optional-metadata tests. All provider discovery in the new regression suite is blocked, and synthetic keys only are used.
Verification: Ruff check passes on all touched Python files; py_compile passes; git diff --check passes; Bandit 1.9.4 on the production endpoint reports 0 findings and 0 errors. New test file formatted with Ruff. Whole-file formatter check reports existing formatting debt in the catalog/readiness files; unrelated historical formatting was intentionally left unchanged. Self-review found no new P1/P2 issue in the scoped change. Parent reports independent Sagan review of code/deployment helper underway and owns live acceptance, immutable image build/backport, publication, and merge.
Final verified base: HEAD before fix commit and origin/dev both c95e41fc62a07e55fd74052023826c71b2a2c789. The clean worktree initially used bf8f2ad6a42ad6396376020876a5f6a709ec6b34 and was fast-forwarded after fetch before task/code edits. Baseline llm_providers.py SHA256 db1724bee1f4ca5d7137237a46c9528944660e13a0915beff166a5695af81f09 matches the parent-reported frozen deployed file. Only production path changed: tldw_Server_API/app/api/v1/endpoints/llm_providers.py. Also touched: new AuthNZ_Unit/test_llm_provider_catalog_overrides.py and the existing Chat_NEW/unit/test_llm_providers_readiness.py test fixture.
Independent Sagan review identified P2: a provider-specific invalid credential error from server_fallback reaches the outer error handler and empties the whole catalog. Reopened for a narrow follow-up: catch only invalid_provider_credentials within each commercial fallback call, clear the effective key without static fallback, preserve healthy peers, and retain global fail-closed capture/cache-unhealthy behavior. Strengthen the existing invalid-override regression with a healthy OpenAI peer and add coverage for store health loss during fallback.
Sagan P2 resolution verified: mixed invalid DeepSeek + healthy OpenAI regressions failed red for both static and encrypted-override peer keys because the entire catalog was empty. The invalid-provider assertion now requires an actual DeepSeek entry with is_configured=false, provider_enabled=false, availability=not-configured, while the peer remains configured/enabled. Catch only invalid_provider_credentials within server_fallback, clear the effective provider key, and do not emit the exception or credential values. Snapshot capture remains outside the catch; credential_store_unavailable from fallback is re-raised to retain global fail-closed behavior. Added a real ensure_healthy path test for store health loss between capture and fallback. This corrects the issue missed by the initial self-review and the earlier permissive empty-catalog assertion.
Follow-up verification: 106 adjacent tests passed, 91 warnings, no failures (23.73s). The focused suite now has 18 cases. Ruff check, py_compile, git diff --check, and Bandit all pass; Bandit has 0 findings/errors. Only production path remains tldw_Server_API/app/api/v1/endpoints/llm_providers.py. Production-only combined backport patch is regenerated from c95e41fc62a07e55fd74052023826c71b2a2c789, covering both the original readiness restoration and this peer-isolation fix. Parent retains deployment/helper deadline handling, independent review, live acceptance, publication and merge.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Restored encrypted server API-key override catalog readiness using existing fallback/readiness semantics. Invalid per-provider credentials disable only that provider without static fallback or suppressing healthy peers; global capture/store-health failures still fail closed. Added 18 offline regressions and scoped healthy-empty readiness fixtures. Final 106 adjacent tests pass; Ruff/compilation/diff checks and Bandit pass. Sagan P2 fixed with red-green evidence. ADR025 remains governing; only llm_providers.py changes in production. Parent owns backport/build, live acceptance, publication and merge.
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
