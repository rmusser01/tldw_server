---
id: TASK-13460
title: Use provider-authoritative commercial model inventories
status: In Progress
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove static pricing/config availability and reject retired cloud model selections using current provider model lists. Private beta blocker, current dev 75ab224081. No live generation or secret changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Only current provider IDs are selectable and callable; pricing remains historical metadata.
- [x] #2 Discovery is bounded, credential scoped and fails closed without generation probes.
- [x] #3 Regression tests and upstream review evidence recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Read-only root cause: llm_providers.py merges pricing IDs; chat_service.py validates static IDs and fuzzy dated aliases. Worktree isolated from dirty UAT checkout.
Design/3-stage plan: Docs/Design/provider-authoritative-model-inventory-20261004.md. ADR check: no new ADR; bugfix within ADR025 adapter/trusted-endpoint boundary. Separate workers own discovery module/tests and shared Chat validation/tests; parent owns catalog/overrides integration. No live provider calls.
Regression evidence: catalog14 passed; catalog/readiness/override124 passed; existing catalog/filter/setup45 passed; multiprocessing rotation passed separately with local socket escalation. Fixed runtime-key precedence (red test reproduced parsed key instead of runtime key). No real provider/generation requests. Update legacy mocks from pricing inventory to provider discovery; disabled/unconfigured cloud IDs absent by design.
Exact current dev remains75ab224081 after fetch. Independent review started. New inventory+sharedselection+catalog combined198passed1blocked: sync HTTP response adapter lacked byte-limit support; required bounded transport passthrough under repair. Existing selected characterization36passed7obsolete alias assertions being updated (local behavior preserved). Hosted66-layer frozen baseline strict forward/fuzz0 applies; backport tracked separately TASK-14.4.71. Invalidencryptedoverride readiness regression confirmed red and fixed; unhealthy store aborts catalog rather than static/partial fallback.
Independentreview found OpenRouter accountfilteredlist requirement, fallbackcandidate bypass, and directadapter callers. Fixed/modelURL /models/user; sharedadaptercredential-binding nowenforcescurrentinventory fordirectspeech/summary/prompt/workflow callers (4sync/asyncboundarytests green); allcommercialasync adapterscurrently delegate toworkerthreads. Payloadoverride/fallbackselection under repair. RequiredsyncHTTPbytecap implemented existinghelper reused;330centralHTTPtests pass. Private encryptedcatalog/readiness33tests pass. Adapterpayloadunits isolateexternalinventory independently from newrealinventoryboundarytests;2RAG runtimecredentialpairing tests supplied mockcurrentinventory and14module tests pass.
Latest focused260pass before finalfallbacksuite; inventory/sharedselection106-worker followupgreen. Directadapterboundary5pass includes sync/async/defaultdeadline; catalog+setup81pass. Fulladapterunitsinitial645pass14fail:8 fixturesneedcurrentinventory/protectedHTTPtestconfiguration now34auditedpolicy+14RAG tests pass;6DNSerrors in unmodified localcustompaths passed on normalDNSrerun51tests. Finalall659adapterunits plus combinedHTTP/inventory tests running. Changed8productionfiles Bandit0findings0errors; scopedlintpassing, HTTPknownbaselinefindingsnotchanged. Sourcebytesstable independentre-reviewpending; privatepatchworkerassembling exactfrozen066stack.
Independent review identified native Anthropic Messages/count_tokens and Slides direct HTTP bypasses plus legacy Cohere/Moonshot config/key mismatch and slow-trickle discovery deadline. Closing these within the same current-model contract. Parent TDD native/legacy reproductions: 10 failed, 6 passed before fixes; HTTP and Slides delegated with disjoint file ownership. Stable earlier combined checks: 596 inventory/HTTP/Chat tests, 659 adapter units, 81 catalog regressions, 33 private-readiness backport tests; final tests and review still pending.
Final native/credential suite: 337 passed; absolute network deadline plus central HTTP checks: 502 passed; changed ten production paths Bandit: zero findings/errors. Integrated run initially 1327 passed/7 failed: catalog filter/tokenizer fixture endpoints inherited the Chat collection loopback configuration despite mocked discovery. Isolating those fixture endpoints to synthetic HTTPS; no production fallback or endpoint validation relaxed. Upstream dev refreshed and unchanged at 75ab224081.
Final exact-source integrated Python3.12 suite: 1477 passed, 300 warnings, 99.78s (/private/tmp/provider-authoritative-final-integrated-green-20261004.log). Separate native Messages/accounting/snapshot suite:337passed. Scoped Ruff, compileall and git diff --check pass; HTTP contains seven unchanged baseline lint findings. Bandit ten touched production paths:0findings0errors. Independent reviewer comprehensive wired directHTTP/SDK generation audit found no remaining production findings;275 independent checks and6 real localhost deadline tests pass (three isolated snapshot cases DNS-blocked in reviewer harness, broader parent337 passed). No live external provider request or generation, no host/config/key changes. Remaining: publish dev PR and normal CI/Qodo/merge; hosted backport/rollout is separateTASK-14.4.71.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Replaced pricing/config-derived commercial availability with credential/endpoint-scoped provider inventories; exact model validation covers catalog, shared Chat/direct adapters, native Anthropic Messages/count_tokens and Slides. OpenRouter account-filtered IDs and fallback candidates verified. Discovery has bounded bodies/pagination/cache and cancellable absolute network deadlines; unavailable/unsupported listings fail closed. Pricing metadata/local provider behavior retained. Python3.12 regressions, security scan and independent review recorded; source not yet deployed.
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
