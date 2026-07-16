---
id: TASK-12972
title: Allow trusted configured local LLM endpoints through scoped egress policy
status: In Progress
assignee: []
created_date: '2026-07-15 19:34'
updated_date: '2026-07-16 01:40'
labels:
  - browser-extension
dependencies: []
references:
  - TASK-605
  - TASK-12020.29
  - Docs/ADR/025-llm-provider-adapter-routing-and-overrides.md
  - Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md
documentation:
  - Docs/superpowers/specs/2026-07-15-configured-local-llm-egress-design.md
  - Docs/ADR/030-configured-local-llm-egress-policy.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the mismatch where guarded setup accepts LAN-hosted llama.cpp and third-party local LLM endpoints but readiness, discovery, adapter dispatch, and the WebUI model cache can hide or block them. Add one exact-origin scope for trusted current server configuration while preserving global SSRF defaults. Resolve paired endpoint/scope context at the configured-local adapter boundary so Chat and every direct registry caller are covered; also cover guarded setup, one-shot typed discovery, TLS/DNS pinning, manual models, and WebUI cache invalidation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Configured local LLM endpoints on loopback, RFC1918, IPv6 ULA, approved CGNAT/overlay, and ordinary public-unicast addresses can use only their exact configured origin and port without disabling global private blocking or opening a global port.
- [ ] #2 Only the authorized setup route and a fresh server-config/environment resolver create scope; reserved request fields, request app config, URL/BYOK overrides, and adapter final URLs cannot create or widen it.
- [ ] #3 fetch, afetch, synchronous stream_response, retries, redirects, DNS pins, and TLS certificate pinning retain the same scope, accepted DNS set, and machine-readable EgressPolicyError reason code.
- [ ] #4 The scoped address classifier denies metadata and all non-allowed special-use classes, honors the global denylist, rejects DNS changes and origin-changing redirects, and leaves no-scope egress behavior unchanged.
- [ ] #5 Setup creates scope only after its write guard; canonical setup readiness fields are used, discovery runs at most once per provider, transient failures are not cached, and explicit/manual model behavior follows the documented readiness matrix.
- [ ] #6 Focused backend/frontend tests, Bandit on touched Python paths, ADR/configuration/user documentation, and migration guidance pass without adding a global SSRF exception.
- [ ] #7 Every registered configured-local adapter wrapper and configured custom OpenAI alias resolves trusted context at its common adapter boundary, so Chat and all direct registry callers use the checked scoped path; numbered custom slots gain transport coverage without new catalog/setup surfaces.
- [ ] #8 The exact enabled and blocked backend metadata records drive WebUI selection correctly; saved-provider events generation-invalidate both cache layers, and pre-save in-flight responses cannot repopulate stale models.
- [ ] #9 Shared model discovery, visibility, and cache invalidation behavior is verified in both the Next.js WebUI and packaged browser extension surfaces.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_scoped_local_llm_egress_TASK_12972.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-15 initial investigation confirmed setup/runtime egress drift, unchecked raw local streaming, and missing runtime manual-model mappings. A 48-test focused baseline passed with 48 passed and 4 warnings.

2026-07-15 first planning review defined exact-origin scope provenance, an authoritative metadata deny set, typed discovery outcomes, a readiness matrix, custom OpenAI coverage, and checked transport propagation. The planning-only commit changed Markdown artifacts; Bandit was deferred to implementation.

2026-07-15 second security/feasibility review found additional blockers: caller-supplied reserved context, direct adapter callers outside Chat, loss of reason codes at EgressPolicyError, unscoped TLS pin checks, stale import-time configuration, incomplete special-use classification, duplicate discovery, stale negative discovery cache, canonical Kobold readiness-key drift, and persistent WebUI model cache. The design and plan were revised to address all of them, remove unused async byte/SSE scope propagation, and keep each implementation stage green. No implementation code has started. The post-review design and plan await requester approval.

2026-07-15 final plan review found that enumerating only three direct callers was incomplete and that certificate-pinning errors could still lose reason_code during network normalization. The plan now resolves trusted context at the common configured-local adapter boundary, structurally covering all registry callers, and adds explicit certificate-pin enforcement/denial tests with typed error preservation.

2026-07-15 final spec review identified an in-flight cache race: clearing references did not stop a pre-save request from repopulating stale data. The design now uses one monotonic invalidation generation per existing model-cache layer and promise-ownership guards; no new cache or event is introduced.

2026-07-15 final post-correction review: independent spec and implementation-plan reviewers approved the current artifacts with no blocking issues or advisory recommendations. TASK-12972 remains To Do and implementation remains gated on requester approval of this corrected design/plan.

2026-07-15 requester approved implementation and clarified that the fix must apply to both the WebUI and browser extension. The implementation remains shared in apps/packages/ui, with explicit verification through both frontend test configurations.

2026-07-15 Stage 5 implementation is ready for independent review but remains In Progress. Frontend TDD RED: 4 failures/36 passes for stale inner persistence, missing inner invalidation, stale outer repopulation, and missing saved event; an additional config-lookup ownership race failed 1/1 before its guard. GREEN: the shared WebUI suite and browser-extension config each pass 43/43. The Stages 1-4 backend union passes 492 tests with 5 warnings. Bandit scanned all touched Python production paths (25,013 LOC) with 0 findings and 0 errors; py_compile passed. Pinned ESLint reported 0 errors (baseline warnings only), and the 8 GB shared-package tsc run reported no diagnostics in touched paths while retaining unrelated repository baseline failures. Ruff passed after excluding only audited pre-existing rule codes. git diff --check passed. Live LAN UAT was skipped because no safe external local-model server/route was available. ADR-030 and LAN/Docker/Tailscale troubleshooting docs were added. TASK-12972.1 tracks the intentionally deferred Novita/Poe/Together checked-egress migration.

Stage 5 touched paths: apps/packages/ui/src/services/tldw/TldwModels.ts; apps/packages/ui/src/services/tldw-server.ts; apps/packages/ui/src/services/tldw/domains/setup-onboarding.ts; their focused tests; apps/packages/ui/src/routes/__tests__/option-quick-chat-popout.test.tsx; Docs/ADR/030-configured-local-llm-egress-policy.md and Docs/ADR/README.md; tldw_Server_API/Config_Files/README.md; Docs/User_Guides/Integrations_Experiments/Setting_up_a_local_LLM.md; Docs/Plans/IMPLEMENTATION_PLAN_scoped_local_llm_egress_TASK_12972.md; and follow-up backlog task TASK-12972.1.

2026-07-15 Stage 5 post-review correction: distinct OLD/NEW scope regressions failed 2/2 before the fix. Stale getModels request A could invalidate newer in-flight request B after old getConfig resolved, and stale getCachedChatModels scope reconciliation could clear B. A single generation-aware scope helper now prevents stale generations from mutating or adopting current scope while allowing a current-generation scope change to own its resulting invalidation generation. Focused GREEN passed 2/2; complete shared WebUI and extension suites each pass 45/45. Pinned ESLint has 0 errors (2 existing warnings in the correction file), and the 8 GB package tsc run reports no TldwModels diagnostics while retaining unrelated baseline failures. The backend remains unchanged; prior 492-test backend and zero-finding Bandit evidence remains valid. TASK-12972 and Stage 5 remain In Progress pending re-review.

2026-07-15 Stage 5 cross-context cache correction: strict RED failed 5 tests with 30 passing under both WebUI and extension Vitest configurations. The existing tldwModelsCache record now carries a unique tombstone token; source, other WebUI tabs, extension sidepanel, and no-window/background contexts apply each token once through the existing Plasmo storage watch, invalidate the inner cache, and notify the outer cache subscription. Source applies synchronously and ignores its watch echo. Serialized writes preserve a fresh post-clear model record after the tombstone; startup tombstones never hydrate models. Window config events and stored-config changes now request this single tokenized path. Focused GREEN passed 35/35; complete shared suites passed 51/51 in both configs. Pinned ESLint passed with 0 errors and 11 pre-existing warnings; full frontend TypeScript passed with incremental output disabled. Backend is unchanged, so prior 492-test and zero-finding Bandit evidence remains valid. Stage 5 and TASK-12972 remain In Progress pending re-review.

2026-07-15 verification evidence correction: the focused cache suite passed 35/35. The exact planned four-file complete shared command from Stage 5 Task 5.1 passed 50/50 under the WebUI configuration and 50/50 under the extension configuration; the preceding 51/51 count included one additional chat-model test outside that planned command. No implementation or test result changed. TASK-12972 and Stage 5 remain In Progress.

2026-07-15 final Stage 5 cache-race correction: strict focused RED failed 3 tests with 34 passing under both WebUI and extension configurations. A delayed clear-A echo after clear B replayed A and invalidated B ownership; clear-before-first-hydration returned a seeded stale record without reaching backend; and the outer cache independently accepted A/B/A replay. Inner seen tokens, locally pending tokens, and outer seen tokens now use bounded 64-entry insertion-order histories while lastAppliedInvalidationToken remains the active tombstone guard. A clear before hydration starts consumes the initial hydration slot; in-flight hydration retains the generation guard. Focused GREEN passed 37/37 in both configs. The exact planned four-file suites passed 52/52 WebUI and 52/52 extension. Full frontend TypeScript passed; pinned ESLint passed with 0 errors and 11 pre-existing warnings. Backend remains unchanged, so prior 492-test and zero-finding Bandit evidence remains valid. TASK-12972 and Stage 5 remain In Progress pending re-review.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
