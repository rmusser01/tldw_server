---
id: TASK-12972
title: Allow trusted configured local LLM endpoints through scoped egress policy
status: In Progress
assignee: []
created_date: '2026-07-15 19:34'
updated_date: '2026-07-15 21:22'
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
