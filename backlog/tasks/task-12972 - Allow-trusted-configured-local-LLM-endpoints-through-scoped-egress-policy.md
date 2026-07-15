---
id: TASK-12972
title: Allow trusted configured local LLM endpoints through scoped egress policy
status: To Do
assignee: []
created_date: '2026-07-15 19:34'
updated_date: '2026-07-15 20:01'
labels:
  - llm-providers
  - security
  - egress
  - local-models
  - webui
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
Fix the mismatch where first-run setup accepts LAN-hosted llama.cpp and third-party local LLM endpoints but provider readiness, model discovery, and chat dispatch apply the global egress policy and hide or block them. Add one exact-origin policy for trusted operator/admin-configured local-provider endpoints while preserving the default SSRF posture for all other outbound traffic. Cover setup validation, readiness, model discovery, non-streaming chat, and streaming chat; preserve manual model fallback when discovery is unavailable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Configured local LLM endpoints on loopback, RFC1918, IPv6 ULA, and approved local-overlay addresses can use their exact configured port without disabling global private-network blocking or globally opening that port.
- [ ] #2 Only guarded setup and server-owned provider configuration can mint the exact-origin scope; request-level URLs, BYOK values, adapter final URLs, and non-custom public-service adapters remain on the default policy.
- [ ] #3 Setup validation, provider readiness, typed model discovery, non-streaming chat, and streaming chat use the same central scoped policy and return stable blocked, unreachable, and unsupported outcomes.
- [ ] #4 The policy rejects malformed URLs, URL userinfo, link-local and authoritative cloud-metadata targets, multicast/unspecified/reserved targets, global denylist matches, DNS resolution changes, and redirects outside the configured origin.
- [ ] #5 Explicit model and probe behavior follows the documented readiness matrix; supported manual models remain catalog-visible when optional discovery is unsupported, while requested-probe reachability and policy failures remain actionable.
- [ ] #6 Focused backend and frontend tests cover every network-backed local provider path, custom OpenAI provenance, sync and async streaming, LAN/Docker/overlay addresses, nonstandard ports, dangerous targets, redirects, request override rejection, discovery, and readiness.
- [ ] #7 Configuration and user documentation remove global private blocking as the recommended local-provider workaround and explain compatibility behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/Plans/IMPLEMENTATION_PLAN_scoped_local_llm_egress_TASK_12972.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-07-15 planning review: confirmed setup/runtime egress policy drift, an unchecked raw local streaming path, and runtime catalog mappings that omit setup-owned manual model fields for llama.cpp/Kobold/Ooba/Tabby. The plan uses an exact configured-origin scope, resolved-address classification, global denylist precedence, checked sync/async/stream transports, and no global private-network or port relaxation. No production WebUI change is planned unless the existing selector contract test fails.

2026-07-15 independent review found and corrected five planning gaps: trusted-scope provenance was undefined; metadata targets were only examples; DNS/policy/reachability outcomes were ambiguous; readiness/manual-model behavior lacked a state matrix; and custom OpenAI plus async streaming paths were omitted. The revised design carries scope separately from URLs, defines an authoritative metadata set and reason codes, adds typed discovery outcomes, enumerates all local-provider paths, and expands transport/provenance tests.

2026-07-15 final review: independent spec and implementation-plan reviewers approved the revised documents with no blocking issues. Advisory improvements were incorporated: runtime summaries now include auth/server discovery failures, and custom-adapter no-scope calls explicitly stay on centralized checked transports with the default policy. Implementation is gated on requester approval of the draft design.

2026-07-15 planning-artifact verification: the 48-test focused Security, Setup, readiness, and local-streaming baseline passed (48 passed, 4 warnings in 14.65s). Staged diff whitespace validation passed after cleanup. Bandit is not applicable to this planning commit because it changes Markdown task/design/plan files only; implementation-stage Bandit remains required by Stage 5.
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
