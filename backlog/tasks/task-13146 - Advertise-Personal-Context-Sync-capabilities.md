---
id: TASK-13146
title: Advertise Personal Context Sync capabilities
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 18:44'
updated_date: '2026-08-30 19:36'
labels:
  - personal-context
  - sync
  - security
dependencies:
  - TASK-13145
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/Design/2026-08-30-personal-context-profile-server-design.md
  - IMPLEMENTATION_PLAN_personal_context_sync_capabilities.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Advertise a complete typed Personal Context capability contract through Sync v2 so Chatbook can enable profile linking only when schema, integrity, cleanup, purge, authorization, key custody, and quota requirements are satisfied.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 advertises all five Personal Context domains with upsert and tombstone operation support.
- [x] #2 The capabilities response includes the exact typed schema, integrity, wrapped-bootstrap, cleanup-acknowledgment, purge-generation, and quota contract.
- [x] #3 Personal Context readiness requires server_trusted_v1 authorization and valid server profile key configuration, with stable blockers when unavailable.
- [x] #4 Existing Sync domains and clients remain compatible when Personal Context is unavailable.
- [x] #5 Targeted Sync model, service, endpoint, static, security, diff, and independent review gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Sync v2 model, service, and endpoint tests for Personal Context domains, operation maps, typed capabilities, authorization policy, key availability, and quota values.
2. Extend the server Sync v2 protocol models and capability response with the exact Personal Context contract.
3. Derive availability from Shared Core schema support and profile master-key configuration while preserving existing Sync behavior.
4. Run targeted Sync regressions, static/security gates, independent review, update documentation/task evidence, and commit.

ADR required: no (existing)
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: ADR-002 already governs the server authority, Personal Context Sync contract, integrity, cleanup acknowledgments, and purge generations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added all five Personal Context Sync v2 domains with exact upsert/tombstone operation maps and a typed schema-v1 capability contract covering HMAC integrity, wrapped bootstrap, cleanup acknowledgments, purge generations, authorization policy, and bounded quotas.
- Derived availability from Shared Core schema compatibility, a valid 32-byte profile master key, server-trusted encryption readiness, and complete v1 adapter/materializer registration. Production fails closed with stable key, schema, authorization, or transport blockers until every dependency is available; unrelated Sync domains remain usable.
- Restricted the five domains to complete personal-dataset enrollment and only advertises writable adapter versions after both capability readiness and dataset enrollment permit them.
- Verification: 356 targeted Sync model/service/endpoint tests passed; Ruff, Python compilation, Bandit, and `git diff --check` passed. Bandit emitted only existing parser/noqa warnings around previously reviewed dynamic SQL sites.
- Independent review found and verified fixes for premature transport availability and writable-version advertisement; final re-review returned CLEAN. The full repository suite was not run under the repository's targeted-test policy.
- ADR required: no. Existing ADR `backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md` governs the implemented contract.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Sync v2 now advertises the complete Personal Context contract without claiming the feature is usable before its key, authorization, schema, adapters, and materializers are ready. All scoped verification and review gates passed with no known implementation blockers.
<!-- SECTION:FINAL_SUMMARY:END -->
