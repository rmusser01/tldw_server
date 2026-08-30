---
id: TASK-13145
title: Expose authenticated Personal Context API
status: Done
assignee:
  - '@codex'
created_date: '2026-08-30 17:37'
updated_date: '2026-08-30 18:37'
labels:
  - personal-context
  - backend
  - api
  - security
dependencies:
  - TASK-13144
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
documentation:
  - Docs/Design/2026-08-30-personal-context-profile-server-design.md
  - Docs/API-related/Personal_Context_API.md
  - backlog/docs/lessons-testing-evidence.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the canonical encrypted Personal Context repository through a single authenticated per-user service and bounded API so Chatbook and tldw_server can operate on the same record contract without cross-user disclosure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Authenticated users can read and mutate only their own Personal Context profile, and cross-user opaque IDs return the same not-found response without decryption.
- [x] #2 Status, manifest, scope, record, proposal, runtime, export, and purge endpoints use strict bounded schemas and stable machine-readable errors.
- [x] #3 All canonical mutations flow through one service with optimistic version checks, semantic-key collision checks, and profile lifecycle enforcement.
- [x] #4 Proposal review, runtime enablement, recovery/plaintext export, and purge confirmation/generation follow the approved Personal Context semantics.
- [x] #5 Targeted service, endpoint, authentication-boundary, and existing Personalization regressions pass.
- [x] #6 Ruff, formatting, compilation, Bandit, diff hygiene, and independent review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Write RED service, endpoint, and authentication-boundary tests against existing auth, router, and error conventions.
2. Implement the single PersonalContextService plus encrypted runtime policy and export/purge semantics over TASK-13144.
3. Add authenticated per-user dependencies, strict request/response schemas, bounded routes, and stable error mapping in the minimal/content router groups.
4. Run targeted Personal Context and existing Personalization regressions, static/security checks, independent review, and commit.

ADR required: yes (existing)
ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
Reason: The existing ADR already governs authenticated service ownership, server authority, encryption, and cross-application contract semantics; no new architectural decision is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one authenticated PersonalContextService over the encrypted repository, strict API schemas/routes, server-local runtime policy, explicit exports and purge barriers, workspace ownership checks, optimistic mutations, proposal review and bounded receipt retention. Hardened purge and expiry races inside BEGIN IMMEDIATE transactions, enforced record lifecycle and immutable kind/scope authority, added storage caps and proposal pagination, and documented the contract. Verified 264 targeted Personal Context, existing Personalization, auth-boundary, and router-contract tests; Ruff format/check, compileall, Bandit, git diff hygiene, and independent review all passed. ADR 002 applies; no new ADR was needed. Per repository policy, the full test suite was not run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Authenticated Personal Context API is complete with encrypted per-user profile access, bounded record and proposal operations, stable errors, runtime/export/purge controls, transaction-safe lifecycle fences, documentation, and targeted regression coverage.
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
