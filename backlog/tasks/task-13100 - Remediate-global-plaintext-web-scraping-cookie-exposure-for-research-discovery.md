---
id: TASK-13100
title: Remediate global plaintext web-scraping cookie exposure for research discovery
status: To Do
assignee: []
created_date: 2026-08-21 19:36
labels:
- security
- web-scraping
- authnz
- multi-user
- research-discovery
dependencies: []
references:
- TASK-12968
- TASK-12964
- TASK-13139
priority: high
documentation:
- Docs/superpowers/specs/2026-08-27-agent-native-web-research-quality-provenance-roadmap.md
updated_date: 2026-08-28 01:00
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Eliminate the existing global plaintext domain-cookie store and cross-user raw cookie management surface in web scraping. Replace ambient global cookie reuse with explicit owner-scoped encrypted references or disable the capability until that contract exists. This is an independent security remediation and a hard dependency for any future credentialed scraping or authenticated-browser program; credentialless discovery must prove it never touches these paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Ordinary authenticated users cannot read, enumerate, overwrite, or cause transmission of another user's cookies or authenticated session state.
- [ ] #2 Raw cookies and credential-bearing headers are removed from global plaintext files, job metadata, logs, API responses, cache keys, and shared singleton state; any retained capability uses encrypted owner-scoped opaque references.
- [ ] #3 Credential resolution is authorized for the requesting owner and bound to an explicit source origin set; cross-origin redirects strip credentials and no ambient cookie jar is consulted.
- [ ] #4 Existing plaintext cookie data receives an explicit migration, quarantine, or secure-deletion procedure with documented compatibility behavior and no silent reuse.
- [ ] #5 Multi-user isolation, single-user compatibility, raw endpoint authorization, job/artifact redaction, redirect behavior, and user-A/user-B regression tests pass.
- [ ] #6 Authenticated and browser retrieval tasks declare this task as a blocking dependency; the credentialless shared-discovery executor has dependency-boundary tests proving zero access to the remediated or legacy cookie mechanisms.
- [ ] #7 Focused tests, API contract checks, lint/format checks, Bandit, and diff hygiene pass, with operational upgrade notes for affected deployments.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory and remove ordinary-user raw cookie read/list/write endpoints and global CookieManager call sites.
2. Disable persistent web-scraping cookies in multi-user mode; retain only explicit request-scoped cookies where an approved current workflow requires them.
3. Quarantine then securely remove or administratively retire the legacy plaintext cookie file with documented upgrade behavior.
4. Prove credentialless discovery, MCP fetch, caches, logs, Jobs metadata, and browser gates never consult ambient cookie state.
5. Defer an encrypted owner-scoped cookie vault until a concrete authenticated-retrieval use case is separately designed and approved.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Created as the research-discovery replacement for the ambiguous active TASK-12969 record. The superseded discovery record is archived at `backlog/archive/tasks/task-12969 - Remediate-global-plaintext-web-scraping-cookie-exposure.md` after every discovery-specific reference was migrated. The replacement was rekeyed from the historical branch-local TASK-13013 allocation to TASK-13100 after the latest-dev rebase exposed unrelated active TASK-13013 claimants; historical commit subjects remain unchanged.
2026-08-27 program reconciliation (TASK-13139): choose the minimal safe first release. Remove raw cookie-management endpoints, disable persistent cookies in multi-user deployments, allow only explicit request-scoped cookies on already-approved paths, and quarantine/retire the legacy plaintext file. An encrypted owner-scoped cookie vault is deliberately deferred until a real authenticated-retrieval requirement exists; do not build one speculatively.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
