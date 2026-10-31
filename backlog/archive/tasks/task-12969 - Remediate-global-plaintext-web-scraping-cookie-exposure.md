---
id: TASK-12969
title: Remediate global plaintext web-scraping cookie exposure
status: To Do
assignee: []
created_date: ''
updated_date: '2026-08-21 20:37'
labels:
  - security
  - web-scraping
  - authnz
  - multi-user
dependencies: []
references:
  - TASK-12968
  - TASK-12964
priority: high
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-08-21: Superseded by research-discovery security prerequisite TASK-13100 after the active TASK-12969 ID collision was confirmed. The initial branch-local TASK-13013 replacement was rekeyed after the latest-dev rebase exposed unrelated active claimants. Discovery documents and inventory follow-up references were migrated without changing scope. This duplicate record is retained in archive for history.
<!-- SECTION:NOTES:END -->

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
