---
id: TASK-12749
title: Design MCP effective permission explain and preview surface
status: Done
assignee: []
created_date: '2026-06-17 03:25'
updated_date: '2026-06-17 03:28'
labels:
  - mcp
  - policy
  - design
  - observability
dependencies: []
documentation:
  - >-
    Docs/superpowers/specs/2026-06-16-mcp-effective-permission-explain-preview-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a read-only MCP policy explain and profile tool-preview surface. The design should cover the shared policy_explain service, admin API, local and remote CLI modes, dedicated admin permission seam, required audit behavior, redacted response contract, degraded states, bounds, and test strategy.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design spec exists under Docs/superpowers/specs with the approved scope and decisions recorded.
- [x] #2 Spec covers shared policy_explain service, admin API, CLI local/remote modes, authorization seam, audit requirements, redaction rules, degraded states, response contracts, preview semantics, tests, and rollout.
- [x] #3 Spec is reviewed for issues and updated before handoff to implementation planning.
- [x] #4 Doc-only validation is recorded, including diff checks and Bandit skip rationale.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created a design-only spec for the MCP effective permission explain and profile tool-preview surface. This task intentionally does not implement runtime code; implementation planning should wait for user approval of the reviewed spec.

Local spec review incorporated fixes: corrected the subject extraction module path, changed profile preview to POST to avoid leaking session data in URL logs, clarified static-policy-only is not inherently degraded, added evaluated_at/skipped_contributors response fields, added argument payload size caps, added CLI --args-json-file/--args-stdin guidance, removed an unrelated FastAPI initialize test note, and required audit failures to prevent successful policy-detail responses.

Second design-review pass incorporated fixes: documented the admin identity gap in the current API-key auth dependency, required a strict audit helper rather than existing best-effort audit helpers, required a public unfiltered admin catalog provider instead of private discovery internals, replaced subject-level boolean redaction with redaction_state, added a stable error envelope, and documented profile-id conflict validation for preview requests.

Verification after second review: git diff --check passed; placeholder-marker scan passed; non-ASCII scan passed. Bandit remains skipped because this is documentation and Backlog metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the MCP Effective Permission Explain And Preview design spec and task record. The spec covers the shared policy_explain service, admin API, local/remote CLI behavior, dedicated mcp.policy.explain authorization, fail-closed audit behavior, redacted response contracts, degraded state handling, preview completeness semantics, tests, and rollout slices. Review fixes were incorporated before handoff. No runtime code was changed.
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
