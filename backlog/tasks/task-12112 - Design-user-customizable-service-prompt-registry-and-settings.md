---
id: TASK-12112
title: Design user-customizable service prompt registry and settings
status: In Progress
labels:
- prompts
- design
- webui
- browser-extension
- backend
references:
- TASK-2341
documentation:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
modified_files:
- Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a governed, per-user Service Prompt Registry that exposes a curated allowlist of backend-owned content-generation prompts through a dedicated shared WebUI/browser-extension settings page. The design must preserve deployment defaults and explicit request overrides, provide strict variable validation, preview, revisions, background-job pinning, context-integrity enforcement, and broad content-facing service migration while keeping reusable Prompt Library, Prompt Studio, security, routing, judge, and machine-protocol prompts separate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document the Service Prompt Registry architecture, ownership boundaries, precedence, persistence, revision, and job-pinning contracts.
- [x] #2 Define the authenticated API and shared WebUI/browser-extension settings experience, including validation, preview, comparison, reset, history, conflicts, capability states, and responsive behavior.
- [x] #3 Define the allowlist eligibility policy, context-integrity behavior, privacy/security safeguards, failure semantics, deployment-default compatibility, and explicit exclusions.
- [x] #4 Define a broad content-facing migration inventory strategy, reviewable rollout slices, verification matrix, and measurable completion gates.
- [ ] #5 Review the written design through the required independent spec-document-reviewer loop and record the approved spec.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Design spec written at Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md and initially committed in 5d55b88cee. Independent spec review round 1 found four planning blockers covering context-integrity approval, multi-part explicit overrides/operator bypass, deployment-default failures, and multi-prompt job pinning. The spec was revised to resolve all four. Review round 2 approved the spec with no blocking issues. Verification: git diff --check passed for the documentation changes; no code or runtime configuration changed. Bandit and runtime test suites were skipped as not applicable to this docs-only design task. No known blockers remain.
Reopened at the human review gate for an additional implementation-risk audit requested by the user. Audit focus: hidden server-default confidentiality in full-bundle execution snapshots, approval-time schema/default races, catalog ETag invalidation, reset safety when defaults are unavailable, explicit request snapshot trust/retention, tenant-scoped deduplication, and broad-release launch semantics.
Independent review pass 3 found two remaining blockers: execution artifacts lacked a cryptographic authenticator independent of their storage, and backup import did not deterministically reconcile exported active/pending revisions with the one-current-pending invariant. The spec now requires externally anchored authenticated component/pin/binding envelopes with verification-key retention, and imports all revisions as unapproved history with no active/pending pointers until owner preview/resubmission. The three-pass independent review cap is reached, so the amended spec is returned to the human requester for final approval rather than starting an unbounded fourth automated review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Designed a governed per-user Service Prompt Registry and shared WebUI/browser-extension settings experience. The approved spec defines curated eligibility, atomic multi-part definitions, per-part explicit override semantics, strict deployment-default failure behavior, versioned pending revisions with explicit context-integrity operator approval, deterministic preview, history/reset/restore, atomic full-bundle job pin sets, API/capability contracts, responsive UI states, security/privacy boundaries, broad domain rollout slices, and verification gates.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
