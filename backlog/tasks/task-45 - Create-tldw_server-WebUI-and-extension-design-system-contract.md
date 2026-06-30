---
id: TASK-45
title: Create tldw_server WebUI and extension design-system contract
status: Done
assignee: []
created_date: '2026-05-04 17:12'
updated_date: '2026-05-04 17:27'
labels:
  - design-system
  - webui
  - extension
  - docs
dependencies: []
references:
  - apps/DEVELOPMENT.md
  - apps/packages/ui/src
  - apps/tldw-frontend/tailwind.config.js
  - apps/packages/ui/src/assets/tailwind-shared.css
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a governance-first design-system contract for tldw_server that covers both the WebUI and browser extension from day one. The approved scope is a contract-first design, with Ant Design treated as an implementation substrate rather than the product design language, and setup/recovery/admin health as the first proof surface. The contract should name apps/packages/ui/src as the shared UI source of truth and define ownership, system layers, AntD policy, product state language, testing/enforcement, and staged rollout guidance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design-system contract document exists under Docs/Design and captures the approved scope for WebUI plus extension governance.
- [x] #2 The contract defines design-system layers for tokens, primitives, patterns, and product surfaces.
- [x] #3 The contract defines component ownership across apps/packages/ui/src/components/ui, components/Common, and feature-local component directories.
- [x] #4 The contract defines the AntD policy: AntD may remain as implementation substrate but tldw-owned wrappers and patterns own product semantics.
- [x] #5 The contract defines canonical setup/recovery/admin health states and the required user-facing structure for recovery and diagnostics.
- [x] #6 The contract defines testing and enforcement expectations for the governance-first v1 rollout.
- [x] #7 The Backlog task is updated with the created document path and verification notes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved governance-first design-system scope in Docs/Design/tldw_web_design_system_contract.md.
2. Ground the contract in current repo anchors: apps/packages/ui/src, tailwind-shared.css, tldw-frontend Tailwind config, components/ui, components/Common, and apps/DEVELOPMENT.md.
3. Define system layers, component ownership, AntD policy, canonical state language, recovery pattern, accessibility rules, testing/enforcement, rollout plan, non-goals, and success criteria.
4. Run document-focused verification and review the spec for scope gaps before finalizing.
5. Update acceptance criteria and final task notes with verification results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created initial contract document at Docs/Design/tldw_web_design_system_contract.md from the approved brainstorming decisions. This is documentation-only work; no runtime code was touched.

Spec review loop completed. First review found three issues: incomplete state contract table, ambiguous tailwind.css/tailwind-shared.css and Button ownership anchors, and overly broad v1 primitive/pattern commitments. The document was revised to address all three. Second review returned APPROVED with no remaining blockers before user review.

Verification: inspected the document sections with rg, reviewed the beginning and tail of the file with sed, checked the file length with wc, and ran an awk line-length scan. Long lines are limited to the canonical state Markdown table. No runtime tests were run because this is documentation-only. Bandit is skipped because no code paths were touched.

Commit attempt was blocked by the pre-existing repository state: `git commit --only ...` failed with `fatal: cannot do a partial commit during a merge.` The new design-system contract and this Backlog task are staged, but no commit was created in this turn because unrelated merge/index state must be resolved first.

Reopened briefly to address post-review spec polish requested by the user: pin the v1 proof surface to concrete screens/files, define state token defaults as aliases to existing color tokens, and clarify Button migration timing.

Addressed post-review findings in the contract: added v1 state-token alias guidance to the existing color tokens, clarified that Button should not be migrated wholesale during the proof-surface slice, and pinned the v1 proof surface to concrete recovery/setup/admin health files while marking other admin routes as later migration candidates. Verification rerun: rg checks confirmed the added sections, git diff --check passed for the doc and task file, and awk line-length scan still only reports the canonical state Markdown table rows.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and refined Docs/Design/tldw_web_design_system_contract.md as the governance-first design-system contract for tldw_server WebUI and browser extension. The contract records the approved scope: apps/packages/ui/src as the shared UI authority, Ant Design as an implementation substrate, tldw-owned wrappers and patterns as the product design language, and setup/recovery/admin health as the v1 proof surface.

The document defines the system layers, component ownership model, AntD policy, canonical state language, state-token alias guidance, recovery pattern, content and accessibility rules, testing/enforcement expectations, concrete v1 proof-surface file boundaries, rollout plan, non-goals, and success criteria. A spec review loop found and then verified fixes for the state contract, asset/Button ownership anchors, and proof-surface scoping. A follow-up review pass added clearer token aliasing, Button migration timing, and concrete screen/file boundaries.

Verification was documentation-focused. Runtime tests and Bandit were skipped because no code paths were touched. Commit creation remains blocked by the pre-existing merge/index state (`fatal: cannot do a partial commit during a merge.`), so the files remain staged for later commit after the merge state is resolved.
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
