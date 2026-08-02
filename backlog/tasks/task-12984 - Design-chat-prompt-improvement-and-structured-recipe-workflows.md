---
id: TASK-12984
title: Design chat prompt improvement and structured recipe workflows
status: Done
assignee: []
created_date: '2026-07-22 22:20'
updated_date: '2026-08-01 22:22'
labels:
  - design
  - webui
  - browser-extension
  - prompts
dependencies: []
references:
  - 'https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_prompting_guide'
documentation:
  - Docs/superpowers/specs/2026-07-22-chat-prompt-improvement-recipes-design.md
  - Docs/superpowers/plans/2026-08-01-chat-prompt-improvement.md
  - Docs/superpowers/plans/2026-08-01-single-text-structured-recipes.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Document the approved design for an Improve prompt workflow in /chat across the WebUI and browser extension. The design covers independent current-system-prompt and unsent-message targets, active-model rewriting, review and Undo behavior, provider-neutral backend contracts, and reusable single-field structured recipes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Design covers Improve now and Review changes for current system-prompt and unsent-message drafts without cross-context.
- [x] #2 Design defines safe active-model routing, response validation, preservation invariants, error recovery, and privacy boundaries.
- [x] #3 Design defines target-specific single-field recipe schema, rendering, persistence, compatibility, and recipe identity.
- [x] #4 Design includes WebUI and browser-extension accessibility, testing, evaluation, and rollout requirements.
- [x] #5 Approved design is written under Docs/superpowers/specs and committed with the Backlog task.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Write the approved interaction, architecture, model contract, error handling, and test/rollout decisions into one parent design spec. Decompose future implementation into coordinated prompt-improvement and structured-recipe tracks. Self-review the spec for placeholders, contradictions, ambiguity, and scope. Record verification and commit the spec with this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design spec written at Docs/superpowers/specs/2026-07-22-chat-prompt-improvement-recipes-design.md.

Self-review completed: scanned for unresolved markers, reviewed all sections for contradictory product/API/state/privacy behavior, and tightened empty-draft behavior, post-apply inspection, client-side preservation checks, preservation error semantics, and external-provider privacy boundaries.

Verification: Markdown heading inventory and unresolved-marker scan passed. Runtime tests were not run because this checkpoint changes documentation and Backlog metadata only. Bandit skipped because no executable code is touched. No known blockers; final approval and task completion remain pending user review of the written spec.

2026-08-01 adversarial review: rechecked the current PromptSelect override behavior and structured-prompt v1 implementation, and re-fetched the referenced OpenAI GPT-5 prompting guide. Amended the spec to preserve selected-template override/reset semantics; prevent stale review application; distinguish Auto from a resolved provider route; bound and validate protected-token side-channel data; separate chat authorization from prompt-library administration; define exact no-change response behavior; harden XML rendering after variable substitution; distinguish capability support from permission; remove schema-aware recipe search from v1; and require separate Track A/Track B plans.

Verification: scoped git diff check and decision/placeholder scans passed. Runtime tests and Bandit remain inapplicable because this pass changes only the design spec and Backlog metadata. The task remains In Progress pending user approval of the amended written spec.

User approved the design on 2026-08-01. Added separate dependency-ordered implementation plans and Backlog children TASK-12984.1 and TASK-12984.2. Final planning review aligned the implementation contracts with the approved operation_id/model_selection/protected_tokens API, adaptive non-stacked UI, exact Undo lifetime, dedicated rate limiting, fail-closed capabilities, schema-v2 rendered_text preview, and old/offline recipe persistence behavior. Verification for this documentation-only checkpoint: plan structure/contract scans and git diff checks; runtime tests and Bandit are not applicable because no executable code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved the chat prompt-improvement and structured-recipe design, then decomposed implementation into two reviewable tracks. TASK-12984.1 covers provider-neutral active-model Improve now and Review changes with preservation checks, stale-result protection, diff/findings, exact Undo, and WebUI/extension parity. TASK-12984.2 depends on Track A and covers schema-v2 target-specific single-text recipes, deterministic cross-language rendering, persistence/interoperability, reusable builder UI, and parity/mixed-version gates.
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
