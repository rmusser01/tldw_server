---
id: TASK-259
title: Review OpenWebUI attachment hydration design
status: Done
assignee: []
created_date: '2026-05-11 05:33'
updated_date: '2026-05-11 05:35'
labels:
  - chatbooks
  - openwebui
  - design-review
dependencies: []
references:
  - Docs/superpowers/specs/2026-05-11-openwebui-attachment-hydration-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Review the approved OpenWebUI attachment hydration design for implementation risks, repo mismatches, and possible improvements before writing the implementation plan. Patch the design spec with actionable findings when needed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Review spec against current Chatbooks, OpenWebUI DB, Jobs, ChaCha message image, allowed-path, and Media DB patterns
- [x] #2 Document actionable design issues and improvements in the spec
- [x] #3 Record verification and known skips
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reviewed the approved hydration design against current repo surfaces: Chatbooks import metadata and worker routing, OpenWebUI DB validation, ChaCha message_images/idempotency, allowed-root path handling, and Media DB registration/dedupe behavior.

Patched the spec with eight design-review adjustments: deep metadata merge, message image source-key limits, Media DB binary registration/dedupe constraints, dedicated hydration job type, hydration-specific file schema validation, original source chat identity for DB fallbacks, preserved-reference limits, and byte-level classification guardrails.

Verification: git diff --check passed, targeted rg confirmed all review findings and the dedicated job type are present. Bandit skipped because this review patch changes docs/task metadata only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reviewed and amended the OpenWebUI attachment hydration design before implementation planning. The spec now documents the implementation risks and guardrails found in the current codebase.
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
