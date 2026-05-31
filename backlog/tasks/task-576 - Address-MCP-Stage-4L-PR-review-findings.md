---
id: TASK-576
title: Address MCP Stage 4L PR review findings
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-31 18:42'
labels:
  - mcp-unified
  - stage-4l
  - pr-review
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix still-valid PR #2195 review findings for Stage 4L editable profile CRUD: narrow Pydantic validation exception handling, normalize create id/name inputs, make create duplicate handling atomic at the store boundary, harden guarded delete error/status handling, and handle patching profiles with no policy_document. Keep changes minimal and validate focused MCP tests plus Bandit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Broad except Exception handlers in create/patch validation are narrowed to expected validation/payload exceptions.
- [x] #2 Profile create rejects or consistently normalizes whitespace id/name values so unaddressable profiles cannot be created.
- [x] #3 Create duplicate detection uses a store-level create-if-absent path rather than read-then-upsert.
- [x] #4 Guarded delete translates store failures and unknown statuses to domain errors with audit events instead of misclassifying as default profile.
- [x] #5 Policy document patch handles profiles whose policy_document is None without crashing.
- [x] #6 Focused MCP profile/FastAPI/CLI tests pass and touched-scope Bandit is clean.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified current PR review findings against rebased branch after syncing with origin/dev. All five reported code-level issues remain valid: broad Pydantic catches, whitespace create id/name behavior, create read-then-upsert TOCTOU, guarded delete unknown/status failure handling, and policy_document None patch crash.

Implemented PR review fixes after rebasing on origin/dev: create now uses store-level create_profile conflict handling; create id/name are normalized; validation catches are narrowed; guarded delete translates store failures and unknown results; policy_document patches handle missing stored policies.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all current PR #2195 review findings from Qodo, CodeRabbit, and Gemini. Added store-level atomic create support for memory and SQLite stores, manager validation/normalization/error handling hardening, explicit unexpected delete status mapping, and regression coverage. Verification: 223 focused MCP tests passed, Ruff touched-file check passed, Bandit touched-scope scan reported 0 results/0 errors, and git diff --check was clean.
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
