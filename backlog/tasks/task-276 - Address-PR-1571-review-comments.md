---
id: TASK-276
title: Address PR 1571 review comments
status: Done
assignee: []
created_date: '2026-05-12 00:09'
labels:
  - vn-play
  - review-fix
  - pr-1571
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1571'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address actionable review comments on PR 1571 for the VN scripted generation runtime branch. Verify each finding against current code, fix valid issues minimally, and record focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Generated profile snapshot lineage is preserved for all profile snapshots
- [x] #2 Branch navigation response schemas preserve generated choice references
- [x] #3 Generation lifecycle errors map to the intended HTTP conflict response
- [x] #4 Checkpoint restore fails fast for unknown active generation revision keys
- [x] #5 Publish idempotency matching does not fall back to legacy hashes for structured payloads
- [x] #6 Capabilities and setup metadata are defensively gated and null-safe
- [x] #7 Generated output parser preserves duplicate choice validation codes
- [x] #8 Tests avoid brittle hard-coded generation request IDs
- [x] #9 Backlog verification notes use portable commands
- [x] #10 Focused VN tests, compile check, Bandit, and diff check are run and recorded
- [x] #11 Raw moderation-blocked debug reveal requires raw-debug authorization for non-owners
- [x] #12 Stale generation request recovery cannot regress the generation status
- [x] #13 Review helper functions have explicit return types
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Addressed PR 1571 review comments for generated profile snapshot lineage, generated-choice branch response schemas, conflict mapping, checkpoint active-generation restore validation, publish idempotency matching, capabilities gating, null-safe setup metadata, duplicate-choice parser error codes, brittle test IDs, portable Backlog verification commands, raw-debug reveal authorization, stale generation status regression, and helper return typing.
- Verification: .venv/bin/python -m pytest tldw_Server_API/tests/VN_Scripts tldw_Server_API/tests/VN_Play tldw_Server_API/tests/VN_Platform -q --tb=short -> 258 passed, 8 warnings.
- Verification: compileall on touched VN endpoint/schema/repository/runtime files -> exit 0; git diff --check -> exit 0.
- Bandit: .venv/bin/python -m bandit -r touched VN backend files -f json -o /tmp/bandit_vn_scripted_generation_review_fixes.json -> 0 results, 0 errors.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Resolved all actionable PR 1571 review comments. The branch now preserves generated profile snapshot links for every profile key, serializes generated-choice branch references without leaking null fields, maps abandoned generation attempts to conflicts, rejects unknown active-generation restore keys, avoids unsafe structured-payload idempotency fallback, gates capability flags by actual scripted-generation availability, handles missing generation snapshots in setup options, preserves duplicate-choice parse codes, removes brittle hard-coded generation request IDs, restricts non-owner raw debug reveal to raw-debug permission, and prevents stale generation request updates from regressing generation status. Focused and full VN verification passed.
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
