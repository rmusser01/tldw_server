---
id: TASK-13866
title: Retire superseded PRs and preserve integration closeout records
status: In Progress
labels:
- documentation
- maintenance
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Retire PRs #2956, #2966 and #3043 after verifying current dev retains their useful behavior and preserving unique historical task records; complete documentation-only PR #3197 without changing production behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Preserve unique PyPI publication task and UAT publication notes with canonical backlog-py records
- [x] #2 Record why the retired CI checker is redundant with existing tested contracts
- [ ] #3 Qualify and independently review PR3197 against current dev
- [x] #4 Restore current-dev CI shard coverage for the newly merged reused-transcript quota regression
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Requester authorized the cleanup and completion on 2026-10-19. Existing PR3197 task: TASK-13461. ADR assessment: no new ADR required; tracking-only supersession records preserve existing ADR-049 and CI decisions. Plan: compare retired records, salvage only unique provenance, verify canonical task format and focused CI contracts, independently review, then close superseded PRs and qualify PR3197. Qodo is unavailable this month; use requesting-code-review.
Prepared documentation-only salvage and #3197 closeout against current dev. Restored TASK-13257 through backlog-py with original dated provenance, retained TASK-13262 publication notes, documented why existing six-gate contract tests supersede #3043 checker. No new ADR; ADR-049 remains governing for historical chat ownership. Required current-head CI and independent final review remain pending.
Fresh connected contract run: 111 passed, 1 failed. Root cause is current-dev ci.yml omission of test_reused_transcript_llm_quota_refusal.py from all five media unit shard matrices after #3277 merged. Existing coverage contract correctly rejects the omission. Scope adds that existing regression to the existing unit-core shard in each matrix and updates its exact expected set; no tests or gates weakened. No ADR needed for a test-file routing repair.
Verification: expected shard assignment failed first, then all 112 connected CI/task/PyPI/license/required-workflow contract tests passed after adding the test path to all five matrices. Ruff and syntax pass. Bandit 1.9.3 on touched Python test with assertion-only B101 excluded: zero findings. Black --check reports pre-existing formatting drift; formatting comparison proves the single added expected path introduces no formatting drift. git diff --check clean. Independent retirement review approves preservation/closures; final combined review and current-head CI remain pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

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
