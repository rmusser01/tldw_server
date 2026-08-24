---
id: TASK-13122
title: Repair automation executor DefinitionRow test fixture
status: Done
assignee: []
created_date: '2026-08-24 19:55'
updated_date: '2026-08-24 19:58'
labels:
  - scheduled-tasks
  - tests
  - cleanup
dependencies: []
references:
  - TASK-13113
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Update the automation executor test helper to construct the current DefinitionRow shape after normalized result-policy fields were added on dev. This is a test-only compatibility repair discovered while running the TASK-13113 adjacent regression gate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The automation executor test helper supplies all required DefinitionRow fields with neutral values.
- [x] #2 All tests in test_automation_executors.py pass without changing production behavior or test expectations.
- [x] #3 The change remains isolated to the executor test fixture and its Backlog record.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED on origin/dev at 2c6553c4ed: 8 executor tests failed before execution because _definition() omitted six required DefinitionRow fields. GREEN: all 10 focused executor tests passed after adding neutral resolution and policy values; git diff --check passed; no production files changed. Delivered in a focused test-only commit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Updated the automation executor test fixture to match the current normalized DefinitionRow contract by supplying neutral resolution and policy fields. This restores the adjacent scheduled-task regression gate without changing executor behavior or production code.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Focused executor test suite passes.
- [x] #2 git diff --check passes.
- [x] #3 No production files are modified.
- [x] #4 A focused commit records the repair.
<!-- DOD:END -->
