---
id: TASK-45.44.6.5
title: Migrate FamilyGuardrailsWizard alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-30 15:52'
labels:
  - design-system
  - webui
  - product-state
  - settings
dependencies: []
references:
  - TASK-45.44.6
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - apps/packages/ui/src/components/Option/Settings/FamilyGuardrailsWizard.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.6
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Migrate the nine FamilyGuardrailsWizard AntD Alert product-state callouts to the shared design-system Alert primitive while preserving family setup, household, template, and relationship warning/info copy. Remove the matching baseline exceptions and verify the scoped Settings/account-security guard state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FamilyGuardrailsWizard no longer imports AntD Alert or renders AntD Alert product-state callouts.
- [x] #2 Representative draft, guardian, dependent, tracker, template-review, and final-review guidance renders inside the design-system Alert container.
- [x] #3 FamilyGuardrailsWizard product-state baseline exceptions are removed and the scoped product-state guard is clean.
- [x] #4 Verification is recorded, including any unrelated baseline guard blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect FamilyGuardrailsWizard alert branches and existing family guardrails tests to identify a focused render harness.
2. Add failing tests that representative family wizard info and warning copy renders inside the design-system Alert container.
3. Replace AntD Alert with the shared Alert primitive while preserving title/body copy and conditional rendering.
4. Remove the nine matching FamilyGuardrailsWizard baseline entries and run focused tests, scoped product-state guard, TypeScript, and diff hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented via TDD: first added DS Alert ancestor assertions to existing FamilyGuardrailsWizard tests and observed the expected red failures against AntD Alert. Migrated all wizard info/warning/success/error callouts to DsAlert while preserving existing title/body copy and dynamic tracker/review variants. Removed the nine FamilyGuardrailsWizard allowed-legacy baseline entries. Verification: FamilyGuardrailsWizard Vitest file red before implementation with five DS ancestor failures, then green with 52/52 passing; scoped product-state guard for src/components/Option/Settings/FamilyGuardrailsWizard.tsx reported no issues; baseline JSON parse passed; TypeScript tsc --noEmit exited 0; git diff --check exited 0. Full verify:design-system-state remains red on unrelated existing blocked findings in WritingPlayground, Notes, ResearchWorkspace, and WorkspaceCapabilityRemediation. Bandit skipped because this is frontend TS/JSON only with no Python touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated FamilyGuardrailsWizard product-state callouts from AntD Alert to the shared design-system Alert primitive, added regression assertions for DS alert rendering across key wizard states, and removed the obsolete FamilyGuardrailsWizard baseline exceptions.
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
