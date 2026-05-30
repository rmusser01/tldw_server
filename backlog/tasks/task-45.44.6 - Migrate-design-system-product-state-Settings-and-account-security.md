---
id: TASK-45.44.6
title: 'Migrate design-system product state: Settings and account/security'
status: In Progress
assignee: []
created_date: '2026-05-14 03:19'
updated_date: '2026-05-30 16:24'
labels:
  - design-system
  - webui
  - extension
  - product-state
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1663'
  - >-
    Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
  - 'https://github.com/rmusser01/tldw_server/pull/1781'
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The linked GitHub issue owns current count and public status.
- [ ] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [ ] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TASK-45.44.6.2 completed on codex/design-system-next-slice-8: migrated TldwConnectionSettings auth notices from AntD Alert to DesignSystemAlert. Baseline evidence: total product-state exceptions 400 -> 398; Settings/account-security exceptions 49 -> 47. PR: https://github.com/rmusser01/tldw_server/pull/1781.

TASK-45.44.6.3 completed locally on codex/integration-policy-alerts-ds at d78018fd9b: migrated IntegrationPolicyPanel policy/pairing alerts from AntD Alert to the shared DS Alert primitive and removed four component baseline entries. Verification included focused IntegrationPolicyPanel DS Alert tests, IntegrationManagementPage tests, scoped product-state guard, TypeScript with 8GB heap, and git diff --check.

TASK-45.44.6.4 completed locally on codex/integration-policy-alerts-ds: migrated TldwBillingSettings billing error/warning/usage alerts from AntD Alert to the shared DS Alert primitive and removed eight component baseline entries. Evidence: TldwBillingSettings baseline count is 0 and Settings-only product-state baseline count is 39. Verification included focused billing DS Alert tests, tldw-review-comments tests, scoped product-state guard, TypeScript with 8GB heap, and git diff --check.

TASK-45.44.6.5 completed locally on codex/integration-policy-alerts-ds: migrated FamilyGuardrailsWizard family setup, mapping, template, tracker, and review guidance alerts from AntD Alert to the shared DS Alert primitive and removed nine component baseline entries. Evidence: FamilyGuardrailsWizard baseline count is 0 and Settings-only product-state baseline count is 30. Verification included red/green FamilyGuardrailsWizard tests, scoped product-state guard, baseline JSON parse, TypeScript with 8GB heap, and git diff --check; full verify:design-system-state remains red on unrelated existing blocked findings outside this slice.

TASK-45.44.6.6 completed locally on codex/integration-policy-alerts-ds: migrated GuardianSettings global unavailable, self-monitoring unavailable, guardian controls unavailable, crisis resources, and offline/auth/setup warning alerts from AntD Alert to the shared DS Alert primitive and removed nine component baseline entries. Evidence: GuardianSettings baseline count is 0, Settings path product-state baseline count is 21, and total baseline count is 165. Verification included red/green GuardianSettings focused tests, scoped product-state guard, baseline JSON parse, TypeScript with 8GB heap, and git diff --check; full verify:design-system-state remains red on unrelated existing blocked findings outside this slice.

TASK-45.44.6.7 completed locally: migrated Evaluations settings auth/setup/unreachable/offline/API-test alerts to the design-system Alert primitive, removed the Evaluations settings baseline exceptions, and verified focused tests plus scoped guard/TypeScript. Full design-system verifier remains blocked by unrelated WritingPlayground, Notes, and ResearchWorkspace findings.

TASK-45.44.6.8 completed locally: migrated General settings extension promotion and OCR asset alerts to the design-system Alert primitive, removed the General settings baseline exceptions, and verified focused tests plus scoped guard/TypeScript. Evidence: general-settings.tsx baseline count 0, Settings path count 14, total baseline count 158. Full design-system verifier remains blocked by unrelated WritingPlayground, Notes, and ResearchWorkspace findings.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
