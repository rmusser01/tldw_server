---
id: TASK-2410
title: Harden Billing module review findings
status: Done
assignee: []
created_date: '2026-06-23 18:12'
updated_date: '2026-06-24 01:40'
labels:
  - billing
  - security
  - review
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address current-code review findings in tldw_Server_API/app/core/Billing: fail-closed usage enforcement, webhook idempotency, checkout URL validation and side-effect ordering, injected compatibility consistency, sanitized logs, overage config validation, and duplicate audit helper cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fail-closed usage-source failures deny or use restrictive usage instead of silently returning zero.
- [x] #2 Webhook compatibility processing is idempotent for duplicate Stripe event IDs and invoice payments.
- [x] #3 Checkout compatibility validates redirect URLs and validates plan/price before Stripe customer side effects.
- [x] #4 Injected-client subscription reads and writes have one consistent contract.
- [x] #5 Billing backend logs do not include raw exception details that can leak secrets or paths.
- [x] #6 Overage policy environment parsing validates mode and percentage bounds.
- [x] #7 Duplicate BillingAuditLogger helper is removed or consolidated without leaving stale exports/docs/tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See IMPLEMENTATION_PLAN_billing_module_hardening_2410.md. Stages: tracking/scope, regression tests, Billing module fixes, verification/closeout.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created implementation plan IMPLEMENTATION_PLAN_billing_module_hardening_2410.md after verifying TASK-2410 file path.

Implemented Billing hardening fixes: fail-closed usage source failures now propagate in closed mode instead of returning zero; checkout and portal redirects are allowlist-validated before Stripe side effects; checkout validates public active plan/price before customer creation; injected compatibility subscription reads now use the same repository contract as writes; webhook compatibility can claim Stripe event IDs before mutating payment history; overage env parsing falls back to safe defaults for invalid modes/percentages; raw exception details were replaced with exception-class logging in touched Billing paths; duplicate BillingAuditLogger helper and stale active references were removed.

Verification recorded: focused touched Billing tests passed with 83 passed, 174 warnings; Bandit on touched Billing core files reported 0 findings; git diff --check on touched paths passed; stale active Billing audit-helper reference search returned no matches. Broader tldw_Server_API/tests/Billing run remains blocked outside this Billing change by app import issues: test_billing_public_api_removed.py imports tldw_Server_API.app.main and hits missing save_and_register_file_export from Storage.generated_file_helpers, and the filtered suite timed out during broad app import/teardown paths.

Moved fixes into worktree .worktrees/billing-module-hardening on branch codex/billing-module-hardening from dev. Worktree verification: focused touched Billing tests passed with 83 passed, 179 warnings; Bandit JSON scan for touched Billing core files wrote /tmp/bandit_billing_worktree_2410.json; git diff --check on touched paths passed; stale active Billing audit-helper reference search returned no matches.

PR review follow-up: rebased codex/billing-module-hardening-origin-dev onto latest origin/dev and addressed review comments by rejecting non-finite overage percentages, normalizing default HTTP/HTTPS ports during redirect origin comparison, and logging settings lookup failures with sanitized exception labels. Added regressions for all three cases.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened the current Billing module review findings and added regression coverage for enforcement failure handling, checkout precondition ordering, webhook idempotency, injected subscription reads, overage config validation, and removed the duplicate audit helper. Focused touched Billing verification passes; broader suite blocker is documented as unrelated app import/runtime setup.
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
