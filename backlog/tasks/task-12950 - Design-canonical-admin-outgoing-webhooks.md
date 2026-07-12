---
id: TASK-12950
title: Design canonical admin outgoing webhooks
status: In Progress
labels:
- admin
- webhooks
- security
- jobs
- design
priority: High
documentation:
- Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the canonical public design for one secure admin outgoing-webhook capability. Reconcile the mounted legacy admin_ops routes with the unmounted admin_webhooks implementation; define the API, repository, encryption, migrations, legacy import, SSRF-safe delivery, Jobs-backed outbox, event protocol, retention, worker operations, and two-PR implementation boundary. Design work only; no product implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Canonical design records one mounted router and removal of the duplicate legacy webhook handlers.
- [ ] #2 Design specifies SQLite and PostgreSQL migrations, secret lifecycle, legacy import, and rollback constraints.
- [ ] #3 Design specifies durable event/outbox delivery through Jobs, SSRF controls, protocol, retries, retention, and operations.
- [ ] #4 Design defines two reviewable upstream implementation PRs and complete verification gates.
- [ ] #5 Independent specification review passes and the user approves the written document before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-12: Conversational design approved and written to Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md. The spec records one final router, server-generated one-time secrets, encrypted target URLs and secrets under a dedicated key ring, new canonical tables, legacy JSON/DB import, six privacy-bounded events, Jobs-only retry, cross-database enqueue recovery, published HMAC protocol, egress controls, feature modes, rollback boundaries, two upstream PRs, and verification gates. Independent written-spec review is pending; no product code changed.
Spec review iteration 1 returned one hosted retirement blocker and three upstream advisories. Upstream spec now enumerates the durable source identity for all six events, defines idempotent secret replay and key-rotation behavior when the dedicated key ring is unavailable/in transition, and defines the AuthNZ delivery state machine plus Jobs transition ownership and in-flight lifecycle races. Re-review pending.
Spec review iteration 2 found that persisted synchronous test deliveries were not assigned a Jobs-free state path. The spec now creates a command-identified webhook.test event plus kind=test delivery, runs one shared DeliveryAttemptExecutor call with a test-attempt token, performs no Jobs enqueue/retry, and marks interrupted tests terminal without retry. Advisory hardening also defines cross-backend partial unique indexes for aggregate versus command event identities and durable forward-resume semantics for interrupted key rotation. Final review iteration pending.
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
