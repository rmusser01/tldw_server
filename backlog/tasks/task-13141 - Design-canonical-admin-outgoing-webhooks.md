---
id: TASK-13141
title: Design canonical admin outgoing webhooks
status: Done
assignee: []
created_date: '2026-08-21 19:48'
updated_date: '2026-08-28 05:22'
labels:
  - admin
  - webhooks
  - security
  - jobs
  - design
dependencies: []
documentation:
  - Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write and review the canonical public design for one secure admin outgoing-webhook capability. Reconcile the mounted legacy admin_ops routes with the unmounted admin_webhooks implementation; define the API, repository, encryption, migrations, legacy import, SSRF-safe delivery, Jobs-backed outbox, event protocol, retention, worker operations, and reviewable implementation boundaries. Design work only; no product implementation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical design records one mounted router and removal of the duplicate legacy webhook handlers.
- [x] #2 Design specifies SQLite and PostgreSQL migrations, secret lifecycle, legacy import, and rollback constraints.
- [x] #3 Design specifies durable event/outbox delivery through Jobs, SSRF controls, protocol, retries, retention, and operations.
- [x] #4 Design defines reviewable upstream implementation PRs and complete verification gates.
- [x] #5 Independent specification review passes and the user approves the written document before implementation planning.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replacement for the webhook design record originally created as TASK-12950 on 2026-07-12. Current dev independently assigned TASK-12950 to Quick Ingest while this branch was open, so this task preserves the webhook history under a unique ID. Conversational design and three automated review iterations are recorded in the linked specification and predecessor record. A 2026-08-21 manual revalidation against current dev is in progress before implementation planning.

2026-07-12: Conversational design approved and written to Docs/Design/2026-07-12-canonical-admin-outgoing-webhooks.md. The spec records one final router, server-generated one-time secrets, encrypted target URLs and secrets under a dedicated key ring, new canonical tables, legacy JSON/DB import, six privacy-bounded events, Jobs-only retry, cross-database enqueue recovery, published HMAC protocol, egress controls, feature modes, rollback boundaries, upstream review units, and verification gates. Independent written-spec review was pending; no product code changed.

Spec review iteration 1 returned one hosted retirement blocker and three upstream advisories. The upstream spec then enumerated durable source identity for all six events, idempotent secret replay and key-rotation behavior when the dedicated key ring is unavailable or rotating, and the AuthNZ delivery state machine plus Jobs transition ownership and in-flight lifecycle races.

Spec review iteration 2 found that persisted synchronous test deliveries lacked a Jobs-free state path. The spec then defined a command-identified webhook.test event plus kind=test delivery, one shared DeliveryAttemptExecutor call with a test-attempt token, no Jobs enqueue or retry, interrupted-test terminal handling, partial unique source indexes, and durable forward-resume key rotation.

Spec review iteration 3 found a crash gap between committing a pending test delivery and assigning its synchronous attempt token. The spec then generated the token before the transaction and committed kind=test directly in processing with attempt/start metadata. Stale test rows become dead:test_attempt_interrupted after the maximum timeout plus recovery margin, guarded by token-conditional terminal updates. The three-iteration automated review cap was exhausted, requiring human review before planning.

2026-08-21 manual revalidation: rebased the docs-only branch onto origin/dev 2e0815c1e4577902a220044619822ab6b1cb395f. The mounted/unmounted webhook defect still exists, but current dev now provides Security/http_hop.py and materially refactored Jobs contracts. The prior TASK-12950 ID collided with current dev's unrelated Quick Ingest task; the webhook record was archived and replaced by TASK-13013 without changing the Quick Ingest record.

Manual design review corrections: added revision ETags and conditional mutations; route-scoped/version-bound idempotency with obsolete-secret replay rejection; encryption of exact event bodies; append-only delivery attempts; durable cross-database Jobs disposition/cancellation recovery; a hard four-network-attempt cap; encrypted legacy backups with a separately held one-time key; webhook-subtree hashing that preserves unrelated file changes; and explicit reuse of the current peer-verified HTTP-hop primitive.

The oversized delivery review unit was split into three upstream PRs: control plane/migration, delivery substrate/recovery, and durable producers/final activation. Canonical mode remains unavailable until all three pass. No runtime code changed and no tests/Bandit apply to this documentation-only review; git diff checks remain pending.

2026-08-21 second manual pass: hardened test operations with If-Match and idempotent one-attempt replay; defined idempotency/precondition ordering and nonpersistent UI replay keys; bound registration fanout and tombstone growth; context-bound every encrypted envelope; made mode-on source mutations fail before domain commit when the key is unavailable; added fail-closed worker admission, no-attempt deferral, lease/expiry horizons, stale-attempt scheduling through Jobs, and a single-owner prepared-disposition contract. This pass also corrected the stale PR-2 activation statement. Verification remains pending; no runtime code changed.

Fresh documentation verification on 2026-08-21: origin/dev resolves to 2e0815c1e4577902a220044619822ab6b1cb395f; the stale-wording scan returned no matches; Backlog CLI parsed TASK-13013; git diff --check passed. The changed set is design/Backlog history only, so runtime tests and Bandit are not applicable. User approval remains pending before implementation planning.

2026-08-21 approval gate: user approved the reviewed design, including the registration bounds, fail-closed mode-on source mutation policy, three-PR upstream activation sequence, and conditional hosted compatibility approach. Implementation planning may proceed; runtime implementation remains out of scope for this task.

2026-08-28 TASK-13013.10 identity normalization: this completed canonical webhook design moved from legacy TASK-13013 to canonical TASK-13141. The public release-readiness program remains the sole TASK-13013 record. Historical commits may retain the legacy ID; current design and implementation links use TASK-13141.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Approved canonical outgoing-webhook design completed. The design reconciles the duplicate implementations into one mounted admin control plane, specifies cross-database persistence and migration, contextual encryption and legacy import, Jobs-backed delivery and recovery, bounded retries and fanout, SSRF-safe transport, durable producers, operations, and three independently reviewable upstream PRs. Verification: Backlog parsing and git diff --check passed; runtime tests and Bandit were not applicable because this task changed documentation and task records only.
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
