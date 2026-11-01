---
id: TASK-15282
title: Design Claims Jobs Stage 2C cluster rebuilds
status: In Progress
labels:
- claims
- jobs
- design
dependencies:
- TASK-9935
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the next Claims Jobs slice for exact and embedding cluster rebuilds. Preserve synchronous HTTP 200 by default; opt-in Jobs mode returns HTTP 202 and a Jobs reference with no local fallback after admission. User selected one active rebuild per owner across both methods: equivalent effective settings reuse the active Job, conflicting settings return 409, and a genuinely new request after completion can rebuild again. Keep every execution/admission lifecycle mechanism in shared Jobs, with Claims retaining clustering and domain persistence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The written design records the approved overlap policy, alternatives, opt-in API compatibility, ownership boundaries and non-goals.
- [x] #2 Atomic shared Jobs admission, retries, cancellation, stale workers, queue changes, owner isolation and SQLite/PostgreSQL domain-write safety are grounded in current implementation.
- [x] #3 The design covers exact and embedding behavior, resource limits, watchlist side effects, rollout/rollback and verification cases without introducing Claims-owned job management.
- [x] #4 An ADR assessment and any required proposed decision are recorded, and an independent builder check is resolved before handover.
- [ ] #5 Design-only verification and commit evidence are recorded; implementation planning waits for explicit approval of the written design.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Started from approved conversation choices; no implementation authorized. Reusing the existing clean linked worktree on a new design branch, preserving local closeout commit 99b234d2e8cb3f4be86c058a3713f633262fb23f on codex/claims-jobs-stage2b-review-metrics-design. Latest fetched dev is 54b321dcec8cdbb6358581f42c1548696260570b. Task ID audit checked all 37 open PRs and full untruncated trees for 12 PRs at GitHub file-list limit; maximum task id is 15281. Parent workstream TASK-9935. Proposed spec: Docs/superpowers/specs/2026-11-01-claims-jobs-stage2c-cluster-rebuild-design.md.
Drafted the full Stage 2C spec and Proposed ADR-085, indexed in Docs/ADR/README.md. ADR required: yes; resource-scoped Jobs admission and cross-store publication fencing establish a new durable persistence/ownership contract, composing with ADR-003 and ADR-058 rather than superseding them. Current Jobs OwnerScopeAdmissionPolicy is queue/job-type scoped and current Jobs detail router requires admin; the draft explicitly proposes generic Jobs-owned resource admission and a read-only owner status projection rather than Claims lifecycle tables or implicit global-admin elevation. User approved concurrency policy and opt-in 200/202 contract only; publication receipts, request-key retention, status projection and resource ceilings remain engineering proposals for full-design approval. Independent builder check dispatched read-only before handover. No production or test files changed; Bandit is not applicable to this documentation-only work.
Independent builder check identified four material design gaps, addressed in the written proposal: source-row/Media ownership locking through publication with later insertions explicitly excluded; absent owner DB as a non-retryable Jobs failure rather than an impossible domain receipt; publication history retained with owner data independently of Jobs pruning; coordinated quiescent restoration with no false promise to detect all same-UUID backup rewinds. Also specified mandatory resource policy on all create/reactivation paths, head-before-row lock ordering, bounded request-key aliases, generation/UUID collision rejection and a separate authenticated Jobs owner-status router preserving admin dependencies. Sent revised sections to the same reviewer for recheck. Design baseline tests: 46 passed, zero skips (clustering, cluster upsert idempotency, Jobs admission facade and scoped reads). Backlog library/format tests: 159 passed, zero skips. Logs: /tmp/claims-stage2c-design-baseline-20261101.log and /tmp/claims-stage2c-design-backlog-20261101.log. Documentation-only; no product or test changes and no Bandit-applicable code.
Independent builder recheck of the clarified spec and ADR reports nothing important left; review agent closed. All four builder questions are resolved in the design, not deferred as implementation guesses. Verification: 46 focused existing-code baseline tests and 159 Backlog/format tests passed with zero skips; exact new task canonical check, whole-worktree official duplicate-task-ID check, linked-file existence and whitespace checks passed. All applicable staged pre-commit hooks passed (normal no-code hooks skipped). Hook log /tmp/claims-stage2c-design-hooks-20261101.log. Final change set is only the new spec, Proposed ADR-085, ADR index entry and this task. No product/test code, no implementation plan, no push or PR, and no prior PR closeout changes. Preparing the verified local design commit; task remains In Progress awaiting explicit approval of the written design. On approval the next action is superpowers:writing-plans.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
