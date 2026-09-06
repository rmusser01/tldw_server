---
id: TASK-13163
title: Resolve Personal Context conflicts through the batched Sync API
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 13:40'
updated_date: '2026-09-06 01:12'
labels:
  - personal-context
  - sync
  - conflicts
  - security
dependencies:
  - TASK-13162
references:
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extend the existing batched Sync conflict API for ongoing Personal Context conflicts while preserving both immutable candidates and routing every mutating decision through canonical Personalization authority. Ordinary conflicts and semantic-key collisions remain narrowly frozen.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Personal Context push conflict creates or reuses a deterministic protected home-authority candidate before returning a terminal conflict result.
- [x] #2 Conflict records carry expected local and remote envelope IDs; stale reviews are rejected without mutating canonical state or resolving the generic conflict.
- [x] #3 The batched endpoint implements skip, overwrite, and duplicate_rename only, with canonical overwrite, merge payload, and duplicate decisions routed through PersonalContextService.
- [x] #4 Mutating decisions use idempotent Personalization replay receipts so interruption cannot duplicate a version, manifest advance, publication batch, merge, or renamed record.
- [x] #5 Ordinary conflicts freeze one object; key collisions freeze both object IDs and only the contested semantic-key slot while unrelated objects continue.
- [x] #6 Candidate retention, replay, stale-review, key-collision, batch-partial-failure, authorization, and plaintext-canary tests pass.
- [x] #7 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs conflict authority and retention.
- [x] #8 A signed valid-next-generation delete-everywhere request with stale lineage is rejected nonretryably with a content-free refresh and explicit reconfirmation instruction, without creating an ordinary conflict candidate; exact idempotent retry and valid purge/generation behavior are preserved.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR. ADR path: backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md. Reason: implement approved ongoing-sync conflict authority and retention. Follow Task 2 in Docs/superpowers/plans/2026-09-03-personal-context-ongoing-sync-02-server-activation-conflict-purge.md after auditing current activation/publication seams. First verify current behavior, identify durable encrypted receipt/freeze storage and migration requirements, and refine exact integration steps without changing approved wire actions. Then add failing candidate retention and replay/stale-review tests, implement canonical batched resolution plus exact freezes, run targeted SQLite/PostgreSQL/canary/authorization tests, and obtain spec and quality review. Keep ongoing_sync_version=0. Isolated branch starts at the verified TASK13192 converter fix 6363466d07; no merge is claimed.

PR #2910 review remediation: verify all ten Qodo findings against approved detail plan and ADR-002; reproduce confirmed relay and duplicate-key bugs before minimal fixes; establish purge reachability and preserve existing retention policy, reporting architectural ambiguity before changing it. Preserve test-only SQL inspection instead of adding production inspection helpers. Run targeted SQLite/PostgreSQL paths with PostgreSQL required, affected Ruff/format/Bandit and diff checks; record per-finding evidence and scoped commit. ADR required: no new ADR for routine fixes; existing ADR-002 governs.

Approved purge-review follow-up: user selected stale delete-everywhere rejection with refresh and explicit reconfirmation. Reproduce retained real signed failed-ingress scenario as a failing assertion, add focused adapter cases preserving normal next-generation validation and exact retry, then reject stale purge lineage before candidate creation. No schema, rollout, UI or ordinary conflict policy changes. Parent updates ADR-002 and spec; this worker owns adapter/tests/task/detail plan. ADR required: existing ADR-002 amendment records approved control-request policy. Run targeted adapter, purge, ingress/recovery checks and affected Ruff/Bandit/diff; require completed process exits and independent scoped review.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Continuation checkpoint: approved detail plan Docs/superpowers/plans/2026-09-05-personal-context-conflict-resolution-detail.md adds missing canonical encrypted journal/freeze and Sync retention ownership; independent plan review approved. Existing ongoing wire-contract baseline: 12 passed. Implementation paused before code edits for an owner decision: on a semantic-key collision between distinct local and shared object IDs, does overwrite/Keep local authorize retiring the shared record and installing the local ID, or must that shared record remain and the local copy use duplicate_rename? Approved spec defines duplicate_rename but does not resolve this destructive identity choice. No default deletion, action restriction, or capability enablement introduced. Investigation found generic encrypted object_versions could own private journals under existing purge/rotation inventory; verify in implementation after choice. TASK13192 committed6363466d07 is this isolated branch base, not yet merged.

Owner clarification resolved: user explicitly chooses deconfliction outcome; no automatic local/server winner. Keep shared, keep local values, reviewed merge, or explicitly distinct keep-both. For same-key distinct IDs, keep-local/merge explicitly targets established shared canonical identity; incoming duplicate is accounted for by exact receipt, not silently installed alongside it. Spec and detail plan updated; continuation authorized. Supersedes prior paused-decision checkpoint.

Implemented user-directed Personal Context deconfliction in the existing batched Sync endpoint. Protected immutable canonical candidates and narrow object/key-slot freezes share existing encrypted storage; completed exact receipts do not consume the internal 1000-active-conflict bound. Keep-local and merged values explicitly target the established shared canonical ID; duplicate_rename requires a new ID and noncolliding key. Canonical mutation, manifest, publication and exact decision receipt commit together, with recoverable Sync finalization. Candidate identity/body authentication and same-transaction activation checks cover capture, attachment, resolution and replay. Invalid linked manifests and invalid purge generations reject before conflict review. Retention guards include real SQLite/PostgreSQL destructive checks. No schema, dependency, shared profile-core, new endpoint/action or rollout changes.

Verification: final amended candidate/replay/race/PostgreSQL selection 16 passed and activation 25 passed, both exit 0; preceding classification fix 2 focused plus 35 adapter/materializer passed. Earlier implementation checkpoints: 48 conflict cases, 60 compatibility/activation, 316 targeted regressions and 71 relay/activation passed; these are not claimed as final-source full reruns. Affected Ruff, formatting, Bandit, docs path hygiene and branch diff checks passed. Independent task review and final whole-branch review, including scoped fixes, approved source commit 62e63bf5124628b3e2bb1d7aec745af6f976c3cf. No full suite or live end-to-end rollout certification was run. Known dependency/config warnings remain attributed; the long 48-test interpreter cleanup exited 0 normally.

API/developer guides, published copies, approved spec/detail plan and testing lesson updated. ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs the existing storage, authority and conflict boundaries. Plan: Docs/superpowers/plans/2026-09-05-personal-context-conflict-resolution-detail.md. Runtime context builder remains an unwired existing scaffold; the unchanged all-domain scan-watermark length limitation remains outside this task and was not independently baseline-reproduced. Ongoing sync stays version 0. Local branch codex/personal-context-conflict-resolution starts at 6363466d07; no push, PR, merge or destructive cleanup performed.

Publication: opened https://github.com/rmusser01/tldw_server/pull/2910 stacked on PostgreSQL prerequisite PR #2909. Fresh candidate/recovery/race selection: 15 passed and one PostgreSQL sandbox skip; the skipped test then passed with PostgreSQL required and database access (1 passed). Both processes exited 0. Diff check passed. No source changes, rebase, merge or rollout. Merge prerequisite into dev first, then retarget/rebase this PR onto dev and verify integration; human Change summary remains a pre-merge requirement.

PR #2910 review remediation: fixed missing concrete active semantic-key validation and centralized invalid-choice exceptions; direct canonical resolution schedules relay, while batches defer relay until the outer Sync fence/savepoint exits. Real-factory diagnostic proved that a naive in-service callback invalidates the open SQLite savepoint. Informational conflict_pin rows remain visible in retention previews without blocking unrelated domain compaction; naturally eligible blockers and domain-prefix guards keep existing all-or-nothing behavior. Added targeted direct/batch relay failure and replay cases, missing-key cases, mixed retention on SQLite/PostgreSQL, and PostgreSQL skip/overwrite/duplicate decisions; completed docstrings/types in the PR test module and fixed its proposal clock. Shared pg_database_config already provisions a unique per-test database with teardown; retained existing test-only SQL inspection/tamper probes instead of adding production inspection APIs.

Final verification: focused conflict/replay selection22passed exit0, PostgreSQL-required; corrected PostgreSQL fixture selection3passed exit0 (removes introduced serialization warning), existing retention/relay suites75passed exit0. Affected Ruff, scoped formatting, AST docs/types and git diff checks passed. Bandit production scope has zero findings exit0; test scan has expected assertion B101 and the same HEAD-baseline closed-parameter tamper-query B608, no new production findings. Independent scoped review approved remediation source and test fixture correction. No full sweep. Python interpreter cleanup after the22-test summary completed normally before commit.

OPEN review finding1: real factory probe demonstrated a signed next-generation purge persisted accepted/failed during temporary canonical unavailability, then a second valid request hit that Sync head and returned retryable storage_unavailable because canonical purge has no object head. Approved spec does not choose rejection with refresh/reconfirmation versus a dedicated deletion-conflict review. Purge behavior remains unchanged pending owner policy; task stays In Progress and PR readiness is not claimed. ADR required: no new ADR for these routine fixes; existing ADR-002 and approved detail plan govern. Detail plan and testing-evidence lesson updated. No schema/dependency/action/rollout changes; ongoing_sync_version remains0. Parent owns prerequisite restack, publication and GitHub response.

Approved purge follow-up corrects the earlier diagnostic attribution: the signed request was durably stored with real projection failure, but the injected apply_sync_ingress service outage was never reached. Existing purge materialization parses a dict then passes it to the model-only canonical serializer before service entry; ongoing canonical ingress purge remains unsupported and gated. The regression now delegates real PersonalContextMaterializer.apply and observes original-request re-entry after rejecting the second stale intent. Parent recorded user-approved rejection/reconfirmation in ADR-002 and spec; no canonical purge lifecycle expansion is implemented.

Completed approved stale-purge review follow-up: adapter and generic insertion-CAS boundary reject new stale deletion intents nonretryably with personal_context_purge_reconfirmation_required and refresh/explicit reconfirmation instruction, before ordinary candidate capture. Exact previously stored client ingress retains materializer re-entry under existing full immutable fingerprint, signature, identity, generation and activation checks; changed same-ID entity metadata is rejected. No canonical purge lifecycle, schema, UI, dependencies or rollout added. Verification: final4 SQLite/PostgreSQL preflight/race/retry cases passed exit0; targeted adapter/materializer/ingress-repair/purge-retention107 passed exit0 on final production before test-only fixture corrections. Fresh Ruff, amended range formatting, production Bandit zero findings and diff checks exit0. Independent scoped review approved with no Critical/Important findings. Prior OPEN purge finding is superseded by user-approved ADR-002 amendment, recorded in detail plan. Original canonical-service fault attribution corrected above; existing gated dict serialization and unsupported ongoing canonical purge remain unimplemented. Parent owns publication and prerequisite restack; no merge claimed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed canonical user-directed conflict resolution with exact encrypted candidates/receipts, guarded retention, restart recovery and transaction-boundary activation checks. Task and final reviews approved all fixes; targeted verification is recorded with baseline/environment warnings. Published for review in PR #2910, stacked on #2909; not merged, and ongoing sync remains gated at version 0.
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
