---
id: TASK-13450
title: Design Embeddings Stage 3 durable workflow store and Jobs runner
status: Done
created_date: 2026-10-04 23:05
labels:
- embeddings
- workflow
- architecture
- stage-3
- design
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2790
- backlog/archive/tasks/task-13421 - Design-Embeddings-Stage-3-durable-workflow-store-and-Jobs-runner.md
documentation:
- Docs/superpowers/specs/2026-07-03-embeddings-workflow-architecture-design.md
- Docs/superpowers/specs/2026-07-18-embeddings-workflow-stage2-concrete-api-steps-design.md
- Docs/ADR/003-jobs-vs-scheduler-default.md
- Docs/ADR/022-embeddings-api-and-media-pipeline.md
- Docs/ADR/051-retry-backoff-schedules.md
- Docs/superpowers/specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md
- Docs/ADR/060-embeddings-durable-workflow-foundation.md
modified_files:
- Docs/superpowers/specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md
- Docs/ADR/060-embeddings-durable-workflow-foundation.md
- Docs/ADR/README.md
- backlog/tasks/task-13450 - Design-Embeddings-Stage-3-durable-workflow-store-and-Jobs-runner.md
- backlog/archive/tasks/task-13421 - Design-Embeddings-Stage-3-durable-workflow-store-and-Jobs-runner.md
updated_date: 2026-10-04 23:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the dormant Stage 3 Embeddings durability foundation after merged Stage 2. Deliver an approved spec and ADR assessment, not production wiring. Scope: Embeddings-owned persistence on the configured SQLite/PostgreSQL Jobs backend; immutable reference-only manifests; owner-scoped generation/item sink receipts; bounded batches, attempts, and typed reconciliation operations; additive domain-neutral prepared Jobs capability; injectable resolver, sink, and accounting contracts. Exclude public/admin endpoints, endpoint/Redis media migration, worker/lifecycle registration, production billing hookup, and legacy behavior changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Document the approved six-table owner-scoped persistence model, immutable manifests, pinned embedding-space generations, and named CAS repository transitions.
- [x] #2 Review and approve admission, execution, lease fencing, normal batch continuation, retry/fallback, cancellation, reconciliation, accounting, and retention.
- [x] #3 Specify backend security/concurrency, migration capability gates, crash-window tests, bounded resources, and unchanged-production-path verification.
- [x] #4 Write and self-review the complete spec and required ADR; resolve placeholders, contradictions, and ambiguous guarantees.
- [x] #5 Obtain requester approval of the written spec before implementation planning.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Approved architecture, section reviews, and three sets of written-spec corrections preserved (complete).
2. Requester approved the consolidated written spec and ADR on 2026-10-04 (complete).
3. Active links moved to collision-free TASK-13450; original TASK-13421 history archived (complete).
4. Documentation/approval metadata verified; handoff to separate Stage 3A planning TASK-13451 (complete).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-02: Worktree .worktrees/embeddings-workflow-stage3-design; branch codex/embeddings-workflow-stage3-design; base origin/dev 9958110df2. Searched existing Embeddings/Stage 3 records; no matching design task found. Stage 2 parent and stale Stage 2C tracking corrected after merged-source/historical-verification review. Fresh runner/type baseline: 48 passed, 8 warnings in 3.34s.
Approved scope: dormant internal foundation; one Jobs root and bounded deterministic batches; at-least-once provider execution; generation/item-idempotent sink receipts; reference-only manifests with immutable revisions; generations pin provider/model/dimensions/policy plus trusted embedding-space fingerprint; fallback identity changes supersede the generation and restart item zero; cooperative cancellation at batch checkpoints; Jobs-only durable quarantine; per-batch accounting hooks fake-tested without production billing; Jobs active/archive UUID lifetime without active-Jobs foreign key.
Approved tables: embedding_workflows, embedding_workflow_items, embedding_workflow_generations, embedding_workflow_outputs, embedding_workflow_attempts, embedding_workflow_operations. Typed transactional outbox separates Jobs transitions, accounting, and superseded-output cleanup. Exact non-null owner on every table, composite owner constraints, PostgreSQL FORCE RLS, explicit SQLite owner scope, SQL under DB_Management, Embeddings-owned migrations, fail-closed capability checks. No raw inputs, vectors, credentials, or unbounded exceptions in durable state.
Verified existing prepared Jobs APIs enforce canonical Admin Webhook identity: require an additive domain-neutral capability while preserving existing behavior. Successful batch continuation must not consume failure retries. Execution/recovery and testing sections remain to be approved; no production code changes.
ADR check: ADR required: yes. This creates durable persistence and Jobs/domain ownership rules. Link ADR-003, ADR-022, ADR-051; draft a new Proposed ADR with the next free number when writing the fully reviewed spec. ADR-022 remains governing for unchanged legacy paths.
Bandit not applicable to tracking-only Markdown changes. MCP cannot resolve TASK-12973 (including as dependency); official CLI fallback used after confirming the failed MCP create produced no task.

2026-10-02: Requester approved the execution/recovery section. Admission persists the immutable manifest and enqueue operation together; provider execution waits for exact Jobs UUID binding. Validate owner, Jobs identity, live lease, generation, and workflow version; reject stale checkpoints. Recover acknowledged generation/item sink receipts and pending accounting before another paid call. Resolve source revisions, acknowledge accounting reservation, and invoke only the pinned provider/model. Shared caching is disabled unless the adapter certifies the generation embedding-space fingerprint. Atomically checkpoint valid sink receipts, cursor, accounting settlement, and Jobs disposition operations. Required acknowledgements gate subsequent execution. Successful batches yield without failure-retry consumption. Only provider-origin failures can select fallback; changed embedding space creates a new generation starting at item zero. Cancellation prevents the next batch while in-flight work settles; completion requires all current-generation outputs and accounting acknowledgements. Operation reconciliation never calls a provider. Ambiguous dispatched attempts settle conservatively; superseded outputs remain non-authoritative with bounded cleanup.
Verification-section investigation: Jobs conftest disables lease enforcement and PostgreSQL RLS by default; durable workflow tests must explicitly enable both and exercise PostgreSQL under a non-superuser/non-BYPASSRLS role. Reuse existing jobs_pg_dsn/pg_temp_db fixtures and backend parity patterns. Proposed retention safeguard for review: unresolved accounting/cleanup/disposition operations pin minimal owner-scoped recovery state beyond Jobs expiry, with finite automatic retry budgets and explicit exhausted status; never silently discard obligations or delete completed current-generation sink outputs merely because Jobs history is pruned. This safeguard and the test/rollout section are not yet approved. No code changed.

2026-10-02: Requester approved incorporating all four validated design-review corrections before continuing. Corrections: separate lease-fenced execution authority from receipt-only recovery settlement after cancellation; define lease-independent logical Jobs intents with explicit applied/rebind/superseded/exhausted resolution and retained exact acknowledgement proof; require atomic first-accepted-write sink receipts and irreversible generation closure before cleanup; specify enforceable byte/dimension/row limits and adapter/handler deadlines. Verification section and dormant rollout boundaries are approved subject to these corrections. Review validation: 79 existing SQLite Jobs prepared-disposition/cancellation tests passed (7 PostgreSQL cases deselected, 171 warnings, 2.74s); direct temporary-DB probes confirmed cancellation clears all lease fields, late canonical completion is rejected, old unapplied lease commands reject as stale, and a later transition replaces the prior root marker. The first probe's expected rejection category was corrected after diagnostics: canonical late completion returned backend_conflict, not no_transition. No production change is claimed. Consolidated written spec is being drafted; final written-spec approval remains outstanding. Bandit is not applicable to documentation-only changes.
2026-10-02: Consolidated the section-approved Stage 3 design into the written spec and Proposed ADR-058; registered the ADR without changing accepted ADR-022. All four validated corrections are incorporated. Inline spec self-review additionally made state vocabularies/monotonic phases explicit, distinguished definitive admission rejection from unknown admission evidence, preserved the first valid same-identity sink winner despite differing recomputed vectors, clamped each adapter deadline to the remaining batch quantum, covered native/postprocessed materialization bounds, and preserved existing Jobs pause/drain semantics. The acceptance matrix covers each clarification. Fresh validation: five local document links, six unique domain tables, limit arithmetic, ADR index/status, review gate and placeholder checks passed; git diff --check passed. Existing Embeddings runner/types: 48 passed, 108 warnings, 1.36s. Existing focused SQLite Jobs disposition/cancellation: 79 passed, 7 PostgreSQL cases deselected, 171 warnings, 2.53s. These baseline tests do not certify unimplemented Stage 3 behavior or PostgreSQL/RLS; both backends remain explicit implementation gates. No production/test code, entrypoint wiring, migration execution, PR, or implementation plan was created. Bandit is not applicable to this Markdown-only scope. AC5 and task completion remain pending requester review of the consolidated written spec.
2026-10-03: Requester approved incorporating the second written-spec review's four corrections before implementation planning. Documentation-only scope: atomically ordered per-attempt accounting with terminal-state protection against delayed reservations; a certified batch-resolution/write timing contract that avoids 64 serial calls exhausting the acquisition quantum before progress; authoritative sink absence versus unavailable/lagging lookup; and renumbering the proposal because origin/dev owns accepted ADR-058 for Jobs completion row identity. Add explicit acceptance cases and verify the revised documents. No production code or implementation plan is authorized by this approval; consolidated written-spec review remains the gate.
2026-10-03: Incorporated the four approved second-review corrections in the spec and Proposed ADR. Accounting now requires atomic per-attempt ordering, absorbing terminal proof/tombstones, rejection of late reserve/replay, and bounded reservation delivery authority through proof retention/pruning. Resolver/receipt lookup/write are whole-batch ports, not 64 serial per-item calls; the conservative timing plan is 60s preflight + 20s local/DB + 9x10s non-provider calls + 120s provider + 5s teardown = 295s, with 5s scheduling margin. Full dispatch budget is reserved before paid work; serial/undersized contracts are rejected and repeated no-progress exhaustion is a typed failure. Sink lookup requires writer-authoritative accepted/absent decisions; lagging/uncertain visibility is unavailable and cannot authorize another paid call. Updated explicit required acceptance cases for each. Renamed our proposal from ADR-058 to ADR-059 and updated active spec/index/task references; accepted upstream ADR-058 was not modified. Local origin/dev ADR inventory still ends at 058; numbering must be rechecked at later integration. Self-review and 65 local links, six-table/scope gates, placeholder scan, limit/timing arithmetic, absorbing-ledger orderings/expired delivery and authoritative lookup contract probes passed; git diff --check passed. Fresh existing Embeddings runner/types: 48 passed, 108 warnings, 1.20s; focused SQLite Jobs: 79 passed, 7 PostgreSQL cases deselected, 171 warnings, 1.90s. Read-only contract models and baseline tests are not runtime certification of unimplemented Stage 3 behavior; new runtime acceptance tests and PostgreSQL/RLS certification remain required future implementation work. Bandit N/A for Markdown-only changes. No production/test code, migration execution, rebase, PR or implementation plan was created. AC5 and final task completion still await review of the revised written spec.
2026-10-04: Requester approved incorporating the third written-spec review correction and carrying two planning requirements forward. Documentation-only scope: new Embeddings roots explicitly select ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT; validate persisted execution controls on admission replay/binding/resume; require finite durable no-progress exhaustion independent of provider retries, retaining Jobs-only root lifecycle/quarantine authority and unchanged legacy defaults/Admin Webhook restrictions. Carry atomic global capacity reservation under FORCE RLS/non-bypass owner roles and private durable-phase mapping to closed inline trace literals into later Stage 3 planning. Update the spec and Proposed ADR, self-review, run fresh boundary verification, and commit. AC5 remains unapproved; no runtime/test code, migration, implementation plan, PR, rebase, or merge is authorized.
2026-10-04: Fresh local origin/dev ADR inventory includes accepted ADR-059 for backlog-py task editor cutover (dated 2026-10-03). Renumber only this branch's Proposed Embeddings ADR to the next free ADR-060 and update active index/spec/task links; preserve upstream accepted ADR-058 and ADR-059. No upstream rebase or accepted-rationale rewrite. Recheck numbering before later integration.
2026-10-04: Incorporated the approved admission correction in the spec and Proposed ADR-060. New Embeddings roots explicitly pin REQUEUE_NO_ATTEMPT and immutable expiry/retry/quarantine/no-progress controls; replay, binding, and resume reject conflicting stored controls. Both acquisition reclaim and integrity sweep must preserve provider/failure budgets. The neutral Jobs capability must durably bound five consecutive no-progress yield/expiry episodes, including pre-handler expiry; exact episode evidence counts once, new fenced semantic progress resets once, and deferral/proof-release bookkeeping or stale/replayed proofs cannot reset it. Exhaustion is Jobs-owned failure/quarantine, never a domain DLQ or root revival, with existing obligations retained. Added acceptance cases. Carried RLS-compatible atomic cross-owner global capacity reservation and exhaustive private-phase/terminal-to-inline-trace mapping into Stage 3B/3C planning requirements. Renumbered only the proposal/index/active references to ADR-060 because origin/dev now has accepted ADR-059; numbering still requires later integration recheck. Fresh verification: 6 existing SQLite lease-budget tests passed, 29 deselected, 24 warnings, 1.34s (test_jobs_lease_reclaim_budget_sqlite.py; retry-consuming and retry-neutral acquisition/sweep cases). Self-review, 65 local links, six-table/scope/approval gates, control/acceptance requirements, placeholders, ADR inventory, and resource/timing arithmetic passed; git diff --check passed. A read-only no-progress contract model covered expiry/yield limits, deduplication, restart persistence, single-use progress reset, deferral/proof-release exclusion, independent provider budgets, and terminal non-revival. These checks are design/baseline evidence, not certification of the unimplemented Stage 3 neutral capability or PostgreSQL/RLS behavior. No production/test code changed; Bandit is not applicable to this Markdown-only scope. AC5 and task completion remain pending requester review; no implementation plan, PR, rebase, or merge.

2026-10-04: Canonical replacement for this branch's original TASK-13421, which now collides with the unrelated Assess Jobs completion SQL ownership task on origin/dev. The requester approved the consolidated written spec and continued into implementation planning. Preserve the original record in archive; use TASK-13450 for this Embeddings design in active links. No new architecture decisions or runtime implementation in this tracking correction.
2026-10-04: Written approval received, followed by continue. Spec status and ADR-060/index now record accepted written approval; accepted rationale unchanged. Canonical record TASK-13450 preserves the original review notes; old Embeddings TASK-13421 is archived because current origin/dev owns a different active task with that ID. Fresh read-only inventory checked 325 distinct ref tips before selecting free IDs 13450/13451. Verification of approved docs/planning links and metadata: 67 local Markdown links valid, approval/index consistent, task sections balanced, original history preserved. Baseline inline runner/types: 48 passed, 108 warnings, 1.20s; selected SQLite lease-policy tests: 6 passed, 29 deselected, 24 warnings, 1.08s. git diff --check passed. No runtime/test code or migrations changed, Bandit N/A for Markdown-only scope. Future Stage 3 and PostgreSQL/RLS certification remain unimplemented; design acceptance is not activation or merge approval.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Requester approved the consolidated Stage 3 durable workflow spec and ADR-060 on 2026-10-04. The design defines dormant Jobs-rooted persistence, exact owner/lease/control authority, immutable generations/manifests, bounded sink/accounting recovery and no-progress exhaustion, plus concrete verification and staged rollout gates. Recorded accepted metadata and preserved review history under TASK-13450 to avoid the unrelated upstream TASK-13421 collision. Handed off to separate Stage 3A planning TASK-13451; no runtime implementation, production wiring, PR, or merge.
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
