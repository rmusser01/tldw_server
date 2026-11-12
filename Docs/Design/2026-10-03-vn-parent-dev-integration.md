# VN Parent And Current Dev Integration

Associated original work: TASK-13369, Address PR 3016 Qodo durability findings.
Human approved publication, the bootstrap fix, current-dev integration and fresh
review/CI. This does not supply the child's separate human Change summary,
authorize PR body edits, bypass merge gates or permit cleanup.

## Inputs

Parent reviewed baseline393c365 and actualdev86e287fee7bfa1a1588639232e35db3666851ded.
Seven real conflicts span service/worker, DB, frontend and their tests. Automatic
merges are also checked: generation routes lost async declarations while keeping
await expressions. Restore their asynchronous owning-thread boundaries.

## Contracts

- Keep per-variant frozen recipes, fenced outcomes, receipt first-completion,
  counters, approved bytes and deliberate-deletion receipts unchanged.
- Independently retain authored recipe, pinned execution recipe and source batch
  snapshots, strict Retry validation, slot failure/latest ownership and reload
  recovery. Never derive a historical missing snapshot from current inputs.
- Positive new work stores both representations; authored zero-work retains the
  legacy path rather than inventing an empty durable ledger.
- V1 enqueue failure stays recoverable on its original batch and Jobs identity.
  Legacy V0 retains exhausted-fanout ownership and zero-work completion.
- A failed V1 variant does not terminalize its unfinished siblings. Slot
  provenance is display metadata, never a second queue or lease authority.
- Retry copies frozen failed-batch inputs; Regenerate uses current inputs for a
  new draft. Transaction/rollback tests use real failed historical sources and
  assert those sources remain unchanged, not an unrelated fresh Start.
- Retry admission uses the existing capacity-aware count inside its transaction:
  hidden active reservations and completed assets count, failed/cancelled
  reservations release capacity. Rejection leaves batch/recipe/receipt/Jobs
  state unchanged.
- A queued zero-work batch does not prove completed fanout merely because
  enqueued/planned counts are zero. Interrupted receipt recovery retains one
  original deterministic Jobs parent, then actual zero-work completion.
- A typed nonretryable legacy failure records its terminal VN failure even when
  Jobs has unused retries; ordinary retryable exceptions retain that budget.
- A child execution snapshot is a fenced worker write. Resolve candidate inputs
  first, then admit the current variant claim and pin the first-writer snapshot
  inside one existing repository transaction on its owning thread. Rejected
  lease, claim or cancellation admission must not persist execution settings.
  Retain the fresh admission before actual backend work and existing inline/
  private-memory/active-caller fallbacks; no new lease authority or schema.
- Legacy fixtures explicitly create V0 batches without per-variant ledger rows.
  Their failure/provenance assertions remain; new V1 work retains recoverable
  enqueue failure, immutable variant outcomes and successful-review precedence.
- Keep caller-owned private memory and active transactions as explicit fallbacks.
  No external exactly-once guarantee.

## Verification

Bounded native tests precede repair of each newly demonstrated integration issue.
Frontend contracts, DB union/rollback, frozen input/receipt thread boundaries,
failure continuation, cancellation and storage replay are owned scope. Apply
scoped lint/compile and Bandit baseline checks, then independent SPEC/QUALITY
review before normal publication. Current dev already has an equivalent narrow
SQLGlot AST fix; retain it and the independently reviewed bootstrap regression.
Matching-dependency local tests are not native CI, PostgreSQL, UI/E2E or whole
repository verification. Exact-head external review and required strict-current
dev contexts must pass before the normally authorized parent merge.

Tracking limitation: current dev introduces three tasks with ID TASK-13369.
The CLI selects an unrelated task; new-task CLI creation failed, and the official
MCP creation timed out without a confirmed outcome. Never edit an ambiguous ID,
manually rename/renumber tasks or mark the original external gates complete.
The exact original task association and recorded approval precede these edits;
the live SDD ledger and new evidence retain current integration progress.

ADR assessment: no new policy. Existing snapshot, ownership and validation
contracts under ADR002/004/006 govern this integration.
