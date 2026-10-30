# Research phase continuation and recovery

Task: TASK-13260.278.18.83.45. S04 remains open.

A phase commits its executable successor state and returns, but the worker never queues that next phase. Autonomous research and an edited approved outline can therefore stay queued without a Job. Existing Resume also loses the original checkpoint generation and locked-outline flag.

Keep phase work in normal Jobs. Before the parent completes, reconcile its durably committed next dispatch, enqueue the existing idempotent successor, and conditionally bind that successor to the still-current Research state. A retry after a committed phase repairs this handoff without repeating that phase's providers. Preserve owner, original checkpoint generation, policy version and approved-outline identity. Internal continuations use the existing fair-share/quota exemption pattern; initial run admission remains unchanged.

Execution and publication must retain exact persisted Job UUID/owner/lease and current phase/control/generation. A narrow Research DB compare-and-set records its event only on a matching transition; pause/cancel/stale work cannot overwrite newer state. An acquired but not yet bound successor defers without consuming retries. Resume reuses authorized dispatch identity. Terminal Jobs failures reconcile only their exact currently owned Research phase; a granted retry remains nonterminal.

The implementation must explicitly record any pending dispatch data needed across the two stores before changing production. This is not a cross-store transaction or universal exactly-once provider guarantee. Do not repurpose Admin-webhook-specific prepared WorkerSDK operations or change generic Jobs admission.

ADR assessment: existing ADR-003 Jobs selection, ADR-021 worker lifecycle, and ADR-058 locked Job identity govern. This repair stays Research-local and adds no service, dependency or generic Jobs contract. Reassess before introducing a new durable architecture rule.

## Validation

Use actual WorkerSDK, Jobs and Research DB lifecycle with controlled provider seams. Establish the missing successor, stale-phase side effects and terminal failure mismatch first. Then cover autonomous full chain, edited outline, checkpoint parking, pause/resume retaining generation, cancel/stale/lease rejection, enqueue/bind/completion interruption and retry, acquisition before binding, concurrent replay, recollection generation, normal admission limits and cooperative stop/drain. Assert provider call counts and durable Jobs/session state. Official isolated PostgreSQL remains required for supported backend/account scope, with no operator SQL.

## Durable dispatch descriptor

Use one internal phase_dispatch_json session field, versioned and validated. Commit it with the phase/status compare-and-set and matching event. Retain phase, original checkpoint generation, policy version, locked-outline flag and exact predecessor/current Job UUID identity needed by reconciliation; keep lease credentials out and keep active_job_id as the numeric binding. Preserve this authorization through binding, execution and pause/resume, replace it on matched phase advance, and invalidate it on terminal/cancel. Do not expose it through SSE or public session responses. Empty legacy metadata may bootstrap only from the verified current bound persisted Job, including its actual payload; ambiguous recovery refuses instead of inventing identity from phase alone.

## Conditional checkpoint and explicit recovery authority

Stage one unique immutable file, then use an explicit DB transaction to commit the matching owned waiting session, still-pending checkpoint, winning patch/resolution, artifact manifest, checkpoint/artifact/status events and pending dispatch together. A lost acknowledgement uses exact manifest readback; it never deletes a possibly committed canonical file. Losing Pause/cancel/concurrent approvals publish no partial checkpoint or manifest.

The closed dispatch authorization distinguishes initial, automatic continuation, checkpoint and Resume. Only a legal automatic phase edge with the inherited checkpoint/policy/locked-outline identity grants predecessor replay or terminal-outcome authority. Explicit decisions cannot be revoked by an old predecessor. An unbound Resume retains the same possibly admitted unknown-ACK candidate nonce and changes its authorization; stale cleanup must recognize that exact current candidate and cannot cancel it. A previously completed bound/parked delivery gets a new delivery nonce while retaining its approved policy. Lease credentials remain absent from this internal descriptor and public responses.

A losing dispatcher does not cancel its candidate on a generic bind-CAS loss: that delivery may already have advanced the owned run, or be requeued for a granted handoff retry. Only the current worker's exact authority guards retire obsolete delivery. Candidate admission uses authoritative scoped UUID readback; an ended candidate cannot be reported executable merely because idempotent create returned its key. A bound delivery ending without its Research transition is reconciled on normal owner reload, without silently granting another retry.
