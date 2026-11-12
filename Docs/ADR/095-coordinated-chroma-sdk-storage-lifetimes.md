# ADR-095: Coordinated Chroma SDK Storage Lifetimes

**Status:** Proposed
**Date:** 2026-11-02
**Backfilled from:** not backfilled
**Decision owner:** Claims workstream; implementation under independent review
**Related task:** TASK-15454
**Related spec/plan:** `Docs/superpowers/specs/2026-11-01-claims-jobs-stage2c-cluster-rebuild-design.md`; `Docs/superpowers/plans/2026-11-01-claims-jobs-stage2c-cluster-rebuilds.md`

## Decision

Coordinate canonical per-owner Chroma access in the shared Embeddings adapter, keeping validation and the complete SDK lifetime inside the same thread/process exclusion. Existing-only maintenance must distinguish valid buffered data from incomplete materialized storage; it never authorizes repair of missing files.

## Context

Claims cluster rebuilds need bounded reads of existing owner vectors. The old process-local manager RLock covered lookup, but returned collection objects could be used after that lock was released, and other processes did not share the lock. The installed Chroma SDK can initialize or repair index files during reads. Deliberately incomplete-fixture probes demonstrate that behavior, not a TOCTOU race or corruption of a healthy store.

A pre-open stat, a Claims-only mutex, or catching a provider exception after a write cannot establish the existing-only boundary. A live cached SDK can also retain background work outside an operation lock. Jobs resource admission remains logical execution policy, not a lock for every physical vector writer. ADR-022 governs API/pipeline ownership and deliberately leaves vector-store policy separate; the Jobs/domain ownership decision remains unchanged.

## Alternatives considered

| Option | Why not selected |
| --- | --- |
| Lock only rebuilds or collection lookup | Misses raw collection operations, other canonical writers and other processes. |
| Keep idle live SDKs outside shared storage exclusion | Leaves initialization, background work or shutdown outside the protected lifetime. |
| Hold process exclusion for every long-lived manager object | Nonclosing callers and retained handles could pin a store indefinitely. |
| Stat first and open the normal SDK afterward | Does not serialize cooperating changes or prevent initialization of incomplete state. |
| Copy a live store or introduce another backend | No consistent-copy protocol was proved, and neither change is required by the current work. |

## Consequences

- Canonical persistent managers cache inert guarded identities, not idle live SDKs. A per-store session covers SDK construction, nested client/collection operations and shutdown before releasing native exclusion. Related vector batches can share one synchronous session on one thread.
- Native file coordination is shared by canonical application processes. No global cache mutex is held while waiting or initializing. Lock metadata is outside the owner persist tree; maintenance must not create absent owner storage. Lock files are not unlinked during normal operation.
- Storage identity cannot depend on an unchecked symlink alias. Inherited sessions and cached handles must fail closed after a process fork rather than reuse a parent's SDK, recursion depth or lock ownership.
- A failed SDK startup is cleaned up under the same lock. A failure to establish shutdown retains the SDK and exclusion in fail-closed quarantine; operators restart the affected process rather than force-unlock a potentially active SDK.
- Existing-only access uses no-create opening after locked schema, identity and structural index validation. Aggregate catalog, SQL, metadata and element work must be bounded and a validated snapshot reused within the held session. Metadata is parsed without executing pickle objects. Proven absence, access failure and incompatible/incomplete storage remain distinct; failures never become successful empty embeddings or a stub fallback.
- The proposed correction supports ordinary empty and buffered collections by reading never-persisted logical state from the existing SQLite log only when checkpoint and metadata coverage establish completeness. Materialized targets continue through the qualified SDK path. This is a bounded read adapter, not a second persisted vector backend or permission to reconstruct missing materialized files. Default writer configuration remains unchanged.
- Qualification is explicit, not a promise for all `chromadb>=0.4.0` versions or disk layouts. The current correction targets the fresh Rust 1.5.5/MD5 quiescent profile, including default buffering; compatibility and coverage evidence remain subject to independent review. Unknown profiles fail closed. Windows execution remains unqualified until exercised.
- Session startup/shutdown and SDK acquisition metadata have overhead. This decision makes no throughput, hard-RSS, arbitrary graph-authenticity or immunity-to-uncooperative-filesystem-writers claim.
- Existing public writer creation behavior remains permitted. Injected clients, external SDK processes and old binaries aimed at the same canonical path are not implicitly coordinated; rollout must drain nonparticipating clients first.
- Jobs retains execution, retry, lease and cancellation authority. This protocol introduces no Claims queue, polling loop or alternative worker lifecycle.

## Follow-up

- Keep this record Proposed until the implementation and its compatibility/cost limits have completed review; do not infer human acceptance from its presence.
- Task 11 must qualify the full touched branch, document SDK/profile support, contention and shutdown-quarantine recovery, and retain disclosed test/lint baseline limitations.
- The private Claims resource/publication record is now ADR-086 after an identifier-only repair of the collision with published AuthNZ ADR-085; accepted rationale is unchanged. The refreshed audit checked current dev and all 39 open PRs on 2026-11-02. Recheck private identifiers before any separately authorized publication.
