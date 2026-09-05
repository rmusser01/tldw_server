# ADR-043: Managed llama.cpp manual slot snapshots

Status: Accepted
Date: 2026-09-04
Decision owner: Requester approved written design and subagent-driven implementation.
Related task: TASK-13159
Design: [Manual slot snapshots (repository source)](https://github.com/rmusser01/tldw_server/blob/3869302734f194b9389b37543c0e25fd6af7d5fa/Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md)

## Context

Managed runtimes can serve more than one user's prompts. Their cache artifacts
are sensitive runtime state, not user-owned conversation records. The current
supervisor owns process lifetime, while upstream save/restore operations can
outlive an HTTP client and cannot safely be assumed replayable.

## Decision

Provide opt-in manual Save/List/Restore/Delete for administrators on managed,
single-model runtimes only. Keep Pause/Resume process semantics unchanged.
Snapshots belong to the managed profile and installation. Enforce admin access
on metadata, mutations and operation receipts; never provide cache downloads or
prompt previews through this feature.

Store immutable binaries and versioned metadata in private service-owned storage.
Give the child a per-launch working directory, not the committed catalog. Publish
metadata last, hash before restore, fail closed on unknown compatibility, and
prune only after a verified save. Default retention is 10 per profile. Explicit
snapshot deletion is required before profile deletion.

As a narrow exception to ADR-003's Jobs default, execute mutations through the
owning supervisor with durable operation receipts and launch-generation fencing.
Do not enqueue restore work for generic workers, retry on lease recovery, or
replay a dispatched operation after restart. Use existing service startup/shutdown
ownership under ADR-021 and checked configured-origin transport under ADR-030.
An unknown outcome requires operator recovery and confirmed child exit before
further snapshot mutation. Ordinary inference availability is reported separately.

## Alternatives

| Alternative | Trade-off and reason not selected |
| --- | --- |
| Generic Jobs execution | Fits ops visibility, but process-local authority and uncertain side effects require non-replay and owner affinity. A narrow supervisor operation avoids unsafe generic retry semantics. |
| Synchronous request only | Simpler, but page disconnects and long filesystem operations obscure durable completion and recovery. |
| Per-user snapshots | Attractive future isolation, but current shared slots lack trustworthy user ownership; an API role check cannot invent that association. |
| Automatic save on Pause and restore on Resume | Changes existing behavior and recovery guarantees; user explicitly selected manual first. |
| Arbitrary paths and external runtime support | Broader usefulness, but weakens storage and lifecycle authority and introduces untrusted binary ingestion. |

## Consequences

Operators receive explicit, inspectable actions without a promise of conversation
resumption. The service must implement private storage, durable receipts,
single-owner fencing, and conservative compatibility checks. Multi-worker
deployments must reject requests at a non-owner instead of acting on a different
process. Disk encryption remains an operator concern. No new database schema is
required by this proposal; manifests and receipts are versioned local files.

This decision does not supersede ADR-003 globally.
Any later Jobs integration must preserve single-dispatch and runtime-owner
constraints, and must not turn a recovery event into an automatic restore.
