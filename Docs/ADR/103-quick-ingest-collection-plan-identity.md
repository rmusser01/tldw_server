# ADR-103 — Quick Ingest collection plan identity

**Status:** Proposed

**Date:** 2026-11-26

**Backfilled from:** not backfilled

**Decision owner:** Quick Ingest maintainers

**Related task:** TASK-15659.1

**Related spec/plan:** [Resume collection setup](../Design/2026-11-26-quick-ingest-collection-reuse.md)

## Decision

Give a Quick Ingest draft and each of its sources a durable identity in the authenticated owner's collection storage. An atomic setup request returns the original collection and source rows when the caller repeats those identities.

## Context

A browser can lose the response to a committed collection write. A second tab may retry the same draft while the first is still running. Users can also edit a source's metadata and move it to a different ordinal. Recreating a collection by name or its items by URL would confuse separate imports of the same material.

## Alternatives considered

| Approach | Consequence |
| --- | --- |
| Look up the editable item metadata or the original ordinal before inserting | Two callers can both see an absent item. Reordering frees the original ordinal for another insert. |
| Keep the server-assigned IDs only in the browser | The first acknowledgement can be lost before the browser learns an ID. |
| Enforce authenticated plan and source identities in collection storage | A repeated write returns the authoritative row, including edits and later processing outcomes. |

## Consequences

The browser retains the draft identity across retries and uses a separate identity for a fresh draft. The collection store commits the collection, its source bindings and the requested additions atomically. Later setup requests preserve existing rows. Clients dispatch work using the authoritative item receipt. Jobs keep their owner-scoped logical submission keys. A direct, non-idempotent transport requires server-side admission before performing its side effect. Ambiguous delivery remains visible to the caller until an authoritative outcome exists. [Jobs and Scheduler](003-jobs-vs-scheduler-default.md) · [Jobs completion row identity](058-jobs-completion-row-identity.md) · [Jobs resource admission and domain publication](094-jobs-resource-admission-and-domain-publication.md)

Existing collections remain addressable by their IDs. Historical rows without a recorded binding keep their existing identities. Matching editable names or URLs cannot establish which earlier draft created them.
