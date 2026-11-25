# ADR-099 · Admit Research Workspace generation through native Data Tables

**Status:** Proposed
**Date:** 2026-11-24
**Decision owner:** Research Workspace workstream
**Related task:** TASK-15639
**Related design:** [Research Workspace Data Tables](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Design/research-workspace-data-tables.md)

## Decision

The authenticated Workspace endpoint qualifies the explicit provider and model
against the server-managed Data Tables route. New admissions check selected
Workspace associations and native Media revisions. Media owns the table and
its source references. Each reference records a digest of the bounded resolved
text because a document revision or transcript can change independently of
its parent Media row. Jobs checks both when resolving evidence and owns the
generation receipt.

The owner, Workspace, and client operation identify one Media UUID. The
generation reservation binds that UUID to the normalized request. Media
commits the table and sources together, then Jobs accepts the exact
reservation idempotently. Replay can recover an existing admission or finish
an interrupted admission if the original evidence and reservation still
match. A new native generation supersedes the earlier reservation. Native
Jobs completion remains the authority that publishes rows.

## Alternatives considered

The legacy Data Tables endpoint accepts broad source inputs and uses the
deployment's default provider. A Workspace tag is useful for discovery and
does not prove membership or authorize generation. A Workspace-owned copy of
the table would split editing and export from the canonical table.

## Consequences

Chatbook keeps a Workspace artifact reference to the native UUID and
published version. Native Data Tables remains the owner of content, Jobs,
and exports. This contract uses server-managed credentials and leaves
request credentials and user-owned keys out of durable Jobs.
