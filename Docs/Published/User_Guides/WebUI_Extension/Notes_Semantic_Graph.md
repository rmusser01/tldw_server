# Notes Semantic Graph

The Notes Graph can add similar-content relationships from an explicitly
enabled semantic index. The WebUI and browser extension use the same workflow.
Semantic relationships are a review aid: they do not change Notes or create
durable links unless you convert one to an ordinary manual link.

## Before You Enable It

Open Notes, select a Note, switch to **Graph**, and open the **Similar content**
tab. The setup disclosure shows the effective:

- embedding provider and model;
- sanitized endpoint destination, when applicable;
- embedding execution boundary as local, external, or unavailable;
- vector storage boundary and backend;
- exact outbound categories: Note titles and Note body chunks;
- active Note count and estimated chunk/run counts;
- resolved dimensions, cosine metric, and configuration details.

Read this disclosure before selecting **Enable semantic index**. Consent is
explicit and revision-bound. The server does not read or transfer Note text for
indexing before consent. If model dimensions are unknown, the disclosure says
that the server will send one fixed non-user probe after consent and before it
reads or transfers any Note text.

The configured provider, model, endpoint, execution boundary, storage backend,
and dimensions are pinned. The service never silently falls back to another
provider and does not keep a durable cross-run embedding cache. A disclosure
boundary change requires **Review consent and rebuild**. An incompatible model
or dimension change requires a rebuild. Changing vector storage requires you to
delete the existing index before setting it up again.

An administrator can disable admission and semantic queries with the semantic
index kill switch. Indexing also requires a dedicated worker; recovery,
incremental updates, retries, and physical deletion require semantic
maintenance. When either service is unavailable, the panel explains which
operation is unavailable. This does not disable the ordinary Notes graph.

## Build And Maintain The Index

Select **Enable semantic index**, review the confirmation, and select
**Enable**. The panel reports one of these states:

- **Off:** no usable semantic generation is published.
- **Preparing:** the first generation is being built.
- **Ready:** the active generation is usable and current enough to query.
- **Updating:** a usable generation remains published while changed Notes are
  processed.
- **Needs attention:** consent, configuration, provider, coverage, or cleanup
  needs action.

The progress display distinguishes indexed, excluded, failed, and pending
Notes. Empty Notes and Notes above the indexing size limit are excluded rather
than failed. A provider or chunk failure reduces coverage; successfully indexed
current Notes can remain queryable. An edit made during a run remains pending so
stale vectors cannot become authoritative. A generation is published only when
its fenced manifest and integrity checks complete.

Use the available actions as follows:

- **Retry failed Notes** retries only failures in the current generation.
- **Rebuild index** creates a replacement generation. The last published
  generation remains available until the replacement is complete.
- **Cancel indexing** requests cancellation. A provider call already in flight
  can still finish or incur cost, but cancelled work is not published.
- **Review consent and rebuild** accepts the current disclosure and starts a
  compatible replacement generation.
- **Disable and delete index** stops new semantic use and queues deletion of
  live derived vectors. Ordinary graph relationships are unchanged.

Management actions require `notes.graph.semantic.manage`. Users with only
`notes.graph.read` can inspect an already-usable index and similar-content
relationships but cannot enable, rebuild, retry, cancel, renew, or delete it.

## Show Similar Content

Semantic relationships are off by default in each Graph session. Focus a Note,
then enable the **Similar content** checkbox. A semantic query is never sent for
**All notes** or another unfocused graph. The controls are:

- **Neighbors:** requested related Notes for the focus Note. The default is 10;
  the server advertises and enforces the effective maximum, never above 50.
- **Minimum passage similarity:** minimum cosine similarity from 0.00 to 1.00.
  The default is 0.75.
- **Reset Similar content controls:** restores the session defaults and starts
  the semantic result at its first page.

These settings and the checkbox are session-only. Changing a semantic control
starts from the first semantic page instead of reusing a cursor created for
different controls or revisions.

The number labeled **Passage similarity** is the strongest current matching
passage pair found between the focused Note and the related Note. It is a
finite, normalized cosine-similarity value, not a probability, factuality
score, confidence score, or whole-document judgment. Raising the minimum can
remove useful broad relationships; lowering it can add weaker relationships.

## Review Relationships And Evidence

Semantic relationships use a distinct dotted treatment and the **Similar
content** label in Canvas. They also appear in the keyboard-accessible
**Relationships** view. Color is not the only indicator. Select a relationship
or expand its evidence disclosure to review the current source and target
passage excerpts, model label, generation, and revision context.

Evidence is bounded and field-relative. Offsets identify positions in the Note
title or body field shown with the excerpt; they are not offsets into a combined
document. The response can include up to three current passage pairs per edge,
and the server applies per-excerpt, per-edge, and total response-byte limits.
Truncation metadata explains when evidence was bounded. Evidence is omitted
rather than served when current Note revisions or ownership no longer match.

Semantic edges are derived and provisional. They do not appear when graph edge
types are omitted; the ordinary graph continues to return only manual links,
wikilinks, backlinks, tag membership, and source membership by default.

## Convert A Relationship

To retain a reviewed relationship, select the semantic edge and use its manual
link action. The client submits the focused Note as the source, the related Note
as the target, a stable idempotency key, and the active semantic generation ID
through the existing Note-links API.

Conversion requires `notes.graph.write`. The server revalidates the owner,
dataset, Note pair, current generation, and current revisions. Success creates
an ordinary canonical manual link; it does not promote, copy, or mutate a
semantic edge. The graph refresh then shows the manual relationship. Repeating
the same accepted request is idempotent. A stale generation or revision, wrong
pair, wrong owner, or already-existing manual link returns a typed conflict and
does not create a duplicate.

## Degraded, Offline, And Read-Only Behavior

If a semantic query fails, the workspace keeps the ordinary graph available and
shows typed semantic status instead of failing the entire Graph view. A
temporarily unavailable provider can pause updates while a current compatible
published generation remains queryable. There is no provider fallback.

When offline, the last successfully loaded graph can remain visible and is
marked offline. Semantic management, control changes that require a request,
and conversion remain disabled until reconnection. The client does not relabel
cached semantic data as current.

Users without semantic management permission see the setup and status as
read-only. Users without graph-read permission do not receive semantic graph or
index data. Users without graph-write permission cannot convert a relationship.

## Deletion, Erasure, And Portability

**Disable and delete index** logically disables the dataset first, then removes
and confirms its vectors from the live ChromaDB or pgvector backend. Until
cleanup is confirmed, status remains cleanup pending or needs attention. Data
subject erasure fails closed: canonical Note or account deletion does not finish
until semantic vector deletion is confirmed.

Live cleanup does not rewrite offline backups. Deleted derived vectors can
remain in ordinary backups until the deployment's normal backup-retention
period expires. Restore procedures must enforce that policy; the live UI cannot
promise immediate destruction of offline backup copies.

Notes Sync, export, and restore do not transfer semantic vectors, provider
configuration, consent, generations, or manifests. Individual Sync edits to an
already-enabled local dataset only mark local indexing work pending. A restored
or imported dataset starts with semantic indexing off and requires a new local
disclosure and explicit consent.

## Troubleshooting

- **Setup is unavailable:** ask an administrator to check the configured
  provider/model, durable credential, sanitized endpoint origin, dimensions,
  vector backend, and semantic kill switch.
- **The build stays queued:** the dedicated semantic Jobs worker is not running
  or the Jobs service is unavailable.
- **Edits or cleanup stay pending:** semantic maintenance or the worker is not
  running. Retry only after the service is restored.
- **Needs attention after configuration changed:** review the new disclosure.
  Renew consent for boundary-only changes, rebuild incompatible generations, or
  delete first when the vector backend changed.
- **Coverage is below the active Note count:** inspect excluded, failed, and
  pending counts. Retry failures; edit oversized Notes; allow pending edits to
  finish.
- **No relationships appear:** focus a Note, enable **Similar content**, lower
  **Minimum passage similarity** if appropriate, and verify that a usable
  generation covers both Notes.
- **A relationship disappeared:** one of the Notes changed, became inaccessible,
  was deleted, or no longer meets the selected controls. Refresh the graph.
- **Conversion fails:** refresh the graph and evidence. The generation, Note
  revisions, pair, permissions, or existing manual-link state may have changed.

Operator route, error, backend, and environment details are documented in the
[Notes Semantic Index And Graph API](/docs-static/API/Notes_Semantic_Index.md).
