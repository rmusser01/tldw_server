# Notes Second-Brain Graph Workspace and Reviewable Suggestions Design

- **Date:** 2026-08-26
- **Task:** `TASK-13138`
- **Status:** Approved in chat after UX, architecture, contract, and operational review
- **Reviewed baseline:** `origin/dev` at `2306c1939f3b460f9c62da8ae83a1aa47c02ee0d`
- **Deferred tasks:** `TASK-13134`, `TASK-13135`, `TASK-13136`, and `TASK-13137`

## Summary

Turn the existing Notes graph from a secondary modal into a first-class Notes
view mode shared by the WebUI and browser extension. Add an explicit, on-demand
workflow that analyzes one selected note, searches the owner's active Notes
library with bounded full-text retrieval, and uses one configured LLM invocation
to propose related-note links and tags.

Every generated result is provisional. A suggestion appears as an overlay and in
a grounded review inspector, but it does not change the authoritative graph,
create a manual link, or apply a tag until the user accepts it. Rejection is bound
to the unchanged source material so the same pair or tag does not immediately
reappear. Accepted results use the existing Sync-aware manual-link and keyword
mutation paths.

This is the first useful Second-Brain slice, not the final knowledge-organization
system. Embeddings, semantic edge types, automatic background organization,
library-wide themes, and saved layouts remain separate follow-up work.

## Current State

The repository already provides the foundations this design extends:

- `GET /api/v1/notes/graph`, `GET /api/v1/notes/graph/orphans`, and focused
  neighbor APIs return bounded authoritative graph projections.
- Explicit manual links are durable, owner-scoped, Sync-aware records. They may
  be directed, but their public create contract defaults to undirected links and
  canonicalizes undirected endpoint order.
- Wikilinks, backlinks, tag membership, and source membership are derived graph
  relationships.
- SQLite Notes FTS5 is maintained in the note transaction by insert, update, and
  delete triggers. PostgreSQL maintains `notes_fts_tsv` with a before-write
  trigger and GIN index.
- The current Notes graph UI is a lazy modal using Cytoscape and Dagre. It offers
  radius, node-limit, zoom, and fit controls, but it is not part of the primary
  Notes workspace and does not support reviewable suggestions.
- `NotesManagerPage` and its supporting components live in the shared UI package,
  so the primary WebUI and browser extension can use one implementation.
- Notes keywords and note-keyword relationships already have owner-scoped,
  idempotent, Sync-aware mutation paths.
- Jobs provides durable admission, owner scoping, leases, cancellation, worker
  execution, and operator visibility. Jobs defaults to automatic retries, which
  this provider-calling workflow must override.

The existing frontend keyword helper is a local word-frequency heuristic. It is
not an LLM-backed Notes organization service and is not reused as one.

## Goals

1. Make Graph a first-class Notes view mode without introducing a dedicated
   route or a routine modal.
2. Keep the existing Notes graph APIs and manual-link storage authoritative.
3. Analyze only the selected note and search the owner's entire active Notes
   dataset for direct related-note candidates.
4. Generate at most five related-note suggestions and five tag suggestions with
   grounded evidence and deterministic cost limits.
5. Require explicit review before any link or tag mutation.
6. Preserve decision behavior across reloads, retries, note edits, cancellation,
   and process interruption.
7. Share the complete experience between the WebUI and browser extension.
8. Preserve tenant isolation, Sync invariants, privacy, accessibility, and
   bounded operation behavior on SQLite and PostgreSQL.

## Non-Goals

- Embedding creation, vector retrieval, or semantic-similarity graph edges.
- Model-selected semantic relationship types or arbitrary edge properties.
- Automatic or scheduled background organization.
- Library-wide recurring-theme extraction or theme nodes.
- Saved named graph views, persisted node coordinates, or synchronized layouts.
- Automatic acceptance, bulk acceptance, or silent canonical mutation.
- A root-level suggestions API or a generic AI-organization API.
- Synchronizing provisional runs, suggestions, or rejection decisions as new
  Sync v2 domains. Accepted links and keywords continue to synchronize through
  their existing domains.
- Replacing the current Notes FTS implementation or adding a new search service.
- Claiming probabilistic confidence or showing model-generated percentages.

## Product Principles

1. **The graph is authoritative; suggestions are overlays.** Existing graph
   responses remain the source of truth. Provisional edges never enter
   `NoteGraphResponse`.
2. **Review is mandatory.** Generation is initiated by an explicit user command,
   and every result requires an individual accept or reject decision.
3. **Evidence is inspectable.** A relationship suggestion shows excerpts from
   both notes. A tag suggestion shows the selected-note excerpt that supports it.
4. **Cost is visible and bounded.** The UI identifies the resolved provider and
   model before generation, and one run permits at most one provider invocation.
5. **Content is untrusted data.** Text inside a note cannot change instructions,
   tools, provider settings, candidate identity, or output bounds.
6. **Accepted state uses existing authority.** The feature does not create a
   second manual-link or tag model.
7. **The focused workflow comes first.** Large libraries start from the selected
   or most-recent note. Whole-library rendering remains available only when the
   existing server graph cap can contain it.

## User Experience

### Notes View Mode

Graph becomes one of the existing Notes view choices. Selecting Graph keeps the
user on the Notes route and retains the Notes navigation/list context. The
existing editor action for opening the graph switches to Graph mode and focuses
the current note instead of opening `NotesGraphModal`.

On desktop, the workspace has three unframed regions:

1. The existing Notes sidebar for finding and selecting a note.
2. A flexible graph canvas for the authoritative graph plus provisional overlays.
3. A fixed responsive inspector with `Details` and `Suggestions` tabs.

The page toolbar provides:

- node search;
- focus-current-note;
- edge visibility toggles for manual, wikilink/backlink, tag, source, and
  provisional suggestion edges;
- a session-local layout menu;
- fit-to-view;
- a `Canvas` / `Relationships` view switch.

Icon controls use the shared icon library and tooltips. Labels appear for the
focused, selected, or hovered node rather than every node. Arrowheads appear only
on directed authoritative edges. Undirected edges remain visually undirected.
Routine content is not nested in decorative cards.

### Focus And Expansion

Entering Graph mode focuses the selected note. If no note is selected, it uses
the most recently opened active note. If the library has no active notes, the
workspace shows the standard Notes empty state and disables generation.

The initial graph uses the existing focused-neighborhood API and server caps.
Selecting another note makes it the inspector target without discarding the
current canvas. An explicit focus command reloads a bounded neighborhood around
that note. Expansion is user initiated and continues through the existing cursor
and truncation contract.

`All notes` is offered only when the active-note count is no greater than the
server's conservative `all_notes_note_cap`, default 100, and never greater than
the effective `max_nodes` cap. The lower note cap leaves bounded room for tag and
source nodes; the response still reports ordinary node/edge truncation if those
projections consume the remaining budget. Above the eligibility threshold the
control is disabled with a concise explanation and the focused workflow remains
available. This slice does not virtualize an unbounded whole-library graph.

### Inspector

`Details` shows the selected node's title, tags, source, and grouped incoming and
outgoing relationships. Selecting a relationship focuses its counterpart.

`Suggestions` shows:

- the resolved provider and model;
- `Generate` or `Regenerate`;
- queued, running, cancelling, publishing, failed, stale, and completed status;
- related-note review rows;
- proposed-tag chips;
- individual accept and reject commands;
- an overflow command to reset dismissed suggestions for the selected note and
  current content fingerprint.

A related-note row contains the target title, `Strong match` or `Possible match`,
a concise rationale, one or two selected-note excerpts, and one or two target-note
excerpts. A tag row identifies whether it will reuse an existing tag or create a
new one and shows selected-note evidence. New tags are visually distinguishable
but use the same review controls.

Pending relationship suggestions appear on the canvas as dashed provisional
edges. They are client overlays with suggestion IDs, not new graph edge types.
When a suggested target is outside the loaded authoritative neighborhood, the
client also renders one ephemeral provisional target node from the bounded
suggestion response. It is removed with the overlay and never inserted into the
authoritative graph response.
Accepting a relationship removes the overlay and refreshes the authoritative
graph. Rejecting it removes the overlay and keeps the rejection available to the
server suppression filter.

### Responsive And Accessible Behavior

At narrow widths, the canvas remains the primary region and the inspector becomes
an in-page bottom region with a stable height and scroll boundary. It is not a
routine modal. The Notes sidebar uses its existing narrow-screen navigation
behavior.

The `Relationships` view is a structured, keyboard-accessible equivalent to the
canvas. It exposes the same selection, relationship details, evidence, and
decision commands. Status never depends on color alone. Focus order, visible
focus, screen-reader names, reduced motion, long titles, long tags, and high zoom
must remain usable. Dynamic labels, loading indicators, and suggestion state must
not resize the canvas controls or overlap adjacent content.

When the client is offline, the workspace may show the last authoritative graph
already available to the Notes client, clearly marked as offline. Generation,
cancellation, acceptance, rejection, and dismissal reset remain disabled until the
server is reachable.

## Architecture

### Component Boundaries

The feature has five focused units:

1. **Graph workspace:** renders existing graph responses, session-local layout
   state, accessible relationship lists, and provisional client overlays.
2. **Suggestion API:** owns nested Notes graph routes, authorization, request
   validation, idempotent admission, status envelopes, pagination, and decisions.
3. **Candidate retriever and prompt builder:** creates a deterministic FTS
   shortlist, bounded evidence windows, and a closed provider request.
4. **Suggestion worker and publisher:** executes one provider call, validates the
   result, stages it in ChaChaNotes, and publishes it through a crash-repairable
   Jobs/Notes handshake.
5. **Suggestion store and decision service:** owns run/suggestion persistence,
   fingerprints, suppression, acceptance leases, retention, and lifecycle hooks.

The graph service does not depend on the suggestion service. The frontend reads
them separately and composes the view.

### Jobs Contract

Generation uses:

- domain: `notes`;
- fixed queue: `graph-suggestions`;
- job type: `note_graph_suggestions`;
- owner: authenticated user ID;
- maximum automatic retries: `0`.

The Job payload contains only a schema version, run ID, owner ID, dataset ID,
source note ID, source content fingerprint, provider/model identifiers, and prompt
contract version. It does not contain note text, excerpts, rationales, tags,
provider responses, credentials, endpoints, or authorization claims.

The safe Job result contains the run ID, result digest, candidate count, validated
relationship/tag counts, dropped-item counts, and bounded usage counts. It does
not contain note-derived text or identifiers for candidate notes.

`max_retries=0` is required because a lease recovery cannot prove whether an
external provider accepted an interrupted request. A user retry is a new run with
a new explicit idempotency key. Preparation code may repeat deterministic local
reads within one handler invocation, but Jobs never restarts the provider-calling
handler automatically.

Application-managed and standalone worker paths use the same registered handler.
The worker obeys existing app/sidecar ownership rules so two worker lifecycles do
not unintentionally consume the queue.

### Admission And Publication Across Databases

ChaChaNotes and Jobs are separate authorities and cannot share a transaction. The
design uses deterministic identity and recoverable publication states rather than
claiming distributed atomicity.

Admission proceeds as follows:

1. Require a bounded `Idempotency-Key`; persist only its digest.
2. Resolve provider/model, permissions, note ownership, content fingerprint, FTS
   readiness, rate limits, and the absence of an equivalent active run.
3. Insert a ChaChaNotes run in `admitting` with a stable run UUID and canonical
   request fingerprint.
4. Enqueue the Jobs row idempotently using that run UUID.
5. Bind the Job UUID and move the run to `queued`.

Replay with the same key and request returns the same run. Reuse of the key with a
different request returns `409 notes_graph_suggestion_idempotency_mismatch`.
Only one run with the same owner, dataset, selected note, content fingerprint,
provider, model, and prompt contract may be `admitting`, `queued`, `running`, or
`publishing`. A new key may create a new run after the prior run is terminal.

If the process stops between run creation and Job admission, replay resumes the
same admission. A bounded reconciler marks abandoned `admitting` records failed
when the matching Job cannot be found or admitted.

The suggestion worker lifecycle runs this reconciliation as a maintenance pass at
startup and at most once per minute, claiming no more than 100 rows per pass. Every
claim uses row revision and lease compare-and-swap, so multiple application or
sidecar processes may run the pass without duplicate publication.

Publication proceeds as follows:

1. The worker changes `queued` to `running` with compare-and-swap and revalidates
   the source fingerprint.
2. It retrieves candidates, calls the provider once, validates the response, and
   revalidates every referenced note fingerprint.
3. One ChaChaNotes transaction writes hidden staged suggestions, their evidence
   references, aggregate validation counts, a result digest, and run state
   `publishing`.
4. The worker completes the Job with the same safe result digest.
5. One ChaChaNotes transaction verifies the completed Job postcondition, activates
   the staged suggestions, applies suppression and supersession, and marks the run
   `succeeded`.

Suggestion reads expose only rows from a `succeeded` run. If a crash occurs after
staging but before Job completion, reconciliation discards the staged set after
the Job becomes terminal. If a crash occurs after Job completion but before
activation, reconciliation verifies the digest and finishes activation. The
public API never reports staged or uncertain output as successful.

### Sync Boundary

Runs, provisional suggestions, evidence references, and rejection decisions are
server-generated review state. They do not become new Sync domains in this slice.

Acceptance delegates to existing authority:

- related-note acceptance uses the manual Notes link coordinator and produces an
  ordinary `manual`, `directed=false`, `weight=1.0`, null-label link with no
  model-selected properties;
- existing-tag acceptance uses the current keyword relationship coordinator;
- new-tag acceptance creates or exact-replays the normalized keyword and then
  creates or exact-replays the note-keyword relationship through one stable
  suggestion-derived mutation identity.

When Sync is active but a required Notes link or organization domain is not ready,
acceptance leaves the suggestion pending and returns the existing safe not-ready
contract. Inactive Sync uses the same product invariants through the existing
legacy mutation path. This feature never writes canonical link, keyword, or
relationship tables directly.

## Retrieval And Grounding

### Content Canonicalization

The content fingerprint is SHA-256 over this exact versioned canonical byte
sequence:

1. title and Markdown content are each normalized from CRLF/CR to LF;
2. each field is NFC normalized;
3. encode ASCII `notes-graph-content-v1`, one NUL byte, the normalized title as
   UTF-8, one NUL byte, and the normalized content as UTF-8.

Tag membership is deliberately excluded. Accepting one tag must not stale sibling
tag or relationship suggestions. An edit to title or content changes the
fingerprint and invalidates all pending suggestions that use that note as source
or target.

Evidence references store `note_id`, `field` (`title` or `content`), fingerprint,
and half-open start/end offsets measured in Unicode code points after the same
line-ending and NFC normalization. The server reconstructs evidence from the
current note only when the fingerprint matches. JavaScript never slices source
text from these offsets.

### FTS Candidate Retrieval

The candidate retriever is a new private DB/service method, not the existing
public exact-phrase Notes search. It never logs the query or derived terms.

It deterministically selects at most 24 bounded lexical terms from the selected
title and content. Title terms receive priority, followed by body frequency and a
stable lexical tie-break. Common stop words are removed, while bounded medical
abbreviations, alphanumeric terms, and hyphenated concepts remain eligible.

SQLite uses parameterized FTS5 OR semantics and BM25 rank. PostgreSQL uses a
parameterized `tsquery` and `ts_rank`. The retriever may fetch at most 60 rows to
apply authority and relationship exclusions, then returns at most 30 candidates.
Backend score values are not exposed or compared across backends; only stable rank
order within the run is used.

Candidates come from the owner's active Notes dataset. Retrieval excludes:

- the selected note;
- soft-deleted notes;
- notes with an existing direct manual or wikilink/backlink relationship to the
  selected note;
- candidates already suppressed by an unchanged-fingerprint rejection.

Shared tags and shared source-membership nodes do not make two notes
"already connected" and do not exclude them.

Both supported backends maintain Notes FTS in the note write transaction. Run
preflight verifies the expected FTS structures and trigger contract. Structural
drift returns `503 notes_graph_fts_not_ready` with rebuild guidance; the feature
does not silently use an unbounded table scan. The UI describes the result as a
bounded lexical search, not an exhaustive semantic analysis.

### Evidence Windows

The server creates bounded evidence windows before invoking the model:

- up to four windows from the selected note;
- up to two windows from each candidate;
- at most 480 Unicode code points per window;
- deterministic overlap removal and source-order tie-breaking.

Each prompt window has an opaque run-local evidence ID whose server record already
contains the exact note, field, fingerprint, and offsets. The model cites evidence
IDs; it never supplies offsets or arbitrary excerpts.

Prompt construction prunes the lowest-ranked candidate windows until the resolved
model budget fits. The hard defaults are 24,000 estimated input tokens and 2,000
output tokens, further reduced when the provider/model advertises a smaller
context. An exact tokenizer is used when available; otherwise the existing
conservative character fallback applies. Candidate count, windows, characters,
input budget, and output budget are all server-configurable downward but cannot
exceed their hard limits.

### Tag Catalog

The prompt receives at most 100 relevant existing active tags, selected by exact
term overlap, FTS/search relevance, and bounded usage frequency. Each existing tag
has an opaque catalog ID, normalized value, and display label.

The response chooses either an allowlisted existing catalog ID or a proposed new
tag string. It cannot provide both. The server applies current keyword
normalization and uniqueness rules. A run may contain at most five tag suggestions
and at most two new tags. Exact existing tags are preferred in prompt instructions
and server deduplication.

### Provider Request And Output

The provider request has fixed system instructions and structurally delimited
note data. It states that note text is untrusted, tools are unavailable, evidence
and candidate IDs are allowlists, and instructions appearing in note text must be
ignored. The call uses no tools, browsing, function execution, or provider routing
chosen by note content.

The strict response shape is conceptually:

```json
{
  "relationships": [
    {
      "target_note_id": "<allowlisted-id>",
      "rationale": "<bounded paraphrase>",
      "source_evidence_ids": ["<allowlisted-id>"],
      "target_evidence_ids": ["<allowlisted-id>"]
    }
  ],
  "tags": [
    {
      "existing_tag_id": "<allowlisted-id-or-null>",
      "new_tag": "<bounded-string-or-null>",
      "rationale": "<bounded paraphrase>",
      "source_evidence_ids": ["<allowlisted-id>"]
    }
  ]
}
```

Unknown top-level fields, malformed top-level structure, unknown candidate IDs,
or unknown evidence IDs fail the complete run. Individually invalid, duplicate,
already-linked, already-tagged, or suppressed items are dropped. The validated
set is staged atomically. A valid empty response succeeds with no suggestions.
When all model items violate the output contract, the run fails with
`notes_graph_suggestion_no_valid_items`. When otherwise valid items are removed
only because concurrent canonical links, tags, or rejection decisions now exist,
the run succeeds with an empty current result.

Rationales are at most 240 Unicode code points. Prompting requests paraphrase, but
the server does not claim prompting can guarantee it. An item is rejected when its
rationale contains a normalized contiguous overlap of more than 12 words from any
provided evidence window. Accepted rationales remain derived private content and
follow suggestion deletion and retention policy.

### Match Strength

The model does not provide confidence or a percentage. The server assigns the
display band:

- a related note is `Strong match` only when it falls in the top third of the
  returned FTS rank, has valid evidence on both sides, and shares at least two
  distinct selected retrieval terms or one exact multiword title phrase;
- other grounded related notes are `Possible match`;
- an existing tag is `Strong match` only when its normalized phrase occurs in the
  canonical selected-note text;
- all new tags and other grounded existing tags are `Possible match`.

The API names this field `match_strength`, not `confidence`, and documents it as a
deterministic lexical/evidence band.

## Persistence Model

ChaChaNotes schema v64 adds versioned SQLite and PostgreSQL tables with equivalent
constraints and indexes. PostgreSQL applies forced owner RLS in addition to
application scoping.

### `note_graph_suggestion_runs`

Each run stores:

- canonical run UUID, owner, dataset, source note, and source fingerprint;
- idempotency-key digest and canonical request fingerprint;
- resolved provider/model and prompt-contract version;
- Job UUID;
- state and revision;
- result digest and safe aggregate counts;
- stable error code and bounded public guidance key;
- created, started, completed, and expiry timestamps.

Run states are `admitting`, `queued`, `running`, `cancelling`, `publishing`,
`succeeded`, `failed`, `cancelled`, and `stale`. Provider error text, prompts,
responses, candidate IDs, note text, and credentials are not stored on the run.

### `note_graph_suggestions`

Each suggestion stores:

- canonical suggestion UUID and run ID;
- kind: `related_note` or `tag`;
- source note and title/body fingerprint;
- target note and fingerprint for `related_note`;
- normalized/display tag plus existing keyword portable identity for `tag`;
- `match_strength`, bounded rationale, and decision state;
- revision, decision reason, accepted resource identity, and timestamps;
- acceptance idempotency digest, request fingerprint, lease token, and lease
  expiry when a decision is in progress.

Suggestion states are `staged`, `pending`, `accepting`, `accepted`, `rejected`,
and `stale`. Superseded suggestions use `stale` with reason
`superseded_by_run`; no separate semantic edge state is introduced.

After rejection, evidence rows and rationale are removed and the row is compacted
to the decision identity needed for suppression. The feature does not retain
review text that is no longer needed to honor the decision.

Relationship identity is the canonical undirected note pair plus both content
fingerprints. Rejection suppression applies to that pair independent of provider,
model, prompt version, or rationale. Tag identity is source note, source content
fingerprint, and normalized tag, also independent of provider/model.

### `note_graph_suggestion_evidence`

Evidence rows contain suggestion ID, side (`source` or `target`), ordinal, note ID,
field, content fingerprint, and canonical half-open offsets. They do not contain
excerpt text. Reads reconstruct at most the configured number of bounded excerpts
and omit them when the fingerprint no longer matches.

### Indexes And Lifecycle Hooks

Indexes support owner/dataset/note/status pagination, active-run uniqueness,
source/target invalidation, suppression lookup, acceptance lease expiry, and
retention scans.

Title/content edit, note trash, and note deletion mark affected `pending`
suggestions stale in the same product transaction. Restore does not reactivate a
stale suggestion; the user regenerates from the restored current version. Tag
membership changes do not affect the content fingerprint and do not stale sibling
suggestions. Hard deletion cascades through runs, suggestions, and evidence.

An existing-tag suggestion binds to the keyword's portable identity as well as its
normalized value. Rename resolves the current display value at read and accept
time. A deleted tag makes the suggestion stale. A concurrent new-tag name collision
resolves the existing normalized keyword and proceeds idempotently to the
note-keyword relationship. Keyword merge follows the existing canonical merge
result; acceptance succeeds only when the selected note has the surviving tag.

## API Contract

All routes remain under the existing Notes graph namespace:

### Runs

- `POST /api/v1/notes/{note_id}/graph/suggestions/runs`
- `GET /api/v1/notes/{note_id}/graph/suggestions/runs`
- `GET /api/v1/notes/{note_id}/graph/suggestions/runs/{run_id}`
- `POST /api/v1/notes/{note_id}/graph/suggestions/runs/{run_id}/cancel`

Generation requires `Idempotency-Key`. The request may select only a configured,
policy-allowed provider/model pair or omit it to use the resolved Notes default.
Clients cannot supply an endpoint, credential, prompt, candidate count, token
budget, or tag catalog.

Admission performs all request-time validation. It returns `202` with a bounded
run envelope. Failures discovered after `202` appear in durable run state and
stable error fields; polling does not reinterpret them as later HTTP errors.

Run-list and run-detail responses expose provider/model, lifecycle state, safe
counts, timestamps, cancellation availability, error code, and guidance. They do
not expose Jobs internals, raw provider errors, prompts, or candidate IDs.

### Suggestions

- `GET /api/v1/notes/{note_id}/graph/suggestions`
- `POST /api/v1/notes/{note_id}/graph/suggestions/{suggestion_id}/accept`
- `POST /api/v1/notes/{note_id}/graph/suggestions/{suggestion_id}/reject`
- `POST /api/v1/notes/{note_id}/graph/suggestions/rejections/reset`

Listing uses an opaque cursor, `limit` from 1 to 100, optional run/status filters,
and a default of current `pending` and `accepting` suggestions. The response
contains reconstructed bounded evidence only for matching current fingerprints.

Accept and reject require `Idempotency-Key`, expected suggestion revision, and the
expected source/target fingerprints returned by the list response. The server
always reloads and revalidates the current notes; client fingerprints are
optimistic guards, not authority.

Reset requires `Idempotency-Key`, the current source fingerprint, and explicit
confirmation in the inspector. It removes only compact rejection keys for that
note and fingerprint. It does not delete accepted links/tags, pending suggestions,
or decisions for another content version.

Request-time errors use stable mappings:

| HTTP | Example codes |
| --- | --- |
| `409` | stale fingerprint, decision race, idempotency mismatch, active run |
| `422` | provider/model disallowed, invalid request contract |
| `429` | admission or decision rate limit |
| `503` | provider unavailable, FTS not ready, Sync mutation domain not ready |

The normal `404` response remains non-enumerating across owner/dataset scope.

## Decision Semantics

### Rejection

Reject is a single compare-and-swap from `pending` to `rejected`. An acceptance
that already won returns the accepted terminal state; a stale fingerprint marks
the suggestion stale and returns `409`.

Current-fingerprint relationship rejection suppresses the same canonical pair as
long as both note fingerprints remain unchanged. Tag rejection suppresses the
same normalized tag while the source fingerprint remains unchanged. Suppression
survives provider, model, prompt-contract, and rationale changes.

### Acceptance

Acceptance proceeds as follows:

1. Revalidate permissions, suggestion revision, source/target fingerprints, and
   current canonical relationship/tag state.
2. If the exact link or tag relationship already exists, finalize `accepted`
   idempotently with its canonical resource identity.
3. Otherwise compare-and-swap `pending` to `accepting`, storing a bounded lease,
   fencing token, idempotency digest, and canonical request fingerprint.
4. Invoke the existing link or keyword/relationship coordinator with a stable
   suggestion-derived idempotency identity.
5. Verify the exact canonical postcondition and finalize `accepted`.

For a new tag, acceptance is complete only when the normalized keyword exists and
the selected note is linked to it. A crash after keyword creation but before the
relationship remains `accepting` or returns to `pending`; it is never reported as
accepted merely because the keyword exists.

Controlled failure checks the exact canonical postcondition. If absent and safe
to retry, the suggestion returns to `pending` with a stable recoverable code. An
uncertain failure retains `accepting` until the lease expires.

A bounded reconciler claims expired acceptance leases with compare-and-swap. It
may only:

- finalize `accepted` when the exact expected canonical link/tag relationship
  exists;
- mark `stale` when a required fingerprint no longer matches;
- return the suggestion to `pending` when the postcondition is absent.

It never creates a link or tag. The normal accept retry performs the same
reconciliation before attempting another mutation.

The worker maintenance pass that repairs admission/publication also scans at most
100 expired acceptance leases once per minute. If that worker is unavailable, an
explicit accept retry still reconciles the selected record before proceeding.

Concurrent accept, reject, edit, regeneration, and external manual mutation have
one deterministic winner through suggestion CAS, canonical link/tag uniqueness,
and exact postcondition checks. Accepting one tag does not stale other suggestions
because tag state is not part of the title/body fingerprint.

### Regeneration And Supersession

A successful current-version run atomically:

- filters suggestions against current canonical links/tags;
- filters unchanged-fingerprint rejection keys;
- activates the new validated set;
- marks older `pending` suggestions for the same source fingerprint stale with
  `superseded_by_run`.

It does not overwrite `accepting`, `accepted`, or `rejected` decisions. Acceptance
finalization also reconciles duplicate pending suggestions for the same pair/tag
so the inspector cannot keep offering a mutation that is already canonical.

## Failure And Cancellation Behavior

Generation is all-or-nothing at the published-run level. No staged or invalid
item is visible. One malformed top-level response fails the run; invalid individual
items are omitted from the atomically published validated set.

Cancellation before provider invocation prevents the call. Cancellation after
invocation is best effort: the provider may still complete and charge, but the
worker checks cancellation before staging and does not publish a cancelled run.
Once a run enters `publishing`, it reports `cancellable=false`. A cancellation race
that was accepted before staging prevents publication; a later request returns the
current publishing state. Digest reconciliation never converts a completed
publication into an uncertain partial result.

Source or target edits during generation make the run stale and discard the
result. A provider timeout, schema failure, unavailable provider, or invalid output
produces a stable run error code and sanitized user guidance. A user retry always
creates a new explicit run; no provider call is transparently repeated.

The workspace preserves the last successfully loaded authoritative graph during a
generation failure. It labels a graph refresh failure rather than presenting the
cached view as newly current, and it never leaves a failed provisional overlay on
the canvas.

## Security And Privacy

- `notes.graph.read` protects authoritative graph reads.
- Suggestion reads require both `notes.graph.read` and the new
  `notes.graph.suggest` permission. The new permission also protects generation,
  cancellation, rejection, and dismissal reset.
- Relationship acceptance additionally requires `notes.graph.write`.
- Existing-tag acceptance requires the existing note-keyword link permission;
  new-tag acceptance also requires keyword creation permission.
- Token scope remains `notes`; multi-user RLS and explicit owner/dataset checks
  both apply.
- Every candidate, evidence reference, model-returned ID, and decision target is
  resolved again under the authenticated owner/dataset.
- The UI states when excerpts will be sent to a remote provider and identifies the
  provider/model before the user starts the run.
- Jobs payloads/results and feature logs exclude note text, excerpts, prompt text,
  response text, rationales, proposed tags, candidate IDs, credentials, and raw
  provider errors.
- Feature code does not enable provider payload logging. Sanitized diagnostics use
  run, Job, suggestion, owner, and dataset correlation IDs plus counts and stable
  codes.
- Rationales are private derived data. Note/user deletion and retention cleanup
  remove them with their suggestions.
- Provider output is never interpreted as HTML. UI rendering uses ordinary text
  escaping and bounded excerpts.

Single-user mode receives the new permission by default. In multi-user mode it is
seeded for administrators and the standard Notes-writing role, not for read-only
graph roles; administrators may revoke it independently. The Graph workspace
remains available to graph readers and hides the Suggestions tab when
`notes.graph.suggest` is absent.

## Limits And Retention

Default operational limits are:

- one active generation run per user;
- 20 generation admissions per user per hour;
- 100 active notes for `All notes` eligibility by default;
- 30 candidates sent to the provider after a 60-row retrieval overfetch;
- five related-note and five tag suggestions per run;
- two new tags per run;
- 100 existing tags in the prompt catalog;
- 24,000 estimated input tokens and 2,000 output tokens;
- 120 seconds provider timeout;
- 100 suggestions per list page maximum;
- 2,000 compact current-fingerprint rejection keys per source note.

Administrators may lower limits. Raising prompt, output, candidate, result, and
catalog limits above these hard values requires a code/schema contract change.
When the current-fingerprint rejection cap is reached, new generation fails with
`notes_graph_suggestion_suppression_limit` rather than discarding a user's prior
decisions or allowing rejected suggestions to reappear. The inspector's explicit
reset command is the recovery path when the user wants those candidates to become
eligible again.

Retention defaults are:

- current-fingerprint rejections remain while their relevant fingerprints are
  current;
- obsolete rejection and stale/superseded suggestion rows: 30 days;
- failed and cancelled runs without retained suggestions: 30 days;
- accepted suggestion audit rows: 90 days after canonical postcondition capture;
- successful run metadata with no retained suggestions: 90 days.

Pending current-version suggestions remain until decided, superseded, or made
stale. Cleanup is bounded, owner-scoped, and never removes a current-fingerprint
rejection merely due to age. Hard note/user deletion cascades immediately.

## Observability

Structured events cover run admission, shortlist completion, provider start and
completion, validation rejection, staging, publication, cancellation, failure,
acceptance, rejection, staleness, and reconciliation.

Local metrics include:

- queue latency and run duration;
- candidate and evidence-window counts;
- bounded provider token usage when available;
- validated and dropped result counts;
- run error codes;
- accepted, rejected, stale, and superseded counts;
- acceptance reconciliation outcomes.

Metrics contain no note-derived labels or content. The feature does not add or
enable telemetry export. Correlation uses safe run, Job, and suggestion IDs.

## Testing And Evaluation

### Backend Tests

- SQLite v63-to-v64 and fresh-v64 migration parity, rollback, indexes, triggers,
  constraints, and deletion cascades.
- PostgreSQL v63-to-v64 parity, forced RLS, cross-owner denial, trigger readiness,
  and live FTS behavior.
- Canonical fingerprints and evidence offsets across CRLF/LF, NFC, astral Unicode,
  title/content boundaries, and Python/JavaScript fixtures.
- FTS term selection, backend rank ordering, active-only scope, direct-link
  exclusions, shared-tag/source non-exclusion, deterministic pruning, and hard
  budgets.
- Prompt injection strings, unknown IDs, unknown evidence, malformed schemas,
  duplicate items, verbatim-overlap rejection, tag normalization, and new-tag
  caps.
- Admission idempotency, equivalent active-run exclusion, explicit retry, Jobs
  `max_retries=0`, cancellation before/after provider start, and all publication
  crash boundaries.
- Concurrent accept/reject/edit/regenerate/manual-mutation races, tag sibling
  validity, new-tag partial mutation, acceptance lease expiry, and reconciliation.
- API RBAC, token scope, dataset isolation, pagination, cursor integrity, rate
  limiting, provider policy, error mapping, and sanitized responses in both auth
  modes.
- Retention, suppression cap, note trash/restore/edit/delete, keyword merge/delete,
  and cleanup behavior.

Property-based tests cover canonicalization, bounded parsers, cursor round trips,
state-transition invariants, idempotency fingerprints, and randomized provider
payloads. External providers are mocked in required CI.

### Frontend Tests

- Graph view selection, editor-to-graph focus, empty libraries, selected-note and
  most-recent-note startup, bounded expansion, truncation, and all-notes gating.
- Search, focus, edge toggles, session-local layout, fit, directed arrows, label
  visibility, provisional overlays, and authoritative refresh after acceptance.
- Details/Suggestions tabs, provider disclosure, polling resume after reload,
  cancellation, stale/error/retry states, evidence rendering, and tag/new-tag
  treatment.
- Keyboard traversal, focus restoration, screen-reader names, non-color status,
  `Canvas`/`Relationships` parity, reduced motion, and high zoom.
- Desktop and narrow-screen Playwright screenshots for dense graphs, long titles,
  long tags, loading, failed, stale, truncated, and empty states with overlap and
  overflow assertions.
- Shared-package contract tests prove the WebUI and browser extension use the same
  workspace and nested API client behavior.

### Offline Quality Evaluation

A checked-in synthetic and hand-reviewed fixture library covers medical,
technical, research, and general Notes examples with related-note targets,
distractors, existing tags, new-tag cases, and adversarial note instructions.

The release gate records:

- at least 90 percent expected-target recall in the deterministic FTS top 30;
- 100 percent evidence-reference validity;
- zero cross-owner, unknown-candidate, already-linked, or rejected-pair output;
- 100 percent tag normalization and duplicate suppression;
- bounded prompt/output behavior on the largest fixtures.

Recorded provider responses exercise semantic review behavior deterministically in
CI. A separately marked live-provider smoke evaluation may report relevance and
grounding for configured providers, but it does not block local development and
never uses real user notes.

## Documentation

Implementation updates:

- the Notes Graph product documentation to distinguish implemented authoritative
  graph behavior, provisional suggestions, and deferred semantic/background work;
- nested Notes graph API documentation and examples;
- Jobs worker/queue configuration and sidecar operation documentation;
- WebUI and browser-extension Notes documentation;
- privacy documentation explaining when excerpts are sent to configured remote
  providers.

The stale `Graphing-Notes-PRD.md` implementation-status language must be corrected
rather than copied into new documentation.

## Rollout And Compatibility

The v64 migration is additive. Existing graph routes and responses remain
compatible. Suggestion routes and tables are dormant unless called, and Graph mode
continues to work without a configured provider by showing authoritative graph
features with suggestion generation unavailable.

The shared frontend may replace the modal after the workspace is verified; it
must not leave two divergent graph implementations. The existing editor graph
command becomes a view-mode transition. No route alias or root API is added.

If the suggestion worker is disabled or unavailable, run admission returns a safe
readiness error before claiming work was queued. Existing Notes editing, search,
manual linking, tagging, Sync, and graph exploration remain available.

## Deferred Work

- `TASK-13134`: owner-scoped embedding lifecycle and derived semantic graph edges.
- `TASK-13135`: opt-in automatic background organization using the review contract.
- `TASK-13136`: source-grounded library-wide recurring themes and theme nodes.
- `TASK-13137`: named saved graph views, filters, viewport, and pinned layouts.

Those tasks may build on the workspace and review contracts established here, but
none may silently convert derived output into canonical user state.

## Definition Of Success

The slice is successful when a user can enter Graph mode in either client, focus a
note, inspect its authoritative neighborhood, explicitly generate bounded grounded
link/tag suggestions, review evidence, accept or reject each item, and observe only
accepted mutations in the authoritative graph and tag model. The workflow must
survive reload, cancellation, duplicate commands, note edits, process interruption,
and Sync readiness failures without duplicate provider calls, cross-owner access,
unreviewed mutations, or false success.
