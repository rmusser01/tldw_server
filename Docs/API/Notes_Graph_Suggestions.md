# Notes Graph Suggestions API

The Notes graph suggestion API generates bounded, grounded related-note and tag proposals for one note. Suggestions are provisional review state. They never appear as a new edge type in an authoritative graph response, and they do not change a link or tag until the user explicitly accepts one.

All routes are nested under:

```text
/api/v1/notes/{note_id}/graph/suggestions
```

Use `X-API-KEY` in single-user mode or `Authorization: Bearer <token>` in multi-user mode. When a token declares scopes, it must include `notes`.

## Permissions And Isolation

- Every route requires `notes.graph.read` and `notes.graph.suggest`.
- Accepting a related-note suggestion also requires `notes.graph.write`.
- Accepting an existing-tag suggestion also requires `notes.link_keyword`.
- Accepting a new-tag suggestion also requires `notes.link_keyword` and `keywords.create`.
- Owner and dataset scope are resolved on every candidate, evidence reference, model-returned ID, and decision target. Out-of-scope resources use the normal non-enumerating `404` response.
- In single-user mode, suggestion permission is included by default. Multi-user administrators and standard Notes writers receive it by default; it can be revoked independently from graph read access.

## Capability Preflight

```http
GET /api/v1/notes/{note_id}/graph/suggestions/capabilities?provider=openai&model=gpt-4.1-mini
```

`provider` and `model` are optional. Omit both to resolve the configured Notes default. Expected provider, FTS, or worker unavailability returns `200` with `generation_available: false`; it does not create a run.

```json
{
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "endpoint_origin_revision": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "data_boundary": "remote",
  "disclosure_external": true,
  "outbound_data_categories": [
    "selected_note_title",
    "selected_note_excerpts",
    "candidate_note_titles",
    "candidate_note_excerpts",
    "existing_tag_labels"
  ],
  "generation_available": true,
  "unavailable_reason": null,
  "limits": {
    "max_candidates": 30,
    "max_relationships": 5,
    "max_tags": 5,
    "max_new_tags": 2,
    "max_tag_catalog": 100,
    "max_estimated_input_tokens": 24000,
    "max_output_tokens": 2000,
    "provider_timeout_seconds": 120,
    "response_candidates": 1
  },
  "allowed_actions": ["generate", "cancel", "accept", "reject", "reset_rejections"],
  "revision": "sha256:abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
}
```

The response includes `ETag: "sha256:..."` and `Cache-Control: no-store`. `local` means the configured endpoint is within the local boundary. `remote` means note-derived categories cross an external boundary. `unknown` must be treated as external. The API exposes only a digest of the configured endpoint origin, never an endpoint URL or credential.

## Start And Poll A Run

Generation requires the capability ETag and a client-generated idempotency key:

```bash
curl -X POST \
  "http://localhost:8000/api/v1/notes/NOTE_ID/graph/suggestions/runs" \
  -H "Authorization: Bearer TOKEN" \
  -H 'If-Match: "sha256:CAPABILITY_REVISION"' \
  -H "Idempotency-Key: 74510d13-6c13-42bd-8e49-c2b416f58341" \
  -H "Content-Type: application/json" \
  -d '{"provider":"openai","model":"gpt-4.1-mini"}'
```

The request body may contain only `provider` and `model`. Clients cannot supply an endpoint, credential, prompt, candidate count, token budget, or tag catalog. Admission returns `202`:

```json
{
  "id": "RUN_UUID",
  "provider": "openai",
  "model": "gpt-4.1-mini",
  "state": "queued",
  "revision": 2,
  "created_at": "2026-08-28T12:00:00Z",
  "started_at": null,
  "completed_at": null,
  "suggestion_count": 0,
  "related_note_count": 0,
  "tag_count": 0,
  "invalid_item_count": 0,
  "cancellation_available": true,
  "error_code": null,
  "guidance_key": null
}
```

Poll or discover active runs with:

```http
GET /api/v1/notes/{note_id}/graph/suggestions/runs/{run_id}
GET /api/v1/notes/{note_id}/graph/suggestions/runs?state=queued,running,publishing&limit=20&cursor=OPAQUE
```

Run states are `admitting`, `queued`, `running`, `cancelling`, `publishing`, `succeeded`, `failed`, `cancelled`, and `stale`. List limits are 1 through 100. `next_cursor` is an opaque owner/dataset/note/filter-bound pagination hint, not an authorization token.

An exact `Idempotency-Key` replay with the same canonical request returns the same admission and never repeats the provider call. Reusing it for a different request returns `409 notes_graph_suggestion_idempotency_mismatch`.

## List Suggestions

```http
GET /api/v1/notes/{note_id}/graph/suggestions?state=pending,accepting&limit=20&cursor=OPAQUE
```

The default states are `pending` and `accepting`. Each item includes its run, kind, revision, current source/target fingerprints, bounded rationale, and reconstructed evidence. Evidence is returned only while its stored fingerprint matches the current note. The page also returns:

```json
{
  "items": [],
  "next_cursor": null,
  "current_source_fingerprint": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "rejection_set_revision": 3,
  "rejection_count": 2
}
```

Related-note items identify an owner-scoped target and evidence from both notes. Tag items identify a normalized/display tag and whether the canonical tag already exists. Rationales are bounded to 240 Unicode code points and evidence excerpts to 480 code points each.

## Decisions

Every mutation requires a fresh `Idempotency-Key`. Accept and reject also require the suggestion revision and fingerprints returned by the list response:

```http
POST /api/v1/notes/{note_id}/graph/suggestions/{suggestion_id}/accept
POST /api/v1/notes/{note_id}/graph/suggestions/{suggestion_id}/reject
Content-Type: application/json
Idempotency-Key: UUID

{
  "expected_revision": 1,
  "expected_source_fingerprint": "sha256:...",
  "expected_target_fingerprint": "sha256:..."
}
```

Use `null` for `expected_target_fingerprint` on tag suggestions. Acceptance uses the canonical Notes link or keyword coordinator, then the client must refresh the authoritative graph. Rejection removes review text that is no longer needed and retains the compact identity needed to suppress the same unchanged pair or tag.

Reset dismissals only after explicit confirmation:

```http
POST /api/v1/notes/{note_id}/graph/suggestions/rejections/reset
Content-Type: application/json
Idempotency-Key: UUID

{
  "expected_rejection_revision": 3,
  "source_fingerprint": "sha256:...",
  "confirm": true
}
```

Reset does not remove accepted links/tags, pending suggestions, or decisions for another content version.

Cancel an active run with its current revision:

```http
POST /api/v1/notes/{note_id}/graph/suggestions/runs/{run_id}/cancel
Content-Type: application/json
Idempotency-Key: UUID

{"expected_revision": 2}
```

Cancellation before provider invocation prevents the call. Cancellation after invocation is best effort, but cancelled output is not published. A run in `publishing` is no longer cancellable.

## Jobs And Publication

Generation uses queue `graph-suggestions`, job type `note_graph_suggestions`, and `max_retries=0`. One run permits at most one outbound provider request; provider transports also disable automatic retries and cross-origin redirects. A user retry is a new run and new idempotency key.

The worker stages a complete validated set as hidden review state, completes the Job with a bounded digest/result envelope, and then verifies the exact owner-scoped terminal Job receipt. Only a matching success state, completion token, run ID, and result digest can activate the staged set. Missing or mismatched authority fails closed; staged or uncertain output is never returned by the public API.

Maintenance runs at startup and at most once per minute with a 100-row budget. It reconciles admission, queued/running/cancelling Jobs, cancellation receipts, publication receipts, expired acceptance leases, and retention. A missing publication receipt may remain recoverable for 30 days; terminal Jobs receipts for this queue must be retained at least that long.

## Data Boundary And Logging

The capability response is the complete authority for note-derived data that may leave the server. The only permitted outbound categories are:

- selected-note title and bounded excerpts;
- candidate-note titles and bounded excerpts;
- bounded existing tag labels.

No tools, browsing, function execution, provider routing from note content, arbitrary endpoints, or client-supplied prompts are enabled. Jobs payloads/results, run rows, operation receipts, structured events/logs, and metric labels must not contain note text, evidence excerpts, prompts, provider responses, rationales, proposed tags, candidate IDs, API keys, credentials, authorization claims, endpoints, or raw provider errors. Logs and local metrics use bounded counts, durations, usage, stable codes, and safe run/Job/suggestion correlation IDs. This feature does not initialize or enable a telemetry exporter.

## Limits And Retention

Default hard or operational limits include one active run per user, 20 admissions per user per hour, 30 provider candidates after a 60-row overfetch, five related-note suggestions, five tag suggestions, two new tags, 100 catalog tags, 24,000 estimated input tokens, 2,000 output tokens, a 120-second provider timeout, and 100 list items per page. Analysis rejects a selected note above 1,000,000 combined UTF-8 bytes and excludes a candidate above 250,000 bytes.

Operation receipts and their bounded replay envelopes remain for 90 days after terminal state unless hard note/user deletion cascades them. Obsolete rejections, stale/superseded suggestions, and failed/cancelled runs without retained suggestions expire after 30 days. Accepted suggestion audit rows and successful run metadata without retained suggestions expire after 90 days. Pending current-version suggestions remain until decided, superseded, or stale.

## Error Contract

Errors use a stable `detail` envelope:

```json
{
  "detail": {
    "error_code": "notes_graph_capabilities_changed",
    "message": "Suggestion capabilities changed; refresh and retry."
  }
}
```

| HTTP | Meaning |
| --- | --- |
| `403` | Required graph, suggestion, link, or keyword permission is missing. |
| `404` | Note/run/suggestion is absent or outside the owner/dataset scope. |
| `409` | Fingerprint/revision race, active-run conflict, or idempotency mismatch. |
| `412` | Capability disclosure changed; fetch capabilities again. |
| `422` | Invalid request, disallowed provider/model, or oversized source note. |
| `429` | Admission or decision rate limit. |
| `503` | Provider, FTS, Jobs worker, or Sync mutation authority is unavailable. |

## Deferred Work

- TASK-13134: embedding index and semantic edges.
- TASK-13135: automatic background organization.
- TASK-13136: library-wide recurring themes.
- TASK-13137: saved graph views and layouts.
