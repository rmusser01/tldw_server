# Research Workspace Study publication

- ADR required is yes.
- ADR path is [ADR-097](../ADR/097-research-workspace-study-publication.md).
- A public creation contract and durable replay receipt establish a new
  persistence boundary. The native Study owner remains authoritative.

## HTTP contract

`GET /api/v1/workspaces/{workspace_id}/study-outputs` returns a bounded page of
visible generated Study outputs and the `research-study.v1` contract marker.
`POST` on the same path accepts a bounded flashcard or multiple-choice quiz
payload with its operation ID, source snapshot, and generation fingerprints.
`GET /api/v1/workspaces/{workspace_id}/study-outputs/{kind}/{owner_id}` resolves
one exact generated output. Responses project the current native name and
version. The Workspace rate limits and expected-account assertion apply.

The request identifies selected Media associations with their current
workspace association versions. Only associations in the snapshot can appear
in entry citations. Catalog versions can be recorded when the client captured
one. Media content and its lifecycle belong to the separate Media store, and
the Study transaction fences the workspace associations. The client performs
generation with its existing qualified provider route. Publication carries
generated Study content and structured references through the same server.

## Persistence

The owner and workspace qualify an operation key. Under the SQLite write lock
or a PostgreSQL workspace row lock, the server checks an existing receipt
before validating current associations for a new save. A changed payload
with the same key returns a conflict. Identical replay returns the live
canonical owner in the same workspace, even after the selected sources have
changed. Missing or moved owners cause a conflict. A fresh operation checks
active workspace and selected source versions, then saves the native record,
its entries, and a receipt before responding.

Receipts hold bounded provenance, a private request digest, and the native
identity. Canonical Study holds the names, cards, questions, and citations.
A repeated deck name uses a stable suffix when required by native uniqueness.
A new Studio operation never restores a deleted deck. Regeneration can
reference an existing generated owner of the same kind in this workspace.

The exact native deck read allows Study to select the saved record outside its
first page. Existing native quiz reads support the same handoff. Studio can
create a separate new version and reopen an existing output. Native Study
hosts edits and deletion.

Older servers lack the contract marker. Chatbook leaves these two Studio
actions unavailable on those servers.
