# Notes Graph Suggestions

The Notes Graph workspace is shared by the WebUI and browser extension. It shows the server's current graph and lets eligible users review grounded related-note and tag suggestions before any relationship changes.

## Open The Graph

1. Open Notes and select an active note.
2. Switch the workspace to **Graph**.
3. Use **Canvas** for the visual graph or **Relationships** for the keyboard-accessible grouped list.
4. Select a note, relationship, tag, or source to inspect it. Use the focus command to reload a bounded neighborhood around a note.

The graph is authoritative. Solid graph relationships come from canonical manual links or current derived projections. Dashed suggested relationships and temporary suggested target nodes are provisional overlays. They disappear when rejected, stale, superseded, or otherwise no longer current.

**All notes** is available only when the active-note count is within the server's bounded eligibility cap, 100 by default and never above the response's effective node cap. When the library is larger, keep using focused neighborhoods and explicit expansion.

## Generate Suggestions

The **Suggestions** tab appears only when the account can read the graph and use `notes.graph.suggest`. Before **Generate** is enabled, the workspace shows:

- the resolved provider and model;
- whether the data boundary is local, remote, or unknown;
- the exact outbound data categories;
- the effective candidate, result, token, and timeout limits;
- any provider, FTS, worker, or permission reason generation is unavailable.

Treat an unknown boundary as external. Depending on the disclosed boundary, the provider may receive only the selected-note title/excerpts, candidate-note titles/excerpts, and bounded existing tag labels. Credentials, arbitrary endpoints, and client-authored prompts are never part of this workflow.

Select **Generate** to admit one run. The status progresses through queued, running, publishing, and a terminal state. Reloading the page discovers and resumes polling the same active run; it does not submit generation again. **Cancel** is available before publication. A provider call that has already started may still finish or incur provider cost, but cancelled output is not published.

The search is a deterministic, bounded lexical candidate search, not exhaustive semantic analysis. Each run returns at most five related-note and five tag suggestions, including at most two new tags.

## Review And Decide

A related-note suggestion includes:

- the current target title;
- **Strong match** or **Possible match**;
- a bounded rationale;
- selected-note and target-note evidence excerpts.

A tag suggestion identifies whether it reuses an existing tag or proposes a new normalized tag, with evidence from the selected note.

Choose **Accept** to create the ordinary canonical relationship through the existing Notes link or tag workflow. The provisional row is removed and the authoritative graph refreshes. Choose **Reject** to remove the overlay and suppress the same unchanged note pair or tag in later runs.

Use the Suggestions overflow menu to reset dismissed suggestions for the selected note's current content version. Confirmation is required. Reset does not remove accepted relationships, pending suggestions, or dismissals from a different note version.

If either note changes while a relationship is under review, the server marks the affected suggestion stale. A stale acceptance cannot create a canonical link or tag. Regenerate against the current note content.

## Navigation And Layout

- **Canvas** and **Relationships** use the same loaded graph and provisional suggestion set.
- Relationship groups are stably sorted and paged at up to 100 rows. Changing pages preserves the selected node and group context.
- Arrow keys move within relationship rows. Activating a counterpart selects it and returns to the corresponding canvas node when Canvas is visible.
- Status and decisions are announced through one polite live region. Focus returns to the relevant review row or owning region after a decision.
- Long titles, tags, evidence, rationales, provider names, and model names wrap within their regions.
- At narrow widths, the inspector becomes an in-page region with its own scroll boundary. The Notes sidebar uses its existing mobile navigation behavior.

## Offline And Read-Only Behavior

When the client is offline, the last successfully loaded authoritative graph may remain visible and is marked offline. Generate, Cancel, Accept, Reject, and reset commands remain disabled until the server is reachable. A generation or graph-refresh failure does not relabel stale data as current or leave a failed provisional overlay visible.

A read-only graph user can use Graph, Details, Canvas, and Relationships. The Suggestions tab and every nested suggestion request remain absent when `notes.graph.suggest` is missing. Non-note selections also make no suggestion requests.

## Privacy And Retention

Suggestions and rationales are private owner/dataset-scoped review state. Every candidate, evidence reference, and decision target is revalidated under the authenticated owner and dataset. The service does not log note text, excerpts, prompts, responses, rationales, proposed tags, candidate IDs, credentials, endpoints, or raw provider errors, and it does not enable telemetry export.

Idempotency receipts retain bounded replay state for 90 days unless hard note/user deletion removes them. Obsolete rejection and stale review state normally expires after 30 days; accepted audit state and successful run metadata without retained suggestions normally expires after 90 days. Current pending suggestions remain until decided, superseded, or stale.

## Troubleshooting

- **Generate is unavailable:** Read the disclosure reason. Configure an allowed provider/model, restore the Notes FTS structures or worker, or ask an administrator for `notes.graph.suggest`.
- **Capabilities changed:** Refresh the disclosure before generating. The server will not silently switch provider, model, endpoint origin, boundary, or limits after review.
- **Note too large:** Split the selected note. Suggestion analysis rejects combined title/content above 1,000,000 UTF-8 bytes.
- **Suggestion became stale:** Refresh the graph and regenerate from the current note version.
- **All notes is disabled:** The active-note count is above the current bounded cap. Use focused neighborhoods.
- **A decision is disabled:** Reconnect if offline, refresh current revisions, or request the additional link/keyword permission required for that suggestion kind.

## Current Scope

The implemented release covers authoritative bounded graph navigation and explicitly reviewed lexical suggestions. The following work remains separate:

- TASK-13134: embedding index and semantic edges.
- TASK-13135: automatic background organization.
- TASK-13136: library-wide recurring themes.
- TASK-13137: saved graph views and layouts.
