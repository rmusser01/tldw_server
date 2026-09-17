# UAT223 / TASK13260.161: supported Note citation route

The actual same-card Deep dive link returns WebUI404 at /notes/<id>. The existing option-notes route accepts /notes and selects sourceNoteId from source_ref_id. Repair only _build_note_route to emit that route, encode the authoritative citation source ID as a query value, and preserve nonempty mapping/string locator values. A locator cannot override source_ref_id. Keep source metadata, locator ranking, media/message routes and authorization unchanged. Granular locator interpretation remains outside this URL compatibility repair; native acceptance proves the correct note is selected, not text-anchor scrolling.

1. Retain actual404 and producer/consumer evidence; add causal route cases and update the three existing endpoint/context expected URLs (In Progress).
2. Minimum note-route builder correction; required PG/SQLite assistant plus provenance controls, Ruff/Bandit and source scope checks.
3. Independent review and same original card source-click/readback without changing source note or job.
