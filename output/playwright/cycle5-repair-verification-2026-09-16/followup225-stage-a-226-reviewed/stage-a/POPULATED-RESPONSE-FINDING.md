# Additional populated evidence response failure

Observed during TASK13260.163 Stage A expanded actual-router controls. No repair in Stage A.

Both official PostgreSQL and SQLite publish real source/target evidence, then the actual pending-list router raises Pydantic ValidationError at SuggestionListResponse construction. `items.0.evidence.0` and `.1`: expected dictionary or SuggestionEvidenceResponse, received SuggestionEvidenceExcerpt. The API facade reconstructs bounded valid excerpts; the endpoint passes the dataclass tuple to a strict BaseModel without attribute conversion. Empty evidence succeeds, so rejected rows (whose evidence is deliberately erased) do not expose this failure.

Source boundary: endpoint notes_graph_suggestions.py list_suggestions and schema notes_graph_suggestions.py SuggestionEvidenceResponse; both unchanged by Stage A. Dedicated test-only plugin changes only TestClient raise_server_exceptions. No provider, native runtime, or DB mutation outside disposable fixtures.

Retained failures: final-green label = 2 fail/58 pass/zero skips; pending-diagnosis label = 2 fail/50 deselected/zero skips. Earlier synthetic setup mistakes are separate: helper default source ID and rejected evidence erasure.

Pending parent task association and bounded repair scope. Do not call Stage A or full UAT225 complete while this surfaced workflow failure remains unresolved.
