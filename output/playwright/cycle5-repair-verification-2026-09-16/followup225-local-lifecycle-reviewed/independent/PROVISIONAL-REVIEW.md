# UAT225 Stage B — provisional source review

This is not final approval or an executable-test receipt. The initial candidate manifest was `7b48ea2a5c31e621873a89c2171e8063e4a2dccc39cf249270582cdae1ff99f4`; its copy is retained alongside this note. Author final checks and corrections are still pending.

## Actionable source finding

**SQLite prior-device keyword labels conflict with the local acceptance adapter.** The independently reviewed persistence resolver intentionally treats SQLite keywords as belonging to the selected file, while PostgreSQL checks the selected owner on every hop. The new local consumer resolves the former successfully, then calls the canonical `NotesOrganizationSyncStore` membership path, which requires `keyword.client_id == selected DB client_id` on SQLite too. Existing or merged prior-device tags can therefore resolve but fail to become accepted memberships. Normalized new-tag lookup and the store's acceptance postcondition have the same predicate mismatch. Publication's `_source_has_tag` also excludes already-attached prior-device keywords from deduplication.

The author confirmed that same-file prior-device tags are in the intended local contract. The earlier foreign-tag control used a separate physical SQLite file, so it did not exercise this case. Required causal controls are actual same-file existing tag, merged survivor, same-label new proposal, and already-present membership; PostgreSQL foreign-owner behavior must stay closed. The correction must be local-only and leave the committed task168 persistence and canonical Sync owner contract intact. No reviewer production edits or probes have been executed; the author is retaining RED and proposing the correction.

Broader prior-device **note** ownership checks already exist in the graph fingerprint and NotesLinkStore paths. They are not being silently rewritten or claimed covered by this bounded keyword finding.

## Other boundaries inspected

- Local authority is the exact selected-owner legacy key with no canonical binding, without inserting a legacy authority row.
- Canonical reservation validates the actual owned default personal dataset and preserves existing task/moodboard/studio flags. Existing PostgreSQL table locks serialize reservation with local guarded transactions before product row locks.
- Profile creation, personal-context binding (including a supplied dataset), and direct link/organization bootstrap call the reservation seam.
- Local product mutations use the existing acceptance fences and transaction finalizers; keyword creation alone cannot finalize acceptance.
- Completed receipts remain immutable. Retired runs/suggestions have a separate cleanup boundary that cannot authorize fresh admission or publication.
- Late enqueue discovery uses exact owner/domain/queue/type/payload and run identity. Failed admission lookup obligations rotate under a bounded budget and keep the original expiry; successful late cancellation does not rewrite terminal receipts.
- Maintenance defers when enrollment wins between an initial check and reconciliation; current scope is checked at the store mutation boundary.

No additional concrete production finding emerged from provisional inspection. These observations await final source hashes and independent tests. The author's separate eager `organization.note_db` constructor regression was already found by its adjacent suite and is being corrected with lazy access; it is not a reviewer-discovered finding.

The full native/matrix gate stays closed. No browser, runtime, database, source/test, task, tracker, or git action was performed by this provisional review.
