# Cycle 5 UAT repairs

Parent: TASK13260. This design continues the user's authorization to repair every identified issue before another full fresh UAT. Current product is frozen at `ab527eb3b4`; subsequent commits contain documentation and evidence only. Both current native matrices must finish and their evidence must be preserved before product edits.

## Outcomes and scope

Preserve the authoritative frontend journeys and real server behavior. Correct the confirmed boundaries below with local changes and behavioral regressions. Do not change scheduler algorithms, authentication grants, provider authorization, saved-history semantics, or the running UAT configuration to hide a failure.

| Finding / task | Observed behavior | Required behavior | Expected implementation scope |
|---|---|---|---|
|126 / 13260.66|Study displays while URL remains Import / Export; reload opens wrong tab|Accepted tab selection updates its route without replaying old handoffs|FlashcardsManager and actual router/private-handoff tests|
|127 / 13260.67|Completed ingest GET is discarded after effect cleanup; resumed wizard stays Processing|One current poll projects terminal results after replay, resume and reload|QuickIngestWizardModal effect ownership and existing session tests|
|128 / 13260.68|Due-date refetch changes the Cram queue under an index; cards are skipped and completion contradicts remaining count|Progress follows intended card identities across scheduled rating and re-rating|ReviewTab and actual multi-card changing-query tests|
|129 / 13260.69|Legacy APP_MODE default overrides configured multi-user AuthNZ in Prompt routes|Canonical AUTH_MODE governs existing authentication validators|prompts.py mode helper and scoped AuthNZ/Prompt tests|
|130 / 13260.70|Bare model catalog IDs do not match the saved provider-qualified selection; Analyze falls back to Ollama|Equivalent, unambiguous catalog identity preserves the selected model/provider|AnalysisModal and actual catalog/shared-owner tests|
|068 recurrence / 13260.15|Saved chatId route restores the old character after a picker replacement|Accepted explicit replacement transitions route and character together while preserving the old saved chat|Character selection/route transition and actual Playground/loader tests|
|131 / 13260.71|An already saved assistant greeting is appended again on the first ordinary send|Creation, mirror and send agree on the greeting's canonical identity|Promotion ACK projection and actual loader/mirror controls, with068 Chat owner|
|132 / 13260.73|An invalid FTS query falls back to all visible owned sources|Fallback preserves the text filter and owner restrictions|MediaSearchRepository and actual SQLite query controls|
|133 / 13260.72|Retry with descending saved history answers an older question|The exact accepted failed user turn is last in provider context once|Chat context assembly and actual final-payload/persistence controls|
|134 / existing13260.24|Readiness polling repeats expired-session401 without refreshing|Safe readiness reads use the existing canonical refresh and terminal-invalidation path|apiSend/direct-runtime connection and actual rotation/polling controls|

Detailed read-only designs and temporary reproductions are retained under `output/playwright/cycle5-full-uat-2026-09-16/controller/`. Existing failures remain historical evidence; a passing comparison case is not a completed repair.

## Design decisions

### Flashcards route and study queue

For126, use the existing router after Scheduler dirty-state confirmation accepts navigation. Change only the canonical tab query while preserving other query parameters, hash and intended history behavior. A tab-only change must not replay a stale deck selection or consume/clear a private handoff. Use the actual MemoryRouter/private-handoff integration boundary, since mocked navigation alone cannot prove reload behavior.

For128, replace progression tied to a mutable array position with progression tied to card UUIDs. Reuse the existing review-scope and authenticated-owner reset boundaries. Either retained ordered run IDs or scoped practiced IDs can provide the small implementation; choose after permanent regressions expose removal, re-rate and mixed-practice behavior. Current card values must remain fresh. Progress and completion must derive from the same pending-card state. Re-rate remains another scheduling event, not server Undo. Practice-only remains free of review mutations.

### Ingest effect lifecycle

For127, cleanup must release only the polling ownership/signature it owns, allowing a valid replacement effect to start. Preserve cancellation, session replacement, owner/server changes, late-response guards and timers. A completed job should not require upload replay, job cancellation or a server restart. Keep result classification and warning deduplication unchanged; the same Warning payload already succeeds in the non-Strict comparison.

### Authentication and model identity

For129, consult canonical AuthNZ mode before any legacy boolean. A canonical settings error must not fall back into a privileged single-user identity. Keep credential comparison, bearer/API-key resolution, claim-first admin policy and per-user database selection unchanged. The isolated diagnostic controls downstream auth/DB boundaries, so permanent validation must include existing real credential and tenant fixtures where available.

For130, retain catalog provider metadata and use the existing recognized-provider parser to compare descriptors. Preserve raw model IDs containing colons and filesystem paths. Do not double-prefix already-qualified IDs. Catalog hydration must not write an implicit preference into shared storage. Preserve legitimate removed-model recovery, but do not choose an ambiguous or conflicting provider by catalog order. Reuse existing visible error/selection behavior to require a choice when identity is unresolved.

### Explicit character replacement

For068, a user-accepted replacement must transition away from the old saved route before that route can restore its old identity. Keep canonical saved-route reload113 and saved ordinary Chat/Note mode123 intact. Preserve the previous saved conversation, real drafts, pending streams, confirmation cancellation, account/connection authority and delayed responses. Do not solve this by weakening canonical-loader ownership, deleting history, or treating every mismatch as a new-character request.

For131, actual loader/mirror probes confirm the missing promotion acknowledgement boundary. Copying history saves each row but keeps its server receipt only in closure-local state; the neutral loader cannot identify the corresponding local greeting. Project verified receipts onto their exact captured source IDs under existing owner/history-generation guards. Preserve later edits, ambiguous-save prefix recovery, pending drafts and genuine equal-content messages. Do not deduplicate by text or infer routing from sender-name metadata. The same Chat owner handles overlap with068 sequentially.

### Search fallback and failed-turn context

For132, an FTS syntax failure must retain a literal title/content fallback filter along with existing visibility conditions. Actual isolated SQLite tests reproduce owner-scoped but unrelated results for hyphenated markers; quoted and simple unmatched controls pass. Keep valid FTS, parameter binding, empty-query behavior and tenant restrictions. No permission relaxation is justified.

For133, move the identified accepted failed-user turn from the historical prefix to the final current-turn position without another database write. Preserve the explicit historical ordering contract for other rows; the existing suite intentionally checks descending order. Retain exact user text/images, one canonical identity, limited/zero history, ordinary repeated sends and conflict controls. Actual private context assembly reproduces2descending failures and2ascending controls; browser and canonical records independently demonstrate the wrong answer. Do not alter live configuration to hide the defect.

For134, connect safe readiness reads to the existing canonical refresh-capable direct runtime. Reuse current token-pair rotation, shared single-flight, authority guards and terminal invalidation. Do not add another token store or broaden automatic mutation replay. Cover actual connection-store transport, multiple rotations, terminal failure stopping polling, owner/server changes, hosted cookies, single-user keys and transient network failures. The prior successful away-return refresh used a different transport; it does not certify this readiness caller.

## Verification and limits

1. Add permanent regressions at actual boundaries before production changes; retain the original failing outcome and positive controls.
2. Use disjoint implementation ownership, independent review and scoped lint/type checks. Compare with the recorded90 existing compiler diagnostics; do not call that a clean typecheck.
3. Run Bandit through the project virtual environment for the changed Python scope, and fix new findings. TypeScript-only units record Bandit as not applicable.
4. After a coordinated source freeze/restart, verify the repaired behaviors in preserved single/multi profiles. Include saved-job results, actual tab reload, multi-card Cram/re-rate, Prompt collection auth, first-open model identity, and actual character replacement/request identity.
5. Only after repairs and required targeted controls pass, prepare another fresh configuration/data/browser matrix on a newly recorded source commit. Fetch/check dev before that freeze; preserve any later movement rather than changing product mid-run.

Existing boundaries remain explicit: dependencies are reused, exact Wikipedia access may be externally blocked, successful vision inference is unavailable, native hidden-tab notification verification exhausted prior tool attempts, and optional subsystems are outside the named journey certification. A tool authorization rejection is recorded separately from product failure; a safer fresh Chat may avoid unrelated test history while preserving the original saved conversation.
