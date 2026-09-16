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
|135 / 13260.74|An outage is described as missing credentials and a CORS denial|Current failure evidence determines truthful connection guidance|Connection store and actual Knowledge SetupDiagnostics controls|

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

For135, generic browser network errors must remain cause-neutral; different frontend/backend origins do not prove a CORS denial. Current auth/UX failure takes precedence over historical onboarding-step metadata, so configured credentials are not described as missing during an unrelated outage. Preserve true401, missing-credential and explicit host/allowlist denial behavior. Use the actual store-to-diagnostics boundary. Coordinate the connection-store scope sequentially with134; do not recommend disabling CORS to work around the test outage.

## Verification and limits

1. Add permanent regressions at actual boundaries before production changes; retain the original failing outcome and positive controls.
2. Use disjoint implementation ownership, independent review and scoped lint/type checks. Compare with the recorded90 existing compiler diagnostics; do not call that a clean typecheck.
3. Run Bandit through the project virtual environment for the changed Python scope, and fix new findings. TypeScript-only units record Bandit as not applicable.
4. After a coordinated source freeze/restart, verify the repaired behaviors in preserved single/multi profiles. Include saved-job results, actual tab reload, multi-card Cram/re-rate, Prompt collection auth, first-open model identity, and actual character replacement/request identity.
5. Only after repairs and required targeted controls pass, prepare another fresh configuration/data/browser matrix on a newly recorded source commit. Fetch/check dev before that freeze; preserve any later movement rather than changing product mid-run.

Existing boundaries remain explicit: dependencies are reused, exact Wikipedia access may be externally blocked, successful vision inference is unavailable, native hidden-tab notification verification exhausted prior tool attempts, and optional subsystems are outside the named journey certification. A tool authorization rejection is recorded separately from product failure; a safer fresh Chat may avoid unrelated test history while preserving the original saved conversation.

## Targeted-acceptance follow-ups — 2026-09-16

The native freeze at `3c30685611` ended after two natural multi-user token rotations and canonical revocation of only Alice session12. Native refreshes were initiated by research-runs; actual readiness initiation remains established by the retained transport regressions. Both UAT frontends are paused for these bounded repairs; server data remain intact.

- **Reopened013 / TASK13260.3:** Home stores an owned `rag_media` handoff, but the consumer never enables the retrieval flag used by the actual send path. Enable effective retrieval when accepting that owned handoff. Exercise the real Form/action/RAG request boundary with an existing different-source conversation, both restoration orderings, owner changes and empty/error retrieval. Preserve normal full-content handoffs and saved history; do not hide the defect by deleting the old conversation.
- **137 / TASK13260.76:** supply remaining count to the existing translation engine and English ICU plural forms for the visible label and screen-reader announcement. Verify the configured ICU adapter, singular/plural and empty queue; leave queue/scheduling logic unchanged.
- **138 / TASK13260.77:** the caught network failure is passed to `console.error`, which creates the observed Next development overlay. Reuse the existing backend-unreachable classifier at this catch boundary; recognized outages use fixed warning text and existing user feedback. Preserve unexpected-error diagnostics, missing-endpoint handling and same-hook retry recovery. Native acceptance must cover an actual owned API outage and recovery.
- **139 / TASK13260.78:** expose current connection verification as derived context, not persisted snapshot state. Distinguish unverified network outage from true authentication failure; gate notification reads/actions until verified. Test the actual provider/route mount, same-false verification transitions, recovery and genuine auth states. Do not claim a last-success timestamp for an initially unreachable connection.
- **140 / TASK13260.79:** replace deprecated numeric-input adornments in Conversation settings with supported labeled markup. Preserve six fields' values, bounds, disabled states and blur persistence. Test actual AntD rendering and native opening, not only mocked component props.

Each unit retains its RED/GREEN evidence and independent review before native acceptance. TypeScript-only changes have no applicable Python Bandit scope. The next full SQLite/PostgreSQL matrix remains blocked until current defects and required acceptance gaps are reconciled.

Independent139 review additionally exposed141 (fabricated freshness after failed first inbox reads despite verified core connection) and142 (View navigation after an awaited mark-read loses authority). Track them separately under TASK13260.80/.81. Use the existing runtime success/generation boundaries: only a real successful inbox read establishes freshness; recheck current operation authority before navigation. Keep the ordinary View and later-success controls. These findings come from actual provider/route probes, not native browser observations or an observed disclosure.

The final013 correction records accepted source-selection intent in the existing session store, outside persisted data. Value equality cannot distinguish a new same-value handoff from old restored state; the revision preserves transcript restoration while preventing obsolete source replay. Keep initial matching-target and later explicit restore lifecycles distinct. For142, a narrow authority revision in the existing synchronous event handlers observes batched server/principal transitions; ordinary same-principal token refresh does not invalidate it. No new stores, listeners or persisted schemas are needed.

UAT143/TASK13260.82 covers failures in six additional Chat regression suites discovered when combined coverage expanded. Establish baseline causes first. Where fixtures omit current required dependencies or guards inspect obsolete implementation locations, correct the tests while preserving their intended behavior. Do not change production for incomplete fixtures, skip tests, or remove failed controls. Retain baseline failure and final combined results.

UAT144/TASK13260.83 comes from actual fresh normal-runtime PostgreSQL initialization. The pool's single-user safety fallback ignores the documented explicit PostgreSQL backend selector, while initialization follows its PostgreSQL URL. Make these existing selection paths agree for explicitly selected PostgreSQL; retain fallback for incidental DSNs and existing SQLite/multi-user/test behavior. Cover the real normal-runtime subprocess with all pytest/test flags removed, official empty PG fixtures, idempotent bootstrap, canonical created user/key and no fallback SQLite file. Do not hide the failure by enabling test mode in the application or changing its configured backend.
