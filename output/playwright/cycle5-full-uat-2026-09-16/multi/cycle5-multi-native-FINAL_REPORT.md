# Cycle5 fresh multi-user UAT — final

**Execution complete; acceptance fails.** Product stayed frozen at `ab527eb3b4`. All 12 rows have outcomes or explicit limits. This agent made no product, Backlog, shared-tracker or Git changes.

## Environment

Fresh private configuration/data, API18501 and UI18581, with existing dependencies reused. First navigation was08:00:01Z; final native result was approximately09:32Z on2026-09-16. Root owned processes/configuration. The operator provider adaptation at08:04:45Z and restart to API PID20095 preceded Alice testing; this is **not a multi-user wizard pass**. Root later stopped20095 for the outage control and restarted the unchanged profile as69593, then paused all UAT runtimes after completion.

Real llama.cpp9099/Gemma generations used serialized leases. Existing unavailable Ollama supplied real failures. The final model lease was released after reanalysis and reload at approximately09:21Z. No mocked responses, token/clock changes, permission changes or permanent deletion occurred.

## Final 12-row matrix

|#|Workflow|Outcome|
|---|---|---|
|1|Fresh setup/provider/first Chat|**PASS with operator adaptation.** Actual admin UI created Alice/Bob with default User roles. Unverified admin Test Connection worked. Alice received a useful real first reply.|
|2|Authentication/reload/logout/offline/reconnect/expiry|**PARTIAL — UAT134.** Account switches and reloads passed. Separate parked expiry1830.947s→refresh200→owned Notes200 passed. Real outage showed guidance and a readiness gate; Retry after restart restored Notes200. A later active expiry looped auth/sessions401.|
|3|Two ordinary turns/reload/failure→Retry|**PASS.** Initial five canonical rows. Separate real Ollama502→Gemma Retry reused the exact user once, excluded display-error content, and reloaded three canonical rows.|
|4|Public file→search→cited QA/source Chat|**FAIL — UAT127/132.** Minimize/navigation/resume preserved job2 without another POST. Backend completed with Warning and saved Media1, while UI stayed Processing0/1. Explicit Media continuation independently passed cited QA, source preview/jump and grounded Chat. Nonmatching search queries returned own items (UAT132).|
|5|Exact Wikipedia→search→Chat|**EXTERNALLY BLOCKED.** Exact URL attempted once; honest Access blocked/0 succeeded/1 failed. Article-dependent steps remained blocked. Approved alternate expiry context preserved broken main job2.|
|6|Biology Note→five cards→five distinct reviews|**PASS.** Exact five facts, five grounded generated/saved cards, five distinct reveal/rating events with settled next-card fetches, session1 completed5 and reload verified.|
|7|Saved Pirate Prompt→applied Chat|**PARTIAL — UAT129/131.** Saved/reloaded Synced#1; actual Use in chat→System Instruction; real pirate reply with literal ARRR and canonical exact system text. Collections401 and duplicate canonical greetings remain. Direct outgoing Pirate body was unavailable; no repeat send manufactured evidence.|
|8|Character creation/replacement/reply/reload|**FAIL — reopened UAT068.** Cold cancel/reopen/create201, fresh TestBot4 entry, real BEEP BOOP, saved route and reload passed. Saved selector→Default Assistant1 did not retain the replacement. Explicit second-click receipt shows same route/mode after2.148s, no dialog. No wrong-identity generation.|
|9|Chat Note/backlink/card/Study/practice/manual End|**PASS for tested reuse, Study and End; Undo limited.** Actual provenance/backlink, reviewed card201→review200/completion1, correct six-card practice pool. Scheduled Cram reviewed once and actual End200 completed session3 with one review/four remaining. Positive multi-user Undo was not established.|
|10|Analysis→Review→reanalysis/save/reload|**PASS with explicit continuation.** Actual analysis saved asv2 and displayed in Review. Unavailable analysis502 preserved it. Restored Gemma reanalysis200/save201v3/reload200 preserved the raw source and prior version. Review Reprocess was chunk/embed only, not analysis.|
|11|Permission-aware Delete/sole admin source Trash/restore|**PASS with API fixture adaptation.** Ordinary Delete remained disabled with guidance. Admin’s sole synthetic source moved to Trash200, empty library retained Trash access, Restore200 returned the same source.|
|12|Reciprocal API/browser ownership and confidentiality|**PASS for tested boundaries; draft limit.** Valid own Notes201/read+update200 preceded reciprocal foreign GET+PUT404; own Chat200/foreign404; job owner200/Bob403; QA history Alice1/Bob0. Bob UI/Back excluded Alice; Alice return excluded Bob. Indigo filtering excluded1/retained0 with no answer/excerpt; public Aurora QA was positive. Dedicated unsaved-draft restoration across accounts was not exercised.|

## Confirmed findings

- **UAT127 / TASK13260.67:** job2 completed08:23:07 with Warning and saved Media1; resumed UI remained Processing0/1. It was never cancelled, reset or resubmitted.
- **UAT129:** Prompt collections returned401 despite other authenticated200 responses; actual Prompt save/reload still passed.
- **UAT068 / TASK13260.15 reopened:** saved TestBot4→Default Assistant1 replacement reverted. Click08:55:05.145Z; settled observation08:55:07.293Z; no confirmation dialog.
- **UAT131:** one weather send, but two identical canonical greetings:68450214… at08:56:28.974Z and2988cf09… at08:57:06.483Z. Creation-request causality was not captured.
- **UAT132 / TASK13260.73:** the documented `{query:<marker>}` search body returned the caller’s own nonmatching item. No foreign content leaked; this is not a query-filter pass.
- **UAT134 / TASK13260.24:** the earlier natural refresh passed; a later active expiry produced two auth/sessions401 responses in a6.5s observation. No token mutation or repeated probing followed.

## Evidence index

All files below are under **`/private/tmp/cycle5-multi-native-`**.

|Control|Key artifacts|
|---|---|
|Expiry/outage/reconnect|`expiry-issued.txt`, `expiry-return.txt`, `expiry-late-status.txt`, `offline-test-wire.txt`, `offline-notes-settled.txt`, `reconnect-notes-canonical.txt`|
|Ordinary/Retry|`ordinary-reload-canonical.txt`, `retry-preretry-canonical.txt`, `retry-success-wire.txt`, `retry-final-canonical.txt`|
|Ingest/QA/source|`ingest-resume-terminal.txt`, `ingest-settled.txt/.png`, `qa-wire.txt`, `qa-source.txt`, `source-chat-wire.txt`|
|Wikipedia|`wikipedia-result.txt`|
|Biology/Study|`biology-generate-wire.txt`, `biology-save-wire.txt`, `study-reviews2-5.txt`, `study-reload.txt`, `chat-card-save.txt`, `scheduled-cram-review.txt`, `scheduled-cram-end.txt`|
|Prompt/Character|`prompt-collections-reload.txt`, `pirate-canonical.txt`, `character-reply-wire.txt`, `character-reload-canonical.txt`, `replacement-receipt.txt`, `default-assistant-dom-identity.txt`|
|Analysis|`reanalysis-failure-wire.txt`, `reanalysis-success-wire.txt`, `reanalysis-reload.txt`, `review-selected.txt`|
|Ownership/permission|`isolation-api.json`, `bob-note-save.txt`, `bob-back-state.txt`, `alice-return-notes-settled.txt`, `admin-trash-wire.txt`, `admin-restore-wire.txt`|
|Confidentiality|`indigo-upload.json`, `indigo-search-wire.txt`, `indigo-result.txt/.png`|

Complete IDs/data impact: **`DETAILED_EVIDENCE_INDEX.md`**. Chronology: **`running-tracker.md`**. Six cards were retained; persisted review events were5+1+1 across sessions1/2/3. Two schedule-off practice events were local only. Admin’s source was restored. Alice’s Aurora source/raw963characters/versions1–3, Indigo source2, Bob’s distinct source1 and all synthetic account records remain preserved.

## Limits

API fixtures for isolation, admin restore and Indigo are explicit adaptations, not UI ingestion passes. Indigo used generation off and unchanged security policy; no9099 request occurred. The positive public control was the earlier actual cited Aurora answer.

Missed observers, unavailable stream bodies and harness corrections are retained and not treated as product failures or reasons to repeat completed actions. The final reanalysis captures both generation and canonical save. Review Reprocess acted immediately during inspection; root verified chunk/embed POST200 and no concurrent LLM call.

Native hidden-tab notification cancellation, a newly completed reasoning-only negative, successful vision recovery, dedicated unsaved-draft restoration and positive multi-user Undo remain unverified. No clean-install or broader platform certification is implied.

A broad request inventory was automatically rejected for credential risk and not retried; safe endpoint observers replaced it. Credential/JWT scanning and hashes accompany the report. No further browser, network or source actions occurred after root paused the runtimes.
