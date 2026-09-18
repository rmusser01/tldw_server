# Fresh UAT matrix — 2026-09-17

**All four configurations have completed the frozen48-row execution with recorded failures. No full UAT acceptance:16new findings231–246 remain unresolved.** All230 prior findings have verified bounded outcomes. Frozen source: `8f8774e6c868b304a96d95ab82e28389c129a78b`; branch `codex/fresh-install-uat-fixes`. Fresh origin/dev remains `59049e094e0845a4611ea725ae19b7c1754ea709` and is included; original branch creation used an older base and dev was merged later.

Each cell gets a separate source archive, app data/configuration and browser session. Cells run serially. Installed Python/Bun dependencies, system interpreter/browser and existing local model services are reused; this is not a clean-machine dependency installation. PostgreSQL uses official fixture databases with a direct non-superuser, non-BYPASSRLS runtime login; final ordinary-user isolation remains an actual workflow check.

The12 named journeys follow the recovered frontend UAT/E2E protocol. No authoritative A/B/C three-loop mapping was found; historical A/B/C names describe coverage tiers. Single-user rows omit genuinely multi-user-only boundaries with an explicit applicability note.

| Row | Journey | SQLite single | SQLite multi | PostgreSQL single | PostgreSQL multi |
|---|---|---|---|---|---|
| 1 | Fresh setup, provider discovery and first real Chat | Functional pass; UX231 | Functional pass; operator setup | Functional pass; UX231 | Functional pass; operator setup |
| 2 | Authentication reload, disconnect/logout, outage recovery and natural expiry | Functional pass; UX237; JWT expiry N/A | Expiry/logout/reload/outage/Retry pass | Functional pass; UX237; JWT expiry N/A | Expiry/logout/reload/outage/Retry pass |
| 3 | Two-turn Chat persistence and real provider failure/Retry | Core pass; UX232; image/visibility limits | Core pass; UX232; image guard only | Core pass; UX232; image/visibility limits | Core pass;232; image guard/persistence only |
| 4 | Synthetic file ingestion, search, cited QA and Media-to-Chat | Core pass; analysis warning; API233 | Core pass after loaded handoff; faults233/241 | Fail238 RLS insert; source-dependent steps blocked | Fail245: quota schema; no source/job |
| 5 | Exact Playwright Wikipedia URL ingestion and grounded QA | Blocked: external source access denied | Blocked: exact source access denied | Blocked: external source access denied | Blocked: exact source access denied |
| 6 | Biology Note to five generated cards and five-card Study | Fail234 twice; five-card Study blocked | Fail234; five-card Study blocked | Fail234; five-card Study blocked | Fail234; five-card Study blocked |
| 7 | Pirate Prompt application, real response and reload | Pass | Pass | Pass | Instruction/reload pass; unverified weather claims |
| 8 | TestBot Character selection, real response and reload | Fail236; completion/reload blocked | Pass with operator default; not236 repair | Fail236; catalogue239 | Fail246 stream timeout; catalogue239 |
| 9 | Chat to Note/backlink and reviewed card/Study controls | Partial; re-rate preview Fail235; mixed/early-End blocked234 | Partial;235/240/242; generated-five blocked234 | Partial; re-rate235/analytics240; source/five-card limits | Partial;235/240/242; source/five-card limits |
| 10 | Source analysis, Multi-Item Review and changed reanalysis | Pass; controlled availability failure | Pass; changed reanalysis/failure preservation | Blocked238: no saved source | Blocked245: ordinary/admin source upload |
| 11 | Permission-aware soft delete, Trash and exact restore | Pass | Permission-aware delete/Trash/restore pass; UX244 | Blocked238: no saved source | Blocked245: ordinary/admin source upload |
| 12 | Reciprocal multi-user metadata/content/draft/job isolation | Not applicable: single user | Bounded native/API controls pass; UX243 | Not applicable: single user | Bounded native/API pass; UX243; Media/job blocked245 |

## Preparation and evidence

- Released at 2026-09-17T12:18:45.051Z. All four source archives/dependency copies and origin preflights are complete and independently audited. All four cells have now initialized and run through normal native workflows. The first three apps/browsers are stopped with preserved data; All four app/browser pairs are stopped with data preserved; both official PostgreSQL fixture holders remain alive.
- API/UI ports: SQLite single18600/18680; SQLite multi18601/18681; PostgreSQL single18602/18682; PostgreSQL multi18603/18683.
- Multi-user administration uses the existing supported bootstrap function, normal admin login and actual admin-UI user creation.
- Record every Pass/Fail/Blocked/Partial/Not applicable with evidence. Preserve exact Wikipedia denial if encountered; no substitute counts as that row. Synthetic automated provider responses do not count as real native inference.
- Optional audio/MCP/evaluation/watchlist/extension coverage is not certified by these12 rows.

### SQLite single-user startup

All four copy/origin preflights and independent archive audit pass. Normal initializer exits0; API18600 and frontend proxy18680 both return health200. New headed browser session has fresh storage and visible first-run wizard. Local llama.cpp9099 discovers the actual Gemma model with blank model field, validates the selected ID and saves through UI. Ingestion defaults remain conservative; The real setup first-chat returned200 and a useful answer. Manual API-key entry through the disclosed restore-access form completed normally; ordinary Chat also returned200 and the requested ORBIT-742 code. New231 records incorrect extension wording in three startup console warnings; no functional failure is established. Frozen source remains unchanged.

### SQLite single-user Chat evidence

Two real ordinary turns return ORBIT-742; the second request includes the prior user/assistant context. A one-shot request-only unavailable-model override reaches the real backend and returns400. Visible Retry sends the original model/context and same client-message identity;200 yields one saved answer. Normal reload retains one system, three user and three assistant canonical IDs. This verifies backend availability rejection/recovery, not an upstream-generation outage.

The public128pxPNG is actually attached. Text-only capability guard and Retry expose the image-support explanation; normal reload retains one local image user and a grouped error at2of2. No image completion was sent and canonical history is empty for this never-sent turn. Vision inference and true-hidden visibility are not certified in this cell.

### SQLite single-user source workflow, in progress

Public synthetic file rowan-observatory-public-20260917.txt (306 words,1914characters) was uploaded normally. Actual ingestjob1 /media1 UUID9023fb11-1882-49e6-b351-d7ee4df1bb32 completed in74seconds with a truthful saved-with-warning result: provider analysis truncated. Original source remains intact, chunking is Completed and vector indexing Pending; analysis is not certified. Minimize and reopen retained the processing job. The backend result repeats the identical warning twice, while the UI displays it once.

Media-only full-text Knowledge QA (reranking off, real configured llama.cpp, answer limit2000 selected in UI) returned the exact director Dr.Mira Vale, Cedar Ridge location and Friday18:00 tour schedule. Citation1 expands to the supporting source paragraph and opens a preview with SourceID1/chunk late_chunk:1:1. Open in Media creates a normal second tab at /media?id=1. Media-to-Chat inserts the source into the current composer; the earlier unsupported-image conversation was still selected, so the independent grounded answer uses a new normal text-only chat.

Independent bounded setup/Chat audit: `.tmp/uat-next-matrix-20260916/audits/sqlite-single-chat-review.md`, SHA25657f2e23c8d0fd045b2ac2d4a31495995ff09c64e8b99b134dd0c115af6135d29 (19inputs). No vision, upstream-generation outage or true-hidden acceptance is implied.

Final media handoff after normal New saved chat and source selection sends the complete1971-character source plus the exact question to real Chat, returning200 in about4seconds. Final answer states the three correct source facts; normal reload retains one system/user/assistant trio in conversation975ea2bb-1844-4bd8-b1d0-79e875f26268. Native full-text Media search for Cedar Ridge preceded selection. Browser-harness attempts that assumed a plain rather than collapsed textarea timed out before sending; clicking the visible collapsed label restored the full text and completed the unchanged product flow.

### SQLite single-user answer reuse

Actual Chat knowledge-save201 creates Note c219a2d1-41c8-42d7-ad03-02803aa7c512 with clean final answer and original conversation975ea2bb-1844-4bd8-b1d0-79e875f26268/message29c95eaa-b8c9-4567-a4b6-0ed23f278bc8. After dismissing the first-use Notes tour, normal Open conversation returns to the exact three-message transcript. Reviewed card save201 creates card2cdc609e-7d72-4f74-882b-c134e9be32f6 and linked Note3b597cff-d69c-4e95-86b6-7e4b4895050b, preserving the same answer and provenance. Study controls remain pending.

Biology Basics — Fresh SQLite20260917 is saved with the exact five protocol facts. Its Generate flashcards action prepopulates the source and source-note context. Generation requests exactly5, unique deck Biology Basics SQLite20260917 and actual llama.cpp model; draft review/study still pending.

Five-card initial generation failed in the browser at29.990s although backendaccesslog records200/30.558s. UAT234/TASK13260.176 tracks the confirmed quickstart proxy budget mismatch. The actual frontend default30s is shorter than flashcard client180s. One unchanged ordinary retry is under observation; first failure remains a failed matrix boundary regardless of recovery.

The one ordinary five-card retry also returns raw500 at30.001s. No further retry or frozen-source change; generated draft/save/five-card study remain blocked. Independent diagnosis SHAde22807d31c5ebc8826ca6ecb01a48805e552c854f2612314c260e7d29ac1933 ties seven frozen source files and installed Next16.1.4 proxy code to the observed failure.

### SQLite single-user Study checkpoint

The one reviewed Chat card is revealed and rated Easy, giving a four-day due date. Practice again with Update schedule off reveals/rates the same card without a review POST or schedule change. Scheduled Cram Good creates review2/version3/10days; Re-rate then Hard creates the intentional third scheduled event/version4/14days. Re-rate incorrectly displayed Hard6days, now UAT235. Normal reload retains Reviewed today3, three completed one-card sessions and next review October1. The native card source link opens Note3b597cff-d69c-4e95-86b6-7e4b4895050b with the exact final answer. Mixed deck/undecked and early-End require additional available cards and remain blocked by234 in this path. No five-card pass.

Independent source/reuse audit: SHA2565893f776a65c5544be258cc8f5285b3db24048bc52b2a3790384d040db05c8f1;27hash-bound inputs. It supports row4 core and row9 Note/backlink/card creation, not the later Study controls.

### SQLite single-user pirate Prompt

Normal Prompt editor saves the exact pirate instruction and syncs as server Prompt1/localpa_6fd4-9289-3a9-cb59. More actions → Use in chat → Use as System Instruction applies it to the existing ordinary source conversation. Actual Chat request13:28:53.957 contains the exact pirate system content and weather question;200 finishes13:29:00.228 with a pirate-style answer containing literal ARRR. Normal reload retains the five-row canonical conversation. The answer references existing Rowan context and honestly lacks live weather; no weather accuracy claim.

Partial durable checkpoint: [native evidence and preparation](../../output/playwright/fresh-matrix-2026-09-17/sqlite-single-checkpoint-1331/README.md), [independent retention review](../../output/playwright/fresh-matrix-2026-09-17/sqlite-single-checkpoint-1331/RETENTION_REVIEW.md). All201payload hashes and four gzip roundtrips pass; known-credential/JWT scan has zero matches. CHECKPOINT_SHA256SUMS additionally binds the manifest, README, retention script and independent review. Later native actions remain outside this snapshot until their next checkpoint.

### SQLite single-user Character entry

TestBot saves the exact instruction but native library Chat blocks on model readiness. Recommended Model Settings selection and a separate actual Chat-picker selection both fail to recover the entry; three attempts retained, no further retry. Actual Chat picker visibly shows LLaMa.cpp/rawmodel after selection, yet library entry still blocks. UAT236 records this product failure; exact BEEP BOOP and downstream Character boundaries remain blocked.

### SQLite single-user exact Wikipedia source

Normal Quick Ingest sends the exact https://en.wikipedia.org/wiki/Playwright_(software) with persist/scrape+auto-balanced chunking, optional summary off,13:41:47.287UTC. Response13:41:49.483 is200 with source_access_denied, media_ids empty and stored_articles0. UI truthfully reports Access blocked/0succeeded1failed. Exact source-content/search/QA remain blocked; no alternative source substituted and no denial page stored. Evidence wikipedia-ingest-observation.txt.

### SQLite single-user analysis and Trash

Real source analysis saves exact LIVE_TIER_ANALYSIS_ONE as version2 (201), and Multi-Item Review displays it. Changed LIVE_TIER_ANALYSIS_TWO saves version3 and survives reload. A one-shot streaming model-availability400 invokes the real nonstream fallback, which successfully saves THREE/version4; this is recovery, not failed-generation preservation. Both subsequent streaming and nonstream attempts return actual backend400; the captured native Failed to generate analysis message is followed by normal reload with THREE/version4 intact and no FOUR/new version. This tests availability rejection, not an upstream outage; exact-token prompts do not certify semantic summary quality.

Metadata diagnosis SHA82f8b596267dfe1dd76024930486874dd3ced5df6f104d9d83b7dd2fd016a726 confirms full original version1 metadata survives. The current processing.safe_metadata field projects the latest version, whose analysis save omitted metadata. No new data-loss finding is established.

Sole media1 is deleted204 at13:55:10UTC. The settled active library becomes0/0 and shows its first-ingest empty state. Trash lists item1 with Sep17,2026 6:55AM deletion time. Native Restore200 at13:56:09 returns the same source and versions1–4; Trash becomes0/0, and normal Media navigation returns the1914-character source and THREE analysis. No permanent deletion.

### SQLite single-user connection recovery

Native Disconnect clears access; reload followed by Media entry displays the credential gate. Three redundant search-failure notifications are new UAT237. Normal Open Settings, manual key entry, Save and Test Connection restore Core reachable/RAG healthy; Media loads original source1 and THREE/version4.

Verified owned API17289 receives SIGTERM14:02:04UTC; port18600 returns ECONNREFUSED14:02:28. Normal reload displays Backend readiness check failed with Retry/diagnostic/settings actions. Same-profile API29748 starts without initialization/reset. Native Retry14:03:24 returns source1 and THREE/version4 without credential re-entry, and no pageerror was observed in this bounded window. Single-user API-key mode has no JWT natural expiry; connection-only Disconnect does not promise local Chat erasure.

Rows10/11 independent24-input review SHA6dedca29d37349c38f96684504e0f3b8f38490058d1fd69b3f62fc0c00aedcb9 confirms analysis outcomes and sole-item restore. The media UUID itself is not re-emitted; identity is supported by ID1, unchanged source and exact four version UUIDs. Seven new issues231–237 remain unresolved, with explicit Wikipedia, exact-five-card, Character, image/visibility and mixed-study limits. This is not full acceptance.

Completed SQLite single-user evidence: [delta packet](../../output/playwright/fresh-matrix-2026-09-17/sqlite-single-completed-delta-1406/README.md), [independent retention review](../../output/playwright/fresh-matrix-2026-09-17/sqlite-single-completed-delta-1406/RETENTION_REVIEW.md). All69 payloads and two gzip roundtrips match originals; merged base+delta covers51 matrix references. Auxiliary hashes bind the README, manifest, retention code and review. Seven findings remain open.

### PostgreSQL single-user startup and Chat

Official pg_temp_db/auth and pg_temp_db_session/content fixtures hold fresh databases under direct runtime role tldw_matrix_14c0da2a18044e37, with no superuser, BYPASSRLS, inheritance, CREATEDB, CREATEROLE, replication or memberships. Normal initializer exits0. Fresh API18602/frontend18682 and browser use the same frozen8f877 source and isolated state.

Visible solo-local setup discovers real llama.cpp9099 with blank initial model, selects/revalidates/saves Gemma. Actual first-chat200 at14:10:56 returns exact fresh PostgreSQL single-user ready. Normal key handoff and ordinary Chat succeed. Two turns reload as5 canonical rows in conversation2f4fa839-f278-4a7b-9397-e1b706a35c72. Real model-availability rejection then native Retry returns one answer; reload retains7distinct system/user/assistant rows. Generic error guidance reproduces232; controlled rejection is not an upstream outage. Image guard/visibility limits remain explicit.

### PostgreSQL single-user source failure

Native Rowan ingest job1/fca92acb-398f-41c8-8379-7376cc655615 starts14:19:00UTC and survives minimize/reopen. At14:19:51 it returns no media ID and a database write failure. The restricted-role PostgreSQL log proves INSERT INTO Media violates row-level security; new UAT238/TASK13260.180 tracks the defect. The active catalogue remains0/0, so source QA, Media-to-Chat, reanalysis and sole-item Trash are blocked. Duplicate provider-analysis warnings reproduce233.

The separate exact Wikipedia URL attempt returns200 with source_access_denied/media_ids[]/stored0 at14:23:36.935; the UI says Access blocked. No alternative source or denial page is stored.

PostgreSQL Biology source ff7eb702-0e0f-4d38-8ced-dea50f180655 saves201 with exact227-character/five-fact content. Generate5 sends14:27:02.421; browser500 at14:27:32.434 is30.013s, while actual backend200 logs39.103s. This reproduces234; no draft/deck/five-card Study is available.

Independent238 diagnosis SHA0757595c34de959537effe3ad5ea772d4165d20c1a84304ba068acb1000ecc76 binds21 frozen source/history hashes: persisted job/client/derived row owner1 is correct, but content authorization context is absent in the worker and is not carried through run_in_executor. Backend empty-user scope explains RLS denial by source inference; exact native GUC values were not captured. Repair must retain RLS and restricted runtime roles.

PostgreSQL Pirate Prompt syncs as server1. Native Use in chat → Use as System Instruction sends the exact87-character pirate system prompt and weather question14:32:20.977 through real llama. Response contains ARRR and honestly asks for location rather than inventing live weather. Normal reload retains system/user/assistant in conversation392c8053-b7bc-45b2-8ef8-693b690b7939. A premature harness Back click before Save settled triggered a discarded-navigation confirmation; it was dismissed and saved Prompt preserved.

PostgreSQL TestBot creates201 as Character3 with the exact BEEP BOOP instruction, but library Chat rejects the working configured model (236). No Character completion/reload is certified. The form also invokes world-book catalogue twice and receives500;239/TASK13260.181 identifies the unsupported connection context in list_world_books, distinct from214's previously bounded read methods.

Because media-source persistence is blocked238, row9 independently reuses the actual Pirate Chat answer rather than claiming a grounded-source reuse pass. Native Save to Notes creates8fd0cca2-b54c-4d93-aec1-b53c6bea8daa; reviewed single-card save createsc780acb5-a76d-4d60-983a-d49d3d508161 and linkedNote16652599-f165-4581-b9ae-9baac1bea92c, both preserving conversation392c8053-b7bc-45b2-8ef8-693b690b7939/message7a6f2b4c-9bd3-436d-a52d-2b1a57c77807. This does not replace the failed five-generated-card requirement.

PostgreSQL row9 Study: initial Easy saved version2/4days. Update schedule OFF practice made no review POST (count1 before/after). The browser subsequently closed for an unknown reason; CLI session and PID were absent while the API/frontend remained running. A fresh browser used normal visible key entry and recovered the same card/review. Scheduled Good saved version3/10days; native Re-rate displayed old Hard6days despite the response preview Hard14days. Hard saved version4/14days, reproducing235. Normal reload retained Reviewed today3, three completed one-card sessions, and October1 due date. Card source link opened Note16652599-f165-4581-b9ae-9baac1bea92c. A harness attempt to reveal the already-revealed re-rate card timed out after successful Good/re-rate; no extra rating was sent. No uninterrupted browser continuity or five-card pass.

Independent239 audit SHA5ac412549f32a88a1efef8a75302005a2d7ca718bc63c1635eac79487c072e20 confirms seven frozen source/history hashes and three safe evidence inputs. Catalogue failure occurs before SELECT; prior214 covered different readers.

### PostgreSQL single-user connection recovery and limits

Normal Disconnect/reload/Media navigation displays Add your credentials to use Media. Open Settings, visible key entry, Save/Test Connection restore Core reachable and RAG healthy; normal Study returns all3 reviews. A harness wait expected different credential wording and timed out after successful navigation; brief redundant toasts are not certified in this PG check. Verified owned API30006 stopped14:55:14UTC, port18602 refused14:55:33, and actual reload showed Backend readiness check failed with Retry. Same-profile API40346 restarted without initialization/reset. Native Retry14:56:46 restored3 reviews, three completed sessions and unchanged October1 due date without credential re-entry; bounded observer recorded no pageerror. JWT expiry is not applicable to single-user API-key mode.

All12 PG-single rows are now accounted for with explicit failures/limits. Ingest238 blocks media QA/handoff, reanalysis and Trash. Exact Wikipedia is externally denied. Generation234 blocks5-card/mixed study. Character236 and catalogue239 fail. Image guard is not vision inference. No single-user pass is full acceptance; both multi-user cells remain pending.

Study analytics additionally misclassifies Hard recall as a lapse (240/TASK13260.182): actual scheduler retains lapses0 while aggregate lapse33.3%/retention66.7%. Both source semantics and live response establish a separate analytics defect. Completed PG apps/browser stopped14:58:42UTC; ports18602/18682 refused14:59:14. The official fixture holder29823 and databases remain for later repairs. Fresh SQLite multi-user normal initializer exited0; supported admin bootstrap is next.

Independent late review recovered a missed237 reproduction: auth-reconnect-form.txt visibly contains three Failed to search media notices after the disconnected Media → Open Settings transition. Their presence is proven; exact transient timing/request count is not. This supersedes the earlier no-claim wording about PG notices. Independent240 diagnosis SHAe7b3c05f7f6bab83c011955b0a05bc6948a525239e814326654d16fbf5092e7b binds12 frozen source/history hashes and two native captures.

Durable PG-single evidence: [completed packet](../../output/playwright/fresh-matrix-2026-09-17/pg-single-completed-1458/README.md), [independent retention review](../../output/playwright/fresh-matrix-2026-09-17/pg-single-completed-1458/RETENTION_REVIEW.md). All125 payloads and one gzip roundtrip match sources; all48 distinct PG row references resolve. Main manifest5d0f60e143811af5d45cfc575ef6b5753aa6e51d391e6a23d5daaecaa9e09652, review edb047112d1450ec1ba336ebf4641371ed5795f49ec3cf1973fa415ebc5eeabb.129 auxiliary hashes bind the complete packet except the checksums file itself. No product source changes; no Python scope for Bandit in this evidence-only checkpoint.

### SQLite multi-user startup and Chat

Normal initializer/bootstrap exit0. Actual admin1 signs in through fresh guide and creates Alice2/Bob3 as User through Server Admin; no direct account inserts. Native Model settings says configure a provider on the server. Supported isolated operator config sets actual discovered llama.cpp9099/model and same-profile API restarts45087; frontend43230/18681 remains. This is an explicit operator adaptation, not a multi-user provider wizard.

Normal admin logout then Alice login verifies auth/me2. Two real Chat turns retain ORBIT-742; normal reload canonical5, then real unavailable-model400 and native Retry original model/context/sameclient pa_45c2-be7e-94d-4e2d preserve one user turn. Settled canonical history7 in246b9dc3-1eda-4f9e-a2b4-61ca132ec789. UX232 reproduces. Actual128pxPNG attached to a new saved chat: capability guard/Retry2of2 makes0additional completions; reload shows one Uploaded Image complete128x128 and canonical0. No vision or true-hidden pass.

Separate Alice expiry context logs in normally at15:07:07.498/lifetime1800, reads owned Notes200, parks15:07:52 with0pages/0serviceworkers, and must return after15:37:17.498UTC. No token/storage extraction or synthetic time. Main account workflows remain separate.

Public Rowan ingestion job1 starts15:20:55, normal minimize/reopen retains queued progress. Earlier wrong-case Start locator timed out before submitting; only one actual job was created. Source outcomes remain pending.

SQLite-multi Biology4b06d0b4-51a5-4857-8d83-4082df846767 saves201 exact227chars as owner2. Generate5 request15:25:00.116→browser50015:25:30.123 (30.007s); backend200in40.829s.234 reproduces; no5draft/deck/Study. Native source job1/ac522301-32e8-403a-b098-811b02351281 owner2 completes15:21:57 with Media1 UUIDf260fcff-c95a-4aeb-9352-7dbc72bad022 and duplicated233 warning; UI truthfully1savedwithwarnings/0failed/1:02. Source1914chars/chunkingCompleted/vectorPending intact. Cedar Ridge full-text search finds it. Media-only/full-text QA real200 answers Dr.Mira Vale/Cedar Ridge/Friday18:00,5chunks/1citation. Native citation excerpt and previewlate_chunk:1:1 match; Open in Media opens actual /media?id=1 tab, which is verified/closed while original expiry anchor remains. Handoff/reuse still pending.

Natural expiry accepted15:39:43UTC: separate context returned1955.146s after normal login/lifetime1800. Initial auth/me401 was followed by refresh200, Alice2 identity200 and owned Biology Note200. No password re-entry, token/storage extraction, injected time or background open pages/workers. Context then closed normally. Main logout/recovery remains pending.

SQLite-multi source Chat revealed new241/TASK13260.183: immediate selection→enabled Chat with this media passed only an ID hint. Waiting in Chat did not recover content. Returning and waiting for visible source produced1971characters; clicking inside the collapsed composer label expands it. Real request with that source and the exact three-fact question returns the correct Dr.Mira Vale/Cedar Ridge/Friday18:00 answer. Initial ID-only draft was not sent. Canonical conversationaf70ef91-059c-44bb-b106-8704f61e8820; reload/reuse continues.

SQLite-multi answer reuse: knowledge/save201 creates Note51470044-f5ea-4e6b-a5ef-3d49dfd224ab and reviewed card56320d51-cc2f-4049-baed-65a888230112/linked Note6174102e-59d7-4c12-8eb2-0e408ef935aa. Both preserve conversationaf70ef91-059c-44bb-b106-8704f61e8820/messagee4aab5d7-333e-4b44-953e-e1b738114bdb and exact115-character answer. Native canonical reload has3rows. The backlink is in More actions as menuitem Open conversation; the initial harness wrongly sought a top-level button.

Study Easy persists4days/version2. Practice Update schedule OFF has review request count1before/1after. Scheduled Good persists10days/version3; Re-rate incorrectly displays old Hard6days although server preview14days, reproducing235. Hard persists14days/version4, repetitions3/lapses0, but settled analytics displays retention66.7%/lapse33.3%, reproducing240. Reload retains3reviewed,3completedone-card sessions and October1due. Five-generated-card/mixed/early-End remain blocked234. Evidence study-easy-practice, study-practice-off, study-good-rerate, study-hard-reload, study-final-note-entry.

SQLite-multi Pirate Prompt syncs as server1. Exact system instruction + weather question use real llama; answer includesARRR and requests location. Reload retains3canonicalrows in870f8d17-44ff-488e-ae03-ee03f0e90f5d. TestBot creates as Character4 with exact instruction, enters CharacterChat successfully under operator-configured default llama/rawmodel, and completes-v2 returns visible exactBEEP BOOP. Reload retains user/assistant inea6efc19-3395-4640-8757-88bd6f0be6ec. This path does not reproduce236; it does not repair the failed single-user qualified model paths. Raw persisted provider text contains think markup that the UI hides; visible exact response acceptance is bounded accordingly. World-book catalogue200 on SQLite. Earlier persona discovery307→429 observations are retained; authenticated Character query/create/completion succeed.

SQLite-multi row10: real analysisONE saves201 as version2, including provider-added quotation marks; initial exact-token harness wait was too strict, not a failed save. Multi-Item Review displays this saved value. Changed TWO (revised instruction excludes quotes) saves as version3 and survives reload with original1914-character source. Request-only unavailable-model controls then produce real400 for streaming and nonstream fallback; native Failed to generate analysis appears. Cancel/reload preserves TWO and no UNSAVED analysis. This proves backend availability rejection, not upstream-generation outage or semantic summary quality. Ordinary Alice still sees Delete disabled with explicit missing-permission explanation.

SQLite-multi confidentiality policy control: native synthetic Indigo ingest job5/aee4f1d8-5801-4d09-bb6d-c3580993b585 owner2 saves Media2 in2seconds, analysisoff. Retrieval-only/full-text/media-only/specific2 returns200 with security_filter excluded_count3/retained_count0 and documents[]; UI explicitly says security settings excluded all sources. Same settings selecting publicRowan1 returns200/excluded0/retained5 with matching source paragraphs. These counts are chunks, not documents; no policy bypass. Exact Wikipedia separate attempt16:00:32 returns source_access_denied/media_ids[]/stored0, truthful Access blocked; no substitute.

### SQLite multi-user final boundaries

- Exact Wikipedia ingestion returned source_access_denied, no stored article. The public Rowan source separately supports cited QA, loaded-source full-content Chat and provenance-preserving Note/card reuse. Early source handoff fails241. TestBot returns visible exact BEEP BOOP and reloads under the operator-configured raw default; this does not close the single-user model identity failure236. Pirate Prompt and real response acceptance use literal ARRR in the retained response.
- Analysis: quoted LIVE_TIER_ANALYSIS_ONE savedv2, changed LIVE_TIER_ANALYSIS_TWO savedv3 and displayed in Multi-Item Review. Real400 stream and fallback attempts preserve TWO and the original1914-character source on reload. Five-card generation234, scheduling preview235, analytics240 and one-card grammar242 remain failures. Late manual cards were created only as ownership fixtures; generated-five/mixed/early-End Study is not certified.
- Reciprocal normal logout/login and settled browser Back through login/Settings/generator clear each prior owner’s source context and deck draft. Alice own QA history/Notes reappear; Bob’s Note URL is404. Separate normal-password API sessions corroborate reciprocal valid Note writes/foreign denials and exact content restorationv3, Chat detail/messages404, card404 and job403. Both Media1 and Deck1 are valid per-user IDs with distinct owned content. Populated deck catalogue responses contain only each own private name. Bob’s virtual selector wait timed out after opening; no final settled selector screenshot is claimed.
- Confidentiality-policy control excludes3private Indigo chunks and retains0; the public Rowan positive under identical retrieval settings excludes0/retains5. This is separate from account ACL isolation. Natural30-minute expiry returned after1955.146seconds and refreshed normally with Alice2/own Note200.
- Ordinary Alice/Bob delete buttons correctly deny missingpermission. Admin1 native ingestjob7 creates own Media1 UUID1589145e-98bf-4c35-900d-5b744672afd3. Delete204 leads to actual Trash; Restore200 preserves the exact1914characters. The initial empty catalogue persisted beside loaded content244; navigation after restore shows1/1. A post-restore observation locator matched two paragraphs and failed after the successful action; the settled follow-up preserves the result.
- Controlled owned API stop at16:47:08 producedECONNREFUSED and native Backend readiness check failed/Retry. Same-profile restart then native Retry at16:49:10 restores admin1 and Media1 with no password re-entry. The browser and owned apps are stopped after recording outcomes; data is retained.

Late independent audit: `.tmp/uat-next-matrix-20260916/audits/sqlite-multi-late-isolation-review.md`, SHA889007f0f6bb0ff54bd0187f413c1e3cddbbf4374f09f31d43065f8216bbebba,43hash-bound inputs; it predates adminrestore/outage. It distinguishes22asserted login/Notes/Chat entries plus10inspected reads in the originalAPI32, and10asserted supplement checks. The early Alice Note editor capture was pre-load; API restoration and later full handoff provide the actual evidence.

### Retained SQLite multi-user checkpoint

[Evidence packet](../../output/playwright/fresh-matrix-2026-09-17/sqlite-multi-completed-1650/README.md) and [independent review](../../output/playwright/fresh-matrix-2026-09-17/sqlite-multi-completed-1650/RETENTION_REVIEW.md):258payloads, two verified gzip roundtrips, all85distinct SQLite-multi row references,15unchanged frozen harness files; known-credential/JWT scan0matches. Manifest SHA bd5462641d65f8c58c7836e553f6073bb12072c9d1a3952b0c6051eae80dcbb4; review f840a3ccb482786c7fb1d737f09c4df97c2f0a38158f350a34dffdc7bce3ead2; checksum index b215af9e43a378f0de1fb86ce22ac4e4485c2e2cc20ec8d718d8ef7e44224254. Two older controller hashes remain historical references, not retained exact controller snapshots. Admin post-restore rootUUID is not independently re-emitted; ID1/title/originalcontent/version1 are confirmed and the ingest receipt establishes the original UUID. No full-matrix signoff.

### PostgreSQL multi-user startup and Chat

Official fixtures hold separate fresh AuthNZ/content databases with runtime role `tldw_matrix_f646b1f478dc4c8d`: LOGINtrue; superuser/BYPASSRLS/INHERIT/CREATEDB/CREATEROLE/replicationfalse; zero memberships. Standard initialize and admin bootstrap exit0; native admin UI creates ordinary Alice2 and Bob3. Supported operator provider configuration is applied to the isolated runtime before first API start. Backend18603/WebUI18683 use frozen8f8774e6.

Alice actual Chat conversation1000f786-333a-4fe1-b48c-6097bc36c71d completes two ORBIT-742 turns200, then a one-request unavailable-model override yields real400. NativeRetry originalmodel200 persists one additional user/assistant pair. Settled readback has7unique canonical IDs; earlier immediate reload capture had5 and is not used as final proof. Actual128pxPNG attachment, capability guard and Retry preserve one loaded image after reload; completion count4before/4after. No successful vision or true-hidden acceptance.

An independent normal Alice browser context receives a real1800-second lifetime at16:58:30.444UTC; all its pages close at16:58:56.793, with0serviceworkers. Earliest natural return17:28:40.444UTC. Other workflows continue while it remains idle; expiry acceptance is pending. Native source ingestion is starting.

### PostgreSQL multi-user checkpoint 17:34UTC

Natural expiry passed after1829.414seconds with a real1800second token: auth/me401, refresh200, Alice2 and ownNotes200. Child context closed after capture. Exact Wikipedia is source_access_denied with no stored article. Pirate answer Note/card saves201, original Note backlink and linked card-source Note pass. Easy4days/practiceOFF noPOST/Good10days/Hard14days persist; preview235, analytics240 and singular242 reproduce. An initial wrong Notes query and Cram radio input locator were harness errors; corrected native actions retained. Study source link opens the actual277-character answer. Generated-five/mixed/early-End are not accepted.

TestBot Character4 is created and enters Chat under the configured raw model. World-book catalogue500 reproduces239. Its actual complete-v2 request at17:28:00.726 gets200headers but native stream times out at~45seconds without visible BEEP BOOP; root cause remains under read-only investigation. No successful Character completion/reload claimed. Evidence: native/pg-multi/expiry-close-testbot-observe.txt and testbot-timeout-log.json. Account-isolation checks continue; no full acceptance.

### PostgreSQL multi-user ownership and upload boundaries

Native settled Alice→Bob and Bob→Alice Back chains show current identity, cleared foreign source/generator draft and only own populated deck. QA/Chat draft fields are empty after each switch. Bob owns Note d5e4cbfb-96e9-4fca-9b2d-c25ead5ade8c, Chat7d28639e-cf24-4da9-b343-75bab49d12d4 and deck2/card7a4417bb-11a2-434d-974f-8a646830de8a. Alice private deck1/card943a2602-e64c-4736-a551-45405885fd95 remain isolated. Native direct foreign Note links404 both ways. Bob ordinary completion again shows Character setup243; canonical reloadcorrects.

Independent normal API logins pass30asserted observations: Notes ownread/write200 and foreignread/validPUT404, Chat detail/history own200/foreign403, cards own200/foreign404, only own populated deck. Initial helper wrongly required404forChat; the legitimate403denial was retained and the assertion corrected. Exact original Notes restoredAlicev5/Bobv3; finalnativeAlice227chars/Bobgeneratororiginalprove content. No browser token/storage read.

Administrator native source upload also413Quota check unavailable at17:46:13, analysisoff. No job/Media iscreated;245blocksanalysis/reanalysis/Trash/restore and positiveMedia/jobownershipcontrols. No source or schema insertedaround thefailure. TestBot246 originaltimeout remains open; afterlogout/reopen canonicalhistorycontains onlyuserturn and itsMoreactions offersDelete/Pin, noRetry. No newturnsent as a substitute. Final controlledoutagecheck is pending.

### Final PostgreSQL multi-user closeout

Owned API98070 stopped17:48:31; actual18603ECONNREFUSED17:48:48. Native reload shows Backend readiness check failed. Same-profile API59324 starts17:49:11 without initializer/reset. NativeRetry17:49:46 restores Alice2 and original227character Note7b4f7117-4139-4f35-9107-a696145f7810/version5 without password re-entry. Finalconsole500s occurred during the intentional outage; originalhistory also confirms231extensionwording.

Browser closes and ownedAPI59324/frontend98121stop17:50:30. Ports18603/18683bothrefuse17:51:04; officialfixtureholders29823/96865remain. All48cells have boundedoutcomes, with blockeddependencies explicit. No full acceptance;231–246 must be reconciled with fixes/verification before anotherfullmatrix.

### Final retained PostgreSQL multi-user checkpoint

[Evidence packet](../../output/playwright/fresh-matrix-2026-09-17/pg-multi-completed-1751/README.md) and [independent review](../../output/playwright/fresh-matrix-2026-09-17/pg-multi-completed-1751/RETENTION_REVIEW.md): all 186 payloads (1,524,044 bytes), 89 row-reference usages / 83 distinct paths, and 15 frozen harness hashes verified. No payload needed gzip. The author-run known-value credential scan and independent JWT-shape scan found no matches; their different coverage is explicit in the review. A row 7 filename reference was corrected without changing any result.

Manifest SHA `1d4d60585e9cb0a48544ce212448559f51a4592f7b1bd93ca04f1be6b4f2c593`; review SHA `4cab6a7cc70de0a2c63e0f2192bbed62f3e72b057c952f14fbbcad5ef73e9254`; auxiliary checksum index SHA `d36de0b89bb1c60c79ad501d9f37f9bab2e683c31de7c183def0bc1db4d437f6` binds 191 files, excluding itself. All 48 outcomes are recorded with failures; sixteen findings remain open. This evidence-only checkpoint changes no product source, so Bandit is not applicable. Repairs continue under [the next plan](../../IMPLEMENTATION_PLAN_fresh_matrix_231_246_repairs.md) before another full matrix.
