# Fresh UAT matrix — 2026-09-17

**SQLite single-user native pass concluded with recorded failures; PostgreSQL single-user native pass also concluded with recorded failures; both multi-user cells remain pending.** All230 prior findings have verified bounded outcomes. Frozen source: `8f8774e6c868b304a96d95ab82e28389c129a78b`; branch `codex/fresh-install-uat-fixes`. Fresh origin/dev remains `59049e094e0845a4611ea725ae19b7c1754ea709` and is included; original branch creation used an older base and dev was merged later.

Each cell gets a separate source archive, app data/configuration and browser session. Cells run serially. Installed Python/Bun dependencies, system interpreter/browser and existing local model services are reused; this is not a clean-machine dependency installation. PostgreSQL uses official fixture databases with a direct non-superuser, non-BYPASSRLS runtime login; final ordinary-user isolation remains an actual workflow check.

The12 named journeys follow the recovered frontend UAT/E2E protocol. No authoritative A/B/C three-loop mapping was found; historical A/B/C names describe coverage tiers. Single-user rows omit genuinely multi-user-only boundaries with an explicit applicability note.

| Row | Journey | SQLite single | SQLite multi | PostgreSQL single | PostgreSQL multi |
|---|---|---|---|---|---|
| 1 | Fresh setup, provider discovery and first real Chat | Functional pass; UX231 | Not started | Functional pass; UX231 | Not started |
| 2 | Authentication reload, disconnect/logout, outage recovery and natural expiry | Functional pass; UX237; JWT expiry N/A | Not started | Functional pass; UX237; JWT expiry N/A | Not started |
| 3 | Two-turn Chat persistence and real provider failure/Retry | Core pass; UX232; image/visibility limits | Not started | Core pass; UX232; image/visibility limits | Not started |
| 4 | Synthetic file ingestion, search, cited QA and Media-to-Chat | Core pass; analysis warning; API233 | Not started | Fail238 RLS insert; source-dependent steps blocked | Not started |
| 5 | Exact Playwright Wikipedia URL ingestion and grounded QA | Blocked: external source access denied | Not started | Blocked: external source access denied | Not started |
| 6 | Biology Note to five generated cards and five-card Study | Fail234 twice; five-card Study blocked | Not started | Fail234; five-card Study blocked | Not started |
| 7 | Pirate Prompt application, real response and reload | Pass | Not started | Pass | Not started |
| 8 | TestBot Character selection, real response and reload | Fail236; completion/reload blocked | Not started | Fail236; catalogue239 | Not started |
| 9 | Chat to Note/backlink and reviewed card/Study controls | Partial; re-rate preview Fail235; mixed/early-End blocked234 | Not started | Partial; re-rate235/analytics240; source/five-card limits | Not started |
| 10 | Source analysis, Multi-Item Review and changed reanalysis | Pass; controlled availability failure | Not started | Blocked238: no saved source | Not started |
| 11 | Permission-aware soft delete, Trash and exact restore | Pass | Not started | Blocked238: no saved source | Not started |
| 12 | Reciprocal multi-user metadata/content/draft/job isolation | Not applicable: single user | Not started | Not applicable: single user | Not started |

## Preparation and evidence

- Released at 2026-09-17T12:18:45.051Z. All four source archives/dependency copies and origin preflights are complete and independently audited. SQLite single-user initialization, API, frontend and fresh browser are running; the other three cells have not been initialized or launched.
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
