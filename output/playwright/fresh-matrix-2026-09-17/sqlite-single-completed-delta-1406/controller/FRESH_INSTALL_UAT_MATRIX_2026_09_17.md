# Fresh UAT matrix — 2026-09-17

**SQLite single-user native pass concluded with recorded failures; PostgreSQL single-user preparation started.** All230 prior findings have verified bounded outcomes. Frozen source: `8f8774e6c868b304a96d95ab82e28389c129a78b`; branch `codex/fresh-install-uat-fixes`. Fresh origin/dev remains `59049e094e0845a4611ea725ae19b7c1754ea709` and is included; original branch creation used an older base and dev was merged later.

Each cell gets a separate source archive, app data/configuration and browser session. Cells run serially. Installed Python/Bun dependencies, system interpreter/browser and existing local model services are reused; this is not a clean-machine dependency installation. PostgreSQL uses official fixture databases with a direct non-superuser, non-BYPASSRLS runtime login; final ordinary-user isolation remains an actual workflow check.

The12 named journeys follow the recovered frontend UAT/E2E protocol. No authoritative A/B/C three-loop mapping was found; historical A/B/C names describe coverage tiers. Single-user rows omit genuinely multi-user-only boundaries with an explicit applicability note.

| Row | Journey | SQLite single | SQLite multi | PostgreSQL single | PostgreSQL multi |
|---|---|---|---|---|---|
| 1 | Fresh setup, provider discovery and first real Chat | Functional pass; UX231 | Not started | Not started | Not started |
| 2 | Authentication reload, disconnect/logout, outage recovery and natural expiry | Functional pass; UX237; JWT expiry N/A | Not started | Not started | Not started |
| 3 | Two-turn Chat persistence and real provider failure/Retry | Core pass; UX232; image/visibility limits | Not started | Not started | Not started |
| 4 | Synthetic file ingestion, search, cited QA and Media-to-Chat | Core pass; analysis warning; API233 | Not started | Not started | Not started |
| 5 | Exact Playwright Wikipedia URL ingestion and grounded QA | Blocked: external source access denied | Not started | Not started | Not started |
| 6 | Biology Note to five generated cards and five-card Study | Fail234 twice; five-card Study blocked | Not started | Not started | Not started |
| 7 | Pirate Prompt application, real response and reload | Pass | Not started | Not started | Not started |
| 8 | TestBot Character selection, real response and reload | Fail236; completion/reload blocked | Not started | Not started | Not started |
| 9 | Chat to Note/backlink and reviewed card/Study controls | Partial; re-rate preview Fail235; mixed/early-End blocked234 | Not started | Not started | Not started |
| 10 | Source analysis, Multi-Item Review and changed reanalysis | Pass; controlled availability failure | Not started | Not started | Not started |
| 11 | Permission-aware soft delete, Trash and exact restore | Pass | Not started | Not started | Not started |
| 12 | Reciprocal multi-user metadata/content/draft/job isolation | Not applicable: single user | Not started | Not started | Not started |

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
