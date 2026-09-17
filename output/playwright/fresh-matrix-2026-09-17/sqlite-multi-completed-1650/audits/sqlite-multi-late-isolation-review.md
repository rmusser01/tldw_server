# SQLite multi-user late native audit — rows 10 and 12

## Disposition

**Row 10 is supported for the named analysis/reanalysis/error-preservation journey. Row 12 is supported for the exercised reciprocal account boundaries, using native browser flows plus explicitly identified API corroboration.** Both settled browser-history Back directions are evidenced. Reciprocal Note and Chat authorization, restored owner writes, cleared drafts/handoffs, distinct per-user Media, owner-specific deck catalogues, reciprocal card/job denials and confidentiality/public retrieval controls are supported. The completed ten-check supplement closes the initially missing reciprocal job and populated card/catalogue evidence. This is bounded acceptance, not exhaustive isolation or a full-matrix pass. Bob's final populated selector snapshot was lost to a harness locator timeout; no successful selector-view assertion is made.

This is an evidence audit under parent TASK13260, on the parent's attributed frozen source `8f8774e6c868b304a96d95ab82e28389c129a78b`. It does not re-hash the application archive or execute the application. Only the report and JSON hash index are written. No browser, runtime, inference, database, credential, private helper/log, task, tracker, git, product or test action was performed. The companion JSON binds every reviewed input; provider reasoning is intentionally omitted.

## Row 10 — real saved analysis and failure preservation

- `analysis-one-result.txt` records real completion200 at **16:02:13.569Z**, then version-save201 at **16:02:15.992Z**. Media1 displays **quoted** `"LIVE_TIER_ANALYSIS_ONE"`. The API stores the quotation marks as HTML entities (`&#34;`). This is a successful save with provider-added quotation marks; it is not exact unquoted output. The earlier strict-token locator timeout must not be represented as a save failure.
- `analysis-review-two-start.txt` contains the actual Multi-Item Review snapshot with the original source and quoted ONE. The revised instruction explicitly requests TWO without quotes. Real streaming completion200 at **16:04:23.568Z** saves unquoted `LIVE_TIER_ANALYSIS_TWO` with201 at **16:04:26.039Z**. `analysis-two-reloaded.txt` shows TWO after reload.
- Media versions are **v1 `fc0ab53e-8a5a-4f1e-825f-b6994b9eef89`**, **v2 `53a04457-f776-454c-9629-3d84098b9879`**, and **v3 `616f0723-e34b-4d61-8334-1ea1ca1767b6`**. The selected full Media bodies before/after reanalysis all contain the same 1,914-character source, SHA256 `a94b1e966d89b7b94e0cd69dafe9ab1c554dc81accf43e57957276b08294225c`.
- `analysis-failure-preserved.txt` records the request-only model/provider override for **both** streaming and nonstream fallback. They reach the real backend and return400 `model_not_available` at **16:05:35.295Z** and **16:05:35.370Z**. Both overrides declare `noResponseFulfillment:true`; this is not a fabricated response or an upstream-generation outage.
- The failure snapshot visibly says **Failed to generate analysis**. Cancel/reload returns Media200 at **16:05:36.189Z**, still TWO/v3 and the same three version UUIDs/source. No UNSAVED output/new version appears in that settled response. This proves failure preservation for the controlled availability rejection, not semantic summary quality or all provider failures.
- Alice's Delete action is disabled with **Your account does not have permission to delete media.** This is an honest permission boundary only; row11 authorized delete/Trash/restore is outside this audit.

## Confidential source negative and public positive

`policy-indigo-result.txt` and `policy-controls-final.txt` preserve actual retrieval-only `/rag/search` requests:

| Control | Request/response UTC | Scope/result |
|---|---|---|
| Indigo | 16:10:48.087 → 16:10:48.213, 200 | `include_media_ids:[2]`, documents0, security_filter excluded3/retained0 |
| Public Rowan | 16:12:00.166 → 16:12:00.509, 200 | `include_media_ids:[1]`, documents5, excluded0/retained5 |

The request bodies differ **only in query and selected media IDs**. Both use full-text Media retrieval with generation disabled. The Indigo UI truthfully says security settings excluded all retrieved sources. The positive result's five `late_chunk:1:*` records identify Media1; these are chunks, not five independent documents. This supports the configured synthetic policy negative/public positive. It does not prove every classifier/policy, invisible documents, successful generation, or a policy bypass. The Indigo ingestion job itself is described by the controller but is outside these two selected policy receipts' independent acceptance.

## API corroboration — inspect the bodies, not only `completed:true`

The reviewed helper `sqlite-multi-api-isolation.mjs` uses independent normal password login, validates Alice2/Bob3 via `/auth/me`, keeps tokens in memory and does not extract browser tokens or query the database. Its private credential file was **not opened by this reviewer**, and the helper was **not executed**. `isolation-api-corroboration.json` contains32 records between **16:24:47.550Z and16:24:48.393Z**: two identity records and30 API observations. Its final ten reads have no status assertion in the helper; the outcomes below were independently checked from their retained bodies.

### Asserted reciprocal controls, independently confirmed

For Alice Note **`4b06d0b4-51a5-4857-8d83-4082df846767`** and Bob Note **`b6ed11ef-8ac1-4b92-82c2-c5fd1e3bab4a`**, each owner reads200/v1, makes a valid versioned PUT200/v2, and the other account's GET and validly shaped PUT both return404. The owner reread remains exactly its own v2 content. Owner restoration PUT200 returns the exact original content at **v3**. These were deliberate writes followed by restoration; they are not unchanged-version or wholly read-only storage checks. Bob's restored content is134 characters; `bob-note-restoration.txt` visibly confirms the original text, All changes saved and Version3. Alice's later Notes list in `alice-return-surfaces.txt` confirms original Biology content/version3.

Both owners' saved Chat detail and canonical messages return200 while reciprocal foreign reads return404:

- Alice: **`af70ef91-059c-44bb-b106-8704f61e8820`**, canonical3.
- Bob: **`e24e6232-aae1-472e-98c5-3c29906ababf`**, canonical3.

### Additional observed controls

- Alice Media1 is Rowan (1,914 characters); Bob Media1 is his own Birch (1,906 characters, SHA256 `cbae9d1e7cf5a4f43ebc728cddd0bcb837b9de0bc8ffb77f98e96e0915c5584f`). A shared numeric ID is **not a shared document** in these per-user SQLite files. Bob's earlier `/media/1`404 occurred before his ingest; the later200 correctly returns Birch, not Rowan.
- The owner-facing job1 response binds Alice2, ingest UUID `ac522301-32e8-403a-b098-811b02351281` and Rowan media UUID `f260fcff-c95a-4aeb-9352-7dbc72bad022`. Bob job1 returns403. Separately `isolation-bob-source-opened.txt` records Bob3 job6 `05febce9-1c8d-4b12-a26d-30af792ab484` completed, creating media UUID **`3ebf97d8-3d9a-4a61-b9b3-154365493b2b`**. The owner/content/UUIDs differ. The original32-record packet lacks reciprocal Alice denial for Bob job6; the completed ten-check supplement below supplies it.
- Alice's card **`56320d51-cc2f-4049-baed-65a888230112`** reads200/v4/client2; Bob reads404. Both deck catalogues in this earlier API packet are empty, so they cannot establish populated-deck isolation.
- Alice's QA list has her two saved QA sessions; Bob's is empty. Native Bob Knowledge QA also says No previous QA sessions yet. No Bob saved QA session or reciprocal Alice denial for such a session is claimed.
- The later native `bob-deck-card-created.txt` proves owner-positive creation through ordinary UI: private **deck1 `Bob Private Deck BIRCH913`**, client3/v1; card **`0665ebfb-836d-46c3-b573-f22260ed9731`**, client3/v1, new/repetitions0. Both return200. This establishes the positive fixture; the completed supplemental API checks below establish the later reciprocal boundaries.

### Completed ten-check supplement

`isolation-api-supplement.json` and its reviewed helper cover **16:40:51.176Z–16:40:51.479Z**. All ten retained GET outcomes are asserted by that helper and independently inspected here; login is normal password authentication, independent of browser storage. Alice2/Bob3 identities are verified. The original32 plus supplemental10 records total42 observations; they are not42 browser actions or42 distinct automated tests.

- Each populated deck catalogue returns exactly its own private deck1: **Alice Private Deck ORBIT742/client2** or **Bob Private Deck BIRCH913/client3**. Same numeric deck IDs are scoped by the separate per-user SQLite stores.
- Alice card **`81abd812-2c27-47d9-b926-f0134b155b81`** and Bob card **`0665ebfb-836d-46c3-b573-f22260ed9731`** each return200 to the owner and404 to the other account; both owned cards remain v1/new/repetitions0. The supplement does not attempt foreign mutations of these cards.
- Bob job6 returns200 to Bob and403 to Alice, complementing the earlier Alice job1 own200/Bob403.
- Native `alice-deck-created.txt` records an empty Alice deck catalogue while Bob's saved private deck already existed, then ordinary UI creation of Alice's private deck1 and new card (both200). Its settled Manage snapshot shows Alice's two cards and her own deck name. These are newly created fixtures, not the earlier reviewed source card.
- `bob-populated-deck-check.txt` is a retained **harness failure**: it waited on the hidden AntD virtual `role=option` element for Bob's deck. The log identifies the hidden option, but contains no final structured result or settled open-selector snapshot. Do not claim a visually verified populated Bob selector from this file. The separate actual API catalogue check is valid and explicit; no product defect follows from this locator timeout alone.

## Browser content, drafts, handoffs and literal Back

### Alice → Bob

`isolation-alice-drafts.txt` records the unsent Alice Chat marker. `isolation-alice-generator-settled.txt` records the full Biology source and linked Note context. Earlier logout/Back attempts include a wrong locator timeout and rapid snapshots taken before auth/page settling; these are retained and are not evidence of a leak or a settled isolation pass. The subsequent Bob identities/empty generator and blank Chat/QA surfaces support the initial account change.

The added **settled repeat** removes the original Back timing ambiguity. `alice-own-restore-switch.txt` captures the actual full Biology generator source, its owned provenance and `ALICE SECOND DECK DRAFT ORBIT742`, then normal Logout/login navigation. `alice-to-bob-history-settled.txt` has **three actual `page.goBack()` calls**, waiting for login, then Settings/Logout, then the generator control. At **16:36:27.768Z**, Bob3 is independently observed; the prior Alice generator has empty source, default deck names, no Alice source context/marker. Bob's card catalogue is empty at this pre-creation point. This supports literal browser-history isolation for this exercised surface.

### Bob → Alice

`isolation-bob-note-save.txt` records Bob's unsent Chat marker and native Note content. `bob-generator-handoff.txt` captures Bob's Note provenance and the actual successful `.fill('BOB ONLY DECK DRAFT BIRCH913')` action. Its body text does not serialize textarea/input values; the intended source/deck state is supported by provenance plus the executed fill, not by a complete structured input snapshot. `bob-switch-out.txt` stops on a wrong-case sign-in heading **after** the logout operation; subsequent Alice2 identities corroborate the actual switch.

`bob-to-alice-history-settled.txt` records the same three literal Back actions with settled controls. At **16:34:01.467Z**, Alice2's prior Bob generator has empty source/default names/no Bob provenance; Alice's own original card reads200/client2/v4. `alice-return-surfaces.txt` shows Alice's own QA history, empty Chat composer without Bob marker and the actual foreign Bob Note404/Linked note unavailable. Conversely Bob's earlier foreign Alice Note404 has the same safe state, with only Bob's own Note in his list.

### Precise snapshot limit

The state labeled **Alice own Biology Note restored** in `alice-own-restore-switch.txt` was taken after waiting only for the textbox. It shows an empty editor/Version pending before content loading settles. Do **not** cite that particular snapshot as a loaded v3 editor. Its subsequent generator contains full Biology content/provenance; persisted restorationv3 is established by the API response and later owner Notes list. This is a capture-timing limitation, not evidence of content loss.

## Remaining limits and evidence hygiene

1. Row12 acceptance is limited to the exercised fixture/route/browser boundaries above, including the completed supplement. No later in-progress result is assumed. QA uses populated Alice history versus empty Bob history, not two populated QA histories. The final Bob populated selector lacks a settled snapshot. These limits must accompany the result; they are not proof of leakage. This audit neither invents extra acceptance requirements nor expands isolation to unexercised domains, workspace sharing, admin roles, PostgreSQL/RLS or raw SQL.
2. Restored Notes are v3, not their original v1. The audit's API actions were the parent's already-completed authorized probes; this reviewer made no requests. The helper does not log out its independently acquired sessions, and no session cleanup is inferred.
3. Original timeout/error receipts remain included. Expected foreign404/403, failed model400 and console references prevent any blanket clean-console claim. Browser text snapshots were reviewed; no new screenshot rendering or pixel inspection was performed.
4. This packet contains no five-card, vision, true-hidden visibility, upstream outage, Wikipedia article, complete row11 or full-matrix acceptance. Parent's earlier source241 and Study235/240/242 findings remain separate and unchanged.
5. The controller still has historical/pending row cells despite later narrative evidence. This audit uses the settled timestamps above; it does not edit the controller or mark tasks complete.

## Verification

The reviewer parsed the approved redacted wrapper results and raw API JSON, checked all32 original records plus all10 supplemental records, compared restored Note content/version transitions, exact source bytes and analysis/version identities, verified policy request differences and both Back sequences' final inputs/current identity, and rechecked input hashes immediately before writing. These are offline receipt consistency checks, not product tests. No private files were read, no raw provider reasoning copied, and no repair was attempted. The companion JSON is the complete reviewed input hash index.

The parent updated the controller during this audit. Its latest SQLite-multi narrative was reread; the JSON retains the previously inspected controller hash as history and binds the current reviewed bytes. Native inputs and helpers remained unchanged. The initial blocked hash recheck wrote no outputs.
