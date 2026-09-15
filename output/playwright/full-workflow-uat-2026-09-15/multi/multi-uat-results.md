# Fresh full multi-user UAT, 2026-09-15
Revision 68863b90b7, backend18101/frontend18181, isolated fresh runtime with reused dependencies and real llama.cpp9099. TASK13260 parent owns tracker. No product edits.
## Setup/account phase
- PASS fresh home displays Multi-user setup guide, documented operator links and Sign in; sign-in navigates directly to settings/tldw with Multi User and Password defaults. No solo setup/CSRF trap.
- PASS documented AuthNZ initialize non-interactive and create_admin bootstrap against isolated runtime, then real UI admin password login.
- PASS logged-in connection shows Core reachable / RAG healthy, no billing errors.
- CLI bootstrap is an intended multi-user operator step, not a browser-only bootstrap pass. Docker/Postgres and clean dependency install not exercised.
- PASS admin UI creates Alice and Bob via Create user with default User role. Alice row verified. Admin logout then Alice password login works; Alice session survives navigation/reload.
- PASS ordinary-user Notes loads, auto-saves synthetic study note, survives reload. Connections remains collapsed; no privileged probe gate. First-use tour appears after initial data entry, dismissed via Skip tour; save had already auto-saved. Minor dev-only Next indicator intercepted bottom sidebar Settings; top shortcuts navigated successfully.
## Candidate findings (parent assigns IDs)
### M-01 P1: Notes→Flashcards rejects grounded source words
Steps: Alice Notes save biology paragraph; Flashcards > Import / Export > Generate, paste same paragraph, 3 basic cards, provider llamacpp and real catalog model, Generate. Actual POST /flashcards/generate422 claim_verification_failed after ~114s. Report marks generated backs Mitochondria and Photosynthesis hallucinations although both occur verbatim in evidence. Expected supported flashcards available for review/save. Save/review downstream blocked for original fixture. No security setting bypassed. Evidence output/playwright/full-workflow-uat-2026-09-15/multi/flashcards-claim-verification-422.json.
### M-02 P2: generation error renders raw JSON
Same failure shows full serialized nested verification report as error paragraph (thousands of chars), also a Next dev error overlay. Expected concise actionable explanation and inspectable diagnostics, not raw JSON. Dev overlay is development-only limitation; raw application error remains.
### M-03 P2: successful prompt save leaves editor dirty and trapped by new=1
Steps Prompts > Create prompt, title Alice Pirate fullmulti20260915, system pirate instruction, Save. Toast Prompt Added; underlying table gains prompt, but URL remains /prompts?new=1 and populated New Prompt stays. Back to Prompts prompts unsaved changes immediately after successful save. Accept clears fields and immediately opens blank New Prompt, still new=1. Expected saved editor closes/clean state. Workaround navigate /prompts directly to inspect saved prompt. Evidence: prompt-save-editor-stays-open.yml. The confirmation was observed live; its attempted standalone capture was empty and discarded during final review.
- UAT-022 fewer-cards recovery also failed: 1 card with focus DNA produced expanded answer adding facts absent from source; this second failure is not claimed false-positive. Original report exact-word rejection remains distinct. Both generated attempts blocked save/review; no manual replacement counted as generation success.
- PASS Prompts > Use in chat > Use as System Instruction sends captured exact system prompt; real200 assistant pirate response. Evidence prompt-applied-chat-request.json. Prompt library persistence verified after navigating directly /prompts.
### UAT-026 P1: persisted tracked Chat fails local provider
Exact workflow picker path with Helpful AI Assistant (existing default character), saved local+server persistence, real catalog model. POST /chats/{id}/complete-v2 with provider llama and save_to_db true400 {detail:Chat provider error}; Retry same model returns400 again. Normal /chat/completions with same model/provider had200 twice. Both shared loops requiring real saved assistant are blocked before Save to Notes/Flashcards, linked conversation and review. No more retries. Earlier character route with initial persistence generated via normal completions but no server artifact actions; toggling off/on persistence established intended complete-v2 route. No API seeding used.
### UAT-027 P1: requested ingest analysis failure hidden
Quick Ingest upload 524B public Cedar Markdown; Standard analysis+chunking enabled; choose llama analysis provider. Job200 completed, Success, warnings:null, UI Succeeded1/failed0 in2sec. Backend Summarization failed logged at same run; inspect visible analysis pending.
### UAT-028 P2: Open in Media shows stale empty first-source splash
Quick Ingest success > Open in Media -> /media?id=1 but original first-content splash and0/0 remains. Skip for now reveals No media found (does not open source). Reload tested next. File is persisted media id1/owner2. Expected deep link opens new source without manual recovery.
- UAT-027 confirmed: after reload Media Analysis displays saved string "Error: Model is required for provider 'llama.cpp'" as if it were analysis. Analyze dialog discovers actual model and enables explicit re-analysis; running recovery.
- UAT-028 recovery requires full reload; Skip for now alone only changes splash to No media found. Reload opens id1 and source content. Ingestion fixture deviation: local public Cedar Markdown replaces workflow Wikipedia URL, preserving ingest/search/source-grounded answer path; external URL ingestion was not claimed at this phase; superseded by the final Wikipedia URL acceptance follow-up below.
### UAT-030 P1: search/QA cannot retrieve ingested ordinary-user source
Alice UI Media full-text search Cedar sends POST /media/search {query:Cedar,fields:[title,content],sort_by:relevance};200 items[] total0 despite id1 owned source content visibly containing Project Cedar. Independent normal password-auth API corroboration returns same empty results. Specific QA selector does list and select id1. Explicit Llama.cpp/actual model, query date+leader ->200 stream contexts[] and no-results answer; no security-exclusion explanation/count emitted. Cited answer and citation inspection blocked for this source. No ACL mapping/bypass performed.
- PASS explicit Media Analyze recovery generates correct three-bullet analysis with actual discovered model. Need subsequent re-analysis and restore.
### UAT-031 P2: tracked retry leaves greeting save IDs inconsistent
After complete-v2 failed and Retry same model created a different conversation id, greeting More actions offers Save to Notes/Flashcards. Both POST /chat/knowledge/save400 Message is not in conversation. Payload conversation7ae58252... and message7adab26b..., snippet default greeting. Partial downstream control only; not a substitute for real-answer loop. No note/card created, linked conversation/review blocked.
## Isolation corroboration
PASS normal password-auth API controls: Alice own note/media200; Bob foreign note read404 and valid versioned update404; foreign media read404 and update404; Bob notes list empty200/media search empty200. Alice original note unchanged. Evidence /private/tmp/full-multi-isolation-results.json; UI Bob checks next.

## Final multi-user acceptance matrix
All bounded named workflows have been attempted; failures and downstream blocks are retained rather than counted as passes.

| Workflow step | Result | Evidence / limitation |
| --- | --- | --- |
| Fresh multi-user setup entry → sign-in | PASS | Multi-user guide and Password defaults; no solo setup trap |
| First administrator bootstrap | PASS operator CLI | Documented isolated initialize/create_admin; not browser-only bootstrap |
| Admin UI create Alice and Bob | PASS | Both created with default User; both password-log in successfully |
| Ordinary-user connection and reload | PASS | Alice/Bob Notes loads, private notes persist across reload |
| Ingest public synthetic source | PASS storage/chunking; FAIL analysis | 524B Cedar Markdown media1/owner2; UAT027 analysis error saved without warning |
| Search ingested source | FAIL | UAT030 Media full-text Cedar200 empty; API corroborates |
| Select specific source → cited QA | FAIL | Selector lists1 document; explicit real model stream returns0contexts, no-results answer; citation/excerpt/original source checks BLOCKED |
| Notes create → persist/reload | PASS | Alice study note; Bob garden note |
| Notes → generate flashcards | FAIL | UAT022 first3card422, second1card422; raw JSON UAT024 |
| Generated card preview/save/review | BLOCKED | No accepted drafts; no manual cards substituted |
| Prompts create/persist | PASS data; FAIL navigation | UAT023 save keeps ?new=1 dirty editor; direct /prompts recovery |
| Prompts apply in Chat → actual response | PASS with navigation recovery | Exact pirate system verified on request, real200 pirate answer |
| Shared Chat → save Note | BLOCKED original; FAIL partial greeting | UAT035 persistence toggle required; UAT026 complete-v2 real turn400; UAT031 greeting save400 after retry |
| Saved Note → linked conversation | BLOCKED | No derived note created |
| Shared Chat → save Flashcards | BLOCKED original; FAIL partial greeting | Same400 knowledge/save with make_flashcard:true |
| Saved shared Chat card → review | BLOCKED | No derived card created |
| Media explicit Analyze → review | PASS recovery | Model discovered, correct3bullets persisted after reload |
| Reviewed Media → re-analyze | PASS | Second real analysis updates to correct concise date/leader sentence |
| Media delete | BLOCKED permission; FAIL capability UX | UAT033 default User lacks media.delete403; permission preserved |
| Media restore | BLOCKED | Deletion denied, Trash empty; no API bypass |
| Logout → Bob UI isolation | PARTIAL | Lists/content isolated; UAT034 Alice recent-note title/UUID leaks locally after logout/login/reload |
| Bob direct browser access to Alice note/media | PASS denial | Clicking leaked Recent note404; /media?id=1 GET404, empty UI |
| Reciprocal Notes API read/write denial | PASS | Alice↔Bob foreign UUID reads404 and valid expected-version updates404; own originals200 unchanged |
| Bob foreign Media API read/write denial | PASS | GET/PUT media1 both404; Alice own media200 |
| RAG confidential-content exclusion policy | BLOCKED fresh-run verification | Public source retrieval already returns0; no classifier/ACL changes or promotions; cannot attribute current no-results to policy |

### UAT-033 P2: unavailable Delete action and misleading confirmation
Alice own source > Delete item > dialog “Delete this item? This cannot be undone.” > Delete. DELETE /media/1 returns403 Permission denied: missing media.delete. UI exposes action without a role/capability explanation; Trash later explains recovery is possible. Preserve existing permissions. Restore remains blocked because nothing was deleted.

### UAT-034 P1: Recent notes exposes prior account title/UUID
Alice opens/saves her study note, logs out using tldw settings, Bob logs in normally, opens Notes. Main list0/empty correctly; Recent notes contains Alice Cell Study fullmulti20260915 and DOM data-testid reveals UUID911dde71-c385-477f-98cf-2b97f57072aa. Click sends GET foreign note404 and shows Failed to load note; content remains empty. Bob reload preserves leaked Recent entry, even after creating his own garden note. Expected all note metadata scoped to account/server and cleared on logout. Screenshot bob-recent-note-title-leak.png visually inspected. Prior Chat tab correctly navigates/login on logout but retains old document title; noted as minor related stale metadata, no claim of content access.

### UAT-035 P2: initial Saved label does not establish server persistence
Normal Chat and Chat as Helpful AI Assistant show bottom Saved and restored character greeting, but request save_to_db:false and no server knowledge-save actions. Same-character composer selection alone does not recover. Exact workaround: New Chat; click top Temp action (clears character/messages); bottom Ephemeral; click Ephemeral once to Saved (tooltip Locally+Server); move mouse away from tooltip; composer Select character/persona > Characters > Helpful AI Assistant; send. This now posts complete-v2 with save_to_db:true. It subsequently fails separately as UAT026. Persisted status should match actual routing without off/on reset.

## Final observations and limits
- Existing Helpful AI Assistant default character used as bounded shared-workflow prerequisite instead of creating unrelated character fixtures. Real inference on9099, no mocks. No optional Evals/Watchlists/Character feature sweeps.
- No literal A/B/C mapping claimed. Initial source control used public synthetic local Markdown; the later Wikipedia URL acceptance follow-up below supersedes the initial URL coverage gap and records its failure. No clean-machine dependencies, Docker/Postgres, STT/TTS, or browser extension sign-off.
- Initial UAT001–019 repairs are not blanket re-certified: multi setup/auth/Notes/provider-selection behaviors exercised; source citations could not be reached due UAT030. Backend API isolation alone is not browser privacy acceptance (UAT034).
- No product code/tests/config/role/ACL changes; no fixes/commits. Only private harnesses, evidence and this report written. Parent owns Backlog and authoritative tracker.
- Runtime remains isolated backend18101/frontend18181, browser full-multi-20260915: tab0 Bob /media?id=1 empty/404, tab1/login after Alice logout. Credentials remain only in private runtime manifest.
- Final ordinary-user own-media lookup remained200 after rejected deletion and denied cross-user writes. Both note original titles were re-read unchanged. Bob’s saved note UUIDb8774e6c-1366-4cf9-a7eb-b92024904cb2.

## Wikipedia URL acceptance follow-up (2026-09-15 04:48–04:52 UTC)
This section supersedes the earlier limitation that URL downloading was not tested. Same frozen revision/runtime; Alice signed in through normal UI after Bob logout. Existing Cedar source was not changed.

### UAT-044 P2: Wikipedia robot-policy response saved as successful article
Exact named journey URL: https://en.wikipedia.org/wiki/Playwright_(software). Quick Ingest > Start a new ingest > paste URL > Add URLs > Configure. Matched `ingestAndWaitForReady` and `applyQuickIngestProcessOptions` in e2e/utils/journey-helpers.ts: analysis and chunking both OFF; Review displayed Custom · Scrape and Storage: Server. Start Processing.

Actual UI: Succeeded (1), Total 1 succeeded / 0 failed, Open in Media offered. POST /api/v1/media/process-web-scraping returned200. Request: `{scrape_method:"Individual URLs",url_input:"https://en.wikipedia.org/wiki/Playwright_(software)",mode:"persist",summarize_checkbox:false,perform_chunking:false}`. Response: `{status:"persist-ok",media_ids:[2],total_articles:1,stored_articles:1,skipped_articles:0,duplicate_articles:0,method:"Individual URLs",errors:null,task_id:"scrape_b4b9b0aa"}`.

Open in Media successfully opened /media?id=2 (no fresh-empty-splash issue on this existing library). GET /media/2 returned200. Title N/A, body only ingestion metadata plus Wikimedia robot-policy message asking crawlers to respect its policy; no Playwright article content. No analysis expected because disabled. UI additionally displayed Chunking: Completed despite request perform_chunking:false; retained as observation, not independently diagnosed.

Expected: explain the unsuccessful article fetch and avoid presenting a denial page as successful article ingestion. The external site denial itself is an environment/site-policy limit; the success/error classification and saved denial text are the product finding. No attempts to bypass site policy or fetch through alternate identities.

### Search and downstream accounting
- FAIL Media search Playwright: POST /media/search?page=1&results_per_page=20&include_keywords=true, body `{query:"Playwright",fields:["title","content"],sort_by:"relevance"}`, returned200 items[]/results[] and pagination total0. UI No media found. This adds a fixture result alongside UAT030, not proof of a separate root cause.
- FAIL named SearchPage route /search (redirect /knowledge): entered exact query Playwright, selected Llama.cpp in UI (auto-discovered real Qwen model). Default selected library sources: Documents & Media, Notes, Characters, Chats; Web fallback disabled. POST /rag/search/stream200, contexts empty, complete upstream_dispatched:true. UI Search complete. 0 sources found; Answer status No results, 0 cited. Real response says context does not contain relevant Playwright information and cannot answer with sourced citations.
- BLOCKED contextual Chat/citation validation: real article never ingested and no context retrieved. Did not repeat known tracked-chat provider400 or count an ungrounded answer as a pass.
- No URL retry needed: deterministic saved denial evidence, no policy workaround. Existing synthetic-source test remains valid as its separate fixture; the exact Wikipedia URL is now attempted but fails article-content acceptance.

Evidence in this directory: wikipedia-robot-policy-saved.png (visually inspected), wikipedia-url-success.yml, wikipedia-url-content.yml, wikipedia-media-search-empty.yml, wikipedia-knowledge-no-results.yml, wikipedia-knowledge-stream.xndjson, wikipedia-url-http-summary.json. All credential-free. Browser final state tab0 Alice /knowledge with Playwright no-results answer, tab1 /login. No product/test/config/permission edits or commits.


## Dedicated Multi-Item Review follow-up (2026-09-15 05:01–05:02 UTC)
PASS: Alice opened /media-multi (Multi-Item Review) in the same frozen runtime. The normal unfiltered list displayed both existing items. Clicked the visible alice-cedar-fullmulti20260915 row, then its checkbox to select it; the viewer entered Focus(1/1), 1 selected. Its Media Content matched the public Cedar fixture and Analysis displayed the saved second result exactly: “Mira Chen leads Project Cedar, which launches on 22 November 2026.” GET /api/v1/media?page=1&results_per_page=20 returned200; GET /api/v1/media/1?include_content=true&include_versions=false returned200. This closes the dedicated Review workspace gap beyond the previously verified Inspector review. Used ordinary visible-list selection; no new full-text search attempt or search success is claimed (UAT030 remains open). No inference, reprocessing, content changes, or API seeding.

Evidence: cedar-multi-review-saved-analysis.png (visually inspected), cedar-multi-review-saved-analysis.yml, cedar-multi-review-source-response.json. Final browser tab0 Alice /media-multi with Cedar selected, tab1 /login. UAT027 severity harmonized to parent-assigned P1; prompt navigation evidence references now explicit.
