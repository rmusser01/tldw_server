# Cycle 4 fresh multi-user UAT — final report

**TASK13260 · completed 2026-09-16 01:36 UTC · no product fixes during this run**

## Provenance and limits

- Product runtime frozen at **7c9409fad2**, which includes corrected dev 2e1a5e58d3 via merge 267c00. The earlier baseline was missing 32 dev commits. The parent subsequently committed documentation and observed newer origin/dev; neither was represented as this runtime's tested code.
- New isolated multi profile, API 18301 and UI 18381, browser session `cycle4-multi-20260916`. Existing installed dependencies were reused: this is a fresh runtime/data/onboarding test, not a clean-machine dependency installation.
- Parent performed documented AuthNZ initialization and CLI admin bootstrap. Alice and Bob were created through the real admin UI with default User roles and verified status. No API-seeded content, mocks, RAG ACL edits, role changes, or product repairs.
- Multi-user WebUI gives operator setup guidance. Parent configured the isolated private INI for the real llama.cpp endpoint `http://127.0.0.1:9099/v1` and its discovered Gemma model, then restarted only the multi API. This is an explicit operator adaptation, not a WebUI provider-wizard pass. Blank RAG provider defaults were retained. Exact safe configuration is in `operator-provider-config.json`.
- Public Cedar, Bob's Copper, admin sole-item, and biology fixtures are synthetic. Indigo is a separate synthetic confidential classification control. Exact journey Wikipedia URL was exercised without a policy workaround.
- Findings stop dependent acceptance, while independent workflows continue. No issue received more than three inference/recovery attempts. Shared provider queueing can affect timeouts; an uncaught timeout overlay remains a UI failure regardless of the original latency cause.

## Twelve named workflow rows

| Workflow | Result | Actual coverage and limits |
| --- | --- | --- |
| Fresh setup / providers / first real Chat | PASS with operator adaptation | Fresh multi guidance and Sign in handoff; admin login and UI creation of Alice/Bob; Alice ordinary login; visible healthy discovered llama.cpp model; real first answer `CEDAR-41`. Provider configuration required the documented operator INI step above. |
| Authentication / reload / logout / offline / reconnect | PARTIAL | Admin and ordinary logins passed. Offline browser logout cleared local credentials/checks and other frontend tabs; reconnect sent those tabs to sign-in. Bob login and subsequent admin/Alice switches passed. Private verifier refresh passed. A deliberate browser access-token-expiry/refresh scenario was **not separately verified**; transient 401 then working prompt UI is not sufficient proof. |
| Saved ordinary two-turn Chat / reload / no duplicates | FAIL UAT-103 | Both real answers correct, one canonical server conversation and five canonical rows. Settled client reload showed duplicate user turns, seven timeline rows. No duplicate server-conversation claim. |
| Public file ingest → search → grounded QA / citations / Chat | FAIL / partial | Cedar file ingested, exact content persisted, full-text search passed. Selected-source Balanced QA timed out with UAT-109 overlay. Fast retrieval found Cedar; generation-enabled recovery still displayed no answer. No numbered-cited-answer pass. Excerpt and actual Open in Media passed. Actual Media→Chat populated exact source text; second character response persisted reasoning only, no final answer. |
| Exact Wikipedia URL → search → Chat | BLOCKED externally; failure presentation PASS | `https://en.wikipedia.org/wiki/Playwright_(software)` through Quick Ingest, analysis off and chunking on, returned explicit Access blocked, zero success / one failure. No article was available for subsequent search or grounded Chat. No denial-page content was accepted as a successful article. |
| Notes → five grounded flashcards → save / actual review / reload | PASS with timing caveat | Biology Note saved/reloaded, five real generated questions/answers matched all five supplied facts, saved into deck 1 with Note source links. All five distinct cards were actually revealed/rated; saved review state survived reload. Early rapid UI actions rerated stale first cards before fetch settled, so seven review events occurred for five distinct cards. This is retained transparently; no clean five-event count claim. Later synchronized mixed review had exact counts. |
| Saved prompt → apply in Chat → real request / answer | PASS | Pirate prompt saved/synced id 1; editor changed to Editing; Back caused no false-unsaved warning. Use in chat → System Instruction produced actual request with saved pirate system and real correct `Arrr` answer. |
| Character context replacement / tracked Chat / history | FAIL UAT-113; core switch PASS | New Cedar Guide id 4 correctly cleared pirate prompt/history, real complete-v2 and persistence returned 200, clean factual answer with separate reasoning disclosure. Reload of creation URL reset to greeting only despite Saved. Normal Server history reopening restored all three persisted rows. |
| Chat → Note / actual backlink; reviewed Chat card → Study | FAIL UAT-111; card loop PASS | Normal real answer saved as Note 201, exact clean content and correct Saved from Chat origin. Actual Open conversation twice stayed on Notes. Real character answer opened a reviewed flashcard draft with clean answer and required question, then saved. Mixed Study included three due deck cards plus one undecked Chat card, four exact reviews and correct completion/next-review. Cram counted all six cards and one actual review left five. Cram exposed no End Session action; no early-End acceptance claim. Actual Study→Note link resolved the correct biology note. |
| Media analysis → Review → reanalysis → reload | FAIL UAT-105; explicit reanalysis PASS | Initial successful ingestion saved a raw provider envelope/reasoning as Analysis. Original evidence retained. Explicit concise reanalysis produced the correct summary. Normal list selection in `/media-multi` displayed it in Review, and reload/reselection preserved it. |
| Permission-aware delete → Trash → restore, admin sole item | PASS | Alice/Bob Delete disabled with permission explanation. Admin ingested own sole synthetic item, soft-deleted with confirmation, retained reachable Trash despite zero active items, saw valid deletion date, restored and reopened original content. |
| Cross-account API / browser isolation / metadata / Back handoff | PASS for exercised boundaries | Bob Back to Alice's source Note gave unavailable/404 without old content/title. Bob QA had no Alice sessions/query/scope. Bob created own Note/Copper through UI. Reciprocal foreign Note reads/valid writes denied; foreign media write denied; reciprocal Media search empty; foreign Chat read denied; ingest-job metadata 403 while own job 200. Media integer IDs are per-user: GET 1 returns each user's own source, not a cross-account global identifier. |

## Findings with assigned IDs

### UAT-102 — P3 — Admin success toast logs context error
Admin `/admin/server` → Create user with valid fields/default User succeeds, but console.error logs the AntD static-message/dynamic-theme warning. No overlay; neither account creation was blocked. Initial fixture `example.test` email rejection was expected validation, corrected to `example.com`, not a product finding.

Evidence: `uat102-admin-create-toast-console.txt`, `admin-accounts-created.txt`.

### UAT-103 — P2 — Saved ordinary Chat duplicates user turns in client timeline
Two real `/chat/completions` requests returned 200 and correct CEDAR-41 answers. Reload initially showed four user/assistant rows; settled reload showed system + duplicate copies of both users + two assistants. Server conversation `a9feabc7-38aa-4b06-b08d-0b673b069819` retained five rows. Expected one copy of each saved turn. Client history acceptance fails.

Evidence: `normal-chat-first-request.json`, `normal-chat-second-request.json`, `normal-chat-two-turns.txt`, `normal-chat-settled-reload-duplicates.txt`, `normal-messages-settled.json`, `uat103-read-only-mirror.txt` (bounded IDs/lengths only).

### UAT-104 — P2 — Minimize leaves ingestion modal blocking navigation
During Cedar job 1, two normal Minimize to Background clicks left the dialog open. Notes click timed out because the modal intercepted pointer events. Expected the workspace to become usable while processing continues. No cancellation/force-click; independent Notes used a separate normal tab. Job eventually completed in 64 seconds.

Evidence: `ingest-minimize-remains-modal.txt`.

### UAT-105 — P2 — Successful ingestion stores raw provider envelope as Analysis
Cedar Standard analysis+chunking with explicit llama.cpp showed one success / zero failures. Analysis contained `choices`, `finish_reason:length`, empty content and `reasoning_content` rather than a usable answer or partial-failure warning. Model truncation may be external/model-dependent; presenting transport data as successful analysis is the UI/product issue. Expected final analysis or explicit failure. Reanalysis workaround is separately recorded, not a retroactive pass.

Evidence: `cedar-ingest-job-result.json`, `cedar-ingest-raw-envelope-analysis.txt`, `cedar-ingest-media.json`; recovery: `cedar-reanalysis-success.txt`, `cedar-review-reloaded.txt`.

### UAT-109 — timeout development overlay
Alice Knowledge QA → exact Cedar id 1 → Balanced / Server default → ask opening date and coordinator. Stream HTTP 200, then idle timeout. UI actionable timeout was overlaid by uncaught Runtime AbortError `BodyStreamBuffer was aborted`, pointing to `background-proxy.ts:1686 controller.abort()`. Expected handled timeout without an uncaught overlay. Selected-source answer/citation acceptance blocked. Shared inference latency is a possible trigger, not an asserted root cause.

Evidence: `cedar-qa-idle-abort-overlay.txt`, `cedar-qa-request.json`. Stream body was unavailable; the empty attempted capture is **not response evidence**.

### UAT-111 — Saved Note's Open conversation does not navigate
Save actual pirate answer via Chat → More actions → Save to Notes returned 201. Note `b89c7b7d-daa6-438a-9fa3-ef4bd6b8b9f9` showed exact answer, correct origin, canonical conversation and message IDs. Notes → More actions → Open conversation twice left `/notes`, no new tab, only owned Note GET 200. Expected the actual linked conversation to open. No API navigation workaround counted.

Evidence: `chat-note-save.json`, `chat-note-backlink-opened.txt`.

### UAT-113 — Character creation-route reload loses active saved conversation
Characters → Chat as new id 4 → real complete-v2/persist 200 created `bd564030-70fe-41bc-9c91-4b661408f8e3`. Reload `/chat?mode=character&characterId=4` reset to greeting only; persisted three rows remained. Expected current saved conversation to survive reload. Normal sidebar Recent conversations → Server → named chat restores it, a visible recovery workaround.

Evidence: `character-cedar-request.json`, `character-cedar-reloaded.txt`, `character-saved-messages.json`, `character-history-reopened.txt`.

## Unnumbered candidates / observations for parent triage

### QA generation-enabled recovery yields no answer
After UAT-109, suggested Fast preset found Cedar. Fast also reset specific-document selection and disabled generation/citations. Used visible Enable in Settings, where Generate Answer was checked, and restored Cedar id 1. Final request 1320 was HTTP 200 with `search_mode:fts`, `include_media_ids:[1]`, `enable_generation:true`, `max_generation_tokens:300`; `enable_citations:false` remained from Fast. Result still said “Found 1 relevant source. Enable answer generation in settings…” and had no generated answer. Three total attempts ended here. No claim that citation generation was enabled on this recovery. Exact final stream body/events unavailable, so do not assert cache, backend generation or model root cause. Prior timeout error text also remained alongside Search complete in the snapshot.

Evidence: `cedar-qa-final-request.json`, `cedar-qa-fast-recovery-no-answer.txt`.

### Media Chat completes with reasoning only and no final-answer warning
Media id 1 → Chat with this media correctly opens Chat and populates full Cedar text in the composer, retaining current Cedar character. Appended a question about volunteer day/time and sign color (facts absent from character instructions). Complete-v2 and completion persistence both 200. Settled assistant row contains only collapsed optional reasoning (49 seconds), feedback/actions, no final answer, no truncation or failure warning. Persisted assistant content is only `<think>…</think>`. Expected useful final answer or clear incomplete-generation state. Model's repetitive deliberation is an observed provider outcome; exact finish reason was not captured, so truncation is not asserted as proven. No further inference retries.

Evidence: `media-chat-request.json`, `media-chat-persist-request.json`, `media-chat-no-final-answer.txt`, visually checked `media-chat-no-final-answer.png`.

### Policy-denied admin background probes — expected denial, UX observation
Bootstrap CLI admin's `/users/storage` and `/users/me/profile?sections=quotas` returned 403 `Email verification required`. Endpoint requires active verified user. No role-authorization defect or bypass is claimed. Background requests log console errors, without overlay. UI-created Alice/Bob were verified. No account verification state was changed to make a test pass.

Evidence: `admin-storage.json`, `admin-quotas.json`.

### Secondary observations not independently isolated
- Settled `/prompts` and `/characters` showed blank document title in native browser output, while Chat/Notes/Media titles worked. Candidate may overlap prior title issue; no separate reproduction or new ID claimed.
- Rapid reveal/rate sequence allowed repeated stale cards before next-card fetch settled. Actual recorded sequence is retained in `biology-five-reviews.txt`; synchronized later reviews avoid this automation timing confound. Do not interpret seven events as seven unique cards.

## Security and evidence integrity

- Indigo owned synthetic confidential source ingested successfully, but normal Fast retrieval returned **Security settings excluded all retrieved sources**, zero sources, no leaked planning answer. No security settings, ACL, or role mapping changed. `indigo-policy-excluded.txt`.
- API isolation probes used ordinary separate verifier logins, never browser-secret extraction or privileged impersonation. PUT requests used valid schema and expected-version headers for synthetic foreign notes; 404/403 denials, not malformed-request 422s. `api-isolation.json`.
- Initial GET `/media/search` was a helper method mistake (405). Corrected POST and real browser search returned 200/zero matches. The 405 is not a product finding.
- No production/previous-cycle ports used. No server lifecycle actions by this runner. No permanent deletion. Admin soft-deleted synthetic source was restored.
- All retained JSON artifacts parsed successfully. Exact private password/access/refresh string scan found zero hits. Screenshot visually inspected and contains only synthetic test content. Final scan receipt: `EVIDENCE_CHECK.json`.
