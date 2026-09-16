# Cycle5 multi-user native execution checklist — PREPARED, NOT STARTED

Parent TASK13260. No cycle5 browser, runtime, account, file ingestion, generation or API action has started. Root must release frozen source commit, fresh runtime readiness, private safe wrapper and admin bootstrap details. No product/task/shared-tracker edits. Root owns API18501/UI18581 and process lifecycle. Use only /private/tmp/cycle5-safe-browser.mjs multi after release; never inspect or print runtime-private credentials. Existing installed dependencies are reused and will be disclosed.

## Authoritative definitions read

- apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts:19 exact Wikipedia URL; named Ingest→Search→Chat.
- apps/tldw-frontend/e2e/workflows/journeys/notes-flashcards.spec.ts:18-25 exact five facts; named Notes→Flashcards.
- apps/tldw-frontend/e2e/workflows/journeys/prompts-chat.spec.ts:14 exact pirate instruction and weather question.
- apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts:34-35 TestBot/BEEP BOOP instruction and actual complete-v2 expectations.
- apps/test-utils/real-server-workflows.ts:114-132 named shared workflows; actual Note backlink3512, Chat→card3970, Trash4274, analysis→Review→reanalysis4425.
- apps/tldw-frontend/e2e/real-server-workflows.spec.ts registers shared workflows. Its onboarding/auth/localStorage seeding is test infrastructure, NOT native onboarding; do not execute these seeds in this run.
- Prior cycle4 multi FINAL_REPORT and /private/tmp/cycle5-native-workflow-protocol.md read. No literal A/B/C labels were established; use concrete named rows below.

## First actions after root release

1. Record exact frozen source and root-owned API/UI PIDs/ports, source freeze timestamp, first browser navigation time. Maintain own private running tracker with Pending/Pass/Fail/Blocked/Partial and artifact links as each result occurs.
2. Operator multi configuration/CLI admin bootstrap is an explicit adaptation. Verify fresh multi welcome/login and provider discovery, no false wizard pass. Create Alice/Bob through real admin UI with default User role and valid synthetic example.com emails. Record returned IDs, verification UI choice, contextual feedback and console. Do not change admin verification or roles to hide policy-denied endpoints.
3. Log in as Alice early. Install response observers before login that retain only endpoint/status/timestamp and numeric expires_in from auth login/refresh. Never retain entire auth response, request credential body, cookies or headers. Derive expected expiry from last observed successful issuance + expires_in, not decoded token contents.
4. Natural expiry proof uses a separate fresh named browser context (proposed cycle5-multi-expiry-20260916), isolated from the main multi context and its account switches. Root approved this method; the private safe wrapper must target this exact isolated session for credential fills. Source TldwAuth.ts493 schedules automatic refresh five minutes before expiry; its initTokenRefresh353 refreshes on page load. Active browsing beyond30min may only prove proactive refresh. Preferred natural away-and-return: after last auth issuance capture, in that expiry-only context navigate the sole Alice application tab normally to about:blank, preserving its context/storage. Ensure no other app tab in this expiry context remains running. Main multi workflow and account switches continue in their different context. Let actual wall time pass beyond last issuance + expires_in +10s while root single work or offline evidence preparation proceeds. No clock/token/storage/timer mutation and no browser-global close. Return normally to UI18581, capture real /auth/refresh200 and subsequent authenticated canonical GET200 with Alice content. Report both last issuance and elapsed time; if another refresh occurred before expiry, recalculate and do not claim expired-token recovery. Root informed of this plan before start.
5. Main multi context may perform logout/offline/reconnect and reciprocal Bob switches while the isolated expiry context waits. Never log out, export/import credentials, mutate storage or open extra app tabs in the expiry context before its return proof. All9099 work uses explicit request→root grant→one bounded generation/job→terminal→release; no concurrency with root single mode.

## Twelve execution rows

###1 — Fresh multi setup/provider discovery/first Chat
- Fresh welcome, multi operator instructions, Sign in handoff, actual admin create-user control and Alice login.
- Exact real configured llama.cpp model discovered/healthy. Document operator configuration and any deliberate API restart by root, never hide adaptation.
- Start ordinary saved Chat through UI; no preseeded messages. Unique cycle5 marker and first real factual/synthetic reply. Capture provider/model/body/status/final text and canonical conversation/user/assistant IDs.
- Related surfaces: Test Connection uses authorized session identity; admin contextual toast; ordinary Notes loads without health/admin/graph privilege probes; titles.

###2 — Authentication/reload/logout/offline/reconnect
- Execute natural expiry plan above (required; proactive refresh alone is Partial). Record numeric expires_in (expected1800; confirm actual) and true elapsed time.
- Ordinary reload keeps owned state; real refresh200 followed by authenticated200. No auth-response body artifacts.
- In the separate main context, actual offline logout clears local auth and owned app tabs; reconnect requires normal login. Preserve unsaved draft/account ownership observations; do not force unsupported private hooks.
- Hidden-tab notification semantics remain known native harness limit; do not patch focus emulation or fake visibility. Named accessible actions inspectable independently.

###3 — Ordinary two-turn canonical Chat + real failure/Retry
- Two actual ordinary turns, distinct synthetic marker and recall. Exactly canonical system +2 user/assistant pairs,5 rows. Settled normal reload one copy each locally and server-side.
- Separate new ordinary conversation: select existing unavailable provider to get real failure. Capture initially persisted one user/correlation before Retry with actual normal history/reload; restore exact Gemma under lease; actual Retry reuses same user ID once, excludes display-error text in model context. Canonical user+assistant stable after reload.
- Unsupported-image negative boundary may be carried as targeted limitation/control; vision success unavailable on current9099. Never claim image generation/recovery pass from text-only or mocked content.

###4 — Public synthetic file→ingest/chunks→search→cited QA→source→Chat
- New short public cycle5 source with unique facts/token (no private/confidential classification). Save exact fixture content and filename; upload via UI, choose explicit supported model/settings if analysis on; acquire lease for entire model job.
- Exercise Minimize/background navigation and resume; terminal saved Warning must show own source/Open Media, never clean analysis success for reasoning/error envelope. Preserve any failure with job ID/result/source ID.
- Full-text search source, QA selected source with generation AND citations enabled, actual provider/model/body, useful final grounded answer and clickable numbered citations/source excerpt. HTTP200 without finalanswer is not Pass.
- Actual source inspection and Media→Chat handoff carries correctsource/text; ask a source-specific fact. One real answer under lease; record finalanswer separately from optional reasoning. Context/account scope must match.

###5 — Exact Wikipedia journey
- Use exactly https://en.wikipedia.org/wiki/Playwright_(software), no alternate URL or retrieval-denial workaround. Use journey-compatible analysis/chunking choices and record them.
- Verify actual article content, not denial page; if denied, mark externalblocked, capture accurate failure and block article search/groundedChat dependent steps. Continue independent rows.
- If article ingests, actual search Playwright, actual source-grounded Chat question: What is Playwright? Use the ingested content to answer.

###6 — Biology Note→exactly5 generated cards→save/reveal/rate/reload
- Create actual Note with these exact five facts, blank-line separated:
  1. The mitochondria is the powerhouse of the cell.
  2. DNA stands for deoxyribonucleic acid.
  3. Photosynthesis converts light energy into chemical energy.
  4. The human body has 206 bones.
  5. Water boils at 100 degrees Celsius at sea level.
- Generate exactly5 through actual Note transfer under lease. Inspect question/answer grounding, count and save all5; retain unique card IDs/deck/Note source ID. Do not accept partial generated count as full journey.
- Review each distinct card exactly once: record displayed card ID/question→Reveal→rating, await actual review response and next-card request/settled NEW ID before next click. Five different IDs and ideally exactly5 review events; if duplicate events occur, record honestly and do not retroactively clean.
- Actual completion screen, saved scheduling state, session reviewed count5 and normal reload. No rapid blind repeated ratings.

###7 — Saved pirate Prompt→actual apply→real output
- Exact instruction: You are a pirate. Respond to everything in pirate speak. Always say ARRR at least once.
- Create/save/sync, correct title, opaque editor cold/warm and Back without false-unsaved warning. Actual Use in Chat→System Instruction; ordinary conversation must remain ordinary.
- Ask Tell me about the weather today. Capture actual system instruction in request, provider/model, finalanswer; judge pirate behavior and literal ARRR casing honestly (Arrr is not literal uppercase acceptance).
- Normal reload retains answer. Save failure/Pending/retry coverage may require root API lifecycle coordination; no self restart.

###8 — Character create/replace→real tracked reply→canonical reload
- Cold New Character drawer open/cancel/reopen without unconnected-form warning; accessible controls.
- Create distinct E2E-TestBot-like card with exact instruction You are E2E-TestBot. Always respond with exactly: BEEP BOOP. Use unique cycle5 name, record ID.
- From prior pirate/ordinary context, actual library Chat as card clears old history/system; real complete-v2 request includes correctcontext/model and real final BEEP BOOP answer, not merely reasoning. Persisted route/history and canonical rows survive normalreload.
- Character→ordinary via real history/Note shows Standardchat/noChoosecharacter blocker; freshCharacter entry stillCharacter. No barechatIdURL test.
- New actual reasoning-only control can use visible supported max_tokens16/temp0/topP0.9/repeat1 under separate lease, retain truthful recovery+reasoning+canonical IDs, restore blank settings. If actualanswer exists classifypositive-only; at most3 attempts, never seed content.

###9 — Chat→Note/backlink; Chat→reviewed card→Study
- Save an actual completed answer to Note. Correctsource conversation/message IDs and clean finaltext (no transport/error/reasoning contamination). Actual More/Openconversation returns right canonicalconversation, even retained historicalerror if exercised; correctordinarymode.
- Save an actual Chat answer to reviewed flashcard draft, edit requiredquestion, verifycleananswer, save card and actualStudy.
- Track mixed deck+undecked totals exactly. Reveal/rate with settlednext-ID gate. Exercise due/completion/reload and Study→Note source. Early End/practice/Undo only when existing UI exposes workflow; absence is reported, not inventedPass. Track actual review-event deltas.

###10 — Media analysis→Multi-Item Review→explicit reanalysis→reload
- Use actual row4 synthetic source; selectedmodel visibly/request-confirmed. Review formattedanalysis matches raw saved source; no providerenvelope masquerading as finalanalysis.
- Actual /media-multi listselection→Review, explicit concise reanalysis underlease, waitterminal, save/reload/reselect sameitem. Failed reanalysis keeps previousanalysis and honest error; no overwriting with transportdata.

###11 — Permission-aware Delete/Trash/restore/admin sole source
- Alice/Bob denied Delete remain disabled/explained or403 withguidance; never grant rights for UI.
- Admin own sole synthetic media via realUI. Softdeletewithconfirmation, reachableTrash atzeroactiveitems, validdate, restoreandreopen exactoriginalcontent/ID. No permanentdelete. Keep admin switches within the main context, isolated from row2 expiry context.

###12 — Reciprocal ownership/isolation API and browser
- Alice andBob eachcreateownNote andpublicfile/media throughUI; preserve IDs anddistinctcontents. Browserlogout/login/reload/Back shouldnotshow previousaccounttitle, Notecontent, QAquery/scope/history, Chatmessages, drafts or sourcehandoff.
- Separateordinaryverifierlogins may testvalid schemas without browsercredentialextraction. Read/write ownNote200 thenforeignGET/validPUT denied404/403; expectedversion honored, owncontent unchanged. Malformed422 isnot authorizationproof.
- Reciprocal mediasearch mustexcludeother'suniquetoken; foreignwrite denied. NumericmediaIDsperuser mayeachreturn ownitem—comparecontent/UUID, donotcallthisleak.
- ForeignChat/read/messageIDs denied, foreigningestjob metadata403 vsown200; QA historyowned. PreserveconfidentialsyntheticIndigo exclusioncontrol; noACL/classification-policy relaxation and no security bypass.

## Evidence/stop discipline

For each row: timestamp, exact input/settings, actual request/HTTPoutcome withoutheaders, canonicalIDs/counts, finalrenderedtext, reload, screenshot when useful, consoleerror attribution, datamutationinventory. Save ownrunningtracker asoutcomesoccur. Final private prefix/outputdirectory assigned byroot afterrelease; no sharedtracker edits.

No more than3failed attempts perissue. Stopdependent steps onfailure, reportroot immediately, continue independentrows. Preserve frozen source/evidence before any laterrepair. No liveproductfix. Existing actualprovider capacity/vision=false, Wikipedia robots access and nativevisibility harnesslimitations mustremainexplicit. Runcredentialscan againstknownprivatevalues withoutprintingthem; retain receipt+SHA256manifest. No missingartifact/body inferredasemptyresponse. Do not waitresponse.finished after navigation; use realstatus+canonicalread and statecapturelimit.

## Readiness uncertainties to resolve from actual released environment

- Confirm login expires_in;30min is current expectation, not yetmeasuredcycle5. Proactive25min timer and reloadrefresh require honestdistinction.
- Root exactsourcefreeze/profile readiness/operator bootstrap pending; no cycle5credentialfile opened.
- ExistingGemma vision unavailable; realQA/analysis/reasoning outcomes andWikipediaaccess remain external/model-dependent.
- APIschema probes will use repository/currentschemas and actualcreatedIDs; prior-cycleIDs are referencesonly, never substitute freshcycle5records.
