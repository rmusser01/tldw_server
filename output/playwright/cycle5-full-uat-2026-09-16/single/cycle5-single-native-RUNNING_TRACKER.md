# Cycle5 single-user native UAT running tracker

Started2026-09-16T07:59Z. Frozen product ab527eb3b4; evidence-only commit a325e01010. Fresh origin/dev59049e094e contained,0missing at07:55Z. API18500 PID16832, UI18580 PID16872. Root /private/tmp/tldw-onboarding-uat-cycle5-single-20260916; fresh config/data/browser, existing dependencies. No provider/test overrides; RAG defaults empty; private0600files. All actions use real UI/API and existing real llama.cpp9099, inference serialized.

| Row | Workflow | Status |
|---|---|---|
|1|Fresh setup/provider/first real ordinary Chat without restart|PASS: visible local wizard provider validate/save200, first-chat150200 Hello!, complete153200, explicitAPIkey handoff, ordinaryChat289200 without restartingAPI16832|
|2|Auth/reload/disconnect/offline/reconnect|Pending|
|3|Two-turn Chat/canonical reload/failure Retry identity|PartialPASS:289/327200, correct WILLOW-85 context, normalreload canonical543200 total5 (system+two pairs), no duplicate user; providerfailureRetrypending|
|4|Synthetic file/chunk/search/citedQA/sourceChat|Pending|
|5|Exact Wikipedia URL/search/Chat|Pending|
|6|BiologyNote/exactly5generatedcards/distinctStudy/reload|Pending|
|7|SavedpiratePrompt/actualapply/realChat|Pending|
|8|Character/contextreplacement/complete-v2/canonicalreload|Pending|
|9|Chatanswer→Note/backlink;Chatcard→Study/mixedcounts|Pending|
|10|Mediaanalysis/Review/reanalysis/save/reload/failurepreserve|Pending|
|11|PermissionawareDelete/Trash/soleitemrestore|Pending|
|12|Account isolation|Single identity only; reciprocal multi checks owned by multi run|

## Findings and limits

No new finding recorded yet. Startup console0errors/1warning requires classification. Prior limits: unavailable vision provider, hidden-tab native tool limitation, exactWikipedia may be denied; do not claim clean-machine install or optional subsystems.

08:07UTC: Save toNotes succeeded. First More actions click triggered hover-replaced DOM control and CLI pointer interception; fresh snapshot exposed normal actionbutton and nextclick succeeded. Record as harness interaction, not submitted action failure. Pre-key setup warnings2 expected, noerrors.

08:14UTC independent single executor resumed. Chat answer→reviewed flashcard saved through visible form/knowledge-save201: flashcard d21a4ee9-ecc8-4d19-9edd-9df593948db4; source conversation5b22027f-193b-4166-9662-bd62ae0032ac/messagee9bb18a6-574e-43ef-af74-b6573adbdf68; synthetic question `What exact readiness message confirms the launch code?`, answer `WILLOW-85 READY.` Associated Note fedc0a60-e31a-458b-ad8e-6a31328d00b7 is expected knowledge-save output. Study remains pending.

Biology Note created through UI: title Cycle5 Single Biology; exact journey five facts,227chars, version1. Normal reload plus actual row selection settles to exact title/body and saved status (041). Immediate prior037 read caught empty pre-hydration fields and is not classified as a product loss; settled040/041 confirm source. Actual Note menu→Generate flashcards transfers exact body privately: URL only /flashcards?tab=importExport (044). Explicit count5, newdeck Cycle5 Single Biology Cards, blank optional provider/model (real server defaults). Generation dispatched with root9099lease; result pending.

Biology generation observer harness error: run-code's response predicate attempted global URL, unavailable in that sandbox. Click had already dispatched actual POST1053 exactly once; loading UI046 and request047 prove in flight. Empty045 capture records no model response. No resubmission. Subsequent terminal response will be captured with CLI response-body.

08:19UTC Biology Study completed: All-decks6 count captured then selected Biology deck1 before any rating. Five distinct real Good reviews069–073 all200, same review_session_id1, version2/repetitions1/due+10min. Exact question/answers independently inspected against five Note facts (049). Terminal074 shows5cards reviewed, Completed Due review. Root independent cards/session/deck reads corroborate. No manual End claim for automatic completion.

Normal reload075 returned Import/Export because tab clicks had left URL /flashcards?tab=importExport;20s Study-list assertion timed out. Evidence076–078 preserves observation, not data loss. Visible Study reselect079/080 restores completed5session and All-decks1remaining Chat card. Root classifying route/tab behavior separately. Saved-card completion053 assertion had strict duplicate-text selector;054/055 actual visible save status confirmed5, no save resubmission.
