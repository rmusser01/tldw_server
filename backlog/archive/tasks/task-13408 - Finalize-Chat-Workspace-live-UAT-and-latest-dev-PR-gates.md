---
id: TASK-13408
title: Finalize Chat Workspace live UAT and latest-dev PR gates
status: In Progress
assignee: []
created_date: '2026-10-01 18:39'
updated_date: '2026-10-02 02:13'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/3071'
documentation:
  - Docs/superpowers/reviews/chat-workspace/2026-10-01-latest-dev-no-mock-uat.md
  - IMPLEMENTATION_PLAN_pr3071_latest_dev_ci_2026_10_01.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue reviewed TASK-13398 Chat Workspace fixes and PR3071. Latest upstream ec86ba4 introduces a different task with ID13398; this unique finalization record preserves both histories and owns remaining integration/UAT gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Complete live Browse/staging/external-link and mobile recovery/composer keyboard UAT plus fresh owner-target visual proof without mocks.
- [x] #2 Integrate latest dev FastAPI0.142.1 and verify backend contracts and original data/tab/stash preservation.
- [ ] #3 Verify production Turbopack token-sync and unchanged bundle budgets, scoped tests/security; push reviewed fixes to PR3071 without merging or inventing the human Change summary.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage1: investigate exact CI causes and latest-dev semantic overlaps; preserve originals/runtimeenvs and resolve only owned merge overlaps. Stage2: reproduce each remaining failure, minimal root repairs with red/green regression, scoped upstream+owned tests and security/review. Stage3: restore actual services, rawChromeCDP no-mock UAT/preservation, update evidence and push; verify remote checks and requester summary without merging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Final no-mock native acceptance is complete: real Chrome raw CDP with actual authentication, SQLite, IndexedDB, Gemma and embeddings; no interception, fabricated app state, plugin or focus emulation. Owner account/backend A/B/A passes three real turns and six canonical rows with no transition/reload autosends. Browse/staging/Clear preserve a typed draft, both real previews return 200, external site loads, and mobile composer/Send fit 390x844 after the bounded transcript recovery-notice fix. Fresh owner and mobile screenshots were inspected. Latest dev ec86ba4e871d844ff4a3a53c607e80eb85bd0e9b merged in 9ebc3bdfb141b8f044e9da927c39a5eb020ab3a6; actual API gracefully restarted on FastAPI0.142.1 with its preserved environment/data and coherent existing OTel SDK/exporter1.45.0. Fresh real grounded Gemma send verifies canonical receipts, citations, saved draft and fresh-document restore with no resend. Original10/eight-row hashes, original tabs, all68stashes and served capability predicates/12negative controls preserved. Direct owning tests:82shared files2228assertions plus3frontend files50assertions; latest upstream backend34pass0skip; fullTSC0; scoped Bandit0new findings; cached precommit0. Independent final9path review48pass/no actionable findings/immutable hashes. Matched isolated final Turbopack539.9KBshared/842.3KBheaviest, unchanged600/900limits andtoken-sync0; direct checkout preexisting node_modules external symlink cannot be followed by Turbopack, no config/budget bypass. GitHub old fd32aad jobs110516627382/110516627352 logs confirm failure at654.1/958.0 and651.1/955.0 bundle gates; browser steps never reached, laterUXhealth failure follows stoppedbuild. Final publication/remote checks next. Known broader neighboring failures and unrelated inherited pip metadata/ML/typer conflicts are recorded, not disabled or claimed green. Archive completed historical13398 family via official CLI to preserve both histories while removing upstream IDcollision; leave new13408 unique active tracker. Acceptance doc owns evidence links; runtime evidence remains private outside PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Official completed-history archive finished: 32 exact-content records moved via backlog task archive; upstream Pin-FastAPI record SHA unchanged. Catalog/result retained externally. Removed only the completed owner-checkpoint plan, preserving unrelated plans. Final acceptance doc and PR body include verified old remote bundle failures and new-head CI limits.

Publication verified using gh pr view: PR3071 OPEN DRAFT, base dev, branch codex/chat-workspace-a11y, head b9b6571e8fdd30005d2d62a410f2f528cd6219e5; final PR body successfully updated and artifact attachment confirmed. Latest remote dev still ec86ba4. Final task-status commit follows; no production bytes change.

Requester supplied Change summary on2026-10-01; add verbatim to PR3071. Completed head792b6CI has281passing/36skipped checks; Onboarding andUXSmoke bothPASS. Three failing backend shards: selected-durable transformative moderation fixture returns mandatory_audit_unavailable503 instead409; SourceV1 malformed locator metadata9 unexpectedlyvalid; Streaming expectedno-cache but nowcorrectno-cache,no-transform. Actual failed joblogs saved externally. Latestdev5f3ed81e88ec44750a5baa7838f7c69672e32694 includes1060changedpaths; immutable mergepreview shows7conflicts in useChatActions/useHistorySelection/service-prompt-scope-error+test/character_messages/chat_service/raw-contenttest. Reopening finalization for root-cause repairs and conservative latest-dev reconciliation, followed by scoped tests/Bandit/review and actual no-mock runtime UAT. Do notmergeGitHubPR.

Human Change summary was published verbatim in PR3071 and readback will verify exact prose. No merge authorization inferred. New plan records3stages. Schema and moderation fixture root diagnoses delegated read-only with no source/tests/services writes while parent reconciles7mergeconflicts. Parent owns runtime preservation and all merge operations.

Latest-dev merge follow-up: source total_chunks bounds reproduced under cached Pydantic 2.13.5 and fixed without pinning; moderation audit override leakage reproduced and scoped fixture cleanup green in both orders; stale SSE assertions updated to require no-cache and no-transform. Fresh live Chrome UAT exposed valid RAG source bookkeeping rejected before inference (zero completion dispatches), now under diagnosis. Independent merge review found protected image read 503 mapping regression under repair. Restored all five exact live services. Local PostgreSQL gate failed on unavailable/unhealthy Docker daemon; retain failed evidence, do not count as a pass. Original stored conversation rows remain exact, but historical Chrome target IDs are absent while same-URL tabs remain; no tabs were created or closed by this follow-up.

Published source integration a4cf2ad and tracking-only latest-dev d81 merge5f52dde to draft PR3071. Human Change summary verified verbatim. Fresh Chrome grounded RAG receipts/citations/draft/reload and six route/mobile checks pass; zero route/reload autosends. Final3APIs restarted gracefully with preserved env/data. Protected GET and bounded503 regressions fixed; exact schema/SSE151, cachedPydantic2.13.5schema142,SQLiteimages33, frontend2282+6 and50,TS/precommit/productionBandit pass. ESLint0errors26inheritedwarnings verifiedbothparents. BroaderChat_NEW237pass11skip1Hypothesis timinghealthfailure; exactseed1pass andpropertyfile17pass unchangedsettings. CurrentCIrunning, remainingfreshnativecancellation/workspace/preview/ownerchecks ongoing. Keepdraft/unmerged; freshPGunavailable andhistoricalChromeIDsabsenceexplicit, notpassed.

Fresh native Stop found keyless owner Verify bug before GET; fixed only callback argument using protected view key with unchanged owner identity/lease/scope/signal. TDD60pass1fail then207pass plusforeignprofile/key/conversation controls. Exact correct stoppedinput51f244 native keyboardVerify GET200zeroinference; explicitReprepare/newSend2totaldispatches onecanonicalinput withverifiedresult, earlierunknownretained. Wrongolderpaneldriverselection producednofurtherinference; bothcombinednativeartifactsrequired,failedscreenshots retained. FullTS8192MBpassed afterdefault4096MBheapOOM; no appconfig/dependencychanges. Freshworkspace/preview/ownerrechecksongoing,remoteCIpending.

Fresh final no-mock Chrome matrix complete: grounded1realRAG/1Gemma; exact StopVerifyGET200zeroVerifyinference/explicitReprepare2dispatch1canonicalinput; route6zeroautosends;workspace11zeroautosends;preview/mobile3twoHTTP200realexternal/nooverflow;freshownerbaseline0sends/account-backend19checks2newGemma4canonicalrows/oldArestored/foreign404/inactivefullSHAunchanged;freshAloadedvisualbefore-afterpassed/inspected. Historical negative fixture staleclient-session failed before anysend; current real pre-switch Acheckpoint captured and identical fullSHAguards pass. Final production sourcebound Turbopack/token-sync/budgets540.3/842.7KBunder600/900; externaldependency roots aligned only in snapshot, repo unchanged. Original10/eightAPIrowSHA/servedcapability12negatives/all68stashes/sixcurrentbaselineTargets retained; historicaloriginalIDs ANDsameURLtabs unavailable, no preservation claim/replacement. Latest fetcheddevd81. Current remoteCIpending/no failures in last snapshot; localPGunavailable/broaderHypothesis timing failure explicit. Publishing final recoveryincrement and evidence next; keepdraft/unmerged andtaskInProgresspendingcurrentCI qualification.

Final publication verification freshly rerun: recovery/transport207pass0fail, fullfrontendTSC8192exit0, fourproductionPythonBandit0findings0errors, normalcachedprecommitapplicablehooksallpass, diffcheck0. Final reviewed recovery increment and corrected current-vs-historical UAT/production claims prepared for commit/push to existingPR3071; human summary retained exact. Stage3 remainsInProgressforcurrentCIqualification, notmarkingallCIgreen or freshPGpass.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fresh latest-dev no-mock Chat Workspace UAT and final production build/token/bundle gates pass. Final reviewed recovery increment fixes keyless-owner Verify with unchanged leases/identity and207freshchecks. PR3071 remains draft/unmerged pending currentCIqualification; requester summary is retained verbatim. Exact original conversation data/all68stashes/current baseline targets retained, while historical original tabs, fresh PostgreSQL and broader local Hypothesis timing qualification remain explicit limits.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
