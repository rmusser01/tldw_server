# A06 Media lifecycle implementation plan

**Goal:** Validate A-06 library search/tags/metadata/delete/restore permissions one step at a time.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md A-06, F-SOURCE/F-DISTRACTOR.
**Task:** TASK-13260.278.18.83.45. Existing frozen523 plus preservedWIP, notlatestdev.
**Architecture:** Real normal UI and existing API/Jobs/MediaDatabase paths. Use dedicated ordinary-user disposable sources, with actual owner/job/media IDs and content retained in functional operator state. No application mocks/rawSQL or credential-store injection.
**ADR check:** No new durable rule; existing ADR022 pipeline and ADR059 task tracking; DBManagement ownership/permissions unchanged.
**Runtime:** Restored original private API32268/owner32266, frontend24533/24507 buildy403t05zQ6dwcJQbge3kl; existing oneChrome/CDP18987. API2880/2874 below is the historical pre-switch owner. No new window/start/install/build implied.
**Constraints:** ACK2s/save-navigation15s/generation180s/background600s. Maximum3 attempts per actual issue, then documented reassessment. Preserve earlier retained users/content/drafts; no proof bundles/model/assets/cache/oldlogs/GitGC/index/fetch/commit/rebase/publication/heartbeatresume. Normal private credentials remain in app/functional private state, never printed.

## Stage1: Disposable actors and source corpus
**Goal:** Dedicated ordinary Alice/Bob plus existing actualAdmin and three owned disposable sources.
**Success Criteria:** NormalCreateUser/Login roles, sourceF-SOURCE/F-DISTRACTOR/neutral control exact text and recorded actual IDs; at least one source has an actual prior analysis/version to test restoration.
**Tests:** Normalforms/chooser/Jobs/canonicalGET readbacks; every source owner/currentjob/result/mediaID agrees. One declared bounded analysis generation for Rowan if needed, no transcription/model downloads.
**Status:** Complete
- [x] Inventory normalAdmin/user creation controls and managed roles; create dedicated actors with private functional passwords.
- [x] Save exact Rowan with declared analysis, then distractor/neutral without analysis/indexing; verify actual content/IDs/artifacts.

## Stage2: Search, batch tags and metadata
**Goal:** Exact search/filter/selection and supported metadata changes without collateral edits.
**Success Criteria:** Cedar Ridge selects only Rowan, clearing and tagfilter counts agree; selectedtwo batchtag add/remove persists afterreload, thirdunchanged; metadata edit retains fullcontent/others; normal keep/discard leave guard behaves.
**Tests:** ActualUI and authenticated canonical readbacks before/after each action; stale selection/doubleclick/confirmationcancel/Back and advertised partialfailure surfaces.
**Status:** Complete
- [x] Search/clear/filter; batchtag exactlytwo, reload and remove.
- [x] Metadata one source, keep/discard unsaved edits, assert other source/content intact.

## Stage3: Delete, Trash, restore and permission boundaries
**Goal:** Actual disposable-only lifecycle with same identities/provenance/analysis versions restored.
**Success Criteria:** Ordinary permitteddelete activecount0 including lastitem, Trash exactIDs/meaningfuldates; restore exactsource/prioranalysisversions and Trashcount/reload; missing-permission denial honest, authorizedAdmin uses actualpermissions; advertised permanentdelete only separate disposableitem/explicitconfirmation/irreversibility.
**Tests:** Actual normalUI confirmations/cancel/Back, Jobs/API canonical results and owner checks; no permission-role change to bypass a test, no deletion of retained A01–A05 evidence.
**Status:** Complete
- [x] Delete declareddisposablecorpus sequentially, verify active0/TrashIDs/dates and exact restore.
- [x] Missing-permission denial and authorizedAdminoperation; advertised disposablepermanent variant separately.

## Stage4: Fix demonstrated bugs and accept selected scope
**Goal:** Minimal root fixes with causal tests and truthful humanUX/functional accounting before A07.
**Success Criteria:** Any actual defect reproducedred and correctedgreen through sharedroot, affectedchecks/security/review; results explicit bymode/coverage, humanUX acceptance; broader commonengineering/release gates remainopen.
**Tests:** Existing configured suites/official fixtures when genuinelyaffected, baseline-currentBandit, independentreview reused only; no unchanged replay. HumanUX review after functional controls.
**Status:** In Progress
- [x] Diagnose/fix each demonstrated defect under approvedscope before retry, retain original failures.
- [x] Verify selected functional matrix/humanUX and task accounting; no full43/release claim.

### Initial actual corpus and first defect
NormalAdminCreateUser Alice4/Bob5/User; Aliceemptylibrary thenONEStart3files. Actualclientper-fileJobs13/14/15/currentbatches/completed/owner4/Media1Rowan422,2Distractor128,3Neutral71/eightstatusGET200/UI3succeeded3saved/library3. Allcanonicaltexts exact; jobbatch3-row expectation wasoperatorerror, noresubmit. ONEactualexistingQwenSummary30.873s/POSTversions201/version2/527analysischars/source422preserved. Userdelete-disabled/explanation agrees actualUserbaseline lackingmedia.delete; do not changeUserrole to bypass. PermittedAdminlifecycle requires actualAdminscope/disposablecorpus if per-userSQLiteisolationexcludesUser4; retainUsernegativecase ratherthan pretendingUserdeleteallowed.
CedarRidge searchactualUIonlyRowan/1of1; subsequentONEClearall unexpectedly0rows/total1 retainedquery. Early rootGET-onlycapturewrong(searchPOST); DOMbutton-onlyobserver wrong(ARIAroles). Prematurecleared3 statement corrected, no validatedclearbeforefix. Rootsource newrandomemptyfilterid inqueryKey/no changednormalizedcriteria, confirmed3causalredcases; sharednormalizedcachekeyfix31UI+31Web/types0 qualified, finalreview/rebuild/livepending. A06only; no delete or A07.

### Clearfilter selected live close / next Tag defect
Clearfilter source/test/operatorreview+31UI31Web/types0/no newlint/security andcorrectedlive scope accepted asnoted inassociatedtask; originalfailure retained. New44bindingfrontend60072/60063, API2880unchanged/oneChrome/User4/fullsource+analysisversions preserved. ONEAddtags selectedMedia1+2 returnedPUT200 butafterreloadallthreeGETkeywords[]; exacttexts/analysisversion intact. BothMediaTag consumers callgenericPUT whichserverexcludeskeywords; canonicalPATCH supportsadd/remove/set. NextboundedrepairroutebothMediatagwriterscanonicallyandexposeRemove requiredbyA06; NotesversionedPUT/owner/permissionguards retained. No delete/A07, tagacceptance notclaimed.

### Tag selected live close
Shared canonical Media keyword PATCH repair passed causal red/green, 69 UI/69 Web overlapping tests, types/lint/security/source review. One47binding build and reviewed startup-only recovery are qualified; original15-second shutdown over-grace remains recorded. Same User4/corpus/API/profile/oneChrome: Add and Remove exactly1+2 persisted afterreload, third3/content/allversions preserved, actualfilterIDs1+2/total2 and cacheclear restores3. Metadata/unsaved/permissions/lifecycle/othermode/humanUX gates stillopen; A07 unstarted.

### Unsaved selected live close
Shared editor/prompt guard and actual Next history rollback causally tested, scoped36 checks/types/lint/security/finalsource review clear. Current50binding build and one guarded frontend replacement qualified. Actual declined Back kept exact453draft and original Notes history entry/full source state; second explicit discard reached original Notes. No Media writes; reload exact422source/identity/keywords/all2analysisversions retained. Stage3 lifecycle/permissions/remaining modes/humanUX open; A07 unstarted.

### SQLite multi-user lifecycle selected result
User4 Delete remains disabled with truthful missing-permission explanation. Separate actualAdmin6 created/loggedin normally; emptylibrary then one3TXT Quick Extract job set16/17/18→Media1/2/3. One realQwen Summary saved201/27.306s/Rowan2versions/601analysischars. Actual DELETEs1/2/3 each204, counts3→2→1→0, confirmationcancel0requests and doubleclick2exactlyoneDELETE. TrashoriginalIDs/sourcefilenames/realmeaningfulUTCdates displayed. Restore1 preserved exactsource/content/metadata/keywords/full2versions andTrashreload2; selected2+3 thenRestore2 prunedstale2, Restoreselectedonly3. Trashreload0/active3/all3exactcanonicalfieldsets preserved. Separate101char permanentdisposableJob19/2.949s/Media4 softdeleted, explicitcannot-be-undone cancel0thenconfirm1DELETE204; Trashempty, normalMedia4deep linkGET404, original3active. RemainingD/PG/partialfailure/humanUXgatesopen; A07unstarted.

Human UX accepted2026-10-14: user accepted current Media/Trash usability; futureUXreview requests include one inline screenshot. Acceptance is separate from the three remaining deployment cells, still unrun.

## Selected functional closure — 2026-10-14
A06 selectedfunctional scope ACCEPTED: originalSQLite-multi/Adminseparatecorpus/permanentdisposable, SQLite-single composedprefix/actualpartialfault/legitimateRetry/remaininglifecycle, fullPGsingle, and changedseedPGmulti normalUser/Admin. HumanUXaccepted; futureUXreviews showone current screenshot. RealPGbootstrap bug repaired withonlymedia.update catalog/usergrant parity; actualrestrictedPGcausalred/green,10SQLitetests/scopedsecurity40identical0new/independentsourcereviewCLEAR; liveAlicePATCH200, ordinaryDelete remainsdenied. FixedPG1passed196.68s/APIshutdown3.213s. Allstartedfixtureownersreleased; originalONErestoreAPI32268/32266ready19.381s, Alice4 normalLogin/all3originalcanonicalsource/content/versions/metadataretained,Rowan422/2versions/oneChrome. Allfailedhistories remainfailed; no freshwholecombinedpass. A07 next. Stage4 remainsInProgress for common finalactual/stagedreview/normalhooks/commit/latest-dev integration/publication/release gates, not a remainingA06 functionalfailure. Completeddeployment/seedhelperplans are removedonlyafteraccountingreview; thismainplanretained.

Accounting precision: A06catalog step6 is permission refusal then actualAdmin operation, which was observed; this is not a nonowner/foreign-private-content acceptance. Earlier plan shorthand nonownerdenial corrected to missing-permission denial. X02 and broader foreign ownership variants remainopen.
