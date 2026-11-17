# A05 PostgreSQL job-status timestamp repair

Task: TASK-13260.278.18.83.45. User explicitly authorizes root-cause diagnostics/retries and quota validation; A06 remains unstarted. Frozen523ef323 plus preserved WIP, not latest dev; heartbeat paused.
ADR assessment: ADR003 Jobs, ADR021 lifecycle, ADR049 owner scope and existing string timestamp wire contract govern. No new architecture rule or OpenAPI schema change.

## Stage1: Identify the actual status-read failure
**Goal:** Distinguish failed status reads from a stalled worker.
**Success Criteria:** Normal upload/status wire and current owned error evidence establish the cause; original failures retained.
**Tests:** New official restricted PG/no-RLS flag diagnostic; real-router queued/completed/cancelled timestamp cases fail before source fix.
**Status:** Complete

## Stage2: Minimal shared conversion and affected consumers
**Goal:** Convert only actual datetime fields to ISO strings at the API boundary.
**Success Criteria:** Existing strings/nulls/schema/owner/principal checks intact; affected endpoints/events/cancel tests and scoped security pass; independent source review clear.
**Tests:** Endpoint/batch19pass, events/cancel4pass; compile, baseline/current lint and Bandit; source review.
**Status:** Complete

## Stage3: Real quota validation and original-session restoration
**Goal:** Corrected PG job status/source lifecycle, ordinary/Admin denials/no orphan and one legitimate Alice recovery.
**Success Criteria:** Actual job/batch/owner/terminal/source readbacks; no HTTP500 polling; official fixture cleanup and same original private API/profile/data/keys/session/draft restored.
**Tests:** One fresh timeout-fixed PG profile with reviewed continuous one-tab driver, existing600/120/15 ceilings and real61s TTL. No RLS/model/CI workaround; actual quota and ASR acceptance separate.
**Status:** Complete (selected timeout/quota and restoration scope)

Diagnostic1: one422B POST200/HTTPACK226ms/job1 then236statusGET500. Currentowned systemlog newest8errorlines read once/sanitized revealed Pydantic created_at/started_at/completed_at realdatetime rejection; one passiveHTTP500internal_server_error confirms. Jobcompleted timestamp present, no canonical source receipt. Client retained300s timeout, no cap increase. Corrected accessible-name driver reached final honestfailure; host396.04s/ASGIexit-15/2.849s/noovercap/fixture released. OriginalAPI85592/85564 stopped2.579s and remainsquiesced pending correctedrun/endrestore; oneexistingChrome/frontend30652/30647 unaffected.

Three new realrouter tests red ValidationError before conversion. Shared _job_to_status now converts four datetime timestamp fields; actualstr/None validation unchanged.19endpoint/batchpass4.02s/41warnings;4events/cancelpass3.81s/23warnings. ScopedBanditprod0/test52unchanged,0errors0new; inheritedRuffprodI001+SIM103/test2I001unchanged. Newtestblockformat ASTidentical. Initialtestlogpermission andsecuritytoolmissing/cache-write invocation failures notacceptedqualification; corrected scoped checks recorded. No tests/runtime replay after pureformat.

Independent actualminimalrepair/all3testbodies/sharedconsumer reviewCLEAR; fixedvalidationhost/helper/driver/originalrestorationclone reviewCLEAR. Correctedliveattempt2started afterallchecks: freshofficialrestricted/noRLS profile a05-timeout-fixed-pg-multi, initialization0/1.686s/Adminprovider0/API57540-owner57409ready12.751s. Original85592/85564 remainsalready-stopped/quiesced, avoiding an extra restart; finalsameprofile restoration is granted and separately guarded. Continuousdriveratreal61s TTL, noliveacceptanceyet; oneChrome/frontend30652/30647/providerunchanged.

Corrected live attempt2: small422B job1 completed in1.656s, HTTPACK212ms/fiveGET200/no500, owner3/batch/job exact and canonicalMedia1 source exact. The timestamp timeout fix is live-confirmed in this selected scope. Above1MiB1250031B job2 then unexpectedly completed/savedMedia2; quota validation honestly failed (2 != null). Host94.93s, ASGIexit-15/2.581s/noovercap, officialfixture released. OriginalAPI remains stopped; no extra stop/restart or quota acceptance.

New quota cause: queued file worker calls document/media processors directly and skips synchronous wrapper byte admission. Minimal worker fix calls existing atomic daily ledger admission before DB/processing with actual staged size/persisted owner/stable jobUUID receipt; typed429 is nonretryable terminal denial. Existing synchronous caller keeps random default receipts; duplicate worker deliveries reuse receipt and do not charge twice. URL/cancellation/owner/defaults/storage cleanup remain. New causal3failed1URLcontrolpassed. Full two files initially36pass1existing mock-file fixture missing, fixed only staged empty file (all original assertions retained); final37pass11warnings2.73s/natural0. Baseline/current Ruff inherited signatures identical; Banditproduction0/test64+21 unchanged/0new/0errors. Independent source review CLEAR.

Fresh quota-fixed final validation prepared: a05-quota-fixed-pg-multi, authoritative a05-quota-fixed-host-outcome.json, normal UI driver and existing61s TTL/600host/120startup/15shutdown; original85592/85564 already stopped with apiA05TimeoutStop proof, final new apiA05QuotaFixedRestore marker. Backend-only delta reuses current reviewed42-binding frontend, no build/install. One existingChrome/tab, officialrestrictedfixture, noRLS/model/cache workaround. Live quota and original restoration pending; A05 ASR Celine/Selene factual mismatch remains open; A06 unstarted.

Quota-fixed live3 partial: Alice smallexact422B1.639s/HTTPACK195ms and oversizedterminalfailed3.324s/HTTPACK141ms/dailybudget/no mediaID/no orphan accepted; five normalGET200/no500. After allowance4MiB/realTTL the driver erroneously clicked backgroundQuickIngest while reload had already restored the dialog; pointer interception ended beforeRetry. Not a productquota failure. Host1failed178.06s/APIexit-15/3.035s/ASGIcomplete/noovercap/fixture released. Originalstillstopped; no newstop.

Next separate operator correction within explicit necessary diagnostic/retry authority: fresh quota-recovery profile/outcome/restore marker, reuse visible dialog and click shortcut onlyifhidden; pure actualoperator block oldfailsvisible/fixedpassesvisible+hidden. Omit acceptedAlice smallcell, establish fresh deniedjob for legitimateRetry, then newAdmin small/denial. Originaltests/oracles/caps/normalUI/officialfixture/ownercleanup preserved. Pythonclone semantic delta names only, JS-onlybranch securityBanditN/A; syntaxpass. No automatic account/HTTP/retry/UX bypass or weakened assertions. Finalquota/restoration pending, A06unstarted.

Remaining quota-recovery run: Alice denial2.814s/HTTPACK391ms/no orphan; legitimate normalRetry job2/newUUID/batch completed3.457s/HTTPACK194ms/exact1250031B source. Admin smallcompleted2.21s/HTTPACK71ms/exact422B. Admin oversizedterminalfailed/HTTPACK151ms/dailybudget/no mediaID; full inventory assertion failed2!=1 BEFORE IDs because driver incorrectly filtered AdminexpectedIDs to owned-only while existingMedia RLS policy permits admin to see all acceptedsources. ActualtenGET200/no500, no source/privilege fix justified. Host1failed168.87s/ASGIexit-15/3.116s/noovercap/fixture released.

Focused Admin orphan correction: actual expected-ID expression purecontrol oldAdmin[2], fixedAdmin[1,2]; ordinaryUser[1] unchanged. Fresh admin-quota profile/outcome/newrestore marker, normalAlice+Admin smallsources as necessary two-owner inventorybaseline only, then Admin oversizeddenial; omitted already acceptedAlice denial/recovery. ExacttotalandIDs stillrejectanyneworphan. Samecaps/officialrestrictedfixture/source/privateoriginalstop proof; source-onlyreview pendingbeforelaunch, no A06.

## Stage4: Combined engineering handoff
**Goal:** Preserve reviewed repairs until the broader UAT candidate is ready for integration.
**Success Criteria:** Combined actual/staged review, normal hooks/commit, latest-dev integration and publication gates qualified under the authorized final workflow; full A05 acceptance separately complete.
**Tests:** Genuinely affected combined-candidate checks and source preservation; no unchanged control replay.
**Status:** Not Started

### Selected timeout/quota functional closure — 2026-10-14
PostgreSQL status datetime conversion and queued-file atomic quota admission are independently reviewed, causal red/green and affected23+37 tests qualified; scoped security0new. Live evidence is composed across separately recorded fresh officialrestricted fixtures: ordinary422B save/exactcontent and oversizeddenial/no orphan; legitimate normalRetry after managed4MiB allowance completed3.457s/exact1250031B/newjobUUID/batch; Admin small save and dailydenial/noorphan. Focused Admin host1passed106.84s, sevennormalstatusGET200/no500, ASGIexit-15/2.58s/noovercap/fixture released. Prior failedhosts and drivererrors remainhistorical, no single full combined-pass claim. OriginalsameprivateSQLite API42628/owner42581 restoredONCE/canonicalready16.531s; originalAlice2/User normalUIlogin and exactoldWAVdraft Celine transcript preserved; sameoneChrome/tab/frontend30652/30647. NoASRredraw/models/build/install/force/commit/rebase/publication/heartbeatresume. FullA05 audiofactualSelene->Celine/remainingvariants and combinedengineering gates remainopen; A06unstarted.
