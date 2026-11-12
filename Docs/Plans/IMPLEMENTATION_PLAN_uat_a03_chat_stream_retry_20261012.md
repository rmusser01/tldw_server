# A03 — Multi-turn Chat, stream controls and Retry implementation plan

> For agentic workers: execute inline with executing-plans, one workflow at a time. Reuse the approved single Chrome window/tab; no new agents or additional windows.

**Goal:** Verify actual conversation history, stable message identity and honest streaming recovery, then continue A04.
**Associated task:** TASK-13260.278.18.83.45; overall TASK-13260.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md A-03; release-catalog-data.mjs v1.1.0.
**Architecture:** Use normal UI actions and real provider/API requests on a fresh ordinary conversation. Read canonical message IDs/order through app traffic; never manufacture API/provider success or manipulate auth storage. A demonstrated defect becomes a separate bounded test-first repair using existing ownership handlers and persistence abstractions.
**Tech stack:** Frozen candidate523ef323 with owned WIP, Python3.12.13 installed source, served repaired production buildEWZTMgJKv6iML-QeE3PUs, existing llama.cpp8010/Qwen, Chrome/CDP/installed Playwright.
**ADR check:** No new ADR. ADR021 lifecycle, ADR025 provider adapters, ADR049 selected history ownership and ADR050 native fork lifecycle govern; this plan changes no durable rule.


## Current acceptance summary

A01 and A02 are accepted. The human accepted A03 UX after viewing the requested current Chat screenshot and replying “looks good”. A03 selected catalog functional acceptance is complete; combined repository/release closeout remains open. Two-turn context and the initial canonical reload ran successfully in SQLite/PostgreSQL, single/multi-user modes. Additional variants have only the scoped evidence recorded below; no entire new cell is accepted automatically from a host/test count.

- SQLite multi-user: reviewed guidance and streaming-header repairs are verified live; Stop, same-input Retry, Regenerate branches, picker persistence and genuine hidden return have evidence. Actual interrupted-stream/reload/same-question Retry and targeted delayed-first-content checks now passed; other mode variants remain open.
- SQLite single-user: history/reload, Regenerate and Stop/reload/Retry passed. The corrected build also passed real rejection/same-question Retry, picker persistence and saved-before-follow reload. Page-unmounted shutdown passed3.167s; the earlier active-client shutdown exceeded15s.
- PostgreSQL single-user: history/reload, Regenerate, real model-rejection Retry and Stop preservation passed. New repaired-build saved-reply reload and actual picker/persistence passed. The earlier stopped-turn replacement produced no final answer and remains unaccepted. New page-unmounted shutdown passed2.348s; active-client shutdown acceptance remains separate.
- PostgreSQL multi-user: actual ordinary User2 history/reload passed. The initial console Regenerate/Retry claim was premature and remains historical. In a new official fixture, the repaired build passed saved-reply reload, Regenerate branch/reload and real rejection/same-user-ID Retry with correct final selected history. New page-unmounted shutdown passed3.120s. The later fresh controls fixture passed Stop/partial reload and picker/request/persistence. Other unrun recovery variants remain open; the earlier failure cause is not attributed retroactively.

A03 selected catalog functional scope is ACCEPTED. No required A03 catalog variant is missing: context/history, canonical reload, Stop/Regenerate, real model rejection and successful same-question Retry, picker, delay, interrupted stream, genuine hidden return and reasoning/final separation all have recorded scoped results. Human A03 UX remains accepted. Actual40/240 interruption/Retry passes preceded the shared watchdog fix; their unchanged canonical/retry paths retain that evidence. The new content-only watchdog is causally qualified by3red/1control→30UI+30WebUI overlapping passes and source review, plus two actual15.009s timeout negatives and a separate saved/reloaded14.186s within-deadline positive. No combined corrected-timeout→successfulRetry in one fresh host is claimed or required by the catalog. All three failed watchdog observer/positive-expectation hosts stay failed; old180s exact cause, literal echo quality and empty-provider sample remain unaccepted historical results. Final positive full passive content equality was not qualified; normal canonical full-content equality across reload was qualified separately with zero new generations.

Remaining engineering/release closeout: final combined current-dev/staged review and normal commit/rebase/publication, public install/upgrade/native/installer/full release,45s nonstream timeout and C01→A03 linked-journey mapping remain OPEN. A04 may now begin in order. This is selected catalog functional acceptance, not an all-attempts/full-fresh-cell or whole-release claim.

## Global constraints

- A01/A02 fresh functional acceptance and human UX are complete; A03 selected catalog functional scope is accepted; A04 is next. Current-dev rebase/combined qualification and public release/upgrade gates remain OPEN.
- One explicitly approved restoration: same existing profile/data/keys/build, one120s startup each for API/frontend/Chrome, one window/tab, no install/build/force stops. All older consumed attempts and STOPs remain; no model/provider restart or downloads.
- Generation180s, navigation/save15s, acknowledgement2s, background/operator host600s, graceful owned shutdown15s. One declared recovery per demonstrated issue; maximum3 distinct attempts then reassess. Declare each runtime lifecycle and preserve every first failure.
- No browser auth-store reads, token/cookie/header injection, fake success, operator SQL, screenshots/proof bundles, broad scans, old logs, staging/commit/rebase/push/PR/merge. Keep concise task/plan results and necessary functional controls only; heartbeat PAUSED.
- Four cells: SQLite single/multi and fresh official restricted-role PostgreSQL single/multi; Admin only for normal bootstrap, Owner/Alice for execution. Fixture held until actual API exit/teardown. Released earlier PG profiles stay consumed.
- Fresh application-state behavior is separate from public clean installation and previous-release/oldest-supported upgrade. Reuse of browser/dependencies/models is disclosed.
- One working text model is currently available. Successful model selection is checked after restoring the working choice from a real rejected choice and sending the next turn; no claim of two working physical models/providers.

## Stage1: Restore and bind the runtime
**Goal:** Restore approved owners and bind actual principal/provider/source without replaying A02.
**Success criteria:** Canonical APIready, served buildID, sole Chrome page, actual llama.cpp modelsHTTP200, ordinary Alice/key entry and a fresh conversation.
**Tests:** Existing runtime monitor/normal app forms; installed-module/free-port/owner guards. No old suites/builds or application import workaround.
**Status:** Complete

- [x] Record old owner absence as unobserved prior teardown; retain old outcomes and distinct new attempt fields.
- [x] Start API18906 with installed Python and same SQLite-multi profile once120s, actual /internal/ready statusready; start existing production frontend18985 once120s with the retained dist; start ChromeCDP18987 once120s with the owned profile and one tab.
- [x] Verify current public identity through normal UI; use ordinary Alice rather than admin for multi-user tests. Preserve all old canonical Chats/Notes.
- [x] Select a fresh ordinary conversation and working Qwen via normal controls. Record only synthetic prompts, public IDs, statuses and selected provider/model.

## Stage2: Actual history, streaming and recovery
**Goal:** Complete the applicable A03 variants in each cell, fixing only demonstrated causes.
**Success criteria:** The real second request includes intended history; canonical message IDs/order survive reload. Stop/Regenerate/Retry preserve documented boundaries and honest partial/failure states. Actual picker, request and saved selection agree.
**Tests:** Normal real UI/provider flows; causal red/green plus affected checks/scoped security and the existing authorized reviewer only for actual repair units.
**Status:** Complete

| Required variant | SQLite multi | SQLite single | PG multi | PG single |
|---|---|---|---|---|
| Two real turns and intended request history | Pass | Pass | Pass | Pass |
| Canonical reload/history, stable IDs/order/no duplicate | Pass | Pass | Pass | Pass |
| Long response Stop and documented Regenerate boundary | Pass | Pass | Pass | Pass |
| Real rejection and Retry of same failed user identity | Pass | Pass | Pass | Pass |
| Actual picker/request/persisted selection alignment | Pass | Pass | Pass | Pass |
| Delayed first chunk, usable draft/progress | Pass: targeted network latency | Pass: targeted network latency | Pass: targeted network latency | Pass: targeted network latency |
| Real interrupted stream and honest partial recovery | Pass | Pass: new observed40 and original240 cancel/partial reload/Retry; old180s failure retained | Pass | Pass |
| Real hidden-window return, draft/focus/scroll | Pass | Pass: actual scroll1182; host later expired | Pass: final case scroll500/lifecycle passed | Pass: actual nonzero scroll; host later expired |
| Reasoning/final separation | Shared parser/source checks + actual original reply reload pass | Pass: actual typed reasoning withheld from final, complete capture===canonical, correct323; failed literal echo quality retained | Pass: real typed wire/canonical/display | Pass: real typed wire/canonical/display |

- [x] Send Remember the tour code ORBIT-742, then What is the tour code? once each. Before submission observe only a safe projection of the actual /chat/completions request: model/provider and presence/order of these synthetic turns; never retain headers/credentials or unrelated prompt bodies.
- [x] Reload/open the same canonical Chat through the existing chat-sidebar toggle/Recent conversations/Server controls. Compare public message IDs, roles/order and completed answers with the first saved result.
- [x] Submit one declared sufficiently long response; Stop after actual content begins, confirm cancellation/partial state and usable composer. Invoke the actual Regenerate control once and compare canonical question/answer/branch identity with the connected handler contract; historical design proposals are not the current oracle.
- [x] Use an unavailable model choice or an expressly disclosed request-only control to trigger a REAL backend/provider rejection. Preserve the first error and failed user ID, restore the working choice normally, Retry once and require exactly one successful answer for that same user turn.
- [x] Change through the actual picker, send a new turn and check the real next request plus persisted/displayed selection after reload. With one physical model, disclose the rejected-to-working selection scope.
- [x] For delay/interruption controls, keep real requests/data and disclose exact affected boundary; never fulfill a forged response. Name dispatch delay separately from first-chunk delay if only dispatch was held. Use actual streaming content/end state, not headers alone, as completion.
- [x] Exercise actual hidden visibility using the single owned window if supported; verify document visibility state and return to the same tab. A viewport resize or emulated timer is not hidden-tab evidence.
- [x] Repeat genuinely required variants sequentially in the other cells with fresh fixture ownership and distinct profiles; never reuse released A01/A02 PostgreSQL databases.

## Stage3: Review and functional acceptance
**Goal:** Record exact completed scope, obtain human UX and close A03 before A04.
**Success criteria:** Every selected fresh functional row has an actual oracle/outcome; changes retain causal/affected/security/review evidence; human UX is separately recorded and all release gates stay truthful.
**Tests:** Final task/plan/ownership accounting and normal touched-scope checks after repairs; no automatic pass or unqualified count inflation.
**Status:** In Progress

- [x] Record every cell/variant, first failure, correction, consumed cap and unsupported/blocked capability explicitly.
- [x] Human A03 UX accepted on the requested current Chat screenshot: “looks good”. Functional completion remains separate.
- [x] Close only the accepted fresh functional scope and continue A04 in order; preserve pending public upgrade/latest-dev/native/installer/full-release gates.

- [ ] Finish combined source/staged review and normal commit closeout with latest-dev qualification; retain this owned plan while that engineering stage is open.

## Initial facts

- Human accepted A02 UX and authorized the one-session restore. Existing build/profile are intact; prior API/frontend/Chrome owners/ports were absent. Earlier ready metadata is historical and does not prove graceful shutdown.
- Actual llama.cpp /v1/models returned200 with unsloth/Qwen3.8-27B-GGUF:Q8_0. No provider/model action was performed.
- Read-only preparation inspected current Regenerate ownership boundaries and existing handler/model-sync test seams; no A03 runtime/send/retry or source repair has yet executed.

## A03 progress — 2026-10-12

- Approved restoration consumed: API5088/owner5085 ready8.690s; frontend10432/owner10430 exact buildS3kF ready0.824s; Chrome10920/owner10918 one page/CDP ready0.440s. Installed native controller attached after sandbox loopback denial; no extra browser. Public Alice2/user verified by normal login/auth/me.
- Fresh native Chat0659afd3-2347-4f57-8993-91c934135497: two real ORBIT-742 turns, second actual request user/assistant/user with intended history; correct final answer. Four canonical IDs/content/parents unchanged after reload; senders user/assistant/user/assistant; no duplicate. Second UI ACK0.140s.
- Long200-line request UI ACK0.165s, HTTP200, no visible answer; net::ERR_ABORTED and accepted-unsent recovery retained original question. No Stop/Retry/Regenerate yet; underlying stall cause not established. Never claim headers alone certify success.
- One CDP minimize control did not produce document.visibilityState=hidden within15s; restored normal. Draft831 preserved. Hidden-tab acceptance remains open.
- Demonstrated no-output recovery dropped actionable error guidance. Optional friendly-only failure_reason repair in five TypeScript paths: causal8red/8green, same admission/identity/Pirate context retained; existing bounded reviewer CLEAR, full configured WebUI types0. Raw diagnostic text excluded. New generator lint fixed with original rejected-yield pattern; inherited25 any warnings retained, no new Python/Bandit scope. Live repair acceptance/build still pending.
- First nine-file UI run199pass/1failure: saved-server interrupted Retry transcript/context/identity pass but final persistence kind serverFailed instead of server. Pre-guidance isolated baseline target passed1/18deselected9.862s; corrected four production files restored exactly. This does not establish flakiness, root cause or regression attribution. WebUI affected run pending; original failure remains unqualified and is not weakened.

## Compression repair — 2026-10-12

- Guidance repair live PASS on SQLite-multi: one request-only nonexistent model control produced real400/model_not_available, rule removed; guidance survives reload. ONE reviewed Retry returned actual final842 in6.160s; original failed user6235 ID unchanged, one parent-linked answer27da, all prior6 IDs intact. No duplicate.
- Streaming cause: actual prior response gzip/no-cache. Last capped unchanged control received1.48MB in two bursts36.788s afterheaders1.783s; firstvisible36.802s already239lines, no honest Stop marker. All three oldlongcontroloutcomes remain unaccepted; no fourthunchangedreplay.
- Exact installed Next compression gated transport causalcheck: no-cache gzip withholds frame beforeend; no-cache,no-transform identity delivers beforeend. Ordinary Chat service five SSEheader sites + macro endpointone now reuse existing Character Chat no-transform policy. No body/provider/config/dependency/schema/lifecycle changes.
- Red3 headercasesfailed8.43s; green affected96pass/10.16s plus realmacroendpoint1pass/10.54s. Source reviewCLEAR; laterfallbackbranches source-only. Banditbaseline/current production0/tests54+96identical/noerrors/no new; RuffoneinheritedserviceI001/endpoint0; newtestblockonlyformattedASTidentical; scopednormalhookspass.
- APIreplacement: original5088 stopabsence15s failed, monitor laterobservedexit-15/actualshutdowncomplete; noforce. New sameprofileAPI76640/owner76634 canonicalready8.941s. Frontend56963/buildL7Ev remains, soleChrome10920/provider74814 preserved. One distinct changed-source live transport/Stop check inprogress; not yet accepted.

## Changed-source live Stop — 2026-10-12

- First repaired check sent no request after reload/preparation; draftempty/noACK/no newrow, operatorresponsewaitinterrupted165.994s. Recorded separately, no providerfailure/Stop claim. One declared recovery required actual restored11-row history beforeSend and ACK2s gate. Possible preparationdraftloss remains unattributed; no workaround accepted as productfix.
- Repaired recovery: UIACK0.225s; real200 identity/no-cache,no-transform; wirefirstchunks8.318s afterheaders8.315s,843updates; visiblefirst875line22.229s; actualStop22.385s kept line+partial, honestStoppedmarker/composerusable. No gzipburst. Originalcappedcontrols remainfailedhistorical.
- Actualreload: canonical12rows, originalquestionpa_9d2a-3629-fff-abb1 singular; no serverassistant for stoppedquestion; keptdevicepartial875 visible withStopped label. NormalRetry same model inprogress, originalquestion/requestcontext usedonce; no finalretryclaimyet.

- StoppedRetry completed45.562s; actualreload13canonicalrows, original875question9d2a singular and oneanswer4f6d linkedtoit, deviceStopped retired. No claimexact160lineproviderinstructioncompliance (118tags).
- Distinct Regenerate completed33.171s; actualreload14canonicalrows/13selectedmessages, original875question9d2a singular, originalanswer4f6d preserved and newanswer82c5 linkedtosamequestion; selectedlastrow82c5/variant2of2. Originalhoverchip click15s missed because normalhoverrevealedtools; actualvisibleRegenerateusedno force/JSinvoke. Source-documentednativebranchbehaviorPASS.

- ActualpickerSQLite-multiPASS: selectedexistingOllama/gemmachoice thenrestoredworkingQwenthroughnormalmenu; no secondworkingproviderclaim/noOllamacall. Real886nextrequestQwen200/final5.539s; reloadselectedQwen+savedsingularuser8d26/assistantf7ab. Exactmenuitemnamefirst15s operatorfailurepreserved; text-matchedrealmenuitem workedwithoutforce.
- Network897separatecontrolledinterruption nowinprogress: realreply, composer898draft duringnaturalfirstchunkdelay, actualCDPofflineaftervisiblepartial/finallyonline; no fabricatedproviderresponse. Hiddenvisibilitycase remainsunaccepted because previousminimizationdidnotchange documentstate.

## Remaining A03 recovery gates — 2026-10-12

- Interruptednetworkattempt1: CDPoffline didnotterminateexistingloopbackstream; responsefinishednormally, firstvisible897content1.351s/898draftintact, normalnetworkrestoredfinally. NoInterruptedacceptance.
- Onepagereloadrecovery899 sentrealrequest200/Qwen/18historymessages/noexplicitmax_tokens. No899answerarrived; appendedwithNo response was returned guidance/retainedquestion/900draft. Controllerwaitinterrupted166.425s toinspectfinishedUI; ChromeERR_ABORTED and responsebodyunavailable, no exactproviderterminationcause. Noactualreloadduringstream/partialInterruptedacceptance; no thirdunchangedgeneration.
- Bothreviewedrepairs remainverified: guidanceerror+sameinputRetry; no-transformliveframes+Stop/reload+sameinputRetry/nativeRegeneratebranches; pickerrequest/display/persistence. This is SQLite-multionly, notallfourmodes or fullA03.
- API76640/owner76634,frontend56963/owner56961/buildL7Ev,Chrome10920/owner10918/onepage retained atlastactualcheck. Ownerstop15s deadlinefailedbutlateractualshutdownprecededreplacement; noforce. No freshwhole-runtimeclaimfromthisnote.
- A01/A02acceptedfunctional+humanUXclosed; A03 Stage2InProgress/Stage3NotStarted. Hiddenvisibility/actualinterruptedstream/otherthreecells/humanUX and latest-dev/rebase/publicupgrade/native/installer/release43remainOPEN. Allworkunpublished, sourceHEAD523frozen.

## Current sequential mode progress — 2026-10-12

SQLite single-user executed two-turn context, canonical reload, Regenerate, Stop, delayed-draft retention and stopped-turn Retry successfully. Chat `0e27a281-d871-4a1f-984e-a522dee944e0`; original stopped question `pa_f33e-3ef6-7a0-6fff` remains singular with one saved answer. Other single-user recovery variants are unrun. The host failed its shutdown gate: 17.351 seconds against 15, although the API exited and confirmed shutdown before the host ended. This is not full-cell acceptance.

The existing Chrome's genuine hidden state and subsequent visible return were read without Playwright focus emulation. Draft900, text-field focus and scroll were preserved; one page remains. Earlier simulated-focus attempts are not accepted evidence.

PostgreSQL single-user is now running under the official restricted fixture at `a03-browser-pg-single`. API96175 / owner96106 became ready in12.269s. No manual database, Docker restart or key/header injection. Normal UI key entry passed protected connection checks; an operator wait wrongly required an auth/me request that this save flow did not issue. Actual Chat two-turn context and canonical reload then passed on Chat `da956411-b3cd-462b-864a-23ae40ff7278`. Regenerate and real model-rejection Retry are in progress, not yet accepted here.

PostgreSQL single-user Regenerate and real model-rejection Retry passed: original failed question `pa_2236-575c-767-aefe` remained singular. Stop began after16.62s of real streaming; question `pa_6f8e-0fb8-7c8-537a` was saved once, and the stopped partial plus draft933 survived reload. Its replacement Retry returned no final answer; the original partial and friendly failure guidance remained. That Retry is unaccepted and was not resent.

After the page navigated normally to about:blank, the PG single-user API exited in2.283s and the official fixture released. Host535.47pytest/536.586outer passed its lifecycle. Earlier active-client stop failures remain open; this is not proof of their sole cause.

A fresh PG multi-user host is now starting using the same reviewed adapter and existing cluster. No new browser or model process is created. A03 is still incomplete; no rebase/publication has occurred.

### Correction to PostgreSQL multi-user results

Admin only created ordinary Alice id2/role user. Actual auth/me verified that ordinary actor before Chat. Two real ORBIT turns completed1.546s/1.029s on7b68f069-fe58-4f5d-8aa2-11a71101368e with canonical four-row reload. A later response-body read after navigation failed in the operator; the already-captured actual public identity was used without repeated login or credential injection.

Subsequent raw seven-row/parent-count checks did not establish final UI completion. Reload showed five selected rows and a device Interrupted partial A03-MULTI-, with Could not save to server. The planned962 interruption send never ran. Regenerate/Retry final acceptance is therefore retracted pending a proper completion/settlement observer and investigation.

The host expired before explicit completion, then waited for actual API exit-15/ASGI shutdown18.092s before fixture release; it failed naturally587.81pytest/588.917outer. A late finish marker does not change browserCellCompleted=false. No force or replay occurred. The retained original SQLite profile is being restored once; that does not close these gates.


## Saved reply / reload boundary repair — 2026-10-12

- The previous streaming flag observer was premature. A deterministic component test holds the real follow step, observes streaming=false with an unresolved action and one recovery, then verifies exact selected history and no recovery only after the returned action completes. New target passed; full then-current file52passed; configured WebUI target passed. This does not establish the historical PG multi failure cause.
- A separate causal regression proved that an ordinary native reply could be acknowledged by its owner while its durable recovery still held an earlier Interrupted partial. Red1failed/120ms. The minimal existing complete() callback now records outcome=complete and the acknowledged assistant ID before following the view. Mounted reload recovery follows the full canonical answer without another save or generation; green1passed/128ms. Existing owner, retry, local and Regenerate boundaries remain.
- Affected six files:232UIpassed/52.79s and232WebUIpassed/52.13s. Declared configured8GiB typecheck natural0; the first raw default-heap compiler exhausted memory and was not a type result. ESLint0errors/eight inherited normalChatMode warnings/test0; normal hooks/diff check pass. TypeScript-only Bandit N/A. Independent actual-source/connected review CLEAR. Live build and qualification are pending.
- Distinct dispatch-delay control used one real request delayed3s without a forged response. The unsent975draft remained; final974answer completed. The operator incorrectly awaited Chat creation after New Chat had already created it; that observer timed out15s and no resend occurred. Read-only scoped history then found the exact final cursor, no pending operation and one completed outcome. Normal reload returned canonical question3028/answer15a6 with correct parent and exact answer, draft preserved, one request total. This is dispatch-delay usability evidence, not a provider first-chunk delay pass.
- One necessary changed-source build declared with the retained eight accepted A01/A02/A03 runtime source bindings, existing dependencies and fresh .next-live-tier-a03-settlement-20261012 output;600s cap/no installs. Original functional source bytes must be restored after compile. Frontend-only owned replacement will retain API/profile/data/keys/Chrome,15s graceful stop/120s startup/no force. No A03 completion, A04, rebase or publication yet.


## Changed-build live settlement cells — 2026-10-12

- BuildQyB8H8AEU0F4dzWNxwRTU compile/token/bundle natural0/0/0 in160.293s; eight retained runtime source bindings restored. Frontend initial launch56075 failed before readiness; operator omitted production quickstart mode and discarded stderr, so no exact startup-log cause is claimed. Existing configuration validator rejected that environment and accepted declared production shaping. The one consumed recovery57113/owner57070 served the exact build in1.017s. No API/Chrome restart or dependency install for this repair.
- Live SQLite-multi saved-before-follow reload PASS: one real generation and only final selection dispatch held; durable record complete, canonical full answer0edb under question39bf, four rows, no Interrupted marker/duplicate,985-era original975draft retained. Initial cross-bookmark observer assertion was overstrict: the original operation tombstone and reopened view cursor belong to different client bookmarks. Corrected read-only checks pinned the original operation separately from actual new selected DOM/canonical reply; no resend. Original Chat IDs/work unchanged.
- Original SQLite API6517 stopped3.258s/exit-15/actual shutdown complete before fixture cells. Existing official-host adapter received only a fixed fresh profile-prefix replacement. Three pure structural guards passed; static Bandit0findings/0errors, Ruff0, independent source review CLEAR. First Bandit invocation in the test environment reported module unavailable; the already installed primary static tool qualified it without installing anything or running below-floor application code.
- New PG-single API61335/owner61300 ready11.496s. Normal key/connection success. Saved-before-follow reload PASS in Chat03a004ae: question4f0d/answercc12, canonical2/full answer/no partial/duplicate, draft980. Actual picker gemma→Qwen then one Qwen981 request and normal reload PASS: canonical4, questione446/answerc3fd singular and parent-linked, displayed Qwen/draft980 retained. No Ollama request/second working model claimed. Incorrect settings-only Server connected selector timed out; real DOM showed successful Test Connection, no resend. Client unmounted and finish222.216s before work deadline; shutdown2.348s/exit-15/ApplicationShutdownComplete, host1passed349.79s/outer350.73s/natural0. Fixture released; no reuse. Lifecycle count alone does not accept all variants.
- New PG-multi API64630/owner64613 ready11.482s. Ordinary Alice2/user created through normal Admin UI; admin logged out, actual public auth-me verified User2 before Chat. Saved-before-follow reload PASS in Chatb67c67f3: question2422/answer0cea, canonical2/full answer/draft985/one generation. Regenerate PASS: same2422 question, original0cea preserved, newc033 parent-linked, raw3/selected2 after reload/no partial/one generation. One bad-model request control produced real400; removed immediately, guidance and failed03fb question survived reload. Normal reviewed Retry Qwen returned in1.492s with dispatch ACK within2s: new04ea to the same03fb, canonical5/selected4, all prior rows unchanged, one linked reply/no Interrupted/save-failed marker/draft985/one retry request. Client unmounted; finish37.734s before deadline; shutdown3.120s/exit-15/ApplicationShutdownComplete, host1passed535.10s/outer535.957s/natural0. Fixture released. Historical failure attribution remains unproven.
- Fresh SQLite-single changed-source cell is now active for the same repair and genuinely unrun rejection/picker variants. All old interrupted/long/hidden ceilings remain; no A03 closure/A04/rebase/publication.


## Human UX result — 2026-10-12

The requester viewed the one requested viewport screenshot of the current corrected Chat and replied “looks good”. A03 UX is accepted for that shown interface. This does not close the remaining functional variants or qualify latest dev, public installation/upgrade, native/installer or release gates. No further screenshot or proof bundle is authorized implicitly.


## Latest completed functional variants — 2026-10-12

- SQLite-single corrected-build saved-before-follow reload, real model rejection/same-question Retry and actual picker/request/persistence passed in Chat6df59f8e. Canonical2/4/6 rows retained prior IDs/content, draft990 and singular parent-linked answers. Client unmounted before finish; shutdown3.167s/exit-15/ASGI complete, host1passed538.79s/outer539.597s. This does not erase the older active-client shutdown failure.
- SQLite-multi interrupted-stream third/final diagnostic passed: Page.stopLoading cancelled the actual active request only after visible nonempty partial at7.255s; actual net::ERR_ABORTED/canceledtrue. Honest Interrupted partial and draft993 survived reload. One normal Retry16.387s saved the final80-line answer under the original question, canonical2/no duplicate/no recovery/draft retained after reload. Current provider emitted no reasoning tags, so reasoning display remains untested. Both earlier failed interruption controls remain historical; no fourth diagnostic is authorized.
- Targeted delayed-first-content variant passed on the same real SQLite-multi API/provider: URL-scoped4s protocol network latency, real200 headers4.962s/first observed meaningful content5.506s, waiting progress busy and editable draft996. No fabricated response or auth mutation. Rule removed and session detached finally. Normal reload returned Chat5822baf0 questiona7ea/answer24e5, canonical2/same IDs/correct parent/exact995 answer/no Interrupted/draft996, one request. This qualifies the disclosed network-delay boundary, not artificial provider compute delay or the other three modes.


## Remaining PG-multi controls — 2026-10-12

- New fixed-prefix pg-multi-only adapter retains official fixture/caps/refusal/lifecycle, pure guard1passed0.24s/Ruff clear/Bandit0findings0errors/independent source review CLEAR. This is operator evidence only.
- Client unmounted; freshly guarded original SQLite API80076 exited2.792s after one TERM. Fresh official PG API76572/owner76029 ready21.138s; ordinary Alice2/user verified through actual public auth-me after normal Admin creation/logout/login.
- One real80-line997 reply stopped after actual visible partial. Initial operator expected Interrupted and timed out; actual UI correctly Stopped with kept text/draft998. Corrected observer and normal reload without another generation: Chatda9c0d3f saved question9569 singular, device partial and draft retained, composer usable.
- Actual picker gemma→Qwen/no Ollama request, one real999 generation: Chatbd9b5127 questionf1cc/answerdd84 canonical2/correct parent/final answer/durable completion; model display and draft1000 retained after normal reload. Exactly2generation requests for this cell.
- Client unmounted before finish; shutdown3.138s/exit-15/ASGI complete, host1passed305.49s/outer306.859s/natural0 and official fixture released. Older active-client shutdown failures remain open.
- Original SQLite same profile/data/keys restored once under apiA03AfterControlsCells: API92236/owner92232 canonicalready13.034s. Existing frontend/build/Chrome retained; no install/model/build/new tab/publication.


## Mode-specific delay checks and restoration — 2026-10-12

- New fixed-prefix delay adapter only: exact existing three cells, fresh profile refusal/caps/official fixture unchanged. Structural3passed0.37s, Ruff clear, Bandit0findings0errors, independent source review CLEAR. Original SQLite92236 guarded/unmounted stop3.298s.
- SQLite-single PASS: API22711/owner22053 ready12.543s; real URL-scoped4s network rule matched. Headers4.504s/first meaningful content4.528s, busy progress/draft1002 before content, one request. Chat292acd2d canonical2/question6b17/answer7f1a/same IDs/correct parent/final1001/draft after reload. Rule removed/session detached; unmounted shutdown3.266s/exit-15/ASGI complete, host1passed136.59s/outer137.92s.
- PG-single first observer UNQUALIFIED: API26433/owner26409 ready14.050s. Real1003 answer visibly saved; draft1004 preserved. Combined page SSE/native-completion observer timed out180s. Later selected native viewf5bce621 had cursorfd7d, no pending and original completed outcome; visible question2d90/answerfd7d survived reload. Two response observers missed15s, including inappropriate networkidle waiting; no resend or whole delay pass. Unmounted shutdown3.971s/exit-15/ASGI complete; host1passed473.09s/outer475.023s is lifecycle evidence only.
- PG-multi first observer UNQUALIFIED: API31523/owner30872 ready14.816s; ordinary Alice2/user verified. Initial mode-change operator tried Save behind existing confirmation for30s; corrected normal Change mode before Save and lowered default locator timeout to15s. One1005 generation missed the2s user ACK; observer saw0 matched Chat requests/tokens and no applied rule, so draft1006 was never entered. Actual UI later showed saved questionbfad/answer7825/final1005. No resend, forged success or latency acceptance. A response observer also missed15s. Client unmounted, then actual shutdown20.563s/exit-15/ASGI complete exceeded15s: NEW lifecycle FAILURE; host1failed477.05s/outer495.844s. Fixture held until actual exit then released.
- Chrome extension worker was inventoried, but exact source and own served bundle contain the WebUI runtime shim with undefined runtime.id; extension-worker transport was not substantiated and is not accepted as cause. Browser transport source permits direct/absolute and hosted /api/proxy routes; exact failed runtime request path remains unproven. No extension or auth-store action.
- Original SQLite restoration apiA03AfterDelayCells FAILED/consumed: canonical readiness not observed within120.100s; API36467 exited-15, ASGI shutdown completion not observed. One newest in-attempt functional log showed ASGI startup complete near the deadline; canonical readiness/failure cause remain unproven. API stopped. ONE declared same-source/profile/data/keys/env120s recovery apiA03AfterDelayRecovery is running, retaining functional probe count/HTTP/payload/error-class/startup/shutdown metadata. No further recovery after it, larger cap or app/environment/model workaround.


### Restoration recovery result

The one apiA03AfterDelayRecovery allowance is consumed and successful: API41974/owner41967, canonical HTTP200/status ready in51.530s, ASGI startup observed,155probes. Last HTTP/payload200/ready; earlier ConnectError is historical within startup. Original Alice2/user was restored through normal login/auth-me; one Chrome page and existing build/profile/data/keys retained. A stale older response-body listener errored after navigation and was removed; current actor was separately verified before navigation. The first120.100s readiness failure remains historical, and no further recovery for that issue is authorized.

A separate operator-budget causal unit is in progress: the real installed Uvicorn shutdown method drains connections for its configured grace before application cleanup. Original first-attempt hosts and historical acceptance are retained. New recovery host will reserve cleanup time inside the same total15s ceiling only after real causal red/green and source review. This is platform/operator evidence, never application/UAT substitution.


## Latest delay and registration results — 2026-10-12

- New observer host reserves lifespan cleanup within the unchanged15s total exit limit: installed Uvicorn controlled shutdown causal red16.104s, green three checks8.66s. Only fresh recovery hosts use5s request drain; original hosts and failures remain. Source review/static checks CLEAR. Platform test is not application acceptance.
- PG-single corrected delay observer PASS: API39038 ready22.181s; one real1007 request, matched4s rule, headers8.349s/content9.027s, busy progress/editable1008draft before content. Chatc2310277 question9918/answer9500 canonical2/same IDs/parent/exact final/draft after reload. Rule/session removed; unmounted shutdown3.299s/exit-15/ASGI complete; host1passed205.01s/outer206.602s. Fixture released.
- PG-multi observer recovery FAILED before ordinary user verification or generation: normal Create user timed out; same attempt log later recorded Aliceid2 created. No resubmission, false actor acceptance or delay pass. Host expired with browserCellCompleted=false; shutdown7.549s/exit-15/ASGI complete, host1failed578.27s/outer580.026s; fixture released. Both observer recovery allowances consumed. Historical shutdown errors/cache warnings do not prove user-creation timeout cause.
- Original SQLite profile/data/keys restored once: API18816/owner18812 ready8.759s under apiA03AfterObserverCells. Original ordinary Alice2/user verified through normal login/auth-me; same build/Chrome retained. This API predates the following source repair.
- Separate registration event-loop responsiveness repair: same validated hash now runs through await asyncio.to_thread; transaction, roles, quota, audit, directories and Argon policy remain. Controlled actual-service regression red[False]/green[True]; three causal/harness attempts consumed, affected15unit cases pass9.91s. Official isolated PostgreSQL original two tests passed27.41s/outer35.211s after correcting TEST_DB variable binding; first incorrect binding skipped2 and is not a pass. Production Bandit0/test12identical/noerrors/new; Ruff2inheritedtest diagnostics only. Added formatting preserves AST; connected independent source review CLEAR. ADR021/025, no new decision. Full late UI timeout causation remains unproven.
- Next qualification is one distinct fresh registration-hash-fix PG-multi profile on changed source, same official restricted fixture,120s startup/600s host/15s shutdown with5s request drain, one existing Chrome tab. This does not reset the failed observer recovery or other exhausted diagnostics. Prepare scripts before host starts; submit one creation only, verify ordinary public identity, then the remaining delay check. No A03 closure/A04/rebase/publication yet.


### Changed-source PG-multi live progress

Fresh API44840/owner40802 ready35.451s. Normal Create user returnedHTTP200 in556ms, ordinary row visible; Admin logout and normal login public auth-me verified Alice2/user/uat_a03_hash_alice_20261012. Shared hash offload responsiveness and current normal creation qualify; historical full timeout cause remains unproven.

Delayed content PASS: real1009 request with applied4s rule, headers13.612s/content14.266s, busy progress and editable1010draft before content. Chatde57e986 questionb5f9/answer0ba2 canonical2/same IDs/correct parent/final/draft after reload; one generation, rule removed/session detached. All four targeted network-delay variants now pass.

Hidden-window remains UNQUALIFIED: first immediate observation still visible; one declared transition-wait recovery saw actual Browser bounds minimized but document visibility still visible for2s. Draft1011 retained, one window restored normally. No hidden-state acceptance or synthetic visibility override. An old closed raw control socket read timed out15s without app action; no new context/window. Remaining PG-multi actual interrupted-stream variant is now running once under the same ordinary actor/fixture/deadline; no replay of SQLite diagnostics.


### PG-multi interruption and fixture completion

One actual long1012 stream cancelled only after visible nonempty partial: net::ERR_ABORTED/canceledtrue, one initial request, no normal finish, unsent1013draft. Normal reload returned only original saved questionff86 in Chat97ad99e7, device partial and honest Interrupted guidance/draft preserved. One actual Retry completed; normal reload canonical2/questionff86/answerf8a5 with same parent, final1012text/no Interrupted marker/draft preserved, exactly2generation requests total. No duplicate question or reply. Original diagnostic ceilings remain.

Client unmounted before finish. Fresh changed-source PGmulti host shutdown2.690s/exit-15/ASGI complete,1passed526.33s/outer528.072s; official fixture released. This host count does not accept hidden visibility or remaining modes. Installed Playwright source explicitly enables focus emulation on its main connection, explaining why another CDP session cannot establish a real hidden oracle by merely minimizing. Safe alternative under investigation: disconnect the CDP control client (installed connectOverCDP close only closes its transport), use raw CDP on the same owned tab for real visibility and reconnect afterward. No Chrome process/window/tab restart or fake visibility is intended. This operator correction has not run.


## Remaining single-user variants

A distinct fixed-prefix adapter admits only the two genuinely unrun SQLite-single/PG-single interruption variants; same official fixture where applicable, fresh profile refusal,120startup/600host/15totalstop with5request-drain. Two structural checks passed0.27s, Ruff0, static Bandit0/errors0; independent source review CLEAR. Prepared browser sequence uses one real stream until visible nonempty partial, cancels only the owned request, requires genuine network cancellation/Interrupted, canonical question/partial/draft reload, one Retry and final canonical identity/parent/no duplicate. Original SQLite-multi diagnostics are excluded. SQLite-single fresh host started once; no original runtime restore between the two fixture cells.

Real hidden oracle correction is prepared but unexecuted and under exact-source review: installed Playwright main client enables focus emulation; connectOverCDP Browser.close only disconnects its transport. Temporarily disconnect that control connection, assert exactly one existing page/current URL via raw CDP, minimize/read actual hidden then restore/read actual visible with same draft/focus/scroll, reconnect to same existing context/tab. This creates no browser process/window/context and fakes no visibility or product data. Previous emulated observations remain unqualified.


### Single-user calibration and driver recovery

Fresh SQLite-single API14025/owner13933 ready9.454s. Normal Test Connection succeeded with persistent success/Core reachable/RAGhealthy; transient-toast observer missed15s, no second connection call. The first1014 generation finished its real network response before visible partial, so unfinished-wire assertion failed before cancellation or Retry. Question9f67/answer0288 and draft1015 preserved; no interruption pass or source causation assigned.

ONE declared same-cell recovery configures URL-scoped real download512bytes/s, latency0/offlinefalse/upload unchanged, retains all busy+visible nonempty+unfinished-wire/realcancel/Interrupted/reload/oneRetry/canonicalidentity/draft asserts and removes the rule before Retry/finally. No fabricated provider response or larger budget. Recovery running once. PG-single will use this calibrated real-transport control for its genuinely unrun first interruption variant rather than repeating the now-known inadequate timing control.

Raw-hidden observer correction had independent review: Important fetch/body deadline and finally coverage fixed, Minor late visibility pass prevented with remaining-budget reads/deadline asserts, no execution before fix. First exact raw check then failed before minimize because actual document already hidden, preserving the truthful gate. ONE declared driver recovery adds same-owned Page.bringToFront and actual visible<=2s gate, with no simulated visibility/new window. Narrow correction and real throttle delta reviewed CLEAR; earlier failures not accepted. Recovery prepared/unexecuted while interruption runs.


### SQLite-single remaining-cell outcome

The one interruption recovery is consumed and UNQUALIFIED: configured real-response rate512bytes/s, one request/no networkfailure/unfinished-wire observation at timeout, but required busy+visible nonempty gate never satisfied within180s. Later UI showed actual1017 answer and draft1018, then Stop control count0. No cancellation or Retry was performed; no source failure cause is assigned. Original first failure and this recovery are preserved; no unchanged third generation authorized.

Corrected raw visibility recovery did observe actual visible→hidden→visible, exact1019draft/focus and scroll1182 preserved, one existing window/tab and transport-only disconnect/reconnect. This is real client-state evidence, not a full host pass. Host work deadline expired before finish (finishBeforeDeadline=false); actual shutdown5.024s/exit-15/ASGI complete, host1failed574.66s/outer575.673s. Late finish does not change browserCellCompleted=false. Same private profile/data kept; original API remains stopped while remaining sequential fixture work is prepared.

Before the genuinely unrun PG-single interruption case, inspect whether the observer’s aria-busy requirement matches the actual content-rendering contract. Do not weaken an assertion without source evidence or replay the consumed SQLite control. Actual network unfinished/cancelled state, honest partial/draft, original-question reload and one successful Retry remain required. No A03 closure/A04/rebase/publication.


### PostgreSQL-single foreground prerequisite

Exact Message487/2129 and PlaygroundChat1607–1627 contract reviewed independently: latest assistant busy flag is correct while streaming; no selector weakening or source failure cause. PG-single first case uses normal network after corrected raw real-visibility precondition, superseding the earlier anticipated512-rate calibration. Consumed SQLite outcomes remain unchanged.

Fresh official API2675/owner2169 ready16.067s; normal private-key/Save/Test Connection returned persistent success. Corrected raw actual visible-hidden-visible with1020draft/textarea focus preserved, exactlyoneexistingwindow/tab; this short empty Chat had no overflow, so scroll preservation is not claimed here. No emulated visibility or new process. First genuinely unrun1021 interruption check now active once, same prepared truebusy+visiblepartial+unfinishedwire/cancel/Interrupted/reload/oneRetry/identity oracle and original caps. Original SQLite profile remains stopped pending completion of sequential fixture cells; source/data/keys/build/Chrome retained.


### PostgreSQL-single interruption result

Real normal-network1021 interruption/reload/Retry PASS in Chat46522c64: visible partial35.406s while latest assistant busy and wire unfinished, actual cancellation/Interrupted and partial/draft1022/originalquestion5730 reload, one Retry57.337s and final answer252f/canonical2/samequestion-parent/no duplicate/draft/no Interrupted. Exactly2generation requests. No timing-cause attribution to the earlier SQLite outcomes.

Actual hidden/visible1023draft/focus/overflow zero-position preserved. A separate genuinely unrun nonzero-position check scrolls down600 and requires measured top>0 before minimize and exact same top after actual return; it passed, onewindow/tab. Current real-provider formatting prompt requests tiny synthetic reason-tag/check and finalmarker to exercise actual returned markup/UI separation; it does not assert internal reasoning quality or replace live provider output. Scope only if actual wire contains the tags/check/final. Await actual result and canonical/native completion before any acceptance or fixture finish.


## Nested reasoning separation repair

**Observed scope:** Actual formatting-control UI rendered reason-related closing-tag/meta text outside multiple reasoning panels. Browser raw-body access was unavailable and canonical reload missed15s; no full live repair/canonical cause claim. The PG-single host expired during this extra check: browserCellCompleted=false, actual shutdown7.423s/exit-15/ASGI complete,1failed577.32s/outer578.607s. Existing earlier passed variants are scoped, not a whole host pass. Client unmounted afterward; fixture released.

**Design:** Existing recognized tags think/reason/reasoning/thought need matched nesting. Reuse one private raw segment splitter in the same library; public parse output/type remains, trim at existing parse boundary, preserve exact whitespace in removal/copy, and keep unfinished reasoning out of final-only output. Root scanner handles renderer, final-only copy/TTS/history/search and optional HTML formatting consumers. Preserve aliases/case handling, normal prose, original functions and tests. No new dependency/config/API/storage schema, no fabricated provider bytes or model quality claim. ADR025/049 govern; no new durable architecture rule.

### Repair stage1: Causal regression
**Goal:** Demonstrate nested trace escaping final/text classification and final-only cleanup.
**Success criteria:** Minimal deterministic recognized-tag nested/mixed/unfinished inputs fail on unchanged library; original tests retained.
**Tests:** Library nested classification, reasoning-only, removal/whitespace/HTML and real Message integration where applicable.
**Status:** Complete

### Repair stage2: Shared minimal correction
**Goal:** Matched nesting in one existing library, all affected consumers retain their contracts.
**Success criteria:** Causal checks green, affected configured UI/WebUI tests/types/lint and independent actual-source review clear; TypeScript-only Bandit N/A.
**Tests:** Relevant library/message/clipboard/history/TTS/RAG consumers and configured typecheck; no replay of unrelated live controls.
**Status:** Complete

### Repair stage3: Changed-source live boundary
**Goal:** Necessary reviewed-source build and one supported live nested-format/separation qualification.
**Success criteria:** Real observed provider output, correct final/trace boundaries/reload/draft, exact runtime source/build ownership; final actual/staged reviews and larger A03 gates remain truthful.
**Tests:** Only genuinely affected current source, normal hooks; no new installs/window/agent, no unchanged exhausted SQL interruption or arbitrary600 cap increase.
**Status:** Not Started

Additional multi-variant host launches STOP after repeated deadline failures until a concrete controller phase-budget/cleanup correction is prepared and reviewed; no larger600/180/15 budgets or silent attempt reset. Original same-profile runtime restoration is separate ownership/readiness, not acceptance or a case replay.


### Nested parser qualification

23new regressions red on unchanged source1.40s, including real Message final-input tail leak. Shared depth scanner green full original/new library+Message63passed3.94s; affected TTS/title/history38passed2.40s in separate UI run (101 across6files, not one combined run). Configured WebUI same6files101passed3.11s; clients overlap. Original tests/fixtures/asserts remain, inherited storage/title diagnostics retained. Public helper APIs, aliases/case/plaintext behavior, raw spacing and HTML policy reviewed independently CLEAR. Unfinished/nested trace now excluded from final-only cleanup; normal renderer keeps outer trace together.

Declared configured8GiB WebUI typecheck0. Proper ESLint actual files0errors: library/test0warnings, Message2identical inherited warnings; initial outside-base ignored invocation was not qualification. Inherited Next pages-discovery diagnostic retained. First formatter tooling invocation used unavailable UI-local binaries; installed frontend tools reused without installation. Two AST-comparison operators overcompared syntactic/source-file whitespace; no edits occurred then. Final changed-block formatter used structural AST excluding parentheses/source-file text, preserving all literal/expression structure, threefiles semantically identical after formatting. No unchanged test replay. TypeScript-only Bandit N/A. Third formatting attempt consumed; no repeated formatting experiment.

Original SQLite API restored once under apiA03AfterSingleCells: API22088/owner22085, canonical200/statusready8.930s; same profile/data/keys/source/Python3.12 environment. This standby readiness does not accept parser UI or UAT. Chrome remains one existing blank tab; old frontend/build retained. Next necessary reviewed-source build uses fresh dist/600s cap/existing dependencies and nine exact runtime source bindings (previouseight plus shared reasoning library), then restores functional source bytes. No install/model/window/agent/publication or unchanged live control replay.


## Residual orphan-delimiter source unit

**Actual evidence:** First necessary parser build failed defaultheap OOM; its single8GiB recovery compile/token/bundle0/0/0,119.315s,154pages/shared595.1KBgzip, build9HdzQPsBwSU1FiJ3b_mQB, ninebindings restored. Frontend459/owner446 startup0.513s after original57113 stopped0.103s; API22088/profile/keys/oneChrome retained. Ordinary original Alice2/user verified normally.

One actual real response200/request1/observererrornull/nativecompletion12.944s, typedreasoning present, saved Chat96c78c95 questiond1e4/answeref88/canonical2/sameparent/draft1030 afterreload. Canonicalcontent exactly equals real content channel: oneopening/twoclosingthinktokens. Parser now keeps check inside onepanel and finaloutside, but extra closingtoken still rendered outside. This is a partial live result, not full A03 reasoning acceptance. Raw stored payload is preserved. No private reasoning text is printed/recreated as proof.

**Minimal correction:** Recognized exact closing aliases are formatting delimiters even when orphaned; strip only outside-reasoning text segments, preserve normal/unrecognized markup and raw spacing. Bare known delimiter-only output gets the existing honest no-final recovery. No producer/storage/API/dependency/ownership rule changes; no newADR under025/049. Fouraliasparser+fourcleanup+fourno-final+unknownmarkuppreservation+twoactualMessage cases red14failed/1passed/63skipped1.84s. Original tests/mocks/asserts retained. Green affected6UI116passed3.16s, sameconfigured6WebUI116passed3.21s (overlap); configured8GiBtypecheck0. Independent complete delta/new15case/connectedconsumer review CLEAR. Changed-block normalformatting semanticASTidentical. Properlint/hooks pending finalreadback; TS-onlyBanditN/A. Two causal attempts used; no replay aftersuccess.

**Next:** One necessary materially changed orphan-fix build, same existing dependencies/known8GiBcompiler resource/600cap/freshdist/nine exact bindings restoredfinally. Normal ownedfrontend replacement15s/ONE120startup, API/profile/Chrome retained. Livequal will reload this existing actualreply under its same ordinaryowner and verify no closingtag/final-only behavior/samecanonicalIDs/draft; no modelresend or rawdata rewrite. BroaderA03/SQinterruption/PGmultihidden/currentdev/release gates remain.


### Orphan build/runtime result

Normal materially changed build compile/token/bundle0/0/0,113.155s under600,154pages/shared595.1KBgzip/routechat795.1; build5wSKJiyylexjHiVcMAulC, ninebindings restored. Source/type/review evidence remains; build type validation skipped is inherited and disclosed. Normal hooks initially normalized an extra EOF newline only; rerun that one file passed, no unchanged tests replayed. Properlint0errors/library-test0warnings/Message2sameinherited; pages-discovery diagnostic retained; TS-onlyBanditN/A.

Original frontend459 was guarded and exited after oneTERM. Plain socket preflight then rejected bind before any startup was marked/executed; actual listener inventory found none, former PID absent, project-style SO_REUSEADDR guard passed. This is an operator preflight correction, not a second startup or proof of a specific socket-state cause. ONE still-unexecuted startup then frontend17967/owner17963/newHTTPbuildIdready0.877s. API22088/source/privateprofile/data/keys and sameoneChromeretained; no APIrestart/model/install/window/agent action. Existing savedreply96c78c95/d1e4/ef88 is being reloaded under changed build; no modelresend/rawstorage rewrite.


### Orphan rendering reload and final controller review
The materially changed build5wSKJiyylexjHiVcMAulC normal reload passed on the existing SQLite-multi reply: canonical two IDs/parent and raw stored content unchanged, one reasoning panel, final outside with no orphan close/check marker, draft retained. No model resend or stored-data repair.

Private operator final-boundary controller causal red demonstrated a failed-runtime write error skipped finish. Correction uses one authoritative finish-runtime JSON failed/errorName; host accepts only failed exactly false. Pure fake-fs tests cover success, workflow failure, unmount failure and marker-write failure plus whole-case budget boundaries; no application/browser/files in those tests. Two static host guards passed0.13s after correcting operator-only PYTHONPATH collection input; Ruff/Bandit0. Independent actual review CLEAR closes the Important finding. Fresh prefixes and600host/569work/120startup/180generation/2ack/15navigation-stop/5drain remain.

Prepared one-variant PG-multi real-hidden case uses normal fresh ordinary-user creation/login, one real reply, canonical reload and actual positive-scroll minimize/restore, finally unmount and failure-aware finish. SQL-single final interruption proposal uses240-line real response, actual foreground/busy/nonempty/unfinished network and real cancellation, partial/draft/question reload, one successful Retry with2s ACK and canonical two parent-linked messages. It remains unexecuted behind a specific final-attempt approval gate; old first/recovery failures are preserved. No A03 closure, A04, current-dev qualification, rebase or publication.


### Final PostgreSQL hidden case — actual completed host
Fresh official PG-multi API93739/owner93604 ready13.942s. Ordinary Alice id2/user created once through normal Admin UI, then actual logout/login/public identity verified. One30-line real reply Chat2268c030-8958-4b51-871c-c4046ad6e69a, questionpa_0e42-a212-977-ee89/answerpa_41e9-6625-dc6-abd7, canonical2/same IDs/parent after normal reload. Corrected raw CDP on the sole existing window established actual visible-hidden-visible, retained draft/focus and positive500pixel scroll. No emulated visibility, screenshot or new window/context.

Client unmounted and authoritative finish JSON failedfalse. Host1passed78.30s/natural0; actualshutdown2.842s/exit-15/ASGIcomplete/noovercap; official fixture released. Old deadline failures are unchanged. Top matrix now separates actual hidden checks from shared reasoning-code qualification, avoiding an all-mode live format claim. Final SQL-single exception question asked once against249, pending; no SQL flag/startup. Original standby restoration is separate same-profile ownership/readiness, not additional UAT acceptance. A03 remains InProgress; A04/latestdev/rebase/publication/release gates remain open.


### Acceptance accounting and first parser-format mode checks
Independent accounting audit distinguishes the catalog Chat variants from extra active-client shutdown stress. Every historical over-cap failure remains unaccepted; no old log/source/control replay or claim of a source repair follows. The current summary now follows the playbook owned-shutdown contract. Already-passed principal/history/Stop/Regenerate/rejection/picker/delay/real-hidden checklist entries reflect the matrix, while the pending SQL interrupted-stream cell remains explicitly unqualified.

Three fresh first live format checks are prepared for the materially changed nested/orphan parser: SQLite-single, PG-single and PG-multi. One controlled real provider reply per host, ACK2/generation180/canonical reload2/same IDs/parent/draft, controlled reasoning marker inside panel and final outside/no known closing token outside. No raw model trace output or stored-data rewrite. No nesting/alias/model cross-product or replay of original IRQ attempts. Same build5wSK/current source/environment/window; fresh exact a03-reasoning-browser prefixes, restricted official PG fixtures and failure-aware unmount/JSON finish. New static guards3passed0.24s; pure fake-fs admission/rejection checks passed, Ruff/Bandit0. Read-only prepared-script review pending before runtime. General600/569work/120startup/15totalshutdown/5drain and original final-SQL exception gate preserved.


### Consumed final SQL attempt and format selector diagnosis
Human explicitly granted the specific ONE final SQL IRQ attempt. Fresh API57574/owner57419 ready11.230s, but its normal mode/key setup failed before any Chat request: the driver waited15s for the exact fallback API-key placeholder or confirmation, while the actual translated field includes longer demo guidance. Authoritative failedtrue; host1failed136.10s/natural1; actualshutdown1.411s/exit-15/ASGIcomplete. This final attempt is consumed/stopped, not an interruption pass or remaining automatic allowance.

One normal unpersisted mode diagnostic on the existing frontend (API already stopped) showed radio switchedSingle, no native dialog/pageerror, old exactplaceholdercount0 and accessibleAPIKeylabelcount1/visible. Helper now uses the actual label. Candidate/functional English sources agree; no key/token/auth-store output or product sign-in bug claim.

First materially changed format SQLsingle API90055/owner89832 ready13.541s failed before Send at another fallback selector: Connection successful! is not the actual Server responded successfully. You can continue. text. Host1failed90.42s/natural1; actualshutdown0.610s/exit-15/ASGIcomplete. Source helper now matches actual locale and scopes the persistent status within the form, with absent-beforeSave freshness and strict15s acceptance, avoiding the identical toast. No prior connection success is inferred.

ONE format-only selector recovery is declared, distinct fresh a03-reasoning-browser-sqlite-single-recovery. Host requires the prior stopped failed first format metadata, preserves600/569work/120startup/180generation/15stop/5drain/strictfailedJSON; cleanup whitelist rejects PG recovery. Static1passed0.27s, purefake-FS whitelist checks passed/Ruff-Bandit0. It sends one format reply only and does not replay/reset the consumed SQL IRQ exception. First PG format cases remain unrun. Current API is stopped; frontend17967/Chrome10920 retained with one blank tab. No A03 closure/A04/rebase/publication.


### Completed PostgreSQL format checks and remaining SQL proposals

PG-single one real reply PASS: Chat1049c7cd-cf51-41a4-a904-6a8d6bc5a8bf, questionpa_92d4-98f4-056-0dd1/answerpa_df80-d6d2-98e-9975; canonical2/same IDs/parent/draft. One reasoning panel contains the controlled check, final outside, no check/known closer outside. Passive typed wire HTTP200/finished/DONE/settlementcomplete/count1, content245/reasoning369chars, content exactly equals canonical. Host1passed82.11s, actualexit-15/ASGIcomplete3.229s/noovercap before official fixture release.

PG-multi ordinary User2 PASS: Chat44a57198-7343-4fb5-aaf2-5a0c4a7f02d9, questionpa_e2c2-fea8-6f3-23c2/answerpa_20b5-2697-dce-2d7c; canonical2/same IDs/parent/draft. Two legitimate reasoning panels, same inside-check/outside-final/no outside-check-or-closer oracle. Passive typed wire200/finished/DONE/settlementcomplete/count1, content401/reasoning321chars, content equals canonical. Host1passed91.75s, actualexit-15/ASGIcomplete2.620s/noovercap before official fixture release. No raw private reasoning/credentials printed; no replay or all-mode/release claim.

SQLite-single format selector recovery FAILED after real generation and canonical2/identity checks, at display assertion. Host1failed150.25s; actualexit-15/ASGIcomplete2.723s. Supported owned CharactersRAGDB metadata/tag-shape inspection only found multiple ordinary controlled-marker segments plus encoded closers in the saved980char reply. No raw provider reasoning output. Model/backend/channel cause remains unproven; no speculative parser repair, arbitrary text removal or acceptance. Both SQL format hosts remain consumed.

Prepared proposals ONLY: one fresh a03-sql-replacement-irq host with the now PostgreSQL-validated normal API-key/status selectors, existing reviewed240-line actual stream/cancellation/partial reload/one Retry/canonical identity oracle; one fresh a03-sql-replacement-format host with unchanged format assertions and bounded passive typed-wire comparison. Each host requires a NEW separate explicit approval flag; old final/recovery approvals cannot authorize either. No execution so far. Preserve600host/569work/120startup/180generation/2ack/15navigation-shutdown/5drain, exact fresh-profile refusal, authoritative failed JSON and one existing window/tab. No A03 closure/A04/rebase/publication.

Prepared replacement guard checks2passed0.21s; pure in-memory exact cleanup success/failure/foreign rejection and four invalid format combinations pass without application/browser/file writes. JSsyntax0, operatorRuff0/Bandit0findings0errors; independent connected reviewer CLEAR/no findings. Specific exception question for both concrete remaining SQL checks asked ONCE; awaiting human answer. No API/browser/model attempt executed, no reused consumed approval. PostgreSQL fixtures released; existing frontend/one blank Chrome tab retained, API currently stopped.

Human approved BOTH prepared SQL checks explicitly. IRQ fresh host started ONCE: API67005/owner66922, canonicalready7.687s; actual owned browser sequence now active. New per-kind flags do not reset any older cap or authorize additional attempts. Format follows only after IRQ host actually exits; no other workflow or runtime draw added.

Approved IRQ replacement CONSUMED/FAILED: normal key/status setup and actual busy/visible partial/unfinished network checks passed. Actual cancellation was ERR_ABORTED/canceledtrue; Interrupted/partial/draft/original-question canonical reload passed. One legitimate Retry acknowledged busy within2s, but completed-boundary assertion remained false within180s; failure location a03FinalSqlInterrupt column3857. No final completion, source/provider cause or whole-case pass claimed. Host1failed308.14s; actualAPI67005 exit-15/ASGIcomplete2.810s/noovercap. Initial REPL driver binding error occurred before invocation/browser action and was corrected within this same owned host; it was not another startup or model draw. No further IRQ attempt authorized.

Separately approved format replacement started ONCE only after IRQ shutdown: API73914/owner73797, canonicalready10.483s. Same sole tab/build/source, normal single-key form and passive typed-wire observer; one real format draw now active, unchanged assertions and caps. No A04/rebase/publication.

Approved format replacement CONSUMED/FAILED: Chat8c068f7c-df08-4207-a803-59eb625e57df, questionpa_a649-e8cc-f71-87c4/answerpa_3721-613a-a4d-0752; canonical2/same IDs/parent/native completed. Four reasoning panels with controlled check inside and final outside, but controlled check and known closing-tag text also outside. Passive wire count1/status200/enabled/DONEtrue, loadingFinishedfalse after15s settlement: unqualified/equalitynull, content1756/reasoning328chars. Raw private trace not printed. Host1failed85.55s, actualAPI73914 exit-15/ASGIcomplete2.672s/noovercap. No additional format draw authorized.

NEW saved-reply diagnosis used supported CharactersRAGDB getters only: canonical2/content1756chars, two controlled CHECK markers outside balanced literal reasoning, three HTML-encoded known closers outside literal reasoning, one orphan literal closer. Actual Message maps parseReasoning text segments through Markdown; saved shapes are consistent with observed outside text, not proof of another renderer bug or upstream source attribution. Local adapter's own DONE break/context cleanup already exists; no demonstrated cancellation/timeout repair inferred. Initial broad source-search display truncated and not accepted as full coverage; only exact affected bounded sections were recovered. Playbook166/168 preserves incorrect model-format results as failed live quality; no reclassification or assertion weakening.

Original standby restored ONCE after both hosts exited under apiA03AfterSqlReplacements: API78766/owner78763 canonicalready11.767s, same original private profile/data/keys/source/Python environment/build. Existing sole Chrome tab normally switched back to Multi User and restored actual public Alice2/user at /chat. Initial username wait under retained Single User mode timed out before credential submission, then normal mode selection corrected it. No new model draw/window/context/install/build/publication. Both specific SQL approvals CONSUMED, none pending; A03 remains OPEN at failed Retry completion and live format, A04 not started. Do not repeat these attempts or present old approvals as unanswered.

Independent latest-outcome accounting review found no material overclaim; two Minor current-summary/row labels corrected to explicitly FAILED and both approvals CONSUMED. Review was prose/accounting only, not independent runtime verification. Task canonical check and owned task/plan whitespace pass. A03 remains OPEN; no source repair/acceptance inferred from this review.


## Authorized root-cause effort — 2026-10-13
Human explicitly authorized necessary diagnostics/retries and both fixes, validating A03 before A04. No additional approval question within that granted scope; unrelated STOPs remain. ADR021/025/049/050 unchanged; no new architecture rule. Up to3 distinct declared causal hypotheses/attempts per issue before reassessment; no automatic loops or budget increases.

**Confirmed operator defect:** direct background-proxy1861 returns on genuine SSE DONE, then reader.cancel1892. Old passive capture incorrectly required HTTP loadingFinished, which can lose the intentional-cancellation race. New operator regression RED (completeDONE/EOFfalse wrongly unqualified), GREEN after semantic terminal qualification; preDONE cancellation remains unqualified. A second RED proved captured content after firstDONE entered equality; byte-ordered terminal cutoff GREEN excludes it, including later feeds. HTTP completion/cancellation remains reported separately, never converted to workflow success. Private capture now reports only per-request headers/first-content/first-reasoning times, counts, cancellation/error enum and configured maxTokens; no raw trace output. Independent source reviewer supports this contract but does not retrospectively attribute the old event.

**Format diagnostic1:** old literalthink echo prompt asked the model to manufacture formatting, which tests inline-tag instruction quality rather than actual typed-channel separation. Source TldwChat/ChatTldw intentionally forwards content tokens while treating typed reasoning as progress. Existing failed echo outputs stay failed. Natural17×19 fact prompt requires323/no contradictory result, actual typed reasoning present, full captured content exactly canonical, private long reasoning body absent from canonical, stable two parent-linked IDs, draft/reload and no reserved tags outside genuine panels. No fabricated reasoning panel requirement on this content-only native path. Existing parser/nesting/orphan causal tests and actual SQLite-multi inline reload qualification remain intact.

**IRQ diagnostic1:** retained failed canonical DB has only original questionpa_2ecf-12f0-6b4-b725 in Chata68acb0d-526c-47be-83b3-c74f589aff75; native metadata has pending1/no completed outcome. Not merely a missed complete observer. Cause still UNKNOWN. Prepared observed40-line workload, explicitly changed from240 (catalog requires sufficiently long, not240), preserves actual visible/busy/unfinished/cancel/Interrupted/partial/draft/question/oneRetry/canonical/no duplicate/180 oracle and captures first/Retry stages separately. Whole two-generation+observer budget required510s before action;600host/569work/120startup/15ownedstop unchanged. A shorter pass alone will not retroactively prove the240 timeout cause.

Only fresh a03-rootcause-irq-20261013 and a03-rootcause-format-20261013 profiles admitted behind new explicit root-cause authorization flag. Normal existing single-key form and sole Chrome tab; same frozen source/build/env. Prepared operator static2passed, Ruff/Bandit0errors/findings, JSsyntaxclear; awaiting connected script review before live execution. No product source patch, build or new model draw yet in this effort.

Preparation findings corrected and independently read back CLEAR before execution: ordered pending bytes/failure markers (both causal race orders checked), no raw IRQ session (only bounded taps, allsetup undercleanup, originalfailure preserved), cleanup-reserved510 admission, and strict numeric-firstline/two-line mathematical verification with wrong/negated/extra/tag controls. Metadata in-banderror and invalidJSON remain unqualified; numeric completion-token count/allowlistedfinish reason now observed privately. Operator test VM TextDecoder omission was corrected as a test setup issue only; no app workaround.

Original owned API78766 guarded/unmounted then exited-15 in2.843s, no force/owner signal. Fresh IRQ diagnostic1 API28074/owner28025 canonicalready7.668s; reviewed bindings verified before actualcase, same oneexistingtab/build/source/env. Actual observed40-line interruption/Retry sequence started ONCE, no result or rootcause claimed yet. Format diagnostic follows only after actual IRQ host exit. Human authorized necessarydiagnostics/retries; no new permission pending, no A04 yet.

Genuine typed-channel format diagnostic1 actual PASS: Chat32b4d212-2266-405a-8f06-110fa1c22cb3, questionpa_2f85-c0ae-a24-5390/answerpa_1bee-1c5d-104-8590, canonical2/same IDs/parent/draft. Realreasoning177chars/content17, exactcapturedcontent===canonical, correcttwo-line323+verification, nofinaltags/panels0 (nativecontent-only). Capturecount1/200/DONE/qualifiedtrue/finishstop, headers10.419s/reason10.427s/content11.488s. Host1passed65.63s; actualAPI32985 exit-15/ASGI2.402s/noovercap. All old fake-tag echo quality failures remain failed and are not retro-pass; this is actual channel separation, not artificial inline formatting.

IRQ diagnostic1 failed BEFORE Send at reserved510 admission; host1failed76.97s/API28074 exit-15/ASGI0.752s. The prepared phase had waited too long afterreadiness; no model draw. Necessary declared recovery uses exactfreshsuffix/stopped-failedprecondition, unchangedcaps and immediate functional-metadata readiness activation, not a budgetincrease. Staticrecovery1passed0.23s/connectedreviewCLEAR.

Observed40-line IRQ recovery actual PASS: Chat50e1ec11-4978-4613-ad2b-cdf083a58be8, originalquestionpa_bc18-4fa8-dfd-d863/finalanswerpa_45f4-6c2e-dc3-d2a6, actualpreDONE ERR_ABORTED/canceledtrue, Interrupted/partial/draft/originalquestion reload, oneRetry/nativecompleted/canonical2/sameIDs/parent/no duplicate. Firstvisible23.143s, Retry28.984s. Firstrequestreason2934/content41chars beforeintentionalcancel; headers0.446s/reason0.449/content22.497. Retryreason2809/content2285, headers0.255/reason0.257/content21.968, finishstop, capturedcontent===canonical. Crucially retryDONEtrue/HTTPfinishedfalse/canceledtrue/failureAfterDonetrue yet semantic capturequalifiedtrue: actual live example validates the corrected intentional postDONE-cancellation observation. Host1passed70.82s; actualAPI40138 exit-15/ASGI3.014s/noovercap. This run validates recovery contract; it does not by itself prove the old240 timeout cause.

ONE final observed original240 workload comparison prepared under existing humanauthorization (thirdrootphase IRQhost including the firstnoSendpreflightfailure). Fixednewbaselineprefix requires stoppedSUCCESSFUL40 recovery, same510/600 caps; only40/240 workload allowed, conflictingprofileflags rejected beforeanyaction. Staticbaseline1passed0.41s, purecleanup/workload40+240/reject60+conflictingflags pass, connectedreviewCLEAR. BaselineusesoriginalA03-REPLACEMENT-INTERRUPT marker and allreal cancellation/oneRetry/identity assertions, per-requeststagecapture; no reset or automaticfallback. Actualbaselinehost started once; waitforreal result/cause before A03closure/A04. No new product patch/build/publication.


### Actual visible-output watchdog repair and build — 2026-10-13
Original240 observed comparison PASS: Chatae9183de-658e-428a-bd06-a2f958b3fee3, questionpa_99d0-9c91-95a-9e8b/answerpa_293f-1dc5-dca-8538. Firstvisible78.517s/Retry102.791s; original question preserved/canonical2/parent/draft/count2. Genuine typed-reasoning delay77.398s first/96.596s retry; content captured at semanticDONE exactly canonical despite intentional postDONE reader cancellation. Host214.63s/natural0, exit-15/ASGIcomplete3.410s. This positive comparison does not prove the historical180 failure cause.

Source confirmed TldwChat counted hidden typed reasoning as visible while ChatTldw emits content-only and Settings explicitly promises visible-output startup/idle deadlines. Four real service/browser-transport tests cover provider/legacy hidden-reason startup, answer-then-hidden idle, and answer within deadline control. Causal3failed/1passed; minimal helper deletion/content-only predicate then all affected30UI+30WebUI passed, overlapping. All4actual callers independently reviewed CLEAR, original14timeout tests unchanged. Proper shared lint0errors/2identical inherited any warnings; changed-block formatting semanticAST identical, inherited wholefile format flags remain. Configured TypeScript baseline/current27 identical inherited errors; no new/changed-file diagnostics, no whole-project TS-pass. TS-only product BanditN/A; Python operator scopes0findings/errors.

Reviewed normal production webpack/token/bundle buildEWZTMgJKv6iML-QeE3PUs134.371s/all3natural0; inherited skipped typechecking disclosed. Temporary10exact candidate runtime bindings restored. Runner metadata/group/deadline cleanup findings repaired before execution with pure causal operator checks. First frontend start omitted quickstart deployment mode and exited1 before readiness; actual networking-validator red/green proves operator setup cause. Same-build corrected recovery frontend12946/owner12941 HTTP200/actualbuild-IDready0.649s. No rebuild/install/newwindow/provider/model action.

First watchdog host setup used primary pytest config and correctly rejected installed-module source binding before profile/API creation. Explicit candidate config corrected only operator invocation. Fresh host17923/17842 canonicalready8.449s; actual15s watchdog canceled15009ms after matchingPOST,1644hiddenreasoningchars/content0/DONEfalse/netERR_ABORTED. Driver wrongly looked for timeout inside assistant bubble; native recovery accepted_unsent already contains exact friendly startup guidance,128characters. HistorySelectionReview correctly renders rolealert and Reviewretry/Retrysavedquestion; no guidance defect established. Host failed48.64s, exit-15/ASGIcomplete2.418s. Original timeout restored normally finally. ONE fresh -recovery observer now follows real alert/review flow, verifies13–17s cancellation and boundedUI, preserves question/draft/reload and then one real Retry under180 after exact original setting restore. It requires stoppedfailedprior/buildready+restored+served, same600/120/15 lifecycle; A03 remains InProgress until actual result.


### Corrected native-review observer recovery
Source-only reviewer CLEAR for rolealert/native recovery, current-model/saved-question context review and strict timing/canonical oracle. New recoveryguard1passed0.24s. ONE -recovery host now uses original240-line workload (changed literal40→240 to exercise sufficiently long real hidden reasoning, no budget increase). API27249/owner27143canonicalready9.197s on18906. Generation/reload/actualRetry results pending; first stopped failedobserver retained, no extra windows or rebuild.


### Genuine empty-answer sample and final reassessment
Second watchdog host initial timeout/guidance/Q-only/draft/reload/context-review/restored-setting passed: canceled15009ms,3278typedreasoningchars/content0. The same240-question Retry genuinely completedSSEDONE/finishstop59.146s,13027typedreasoningchars/content0, so native Chat correctly removed its empty assistant stub and retained accepted_unsent with94-character empty-response guidance/no completed outcome. Driver positive completion failed; host111.88s/exit-15/ASGIcomplete2.317s. No forged final, reasoning promotion, weakening or renderer/upstream-budget-cause attribution. Actual LlamaCpp adapter forwards normalized data frames; sse normalization preserves data payload; native pipeline967/980 explicitly rejects an empty final. This remains a failed provider-output sample, not a qualified successful Retry or demonstrated new display defect.

Reassessed ONE final distinct40-line workload, which previously supplied actual long response/real reasoning/answer. Exact fresh -final requires stoppedfailed-recovery; conflicting recovery+final flags refused;600/569/120/15 and15s configured deadline/13–17 cancellation/one realRetry180/canonical2/Q-parent/draft/capture equality unchanged. Independent source review CLEAR; new operatorguard1passed0.26s. No fourth automatic validation after this final. Current result pending, A03 remains InProgress/A04 not started.


### Final retained-answer readback and catalog acceptance — 2026-10-13
Final host40564/40524ready9.179s used40lines. Actualfirstanswer14186ms arrived BEFOREconfigured15s; completedSSEDONE/finishstop with1897contentchars/2236typedreasoningchars/onePOST. Its mandatory-timeout-only observer was therefore the wrong oracle and failed; host49.51s/exit-15/ASGIcomplete2.520s remains failed. No fourth generation/Retry/timeout experiment occurred.

ONE same-retained-final-profile read-only API48077/48074ready9.431s allowed normalUI inspection of the already completed Chatb5c28837-82bb-47e6-b35a-01791c4c145a. Questionpa_806b-4d86-e92-12a6 and answerpa_c35a-c783-4e1-03ce: canonical2, correctparent/content1897/nofinalreasoningtokens, sameIDs/parent/fullcontent after actualreload/draftretained. This qualifies the within-deadline positive; it does not invent passive full-content equality or turn the failed original host into a pass. Clientunmounted; one guardedTERM/exitobserved0.814s, monitoractual-15/shutdowncomplete. Original standby restoredONCE via new apiA03AfterWatchdogValidation20261013, supportedinstalledmodules/noapp-dir-PYTHONPATHshim, same privateSQLite-multi data/keys/source; API52506/owner52501canonicalready8.635s. Correctedfrontend12946/owner12941 EWZ and oneChrome10920 retained.

Independent source/accounting audit finds no further REQUIRED A03 catalog functional variant missing. Playbook235–241/catalog cases do not require configured-timeout→successfulRetry in one fresh host. Composition is explicit: existing pre-watchdog40/240 realinterrupted/reload/same-questionRetry passes + changedsharedtimer causal/affected/sourcechecks + new actual negatives and saved/reloaded within-deadline positive. Human UX already accepted. Selected catalog functional A03 CLOSED; all failed diagnostics/quality samples and historical exact causes retain their original status. Global45s nonstream and C01linked journey/publicD-L-U/native/installer/currentdev/rebase/fullTS/combined-staged-source/hook/commit/publication gates remainopen. Keep this owned plan for engineering closeout rather than deleting its unfinished Stage3. A04 may now start sequentially.
