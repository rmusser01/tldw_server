# Cycle 3 multi-user UAT results

Product frozen at d40e17dc81. Fresh initialized multi-user profile; migrations98 and bootstrap admin existed before this run. API18201 / WebUI18281; real local llama.cpp9099, exact Qwen model. Execution is complete with failures, dependent blocks, and the explicit mixed-deck coverage limit below; this is not an all-pass acceptance claim. Browser actions stopped at 2026-09-15 11:14:21 UTC. No application/test repairs during this run; the isolated build-cache configuration exception is documented below.

Runtime started 2026-09-15: backend PID31487 (exec session77186); frontend PID31535 (exec session94173). Private logs/config remain outside retained evidence. Initial disk availability3.6GiB.

| Step | Workflow | Final outcome |
| --- | --- | --- |
| 1 | Fresh setup, admin login, provider UI, real first Chat | Executed with failure077: Provider Keys loading crash. Documented isolated operator configuration prerequisite used; actual model picker + real first Chat passed. UI provider configuration is not claimed. |
| 2 | Admin UI creates Alice/Bob ordinary Users | Pass: both UI-created with role User; reserved .test email rejected, corrected through UI to synthetic example.com. |
| 3 | Alice normal Saved Chat, two turns, reload/server persistence | Fail062: correct answers retained, but first pair split/duplicated across normal and tracked conversations; mode flip and persist400/fallback201/runtime overlay. |
| 4 | Public Cedar ingest/search, Media Chat, default QA/citation/source; confidential negative | Positive and negative controls passed with UX recurrences056/057/060/066. Canonical llama.cpp retry savedCedar1; contentsearch, MediaChat, default specific-source QA, citationpreview and actual Mediajump passed. Specific confidentialIndigo2 excluded1/retained0, noanswer/source. Initial underscore provider was a harness deviation. |
| 5 | Exact Wikipedia URL, search and grounded Chat | Fail065: exact URL extraction failed, stored_articles0. Dependent Wikipedia search/groundedChat blocked; no alternate URL or denial bypass. |
| 6 | Biology Note → five generated Flashcards → save/study/reload/backlink | Bob executed real default-provider generation, inspected all5groundedpairs, saved/linked, reviewedonce and corroborated schedules/session afterreload. Actual source-note link fails081; Studydisplay083/084. See bob/REPORT.md. Mixed deck+undecked076 control was not executed in multi and is not claimed passing. |
| 7 | Pirate Prompt create/save/Back/apply → actual saved Chat request/reply | Pass for Prompt flow: exact systeminstruction/request699, realARRRreply, reload retained. Existing062modeflip recurred afterward. |
| 8 | UI characters, tracked reply, active replacement, old history/settled reload | Direct UI-createdCedar4 andRobot5 replies/persist passed. Fail068activepickerreverts; fail070Cedar2serverrows vs1visibleafterreload; fail085Note-backlink restoresrows but wrongactivecharacter and missing saveactions. |
| 9 | Tracked reply → Note/backlink and reviewed Q/A Flashcard → study/reload | MediaChat groundedanswer Note clean/backlink and reviewedQ/Acard65e74e34… saved/studiedGoodonce/reload/server version 2 passed. Card-sourceNote link fails081. DeliberateCedar ownNote saved; Cedar→Flashcard/study blocked085missingloaded-messageactions, no workaroundgeneration. |
| 10 | Media analysis → Review → grounded second analysis → reload | Pass: Reviewfull-contentsearch and exactcontent; distinct realsecondanalysis request215, persistedversion2 withversion1retained; independentGET andUIreloadconfirmed. |
| 11 | Ordinary delete permission explanation; admin-owned delete/Trash/date/restore | OrdinaryAlice disabledDelete +explicitpermission explanation passed. SeparateBob runner admin-ownedsoleMedia1delete/restore/date passed; emptylibraryTrashnavigation071 anddeleteconsole072 recur. See bob/REPORT.md. |
| 12 | Two-tab offline logout/draft recovery, Alice/Bob auth/API/browser/QA isolation | Offline048boundary/reconnect/A→B→AqueuedNote recovery and independentforeignNotes/Media404controls passed. APIoriginals unchanged. Fail080naturalrefresh,086cross-accountQAquerymetadata,064fullprivateNotehandoffbodyvisibleunderBobviahistory. BobdraftdidnotreplaceAlicedraft. No remote-revocation claim forofflineLogout. |

## Findings

- UAT077: Provider Keys route crashes during loading with ICU `res.replace is not a function`. Separate GET users/keys403 is expected: BYOK is disabled. UI and console evidence retained in temp.
- UAT058/059 recur on fresh Home: blank title and Reading Queue unavailable copy.

## Setup prerequisite and progress

Multi-user wizard intentionally offers an operator guide/sign-in exit. API setup editor exposes llama_api_IP but only a comment for llama_model. Followed the documented Local LLM configuration prerequisite: modified only API.default_api=llama_cpp, Local-API.llama_api_IP=http://127.0.0.1:9099/v1, and Local-API.llama_model exact Qwen3.8 path in the isolated0600 config. This is not a successful UI provider-setup claim. Whitelisted before/after recorded in uat-cycle3-multi-operator-provider-config.json. API restarted as PID37158/session83644; database/users retained.

Admin Chat selected the exact model through the normal picker and returned MULTI CEDAR READY. Request196 POST /chat/completions200 used the exact llama:Qwen model, stream=true, save_to_db=true. Model defaults initially lagged the catalog after operator restart; verifying visible Refresh, no new defect assigned. Full acceptance remains pending.

## 09:58 checkpoint

- Independent Alice GETs200: normal chat7ab86ad0-6e36-4fa3-b1e6-61ca50ef8f00 has first user/assistant pair; trackeda68ad772-d28d-43f1-b948-2b97119788e0 has system+bothpairs. First pair duplicated across conversations, not twice within tracked chat.
- Cedar first analysis entered noncanonical llama_cpp from handoff, yielding savedMedia1/Warning missingmodel. This is a harness input deviation, not a canonical provider regression. Actual UI suggestion llama.cpp selected on unchanged-file retry with explicit overwrite; job2Success after87s, sameMedia1/UUID.
-056lateprovider validation and057~3sec estimate recur. Failed-attempt generic output hides actual chunk warning (065partial-failure-copy observation).
- API.default_api normalized from the handoff alias to actual offered llama.cpp, with separate whitelisted evidence and no base/model changes. API restart42500/session79083 coordinated after both agents completed mutations.

## 10:16 checkpoint — paused for environment repair

Public QA used Documents & Media only, specificMedia1, Server default, webfallbackOFF. Real result correctly names JonahPatel/ninereadingtables/northentrance, onecitation. Sourcepreview returns exactCedartext; typeOther while sourcecardDocument recurs066. The OpeninMedia action did not execute because npm failed beforePlaywright withENOSPC.

Tracked MediaChat answer persistence triggered062 runtimeoverlay (`speaker_character_name must reference a selected participant in this chat`). Dismissed usingEscape aftersnapshot/screenshot; fallback/messages201 retained answer. AnswerNote af0c68d3-69a1-4860-b369-4ccace5897ca version1 has only visible116charanswer, no reasoning; Openconversation returns actualtrackeda68ad772-d28d-43f1-b948-2b97119788e0. ReviewedQ/Acard65e74e34-b5f3-49c7-80fb-a59f135e1d65 saved201; endpoint also createdNotef9685841-15a9-4694-a438-a8559859458e.

Bob independentGET/validversionedPUT of AliceNote404. Subsequent independentAliceGET200 confirmsversion1/title/contentunchanged. BobownNote a80707b3-c761-4d8a-8bbe-fe2222886bc2 and ownMedia1UUID1430f1e4-37e3-473e-b6d0-764f8f922f8d are separate; numericMedia1 overlapsAlice1 by design. AliceMedia2 foreigncontrol pending.

Environment: first ownedUIcache recycle31535/31537→43042 at10:01 restored3.6GiB. At10:10 ENOSPC recurred; ownNextdist3.4G, dev/cache2.6G, static609M, server247M. Both agents paused with no model/save in flight. Parent stopped43042/43043 and removed only ownedmultiNextdist; APIs/data/browserprofiles retained. Parent is applying a documented config-only exception disabling dev filesystem cache for isolatedTLDW_NEXT_DIST_DIR runs; application/test product remains frozen. Browser stays online; no newmodelcall until coordinatedresume.

## 10:30 checkpoint — resumed, source controls complete

Parent resumed frontend48482/session52335 with the isolated development filesystem cache disabled. This documented build-configuration exception was committed c10e1464fc; functional application source remains d40e17dc81. Cache remained8KiB and free disk2.9GiB after the resumed Media/QA/Prompt routes. API42500 and all data/profiles retained.

Public QA recent-session recovery restored the real cited result without another inference. Source preview Open in Media created tab2; selecting it verified actual /media?id=1 and Cedar content. The earlier preflight ENOSPC interruption is preserved separately.

Exact Wikipedia Playwright URL attempt returned process-web-scraping200 with stored_articles0 and “Failed to extract” (065 generic-error recurrence). No article was persisted, so Wikipedia-dependent search/chat are blocked. Unchanged confidential Indigo fixture uploaded through UI with analysis/chunking off became AliceMedia2 (UUIDe5d254c0-5ff2-4b69-8f1e-6d7496872558), version1. SpecificMedia2 + Documents&Media only + Serverdefault/WebOFF QA returned contexts[], excluded_count1, retained_count0, output_emitted:false. UI explicitly says security settings excluded all retrieved sources and shows0sources; no answer or token text. This pairs with the successful Cedar positive control.

At10:19 the normal browser session expired and refresh401 caused sign-in routing. Parent diagnosed UAT080: refresh's SQLite transaction conflicts with a second SessionManager write transaction; no session-eviction claim. Alice recovered through normal UI login10:21:57. The original independent GET helper had six successful fresh verifier logins and one ENOSPC500 login attempt; its replacement preserves and reuses one private verifier session per identity, refreshing only when needed. Both helper provenance and credential cache remain private outside repository evidence.

Pirate Prompt “Cycle3 Alice Pirate” saved with the exact required instruction and Synced#1 row. Save adopted edit identity; Back returned to /prompts without discard confirmation. Apply/request/reply still in progress. Bob's real default-provider five-card generation completed10:19:35; he is reviewing the original deck session.

## 10:42 checkpoint — prompt passed; character control underway

Pirate Prompt Use in chat → Use as System Instruction, explicit New saved chat produced Standard mode with character off before send. Actual request699 /chat/completions200 contained exactly the required pirate system message and exact local model, save_to_db:true. Reply “Ahoy, welcome aboard Project Cedar, ye landlubber—ARRR!” remained after reload. The existing062 mode flip to Helpful AI Assistant recurred after the first reply.

UI-created Cycle3 Cedar Guide character4 and Cycle3 BEEP Robot character5 without API seeding. Direct Chat as Cedar Guide opened /chat?mode=character&characterId=4. Grounding request1106 complete-v2 on conversation812385d9-f19b-4bdf-82a0-1d850884a4b3 returned “Jonah Patel coordinates Project Cedar, and volunteers meet every Tuesday at 18:15.” Persist1111 returned200; replacement and history controls continue.

Bob GET404/schema-valid Media PUT404 of AliceMedia2 completed10:31:07. The Media update schema has no expected-version field, so this is not an optimistic-lock control. Independent AliceGET20010:32 returned the entire original body unchanged, including version_number1. Bob ownMedia1 remains unchanged. Bob's five original Biology cards reviewed Good; his final packaging/reload checks continue. Admin-owned Trash workflow delegated to Bob; main has not changed admin data.

## 10:49 checkpoint — Review/reanalysis and Alice study completed

068 recurred: picking Robot5 from the active Cedar character picker settled back to Cedar4 and cleared the visible old history. The next real request used newconversationdcd5ec14-e564-4372-bc2f-18f1106dbefc and answered “I do not know.” rather than BEEPBOOP. Saved CedarNote0d6a6904-c5b4-4d38-a8c6-18b1c36f46e2 retained only the clean visible answer and opened originalconversation812385d9-f19b-4bdf-82a0-1d850884a4b3. Before reload both visible rows appeared. After settled reload the assistant disappeared (070): UIone user row/Chat1message, independentGET200 has two serverrows, including senderCycle3CedarGuide and the original answer. This supersedes any earlier provisional reload-pass wording. No local mirror read was performed.

Media1 Actions→Open in Multi-Item Review opened /media-multi. Full-content search “north entrance” with the explicit full-content option enabled returned only Cedar and showed complete274charcontent plus originalanalysis. Returned to the same Media1 and generated a distinct two-line analysis with actual local model request215200. Storedversion2 and originalversion1 retained; independentGET200 and settledUIreload show “Opening: 22 November 2026 / Volunteers: every Tuesday at 18:15”.

Alice's savedQ/Acard65e74e34-b5f3-49c7-80fb-a59f135e1d65 displayed exact inspected question/clean116charanswer. ReviewedGood once; reloadshowsReviewedtoday1, completedall-deckssession1card and nextreview10minutes. IndependentcardGET200 confirms persistedscheduling. This one-card session does not test mixed-deck076. Sourcebacklink control and final auth/isolation journey remain.

## 11:03 checkpoint — final offline boundary / metadata isolation

Direct Robot5 baseline passed: /chat?mode=character&characterId=5, complete-v2/persist200 on2b88abd0-9bcb-4782-95cc-22077dc5145d, realreplyBEEPBOOP. Deliberate Cedar answer→card remains blocked085: Note backlink restores its two visible rows but leaves Robot5 active, and the assistant action menu lacks Save to Notes/Flashcards. No extra generation workaround/card/study pass is claimed.

Alice ordinary /admin/server shows basic connection details and Admin access required; admin dashboards/install controls unavailable. Independent Alice cached-verifier GET404 and expected-version1 PUT404 for BobNotea80707b3-c761-4d8a-8bbe-fe2222886bc2; subsequent independent BobGET200 retains exactoriginalcontent/version1.

Natural080refresh failure recurred before the final offline setup; preserved then recovered through normalAliceUIlogin. Both /notes and /settings/tldw fully loaded online. Browseroffline; Alice typed newprivateJuniperLantern draft and clickedSave. ExplicitSave shows error modal and locallyqueued1; no server-save claim. SettingsLogout clearslocalcredentials with bounded warning that remote revocation is unconfirmed. OtherNotestab remains /notes, nativeSignedout/Reconnecttosignin, titleSignedout|tldw; oldprivatecontentunmounted. Screenshot andconsole retained. Console contains expected offline fetch failures and warnings, noNextRuntimeError/uncachedHelpchunkfailure. ReconnectautomaticallyroutesNotes to/login.

NormalBobUIlogin in sameprofile: Notes shows onlyBob's two ownrecords and blankeditor, noAliceJuniperdrafttitle/body/queuedcount. However /knowledge leaks Alice's exactCedar/Indigoquery history pluscitationcount/status/time (086). OpeningAliceCedarRecent triggers foreignmessages-with-contextGET404 and displaysUnabletoloadconversation,0sources; noanswer/sourcebodyrestored and noRAGsearch/inference dispatched. The resulting404also produces aNextoverlay, retained inbob-alice-recent-restored.txt. Boblocaldraftboundary was then typed distinctly; finalAliceownqueuedrecoverycontinues.

## Final boundary outcome and cutoff — 2026-09-15 11:14:21 UTC

Alice return automatically synchronized her queued draft: realPOST161201 created Note01241eac-571d-4631-9802-9cbc7fdfe769, version1, exactJuniperLanterncontent. UIselectedthatnote withlastsavedtimestamp; Boblocaldraft was absent. ExistingBobverifierGET ofnewprivateNote404. This verifies recovery and APIownership, not overallclientprivacy.

064account-boundarycontrol failed: AliceopenedtheactualNote→Generateflashcardsaction. ItsURLcontainedfullprivatebody/title/sourceUUID. After normalAliceLogout andBobLogin, two normalbrowserBack actions traversedtheinterveningLoginentry and reopenedtheactualhandoff. UnderbrowserprincipalBob(user3, auth/me367200), GenerateFlashcardsshowedAlice'scompleteJuniperbody andsourceNote title. NoBobgeneration/savewasperformed. IndependentlyBobGETsameNote404, soAPIACLheld. Snapshot private-handoff-bob-history-result.txt and PNGprivate-handoff-bob-visible.png confirmvisiblebody. Principalresponse was checkedagainstprivatecredentials thenusername/emailredacted.

Finalbrowser remains online, authenticatedBob on the historicalprivatehandoff. Browser actions are stopped; parentowns runtime retirement and evidence sealing. API42500 andfrontend48482/session52335 retained. Lastobserveddisk2.5GiB. Parentcommittedisolatedcacheexceptionc10e1464fc; functionalappsource stayed d40e17dc81.

Evidencecutoff11:14:21UTC applies to browsercaptures. Report and filenameinventory finalization afterward is documentation only. Allcaptures are top-level /private/tmp/uat-cycle3-multi-*; auth-refresh-diagnosis.json/txt are explicitly included. Main-ownedbob-prefixedcaptures are listed in /private/tmp/uat-cycle3-multi-main-owned-bob-captures.json; do not exclude those with the independentlysealedBob bundle. Parentpackages maincaptures in multi/main/, keeps thisreport here, and preserves bob/REPORT.md. Nooutsideprefixbrowsercapture is required for this report; referenced .playwright-cli paths are provenance only where a copied capture exists.

PNGvisualcheck: offline-notes-signed-out.png shows onlynativeSignedout/Reconnecttosignin; private-handoff-bob-visible.png showsAlice'ssyntheticprivatebody intheFlashcardsourcefield. Bothinspected. No credentials visible.
