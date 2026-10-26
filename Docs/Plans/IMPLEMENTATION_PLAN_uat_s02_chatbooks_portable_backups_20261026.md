# S02 Chatbooks and portable backups UAT

Task: TASK-13260.278.18.83.45. S01 selected closure precedes this unit; selected38/43. Initial pre-repair binding: source2a0245057b17a100442b33ce395f6bea4636e49c on frozen deva412f6580b7ed0cf48fd7d222a47a20cfc0dde92, including3309/3310/3311; installed2c50 API matched the relevant backend at that point. Count repair dac2fdca49ea30264c154367a0e46b3bb9ae2c77 is installed privately at /private/tmp/tldw-s02-counts-20261026/site; inline-link repair14cf2285c89397de2ba7403433a5a5861ecc5dca is now privately installed at /private/tmp/tldw-s02-links-20261026/site. Private frontend54RnDCNE10vzciCahlGRn is built from2a024. Verify affected source bindings before actual use; no fresh live-dev or release-wide claim.

ADR required: no. Exercise existing Jobs, per-user storage and supported archive/import contracts under ADR-003, ADR-020 and ADR-059. Reassess if a proven repair creates a durable rule.

## Scope and acceptance

Source of truth: RELEASE_UAT_PLAYBOOK.md S-02 and current release-catalog-data.mjs. Five selected variants: chat-note-source-roundtrip, advertised-import-formats, full-account-export, corrupt-import and failed-import-retry. Current UI advertises Chatbook archive, OpenWebUI JSON and OpenWebUI database, with ten selectable types: conversation/note/character/media/embedding/prompt/evaluation/world_book/dictionary/generated_document. Inventory actual collectors, exporters/importers and UI mapping before declaring fixture completeness; do not silently exempt advertised data or accept a title-only record.

Use one synthetic Rowan Chat, Note and each advertised supported source type; pin exact text, linked IDs, metadata and necessary assets. Export through actual UI, independently inspect the downloaded manifest/member integrity and disclosed scope, import through the actual supported mapping into a distinct account/profile, then verify stored content/links/assets/owner/counts and true reload. Verify full-account inventory against source stores and secret exclusions. For each advertised import format, use a genuine format fixture with explicit supported mapping; test malformed input and one legitimate corrected Retry, truthful job outcome and absence of unexplained duplicates. Prior successful controls are not replayed unchanged; failed attempts remain failed.

Native functional UI and actual background Jobs are required. SQLite/PostgreSQL single-/multi-user composition must use real storage, own-positive/opposite-owner controls and official isolated restricted PostgreSQL fixtures; declare controlled boundaries rather than implying every variant is replayed in every mode. Human UX review uses a settled screenshot and is separate from functional acceptance. Existing full-account fixture and verification code can be reused narrowly after review; do not execute its auth-storage injection, unsafe reset paths, extra browsers or proof/log-bundle behavior as native qualification.

Original API/frontend/Chrome/URL/five messages/draft/private profiles/keys and Splash8010 remain protected. One owned private Chrome window/tab, normal UI login/connect, no auth-store injection. Current tools/dependencies/artifacts reused; changed scope gets causal red/minimal shared fix/affected green/security/independent review/normal hooks and necessary packaging. User all-necessary diagnostics/retries/runtime authority persists; no repeated permission gate from consumed earlier units. Reassess after three failed attempts using distinct supported evidence before further work.

Caps:600s workflow/build/host,120s startup,180s generation if genuinely required,300s test case,15s operations/graceful cleanup. Reserve independent cleanup and original handback; no force, shared service/model changes, GitHub/publication/queue action or heartbeat change. Functional minimum synthetic files only; dispose completed payloads rather than keeping proof bundles.

## Full-account and OpenWebUI scope to freeze before live writes

The full-account contract contains19 categories beyond the ten picker types. Require explicit source count and destination/content or documented warning disposition for each: account_profile, account_settings, conversations, notes, characters, world_books, dictionaries, prompts, evaluations, generated_documents, explainer_sessions, media_records, media_transcripts, media_chunks, media_stored_artifacts, media_pointers, embeddings, tags_categories_relationships and sensitive_user_values. Restorable populated categories must have exact body/identity/ownership/byte checks; empty categories require exact zero. media_pointers is reference-only. tags_categories_relationships and sensitive_user_values are declared non-restorable; require truthful warnings and actual secret exclusion rather than claiming complete relationship/credential recovery.

OpenWebUI database input must include two distinct source users; select one through the normal preview and prove the other user's conversations/assets are absent. JSON and database fixtures each include a known attachment reference plus independently known local bytes. Verify reference preservation separately from supported hydration, use the actual import-scope/hydration UI for byte claims, and pin missing/unsupported attachment warnings. A successful text/title import does not accept asset preservation.

## Stage 1: Inventory and freeze roundtrip oracles
**Goal**: Map actual advertised source/format/account contracts and prepare minimal reviewed private fixtures/operators.
**Success Criteria**: Exact fixture bodies/assets/links/owners/counts and independent manifest/import oracles; supported environment/current artifacts/original-owner/deadline/cleanup guards.
**Tests**: Small operator refusal/privacy/deadline/release checks for genuinely new logic; no unchanged product-suite replay.
**Status**: In Progress

## Stage 2: Execute and repair
**Goal**: Actual UI roundtrip, full-account export, advertised imports and corrupt/failed recovery with declared four-mode composition.
**Success Criteria**: Real worker/job outcomes and independently verified saved mappings/contents/assets/ownership/counts/reload; causal fixes and affected qualification for demonstrated bugs.
**Tests**: Five selected catalog variants and named ten-type/three-format instances; genuinely affected application tests/security and real official storage fixtures.
**Status**: In Progress

## Stage 3: Accept, preserve and dispose
**Goal**: Human screenshot UX, independent final review, original preservation and completed owned cleanup.
**Success Criteria**: All selected variants have explicit passing scope or owner disposition; exact data/browser/process/request release; concise durable task/master results and owned completed-plan removal.
**Tests**: Final canonical/URL/draft/owner/control checks, exact touched normal hooks and guarded completed-root disposal.
**Status**: Not Started

## Current outcome

Local count repair is qualified by explicit 38+1+1 composition: 38 affected archive/scope controls, SQLite visibility and restricted PostgreSQL visibility. These real-store archive tests do not by themselves qualify native S02 UAT; the count package has since been refreshed as recorded below. Ordinary PostgreSQL deletion remains open for caller-authority validation. Actual native source export/download has since run, with independent archive link failures and browser cleanup failure below; no S02 acceptance yet. S01 is selected accepted; S02 In Progress; S03-S06 unstarted. Broad task/AC/DoD remain open; heartbeat paused.


### Scope-count diagnosis and causal-test design

S02sourceinventoryfound two concretepreview/archive countdisagreements toprovebeforefix: _scope_count_for_category neverroutes tags_categories_relationships to existing realDBkeywordcounter; embeddingsscope sums vectorrows whilearchivecontent/statistics countscollectionbundles. Callersenumerated: publicfullaccountexportscope→API/UIplusfixtureconsumer; archivecollector emitsonepayloadpercollectionincludingempty. Proposalonlyifrealcausalred:routeexistingtagcategorycounter andcountcollectionunits, keepingallpath/ownership/unavailable-store/errorfallbackcontracts. Prepare actualCharactersRAGDB tag and persistentChroma populated/emptycollection regressions usingexistingofficialSQLiteAuthfixture/fullv1.1export, nofakeDB/model/provider/controlworkaround. NoS02nativeacceptanceyet.


### Actual count RED and complete bounded repair

S02correctedreal-storecausalrun records4actualcountassertionfailures: tags0vs1, populatedChromarows3vscollection1, emptycollection0vs1, media-vector0vs1. Targetedfivecases includeunchanged(2,0)positivecontrol; exactpytestsummarynotemittedbyfilter, outer7.374/natural1; noprecisepytesttimeclaim. Actualv1.1archivesandpersistentchroma.sqlite3passedbeforefailingpublicscopecounts. Minimalsharedrepairnowapprovedbyexistingall-necessaryauthority: existingkeywordcounterdispatch; fullaccountembeddingunit=countofChroma collectionbundles PLUS activevisible nonemptyMedia vectorpayloads via existingDBManagementallowlistedcounter. Preservepath/ownership/RLS/system-operation/trash/deletionfilters and independentoptional-storefallback. AddnormalSQLite/officialrestrictedPG DBcounterqualification and keepnativeS02roundtrip/19category/format/UXgatesopen.


### Qualified count controls and distinct PostgreSQL deletion failure

S02focusedvisibilityrunSQLitePASS/PGFAIL:1failed1passed6warnings2.45pytest4.654outer;38earlierpassingcontrolsunreplayed. ExactnewboundedPGstackprovesordinaryowner soft_delete_media UPDATE deleted=1 rejected42501, independent ofcounts. CanonicalFORALL RLSWITHCHECKincludesnotdeleted; MCPordinarysoftdeletepubliccontractandemailwritercallersmake thisaseparateactualpermissiondefecttoreview/causallyrepairbeforeS02closure. CountfixturewillseedlegacytombstoneviaexistingplatformAdminscopeonlyonPG, retainingeveryordinarycount1/2/foreign303zero assertion; thislegitimatefixturepreparationdoesNOTfixoracceptordinarydelete. Independentsecuritysourcereviewrequested; nonewRLSchangeimplementedyet. No repeatSQLitepass; onlyPGcountfinishpending.


### Count repair qualification and permission boundary

S02countrepair localqualification COMPLETE by explicitcomposition:38affectedchecksPASSinwhole17.56srun, SQLitevisibilityPASSin2.45srun, finalPGcount-onlyPASS1case6warnings2.55pytest4.855outer/natural0. RestrictedPGcount101=1/202=2/303=0 retainsordinaryscopes; preexistingdeletedfixturepreparedunderlegitimateplatformAdminscope only. No passing-case replay. FourtouchedPythoncompile/originaltestASTpreservationPASS; Ruff17identicalinheritedservicefindings/other3files0/0new. BaselinecurrentBanditprod0/0, originaltest99/99,newtest0,0errors0new; narrowlyexplainednewpytestassertionsonly. Actual/sourcequery/testreviewCLEAR; normalhooks/staged/localcommitpending. NativeS02stillNOTexecuted; countfixrequireslaterprivateinstalledpackage. Directordinary-PGtombstoneUPDATEfailure remainsOPEN/callerauthorityandnormalpubliccontractvalidationpending, notconfusedwithcountfix; PostgreSQLSELECTnewrowchecksmeanUPDATEpolicy-onlyproposalinsufficient. NoRLSchangeornewtrustboundaryauthored; do notweakenvisibility. Humanallnecessaryauthority persists, noapprovalpause. S01closed38/43, S02InProgress.


### Native archive and inline-reference repair

Corrected official preparation PASS9.559s; source direct pre-API19-category oracle and independent destination baseline passed. Normal API bootstrap subsequently adds bundled pixel-migu and Helpful AI Assistant; post-API4-card count/exact full archive card content and bundled image passed. Existing actual UI export Job6f28b96f-c07a-49ce-abd6-8c60d9608b3a completed/downloaded38469bytes. Independent archive inspection verified19categorycounts/11types/20inventorymembers/source-secret exclusion, but FAILED message_parent and note_links; both inline fields were absent despite genuine source links. No unchanged export replay. Actor remains FAILED at Disconnect with2/40requests pending; exactownedChrome released. HostPASS130.714s, private API/frontend fullyreleased/ASGIzero/current-controlLogout/originalcanonicalandowners preserved. Operator admission-before-app-import/require-existing-DB was subsequently hardened; no unchanged reread claimed.

Four real-store causal regressions RED4failed5warnings5.17pytest7.249outer also demonstrated generated documents binding old Chat IDs to unrelated destination content. Minimal shared repair exports message parents and Note inline references; import-run successful-write maps remap conversations/messages, including Notes sync capture and generated documents. Unselected/unmapped references clear with warnings rather than destination old-ID lookup. Parent-first topological ordering handles out-of-order timestamps; duplicate IDs/cycles reject, second-message failure rolls back and exposes no map. Exact Note body whitespace survives. No new durable rule/ADR, dependency, schema, endpoint, frontend or startup-authority transport.

Qualification: first affected run81passed5fixture-errors7warnings24.15pytest26.705outer; five new edge tests failed before production because ContentItem fixture omitted its required title. Corrected only that fixture; five incomplete cases plus new active Notes sync capture case PASS6passed4deselected6warnings6.97pytest9.246outer. Explicit81+6=87 passing composition, not one green87run; passing cases not replayed. All unrelated service method ASTs identical; seven intended functions changed/added. Supported compile and Ruff17identicalinherited/0new; new test Ruff/formatclear. Baseline/current production Bandit0/0, new tests0, errors0; narrow new pytest assertions only. Independent actual-source review in progress; normal hooks/staging/commit/private packaging and changed-source native export/import/reload remain open. Distinct browser Disconnect and ordinary restricted-PG tombstone authority gaps remain open.


### Reviewer newline finding closed

Independent actual-source review found universal-newline import reading still converted CRLF/lone-CR bodies. Two additional genuine public round-trips RED2failed10deselected5warnings3.58pytest5.859outer. Both Note export write and import read now use newline=""; only those two new cases GREEN2passed10deselected5warnings2.77pytest4.518outer. Total89PASS by explicit81+6+2composition, never one green89run. Reviewer final production/all12newtest-body assessment CLEAR in covered scope. Supportedcompile2PASS; final production/newtestBandit0/0/errors0; newtestRuff0/format0. Initial final lint attempted a cache write in the protected worktree and was refused; normal --no-cache lint then PASS without source changes. Genuinely changed backend still awaits normal hooks/staged review/commit/private install and live validation; no S02 closure.


### Current private binding and cancellation oracle

Count commit dac2fdca and link commit14cf are local/unpublished, exact independently reviewed patches with normal hooks. Link package PASS15.238s:3817 package/data files,3roots and337dependency nodes/722edges verified. A preceding sandbox Git stderr warning correctly failed preflight before any setup; normal authorized guard then passed without weakening parsing. Private frontend54Rn is unchanged and appropriate because no frontend source repair is yet authored.

Two read-only native transport attempts remain FAILED; latest actual Logout acknowledgement succeeded while both browser ledgers retained3/34 requests. All selected API lifecycle receipts ended and private ASGI shutdown reached zero; host abort99.692s preserved original canonical data/current-control logout/owners and released services. Chrome group52602 was separately absent afterward.

Isolated installed-Chrome hard-navigation calibration independently reproduced the oracle failure: all5 real server responses closed, but both PW/raw-CDP ledgers retained5. Cleanup clear, original owners unchanged, outer15.94s. Distinct normal Cancel-before-anchor control PASS: positive5/5/5 precondition, all5server closures and both terminal ledgers0 BEFORE navigation, pagehide1, cleanup clear, original owners unchanged, natural0 outer1.138s. This qualifies active-document cancellation only. Native actor now uses actual Settings/Chatbooks ReactRouter links and proves unchanged document identity, then Logout acknowledgement and both-zero drain before about:blank; no map clearing, fake terminal receipt or product transport change. Changed-source native export/import/reload and all remaining S02 gates stay open.


### Destination failure diagnosis and next causal unit

Actual native import40cf6954-a87b-4651-b6cf-60fa118e0559 FAILED account finalization after one dispatch. Browser36/36ended/0pending and host205.292s preservation/release PASS. Exact real ProfileCommand dry-run422 rejects synthetic source example.test email as invalid_email; theme paper is accepted. This is an operator fixture error, with no auth-validation weakening justified. Correct only source synthetic email through normal ProfileCommand before a genuinely changed archive and legitimate recovery.

Independent partial destination inventory found world_books0. Chatbooks calls nonexistent WorldBookService.get_world_book_by_name; the actual service exposes get_world_book(name=...). New real public archive tests cover fresh/Skip/Rename/prefix and failure truth after one selected item fails. Minimal correction must retain both top/nested archive names, existing destination content, partial-write disclosure and truthful failed Jobs results; no cross-store atomicity claim. Existing task association/ADR003/020/059 apply. Causal red precedes production edits, affected green/security/review/hooks/private packaging and native destination validation remain open.


### World-book and truthful job-failure repair qualified locally

Real public archive RED4failed1passed6.56pytest10.334outer and three actual dispatcher RED3failed5deselected3.66pytest5.488outer. Minimal correction: existing owner-filtered get_world_book(name=...), normalize both nested/top names for Prefix/Rename, return failure when any selected content fails, retain allowlisted partial counts/warnings/three public metadata fields in all three worker failure lanes. Startup worker initializes result=None for each import; cancellation and private account-payload removal remain protected. No cross-store rollback or automatic retry is claimed.

Causal GREEN8passed5warnings10.26pytest12.667outer. Connected full-account/import-worker/cancellation suites plus three new recorder privacy/cancellation/refusal guards:28passed1failed8deselected8warnings14.00pytest16.034outer. Only remaining failure was the exact empty-import result contract missing five new zero counters; preserve all previous expectations and explicitly assert five zeros. Target-only finish1passed5warnings1.80pytest3.707outer. Qualification is8+28+1=37passing composition; no one green37run or passing-case replay. All protected-owner guards passed. Compile4PASS, lintservice17/coreworker1 inherited/others0/no new; Bandit3production0/0/newtest0/0/modifiedtest63/63/errors0/no new. Independent production/all11newcase review clear; final expectation/staged review, hooks, local commit, private packaging and real native recovery remain open. S02 remains In Progress.


### Retry identity and collection references

Actual native recovery completed world-book and account restoration, but the independent destination check found two Explainer sessions and a collection retaining the old Media UUID. The message bytes were exact; the original tiny synthetic PNG had invalid CRCs, so the canonical API reader correctly refused it. The fixture correction changes checksums only and preserves pixels, text, message identities and parent links.

Real public-archive failures established Explainer duplicate/rename and partial-tree behavior plus collection reference problems. One unselected-collection test initially used the wrong manifest selection ID and is not causal defect evidence. The minimal shared repair honors Skip/Rename and owner/active-title lookup in both Explainer formats. Fixed nested savepoints keep session creation, nodes and reload in one transaction; failed or interrupted trees cannot become later Skip candidates. Collection links require consistent aliases to successfully admitted Media; Skip reuses only exact-owner records with matching provenance from the actual document-version API. Conflicting aliases, malformed provenance, unselected and foreign records remain unlinked with warnings. Existing destination vectors remain unchanged under Skip.

Qualification is 18 new cases plus31 connected existing cases, passed across targeted runs; no one green49-run claim. Initial green9passed/1fixture-failure12.01s; connected37passed/1incomplete-provenance-failure13.49s; focused3passed/1alias-fixture-failure5.55s; alias finish1passed2.61s; strengthened foreign-owner finish1passed2.45s. Normal content deduplication had collapsed identical alias fixture bodies; distinct bodies and an explicit two-record invariant corrected that setup. Four production baseline/current Bandit0/0, newtests0/errors0; compile passes, no new lint findings, and changed-function formatting preserves ASTs. Independent actual-source and complete new-test review CLEAR. Normal hooks/staged review/commit/private packaging and actual corrected-candidate native validation remain required.

The old destination and failed/completed Jobs remain preserved. Its existing duplicate/stale references are not retroactively repaired or accepted. A fresh private destination must demonstrate honest partial failure and a legitimate corrected-archive Retry, with exact final content/count/reference/image checks. Cleanup error metadata now includes the fixed error code; the earlier230ms group-release error was not a15-second timeout. No cap or presence guard is weakened.
