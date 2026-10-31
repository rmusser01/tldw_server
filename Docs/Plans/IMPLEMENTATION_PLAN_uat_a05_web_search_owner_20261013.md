# A05 web scrape ownership and search

Associated task: TASK-13260.278.18.83.45. Parent workflow plan: IMPLEMENTATION_PLAN_uat_a05_ingest_progress_20261013.md.
Candidate: frozen523ef323e9f56bc31559f6c5b8e506c27e599dcb plus preserved WIP, not latest dev. A05 only active; A06 unstarted.
ADR required: no new rule; existing ADR049 ownership/tenant boundaries govern restoring authoritative user stamps. Reassess any legacy migration separately before implementation.

## Stage1: Confirm the actual writer-to-search cause
**Goal:** Reproduce the observed saved article becoming invisible to its own search scope.
**Success Criteria:** Real SQLite persistence/FTS and actual service show owned search failure; foreign isolation and single-user fallback remain testable. No fake application import or raw operator SQL.
**Tests:** Actual _store_persistent with temporary canonical Media DB, synthetic article and User2/None; normal search API under own/foreign scope. External extraction not involved in this persistence test.
**Status:** Complete

## Stage2: Minimal owner repair and affected qualification
**Goal:** Stamp the authoritative effective user on new scraped sources without changing visibility, deduplication or content; use numeric authenticated client identity for restricted PostgreSQL and retain producer provenance in metadata.
**Success Criteria:** Causal greens, affected persistence/search/metadata tests, official restricted PG as applicable, syntax/lint/Bandit baseline/current and independent source review. Existing corrupt ownership/recovery must not be silently treated as repaired by a new-write fix.
**Tests:** Owner/search regressions plus genuine affected suites; review retained data path before any supported repair/migration.
**Status:** Complete

## Stage3: Live validation and engineering handoff
**Goal:** Qualify actual source persistence/search/ask and retained identity recovery before broader A05 acceptance.
**Success Criteria:** Corrected owned API guarded replacement, real source/query success with owner IDs, original failedMedia6 preserved or repaired only through a reviewed supported abstraction. No manual SQL, weakened filters, hidden retries or duplicate-artifact acceptance. Combined engineering/latestdev/publication gates remain open.
**Tests:** One bounded real live validation after source fix, source/query/owner/identity readbacks; official task and parent plan current.
**Status:** In Progress


Causal red: real initialized SQLite/service/FTS under actual owner and foreign scopes,2failed/27deselected/11warnings/1.36pytest/3.089outer/natural1. OwnUser2 and None→singleUser1 both fail; first actual tuple(ownerNone,ownRows[],ownCount0,foreignRows[],foreignCount0) versusexpectedowner+oneownresult. Original test permission warnings for system_log_buffer did not prevent these assertions; no runtime/native gate accepted. Minimal explicit owner_user_id=effective_user_id argument implemented only after red; producer client ID/visibility/FTS/filter/dedup/content unchanged. Existing incorrectly stamped liveMedia6 is not yet repaired or accepted by this new-write fix.


Final source unit: first owner-only repair passedSQLite but corrected existing-only test exposed real restrictedPG SyncLogRLS42501; complete fix uses numeric effective-user client_id plus explicitowner. Finalthreeaffectedfiles58passed12warnings/3.56pytest6.112outer/natural0 inclSQLite+officialrestrictedPG4own/foreign/defaultowner cases. Legacywhitelistnew8cases red2failed6passed9deselected, thenfull17+focusedbootstrap18passed12warnings/3.86pytest7.123outer. All17version/idempotency/explicitowner/invalid/overflow assertions retained. Reviewer Important reader-bootstrap masking corrected with existing_only=True and separateactualforeignhandle; final4pathsourceCLEAR.
FinalscopeBanditproduction0→0/schema0→0/tests67→67/backfilltests2→2/noerrors/no new; narrow newpytestB101 annotation restored afterformatting movedit. Existing serviceRuffI001 identicalbaseline/current; other touchedscopeRuff0/syntax/diffclear. No constructor/application import shim or RLS weakening.
OwnedAPI80752/80744 stopattemptoneSIGTERM failed15.039s absence/lockgate, no startup. Monitor25703 returned0, freshbothPIDs/listenerabsent; actualoldmarkerapiA04CompletionReturnExitCode=-15, apiReadyfalse/apiStateexited. GenericapiExit historical; no ASGIclean-exit/stopdelaycauseclaim. Prepared one no-signal after-exit recovery with strictstate/knownexit/absence/ownerlock/freeport/source/profile/env guards and first120startup, separateRecoveryConsumed; original over-grace retained. Operator-onlyadmission10pass/Ruff0/Bandit0, reviewpending; not executed yet. LiveMedia6sameID/content/query andallbroaderA05gatesremainOPEN.


Live validation completed: independently reviewed no-signal recovery consumed once, API18354/18339 canonical ready19.802s. Original over-grace failure retained. Supported startup bootstrap repaired original Media6; ID/content6091characters/URL/title unchanged; own full-text total1. Existing cited answer preview SourceID6/late_chunk:6:7 matches exact stored excerpt; first terminal observation135s within180s. No re-ingestion/manual SQL/duplicate acceptance, source data unchanged. Stage3 engineering handoff remains In Progress; combined review/hooks/commit/latest-dev/publication and broader A05 gates remain open. Earlier prepared/not-run paragraph above is historical and superseded by this live outcome.
