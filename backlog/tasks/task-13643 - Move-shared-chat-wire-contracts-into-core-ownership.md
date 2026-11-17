---
id: TASK-13643
title: Move shared chat wire contracts into core ownership
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The architecture ratchet rejects core stores importing API-layer native fork and Resume schemas. Give pure shared wire contracts one core implementation, retain API re-exports, and keep canonical OpenAPI and original validation behavior unchanged.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Existing architecture failure reproduced without adding baseline exceptions
- [x] #2 Shared native/history/Resume contracts live in core and API imports retain compatibility
- [x] #3 Canonical OpenAPI and connected protocol tests preserve wire behavior
- [x] #4 Scoped checks, independent review and normal hooks pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR3229 publicCI core-utils-tooling job115377521162 reports6newcore-to-API imports. Highest13642 acrossorigin/open/local beforecreate. LatestremoteDev c65df167 advanced; finalrebase remainsnearmerge. ADRrequired:no; implementationrepairs existing Docs/Architecture.md dependencydirection, no new lasting rule or API shape. CoreWorkspaces/chat_startup_schemas.py isexistinghousepattern. Keep userhumanChange summary intact.
ArchitectureRED1failed2passed→GREEN3passed, removing2fixedlegacybaselineentries. Puredefinitions relocatedunchanged beforeformat; newResume9Optionalannotations and1quotedreturn converted toequivalentPEP604underfuture forlint. All8corecallers usecoretypes; APIfacadesreexportcustomnames/privatehelper/nativehistoryaliases. FullcanonicalJSON+bytes identical2129paths3301schemas; fingerprint/pathunion unchanged, generator0. Bothboundedindependent reviewsCLEAR. ProductionBanditcurrent0findings/errors0 equalsparent0. ScopedRuff7I001 importsort findings tofix AFTERactiverunner; othercodes0. Connected7suites active, no importedsource/test edits whileactive. Latestremote c65df167 retainedforfinalrebase; publichead53065447c5.
Finalconnected661passed0skips1896runtimewarnings639.68s process0. Afterrunner7I001fixes preserveallnon-importAST; finalRuff0. Coldcanonicalexport repeatsbyte-identical OpenAPI/fingerprint/pathunion2129/3301, generator0. ScopedBandit0/0errors equalsparent; bothboundedreviewsCLEAR. Clientcompilersrunning; exacthooks pending. Docs/Reviews/CHAT_CORE_WIRE_CONTRACTS_13643_2026_10_13.md.
Bothnormalclient sourcecompilers exit0 (not dedicatedtestfiletypes). Scoped14productionfilesBandit0/0errors; exact17ownedfile normalhooks status recorded. APIcontractcoldexport afterimports byteidentical. Completedunitreadyforincrementalcommit; no wholePRreadinessclaim.
Exact17ownedfile normalhooks allpass withoutbypass; Black6moved/facadefilesclean, finalRuff0. Owncompletedplan retired. WholePRnew530CI stillhasfrontend/UX/migrationfindings; finalreview/rebase/merge pending.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Core owns the shared history/native/Resume model implementations; API directreexports retaincompatibility and eightcorecallers usecoretypes. Ratchet removes2resolvedbaselineentries withnoexemption. CanonicalOpenAPI/fingerprint/pathunion byteidentical2129paths3301schemas. Actualconnected661passed0skips; bothnormalclientcompilers0; scopedBandit/Ruff/Black/hooks clean. Two boundedindependentreviews clear; ADR nonewdecision, existingArchitecture direction. WholePR findings/review/finalrebase/merge remainopen.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
