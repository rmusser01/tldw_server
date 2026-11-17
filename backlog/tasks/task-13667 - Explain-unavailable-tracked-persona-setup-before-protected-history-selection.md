---
id: TASK-13667
title: Explain unavailable tracked persona setup before protected history selection
status: Done
labels:
- chat
- ux
- capabilities
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Protected H1 and complete-v2 explicitly reject tracked persona chat, while the picker/cockpit still offers it. User explicitly chose to make that unsupported choice unavailable with clear guidance rather than add native persona authority. Preserve ordinary persona overlay, character tracked mode and zero-write refusal guards; never silently convert a tracked selection to an overlay.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Unsupported tracked-persona choices are unavailable with understandable guidance before selection or native writes
- [x] #2 Supported persona overlay and tracked character behavior remain available and source-bound
- [x] #3 Native/WebUI regressions and actual cockpit persona workflow prove guidance, preserved draft and zero unsupported writes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Human decision in this chat2026-10-13:Make the unsupported choice unavailable with guidance. ADR assessment:no new persona protocol or durable rule; UI will match existing protected-history capability limitation. Existing historical TASK13124.5/TASK487 Done are prior legacy success work, not a current capability implementation. Current actual cockpit persona success expectation is unsupported. Highest global task ID verified13665, new ownedfocus13666, chosen13667. PR https://github.com/rmusser01/tldw_server/pull/3229.
Independent read-only scope assessment: use protected H1 context presence, including idle/local/native, rather than owner.kind. Restrict shared AssistantSelect before live and staging callbacks; preserve reactive external overlay event intent. Cover compact tracked-persona launch, saved setups/startup preview Apply actions, and full central apply guard before identity/model/prompt/RAG/address writers. Preserve stored bundles, overlay persona, tracked characters, and no-H1 legacy behavior. User explicitly selected disabling unsupported tracked persona with guidance; no native persona protocol expansion.
Design Docs/Design/2026-10-13-protected-chat-persona-capability.md and plan Docs/Plans/IMPLEMENTATION_PLAN_chat_persona_capability_13667.md created after user preference and independent scope review. ADR assessment: no new durable authority; align choices with existing protected H1/complete-v2 refusal.
Stage1 mapped all picker/live/staged/compact/saved/startup and central application paths; policy8RED then pass. Picker H1idle/local/native +same-open mode staging4RED then pass. Shared Button dropped aria-describedby, reproduced via compact integration and fixed centrally. All5native connected picker/types/tabs/contrast suites90pass0skip after synchronous new-intent retirement; held character ACK event/preference pair RED then GREEN. New guidance uses existing text-text-muted. Saved/startup6cases initially2protectedPersonaRED/4characterlegacypositive; central retained drawer/startup callbacks2RED resolvedtrue before guard. Implemented central identity-adoption refusal before writes (does not block unchanged identity editing), disabled described saved/startup Apply and ModalFooter description forwarding. Focused8pass, preserve identity/model/prompt/RAG/address/records and read/delete/legacy/character controls. Retained staged persona context-change Apply RED; addressed, final positive underverification after correcting ambiguous Guide Persona menu locator to actual selected trigger. English source key plus only1publicChrome key generated via existing flatten helper, unrelated keys preserved. ADR049 governs; no new durable rule or native persona authority. No full-unit/browser/build/merge completion claim.
Whole native12suite run348pass32fail exposed obsolete H1trackedPersona positivefixtures and footer puredesign mock import(getProvidersByCapability); productionguard not weakened. Independent fixture assessment preserved completefamilies inclGREENnegative barriers: inline explicitoverlay onneutralregularsource; saved model/Scene identity=null supportedguardedclear, coherentexistingtrackedsource retainedfordetach/hydration. Explicitlegacytracked vsprotectedoverlay stubcommands separate. RootParent198existingpass; fiveownedpersonaRefusal/supportedcleardetach/lateScene controlspassfocused. Correctednewrefusal snapshot actualfieldnames, noassertion removal. H1metadata-loaded persona Sendgap confirmed; full/compact nowvisibleguide/disabledSend andearly Enter/direct/queuevalidation/run gating; useMessage actualeffectiveState surfacedforcompact consistency/restoredprefs. Focusedfull3pass; compact4pass; combined36model/Send/a11y/actualdispatchguards pass. Reassessedcompactprep exactattributes after assertions: threeSendbuttons plusfourthhistoryretry; onlythreeSenddisabled, reader/recoverypreserved. LocaleEnglish+oneChromeFlattenedkey only. Broadprofiles/static/build/actualbrowser stillpending.
Independent review found compact captured-pair regeneration bypassing the later persona guard. Reproduced a submitted unsupported reply, moved refusal before preparation and dispatch, retained exact captured owner/pair/draft and supported-character controls. Public Regenerate and Alt+Shift+R now need matching unavailable guidance. Source mapping also found the persona overlay guidance had no public chooser action: add an explicit user-selected overlay action for ordinary chat; never rewrite stored tracked bundles or allow overlays in an owned tracked conversation.
Final bounded source rereview clear after fixing compact captured-pair regeneration, all Message retry/fallback/continue paths before provider-setting writes, and explicit public ordinary persona-overlay choice. Native/WebUI connected20 suites each763 pass/0 skips; latest Message connected11 suites each220 pass/0 skips after recovery changes. Earlier source guard failure was pre-existing HEAD onDynamicUIAction forwarding assertion; migration now checks existing resolved bridge assignment plus actual forwarded handler and keeps both main/compare surface assertions. Final17-root production type profiles and separate cockpit E2E source profile exit0. Actual root lint30 TypeScript files:0 errors/298 unchanged HEAD warnings/0 ignored/0 new; only pre-existing incompatible-library diagnostic source frame line numbers normalized for comparison. Build, real browser and preservation qualification pending; no whole-PR approval or merge claim.
Source-matched production build icj9ccdvWZDSuApuhAMKa qualified:154 pages, all38 changed production files covered by41 source-delta entries,969 emitted JSON/1130 static refs/0 missing, actual new-dist tokens0 and shared590.3KB<=600/heaviest869.0KB<=900. Actual2 persona browser workflows pass5.47sec, one headless worker/0 retries/skips/flakes, public unavailable zero-write preserved draft/session/transcript plus distinctive persona prompt in actual ordinary completion request. Final preservation9 original conversations/28 rows deepEqual with immutable baseline; synthetic handoff4 canonical rows unchanged. Independent bounded source clear. Review Docs/Reviews/CHAT_PROTECTED_PERSONA_CAPABILITY_13667_2026_10_13.md. Bandit N/A TS/JSON/MD; whole-project typing/package CI/installed extension/physical AT/studies/master/H2/allremaining cockpit unqualified. Exact staging and normal hooks remain before commit.
All36 exact-owned initial normal hooks pass. All3AC/all6DOD complete for this bounded unit. Only own completed plan is retired privately after its stages complete; final metadata hooks/commit/push follow. No remaining source/build/browser finding in TASK13667; TASK13656 and master/H2/final whole-PR gates remain open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Aligned protected-history persona choices with the existing server capability without adding native persona authority. Shared chooser, compact launch, saved/startup application, loaded Send/Enter/queue and compact regeneration refuse before writes; recovery fallback cannot mutate provider settings first. Explicit public ordinary overlays and supported characters remain available; bundles and canonical records unchanged. Native/WebUI capability763 and final connected message220 each pass;17-root types, cockpit types, actual root lint, source-matched production build, two real browser workflows and immutable9/28 plus synthetic4 preservation pass. Bounded independent source clear; remaining master/cockpit/H2 and final whole-PR review/rebase/seven-gate merge remain open.
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
