# Persona/Buddy Current-State Audit

Date checked: 2026-05-10

## Summary Verdict

The Persona/Buddy system is no longer a single placeholder surface. It has concrete server-owned profile/session contracts, a route-owned Persona Garden/Live UI, a floating Buddy shell, browser wake detection, visual packs, review-gated visual generation, a personal visual library, and internal MCP tools for visual-pack draft/runtime actions.

Stage 1 should stay focused on reliability and product-hardening around the flows that already exist:

- Add a small Persona/Buddy diagnostics surface that gathers profile, Buddy, visual pack, websocket, wake, and MCP-readiness status in one place.
- Tighten recovery copy and recovery actions for live voice, websocket reconnect, wake rejection, visual-pack load/render failure, and setup detours.
- Add smoke/E2E coverage that exercises the known-good happy paths and existing failure fallbacks together, not just one isolated component at a time.
- Preserve the existing VN/CYOA boundary. Persona Visual Packs are Persona Buddy/Persona Live assets; VN asset-pack work is a portability precedent, not the runtime owner.

Do not start Stage 2/3/4 work from this audit. Persona chat quality, broader MCP behaviors, native/background voice, richer renderer adapters, and shared-library/library-market features are follow-up stages.

## Tracker State

Live GitHub state was checked with `gh api graphql` against `rmusser01/tldw_server` on 2026-05-10.

| Issue | State | Updated | Role in this audit |
| --- | --- | --- | --- |
| [#635 Tracking: Persona Chat enhancement](https://github.com/rmusser01/tldw_server/issues/635) | Open | 2025-11-23T01:12:14Z | Legacy umbrella. Body is broad but still has useful Persona Chat reference links in comments. |
| [#1388 Track Live Personas visual identity and runtime assistant effort](https://github.com/rmusser01/tldw_server/issues/1388) | Closed | 2026-05-09T17:50:29Z | Prior visual identity/runtime tracker; closed. |
| [#1389 Add internal persona_visuals MCP module and runtime visual overrides](https://github.com/rmusser01/tldw_server/issues/1389) | Closed | 2026-05-09T04:21:42Z | MCP visual tools/runtime override slice; closed. |
| [#1428 Track Persona/Buddy visual pack reliability and product hardening](https://github.com/rmusser01/tldw_server/issues/1428) | Closed | 2026-05-09T20:59:05Z | Visual-pack reliability tracker; closed. |
| [#1449 Epic: Persona/Buddy visual-pack reuse and libraries](https://github.com/rmusser01/tldw_server/issues/1449) | Closed | 2026-05-10T05:48:08Z | Visual-pack reuse/library epic; closed. |
| [#1497 Research: Persona visual-pack renderer and provider adapter evaluation](https://github.com/rmusser01/tldw_server/issues/1497) | Closed | 2026-05-10T05:45:53Z | Renderer/provider research; closed. |
| [#1391 Track character/persona CYOA-VN mode effort](https://github.com/rmusser01/tldw_server/issues/1391) | Open | 2026-05-10T01:25:36Z | Separate VN/CYOA tracker. Should not own Persona/Buddy runtime hardening. |

Useful `#635` references preserved before any tracker rewrite:

- [#635 comment: evaluations](https://github.com/rmusser01/tldw_server/issues/635#issuecomment-3454426336): StickToYourRoleLeaderboard, UGI-Leaderboard, NovelChallenge.
- [#635 comment: Guided Generations](https://github.com/rmusser01/tldw_server/issues/635#issuecomment-3454426938): Samueras/Guided-Generations.
- [#635 comment: 3D model assistants](https://github.com/rmusser01/tldw_server/issues/635#issuecomment-3454428039): semperai/amica.
- [#635 comment: LlamaTale](https://github.com/rmusser01/tldw_server/issues/635#issuecomment-3567289751): neph1/LlamaTale.

Recommendation: leave `#635` open until a new broader Persona/Buddy epic links this audit and explicitly splits Persona Chat quality from Buddy/Live reliability. After that, add a comment preserving the links above and either retitle `#635` to Persona Chat quality or close it as superseded by the new epic plus a Persona Chat sub-issue.

## Contract Inventory

| Flow | Server contract | Client owner | Persisted state | Session/runtime state | MCP/tool surface | Tests | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Persona Chat | Chat can resolve `assistant_kind=persona`, `assistant_id`, and `persona_memory_mode`; persona profiles project into chat assistant cards through `tldw_Server_API/app/core/Chat/chat_service.py:450` and `:474`. Persona exemplar prompt assembly is shared in `tldw_Server_API/app/core/Persona/exemplar_prompt_assembly.py:111`. | Shared chat hooks plus `apps/packages/ui/src/hooks/chat/personaServerChat.ts:96`. | Conversation assistant identity and memory mode; persona profiles in ChaCha. | Chat request/runtime guidance and optional memory/exemplar metadata. | Not Buddy-specific. Existing persona exemplar/runtime paths only. | `tldw_Server_API/tests/Chat/integration/test_persona_backed_chat_conversations.py:187`, `tldw_Server_API/tests/Chat/test_persona_prompt_assembly.py:41`. | Stage 2 is the right place for persona quality/evals; Stage 1 should only cover entry-point reliability/copy if Persona Chat is presented as a Buddy path. |
| Persona Live stream | `/api/v1/persona/stream` websocket at `tldw_Server_API/app/api/v1/endpoints/persona.py:7177`, with auth/feature gate at `:7190`, event metadata at `:7321`, notices/deltas/tool frames at `:7331`, `:7349`, `:7364`, `:7385`, `:7418`. | `apps/packages/ui/src/routes/sidepanel-persona.tsx:464` via `usePersonaLiveSession`; WS URL builder in `apps/packages/ui/src/services/persona-stream.ts:4`. | Persona sessions via `persona_sessions`; preferences include live voice runtime. | Active WS, pending plans, tool calls/results, recovery state, live voice status. | Persona tool execution path; visual override payload can arrive through tool results. | `tldw_Server_API/tests/Persona/test_persona_ws.py:221`, `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx:3879`, `apps/tldw-frontend/e2e/workflows/persona.spec.ts:136`. | Main Stage 1 target: aggregate connection/recovery diagnostics and make reconnect/copy-command states easier to interpret. |
| Buddy shell | Server embeds `buddy_summary` in profile/catalog responses with `PersonaBuddySummary` at `tldw_Server_API/app/api/v1/schemas/persona.py:437`; dedicated buddy endpoint at `tldw_Server_API/app/api/v1/endpoints/persona.py:3617`. | Route-local render context in `apps/packages/ui/src/routes/sidepanel-persona.tsx:241`; shell resolution in `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx:87`. | `persona_buddies` derived from persona profile state; overlay preferences preserved by DB tests. | Floating shell open/position bucket and live render context; not a server session. | Visual runtime overrides can influence state through the visual runtime store. | `tldw_Server_API/tests/Persona/test_persona_buddy_api.py:38`, `tldw_Server_API/tests/ChaChaNotesDB/test_persona_buddy_db.py:34`, `apps/packages/ui/src/store/__tests__/persona-buddy-shell.test.ts:12`. | Stage 1 should expose dormant/no-buddy and stale context diagnostics rather than adding new Buddy personality behavior. |
| Persona Garden setup/profile | Profile CRUD at `tldw_Server_API/app/api/v1/endpoints/persona.py:3441`, `:3475`, `:3550`; setup state and analytics schemas at `tldw_Server_API/app/api/v1/schemas/persona.py:558` and `:570`. | `apps/packages/ui/src/routes/sidepanel-persona.tsx:294`; Persona Garden panels under `apps/packages/ui/src/components/PersonaGarden/`. | Profile fields, voice defaults, setup wizard status, setup analytics events. | Setup detours/handoff cards and in-route wizard state. | MCP setup-related controls are mediated through policy/tools panels, not a single setup MCP tool. | `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx:881`, `:1187`, `:1349`, `:1517`, `:1795`. | Stage 1 should keep setup recovery scoped to existing detour/handoff paths and clarify what failed and where to retry. |
| Wake/voice | Voice defaults schema at `tldw_Server_API/app/api/v1/schemas/persona.py:489`; runtime config normalization at `tldw_Server_API/app/api/v1/endpoints/persona.py:9138`; wake activation/rejection at `:9244`, `:9279`, `:9305`; deactivation at `:9313`. | `apps/packages/ui/src/routes/sidepanel-persona.tsx:557`; browser detector in `apps/packages/ui/src/hooks/personaWakeDetector.ts:76` and `:117`. | Saved trigger phrases and wake behavior on persona profile voice defaults. | Browser speech recognition state; runtime trigger phrases; one-shot/continuous wake gate. | Native companion is accepted as a detector kind on the server, but V1 browser docs say no background always-on mode. | `tldw_Server_API/tests/Persona/test_persona_ws.py:2303`, `:2519`, `:2605`, `apps/packages/ui/src/hooks/__tests__/personaWakeDetector.test.ts:59`, `apps/packages/ui/src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx:156`. | Stage 1 should make wake rejection reasons visible and link them to saved-profile vs runtime mismatch. Native/background wake is Stage 3. |
| Visual packs | Visual pack REST endpoints at `tldw_Server_API/app/api/v1/endpoints/persona.py:3844`, `:3896`, `:4017`, `:4100`, `:4196`; schemas begin at `tldw_Server_API/app/api/v1/schemas/persona.py:25` and `:73`. | `apps/packages/ui/src/components/PersonaGarden/VisualPackEditor.tsx`, `VisualPackReusePanel.tsx`, and Buddy shell active-pack loader at `apps/packages/ui/src/components/Common/PersonaBuddy/BuddyShellHost.tsx:312`. | `persona_visual_packs`, assets, candidates, portability jobs. | Active pack loaded into Buddy shell; render state derived from voice/tool/wake/recovery and runtime overrides. | `persona_visuals.*` MCP tools for draft and transient runtime actions. | `tldw_Server_API/tests/Persona/test_persona_visuals_api.py:190`, `:528`, `:853`, `apps/tldw-frontend/e2e/workflows/persona-live.spec.ts:582`, `:627`. | Stage 1 should retain the current fail-open UI rule: broken packs must not block Live controls. New renderers remain Stage 4. |
| Personal visual library | Reference-backed schema at `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py:5270`; source lookup/upsert/list in `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py:2603`, `:2641`, `:2768`. | `VisualPackReusePanel.tsx` and `apps/packages/ui/src/services/persona-visuals.ts:176`. | Library items reference source persona and source pack; no V1 snapshots. | None unless a library item is used to create a target draft. | `persona_visuals.library_items` and `persona_visuals.use_library_item`. | `tldw_Server_API/tests/ChaChaNotesDB/test_persona_visual_library_db.py:49`, `tldw_Server_API/tests/Persona/test_persona_visual_library_service.py:100`, `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py:159`. | Stage 1 should only improve stale-source diagnostics/copy if surfaced in Persona Garden. Shared libraries/import-export expansion is later. |
| MCP persona visual tools | Tool definitions at `tldw_Server_API/app/core/MCP_unified/modules/implementations/persona_visuals_module.py:63`; runtime override at `:324`; draft mutation at `:346` and `:368`; library use at `:402`; generation enqueue at `:433`. | Persona Live incoming payload handling in `apps/packages/ui/src/routes/sidepanel-persona.tsx:609`; runtime override store in `apps/packages/ui/src/store/persona-visual-runtime.ts:20`. | Draft packs and jobs when mutation/generation tools run. | Transient visual override expires in client store. | `capabilities`, `library_items`, `trigger_state`, `create_draft_pack`, `update_manifest`, `use_library_item`, `enqueue_generation`. | `tldw_Server_API/app/core/MCP_unified/tests/test_persona_visuals_module.py:57`, `:114`, `:218`, `:267`, `:305`, `:393`; `apps/packages/ui/src/store/__tests__/persona-visual-runtime.test.ts:10`. | Stage 1 can add readiness diagnostics around missing context/user/session and draft-only restrictions. Do not add new tools yet. |
| Docs and test surface | Product/docs split Persona/Buddy from VN/CYOA in `Docs/Code_Documentation/Persona_Visual_Packs.md:167`; wake docs at `Docs/User_Guides/WebUI_Extension/Persona_Live_Wake_Phrases.md:1`; PRD reliability goals at `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md:68`. | N/A | N/A | N/A | N/A | See test rows above plus Persona Garden component tests under `apps/packages/ui/src/components/PersonaGarden/__tests__/`. | Stage 1 should add a short user-facing troubleshooting/diagnostics doc only after the diagnostics shape exists. |

## Evidence Table

| Flow | Journey | Evidence files | API/runtime contracts | Existing tests | Issue links | Observed or inferred gap | Severity | Stage 1 recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Persona Live connect/send/plan | User opens Persona Garden, connects Live, sends text, sees plan, approves/cancels. | `tldw_Server_API/app/api/v1/endpoints/persona.py:7177`, `apps/packages/ui/src/routes/sidepanel-persona.tsx:464`, `apps/packages/ui/src/services/persona-stream.ts:4`. | WS auth, event sequencing, `notice`, `assistant_delta`, `tool_plan`, `tool_call`, `tool_result`. | Backend WS tests and UI/E2E tests listed above. | #1388 closed; #635 still broad. | Diagnostics are distributed across logs, route state, voice card, and test coverage; user cannot inspect one status object. | P1 | Add diagnostics aggregator/card for websocket auth, selected persona, session id, last event, pending plan, tool status, and reconnect action. |
| Live voice and wake | User saves trigger phrases, connects Live, arms wake, phrase gates next spoken turn. | `schemas/persona.py:489`, `persona.py:9138`, `persona.py:9244`, `personaWakeDetector.ts:117`, wake guide. | Saved profile phrases plus runtime `voice_config`; server rejects if phrase is not both saved and active runtime config. | Wake backend and detector tests cover accept/reject/restart/fatal cases. | #1388 closed. | Rejection reasons exist server-side but are not clearly mapped to user remediation copy in the audit evidence. | P1 | Surface `WAKE_ACTIVATION_REJECTED` details in Live voice UI with exact saved-vs-runtime mismatch guidance. |
| Voice recovery | User encounters stalled listening/thinking, then can wait/reset/copy/reconnect. | `sidepanel-persona.tsx:723`, `AssistantVoiceCard` tests around recovery actions. | Client-owned recovery mode; reconnect forces disconnect and triggers session recovery. | `LiveSessionPanel.test.tsx` recovery tests. | #1428 closed. | Recovery actions exist, but end-to-end reliability around actual WS reconnect + preserving draft/last command should be smoked together. | P1 | Add one smoke test for stalled voice recovery path and document what each recovery action preserves or resets. |
| Buddy shell selection | User changes selected persona; Buddy shell shows route-local persona summary instead of stale persisted assistant. | `sidepanel-persona.tsx:241`, `BuddyShellHost.tsx:87`, `BuddyShellHost.tsx:312`. | Route-local context and selected-assistant fallback gate. | `sidepanel-persona.test.tsx:533`, `:619`, `:748`, `:816`; shell store tests. | #1388/#1428 closed. | Good coverage exists for stale assistant context; missing user-visible diagnostics for no Buddy row, disabled shell, or inactive surface. | P2 | Add a compact shell diagnostics/debug row in dev or diagnostics mode showing why Buddy is hidden/dormant. |
| Visual pack load/render | Active visual pack renders in Buddy; broken pack does not block Live controls. | `BuddyShellHost.tsx:312`, `:393`; `persona.py:3844`, `:4196`; PRD lines for fail-open behavior. | REST active pack plus client diagnostics. | `persona-live.spec.ts:582`, `:627`, visual API/core tests. | #1428/#1449/#1497 closed. | Failure states are handled, but diagnostics are mostly local to shell rendering and not tied into broader Persona health. | P2 | Include active-pack id/load status/render diagnostic in the Stage 1 Persona/Buddy diagnostics surface. |
| Persona Garden setup detours | First-run/setup moves between command creation, safety connections, live test, and handoff cards. | `sidepanel-persona.tsx:609`, Persona Garden setup components/tests. | Profile setup state, setup events, command dry-run, connection tests, live detour state. | Extensive route tests around setup and handoff failures. | #1388 closed. | The flow is well-covered but complex; smoke candidates should assert the core happy path and one failure detour. | P2 | Add a focused setup smoke checklist/test matrix rather than changing runtime behavior. |
| Persona Chat | User starts a persona-backed chat from assistant selection and gets profile/exemplar/memory projection. | `personaServerChat.ts:96`, `chat_service.py:450`, `exemplar_prompt_assembly.py:111`. | Conversation assistant identity plus persona profile projection. | Chat integration and prompt assembly tests. | #635 open. | Chat quality/eval work is distinct from Live/Buddy reliability; current tracker conflates them. | P3 for Stage 1 | Create or retitle a Persona Chat quality sub-issue under broader epic; do not implement in Stage 1. |
| MCP visual tools | Tool can inspect/modify draft visual packs or emit transient visual state override. | `persona_visuals_module.py:63`, `:324`, `:368`, `:402`, `:433`; `persona-visual-runtime.ts:20`. | Context-scoped persona/user/session; draft-only mutating tools; generation uses Jobs review queue. | MCP module tests. | #1389 closed. | Missing-context/draft-only failures are technically enforced, but not visible as a Persona readiness checklist. | P2 | Add MCP visual readiness diagnostics to Stage 1; hold new tools for Stage 3. |
| VN/CYOA boundary | Persona Visual Packs are Buddy/Live assets, not VN runtime assets. | `Docs/Code_Documentation/Persona_Visual_Packs.md:167`, `Docs/Product/WebUI/Persona_Live_Visual_Packs_PRD.md:27`, `Docs/Product/WebUI/Character_Chat_Terminology_Taxonomy_2026_05_09.md:14`. | N/A | Docs-only boundary plus previous visual-pack tests. | #1391 open. | Risk is roadmap drift, not current code failure. | P3 | Keep Stage 1 language and issue labels anchored to Persona/Buddy; link #1391 only as related/out of scope. |

## Known-Good Flow Checklist

| Flow | Current known-good path | Smoke/E2E candidate |
| --- | --- | --- |
| Setup happy path | Select persona, complete starter command/safety/test steps, record setup analytics, land in Live or Profiles with handoff card. | One route-level smoke that completes setup with dry-run success and verifies analytics event calls. |
| Setup failure detour | Live test failure or command no-match detours to Live/Commands and returns after retry/save. | One route-level smoke for `dry_run no match -> command draft -> save -> return to test`. |
| Persona Chat | Select persona assistant, ensure server chat with `assistant_kind=persona`, send message, verify persona memory mode. | One existing chat integration check is enough for Stage 1 unless Buddy entry points add a new Persona Chat CTA. |
| Persona Live text | Connect WS, send `user_message`, receive plan, approve/cancel. | Existing Playwright mocked-WS flow plus backend WS tests; add diagnostics assertions when Stage 1 card exists. |
| Live voice manual | Connect WS, send voice config, commit transcript, receive processing notice/plan or fallback notice. | Backend WS voice tests already cover; add UI smoke around recovery copy/reconnect. |
| Wake | Save trigger phrase, arm wake, detector matches phrase, server accepts activation, next voice turn bypasses trigger gate. | Existing detector/backend tests; add UI assertion that rejected server reason appears in Live panel. |
| Buddy display | Route-local selected persona supplies Buddy summary; shell opens, persists bucketed position, falls back cleanly when dormant. | Existing shell/route tests cover; add diagnostics visibility if shell hidden/dormant. |
| Visual fallback | Active pack loads and follows state overrides; broken pack does not block Live controls. | Existing E2E covers active and broken pack. Stage 1 should add diagnostics detail assertion. |
| MCP runtime trigger | `persona_visuals.trigger_state` returns `visual_state_override`, client stores transient override, Buddy resolves visual state. | Existing MCP and runtime store tests cover pieces; add integration assertion only if diagnostics reads MCP readiness. |
| Recovery | Stalled voice or reconnect path lets user wait, reset, copy command, or reconnect. | Add one smoke for `recovery_mode -> copy last command -> reconnect` preserving expected text/session state. |

## Flow-By-Flow Findings

### Persona Live

The websocket is server-owned and eventful enough to support diagnostics. It already carries session id, event sequence, reason codes, tool status, and processing notices. The frontend owns composition of route state, approvals, setup detours, voice controller, and Buddy shell context. Stage 1 should avoid changing stream semantics unless diagnostics reveal a missing field; a read-only diagnostics projection is lower risk.

### Wake/Voice

Wake support is explicit V1 browser-side speech recognition, not always-on background capture. The server correctly requires the phrase to be both saved on the selected profile and present in the live runtime config. The likely user-facing gap is explaining which side is missing and how to fix it. Stage 1 should make server reason codes actionable in the Live UI and setup docs.

### Buddy Shell

The shell has a clean selection model: route-local persona context wins, selected-assistant fallback only applies when explicitly allowed. Active-pack loading is isolated so visual failures do not block Live controls. Stage 1 should expose hidden/dormant reasons and active-pack diagnostics rather than inventing new shell behaviors.

### Persona Garden

Persona Garden is the owner of setup, profile defaults, voice defaults, commands, policies, connections, state docs, visuals, and setup analytics. Existing tests cover many detours and failure paths. Stage 1 should keep work concentrated on observability and smoke coverage for the first-run route rather than adding new setup steps.

### Persona Chat

Persona-backed chat exists through ordinary chat infrastructure and has real prompt/memory/exemplar integration. This should be tracked separately from Buddy shell/Live reliability. The current `#635` content is useful as research/input, but the issue title/body are too broad to drive Stage 1.

### Visual Packs And Library

The current implementation matches the accepted product direction: assets are user-owned, attached to a persona by default, stored as packs/manifests, and personal-library items are reference-backed. V1 should not add snapshots back. Stage 1 diagnostics can show active pack/library/source status, but import/export/shared-library expansion should stay later.

### MCP Visual Tools

The MCP module provides enough current capability for composability: read capabilities/library, transient runtime trigger, draft pack creation/update, library use, and review-gated generation enqueue. Stage 1 should only add readiness/error reporting around these existing tools. New tool categories belong in Stage 3.

## Stage 1 Issue Recommendations

1. Persona/Buddy diagnostics surface
   - Scope: read-only diagnostic card/API helper, selected persona id/name, profile load status, buddy summary presence, active visual pack id/status/diagnostic, websocket connected/session id/last event, wake armed/state/rejection reason, MCP persona_visuals readiness.
   - Out of scope: new MCP tools, new renderer behavior, Persona Chat quality changes.

2. Live voice/wake recovery copy and reason mapping
   - Scope: map existing server reason codes and client recovery modes to user-facing remediation in Live Session, including saved-profile/runtime wake mismatch and reconnect/copy/reset actions.
   - Out of scope: native/background wake, new STT/TTS providers.

3. Persona Garden setup smoke coverage
   - Scope: one happy-path setup smoke and one failure-detour smoke using existing flows; preserve route-local Buddy context assertions.
   - Out of scope: new setup wizard steps or new command semantics.

4. Visual pack fail-open diagnostics
   - Scope: centralize active-pack load/render diagnostics into the broader Persona/Buddy diagnostics surface and assert Live controls remain usable.
   - Out of scope: renderer adapters, Live2D provider work, shared libraries.

5. Tracker hygiene
   - Scope: create/refresh a broader Persona/Buddy epic, link this audit, preserve `#635` references, split Persona Chat quality into a separate Stage 2 issue, and link `#1391` as VN/CYOA-related but out of Stage 1.
   - Out of scope: closing `#635` without first preserving its useful references.

## Explicit Non-Goals And VN/CYOA Boundary

- Do not route Persona/Buddy work through VN/CYOA runtime ownership.
- Do not add Live2D or additional renderer/provider adapters in Stage 1.
- Do not add always-on/native/background voice in Stage 1.
- Do not broaden MCP persona tools beyond diagnostics/readiness in Stage 1.
- Do not change personal visual library storage away from reference-backed V1 semantics.
- Do not make Persona Chat quality/eval work a dependency of reliability diagnostics.

## Verification Commands And Skipped Checks

Fresh verification for this docs-only audit:

```bash
git diff --check
git status --short --branch
```

Result: `git diff --check` passed with no output before staging. `git status --short --branch` showed only the new audit report and the Stage 0 Backlog task as untracked changes.

Not run:

- Pytest/Vitest/Playwright were not run for this audit because no runtime code or tests changed. Existing tests were inspected by source path and line references.
- Bandit is skipped for this task because the touched scope is Markdown-only documentation and Backlog metadata. No Python production or test code changed.
