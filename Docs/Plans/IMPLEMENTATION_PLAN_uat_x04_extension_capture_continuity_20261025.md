# X04 extension capture and continuity implementation plan

> Execute inline with the existing executing-plans workflow and the one authorized read-only reviewer. No additional agents.

**Task:** TASK-13260.278.18.83.45; related historical TASK-13260.277.18 does not supply current acceptance.
**Goal:** Validate and repair the packaged extension's complete selected X04 flow before S01.
**Architecture:** Use existing WXT Chrome production packaging, real extension background/content/options/sidepanel entries and the same current API/WebUI contracts. A disposable browser profile owns extension storage; original runtime and browser remain protected.
**Tech stack:** installed Bun1.3.2, Node20.19.5, WXT, React18.3.1, Chrome156 and supported Python3.12 API; official restricted PostgreSQL fixtures when used.
**Spec:** Docs/Development/RELEASE_UAT_PLAYBOOK.md X-04 and apps/tldw-frontend/scripts/live-tier-uat/release-catalog-data.mjs X-04.
**ADR check:** No new durable decision for validation; retain ADR049 identity and ADR059 task workflow. Reassess each actual repair before implementation.

## Fixed constraints

- Candidate f0eb7c175ccf887a0adc302eb440e804bd4dd485 includes all current fixes on freshly verified dev a412f6580b7ed0cf48fd7d222a47a20cfc0dde92, including merged3309/3310/3311.36/43 selected families accepted; X04 open.
- Preserve original API50010/49993, frontend50863/50855, Chrome54716/54710, profile/data/keys, original URL/five messages/draft/canonical objects and restoration dependencies. No original cycle/signals.
-600s host/build/run;120s startup;180s actual generation;300s case;15s operation/graceful shutdown. No force. Diagnose actual failures, record attempts and review a changed approach; broad human necessary diagnostics/retries authorization persists.
- Existing dependencies only; no installs/downloads/models/Splash process changes, raw operator SQL/homemade PostgreSQL, assertion weakening, proof bundles or foreign private evidence/log recovery.
- One fresh private extension Chrome window/profile; at most two page tabs plus its real native sidepanel. This necessary X04 scope is covered by the human's all-necessary-UAT authority. Close the private window/profile and dispose completed payloads. Never mutate original extension/browser storage.
- No storage/auth/onboarding injection, seeded offline/readiness bypass, handler calls pretending to be context-menu events or ordinary-tab sender identity pretending to be a native panel. Existing fixture-only sidepanel tests remain controlled source checks.
- Review screenshots are temporary and removed after human UX acceptance. Normal outputs are functional package artifacts, not proof copies. No GitHub publication/queue/merge; heartbeat paused.

## Stage1: Bind source and package the actual extension
**Goal:** Produce an unchanged normal candidate extension with declared permissions and working installed tooling.
**Success criteria:** Exact source/head/lock/public-link bindings, installed dependency resolution and protected owners/build unchanged; Chrome production build and token verification pass inside600s; built manifest/options/sidepanel/background/content entries inspected.
**Files:** apps/extension/package.json, wxt.config.ts, scripts/build-with-profile.mjs, scripts/post-build-tasks.mjs, entrypoints; shared apps/packages/ui/src/entries and assets; existing build helper patterns in private x01_current_frontend_build.py.
**Tests:** Guard exclusive output paths, current source and canonical symlink; snapshot tracked app source in memory before/after the normal command; no source edits merely for packaging. Inspect actual manifest permission scope before native install.
- [ ] Run installed normal `bun run build:chrome:prod` from apps/extension with pruned nonsecret environment and existing Node20/Bun.
- [ ] Confirm exact source preservation, native package entries, normal token check, protected owners and original build.
**Status:** In Progress

## Stage2: Native core workflow
**Goal:** Validate five declared core variants with actual browser extension APIs and real server persistence.
**Inputs:** Serve the playbook's exact F-SOURCE Rowan text on an owned loopback page. Select `The observatory's main telescope is named Selene.`; preserve selection bytes and source URL. Generate two small factual card drafts and ask which telescope Rowan uses, using the already running configured provider.
**Source map:** entries/background.ts save-to-notes-pa/contextMenus handler; routes/sidepanel-chat.tsx, sidepanel-note-capture.ts, sidepanel-flashcards.tsx; services/sidepanel-chat-handoff.ts and services/tldw/sidepanel-chat-webui-handoff.ts; normal Options connection/auth UI and flashcard APIs.
| Variant | Actual oracle |
|---|---|
| packaged-install-login | Fresh profile installs current production package, normal connect/login identifies intended account/server; required/optional permissions correspond to selected server/source. |
| capture-web-handoff | Real selection/context-menu opens editable draft with exact selection+URL; normal Save and full WebUI show identical canonical content/provenance. |
| sidepanel-card-study | Real provider returns reviewed two-card draft; edit/save exact count, reload same IDs, one real due-card scheduling event, fuller WebUI controls reachable. |
| sidepanel-chat-continuity | One real grounded answer, no invented facts; native panel close/reopen and actual source reload retain correct conversation/draft with no duplicate messages or stale capture. |
| cross-surface-account-switch | Alice/Bob own-resource positive controls plus opposite denials; normal logout/login and WebUI/extension entries preserve documented origin/account/credential scope without shared-profile assumptions. |
**Modes:** Record actual SQLite/PostgreSQL single-/multi-user bindings and which unchanged-contract observations compose. No whole-family pass from one backend or seeded identity; fresh official per-test restricted PostgreSQL only.
- [ ] Install and connect normally; execute each core row sequentially and record saved IDs/content in memory/concise notes.
- [ ] Validate required account/backend bindings with exact API/browser lifecycle and original preservation.
**Status:** Not Started

## Stage3: Recovery and minimal causal fixes
**Goal:** Validate all six recovery variants and fix actual defects at shared roots.
| Variant | Actual oracle |
|---|---|
| restricted-page | Native restricted page reports unsupported capture, requests no unrelated permission and creates no false draft/content. |
| no-selection | Normal action without selection gives useful capture guidance and no fabricated source text. |
| permission-denied | Deny the requested optional source/server permission normally; operation remains denied, draft retained appropriately, legitimate permission grant recovers. |
| offline-server | Actual owned context/service outage gives actionable connection state, preserves capture draft, normal reconnect works without duplicate save. |
| expired-session | Real supported credential/session expiry gives normal login recovery and no prior-account content exposure. |
| narrow-sidepanel | Actual narrow native panel retains readable essential controls, keyboard/focus/cancel and truthful recovery, with measured layout and axe triage. |
**Files/tests:** Existing shared route/entry/transport/ownership tests and actual extension tests/e2e are selected only when their changed behavior is relevant. Existing UXR sidepanel helpers inject auth/sender identity and cannot certify these native rows. Extension language metadata uses split entry writers; inspect actual rendered locale before assuming X03 WebUI metadata qualification transfers.
- [ ] Record source/caller map and causal failing regression before each behavioral repair.
- [ ] Implement smallest shared correction; run connected configured checks, parse/lint and touched-scope security; independent actual+staged review, normal hooks/local commits.
- [ ] Package changed source and confirm failed native workflow without replaying unrelated qualified controls.
**Status:** Not Started

## Stage4: Human UX, dispose and close
**Goal:** Finish X04 before S01 with honest variant/mode limits and no retained completed bulk.
**Success criteria:** All eleven variant dispositions, actual saved/provenance/ownership/provider/recovery and lifecycle checks; separate human UX screenshot review; original session preservation, independent final review and guarded completed-fixture/package/operator cleanup.
**Tests:** Exact original canonical/owner/session/draft checks, actual graceful private service/browser release, task format/links/diff check and normal hooks. Do not infer terminal reads from browser detachment.
- [ ] Present settled native extension and handoff screenshots for human review while independent checks continue.
- [ ] Update existing task/master count only after all selected X04 gates pass; remove only this own completed plan and completed disposable payloads.
**Status:** Not Started


## First normal build and causal repair scope

X04 normal production Chrome build attempt1 FAILED naturally2.829s before packaging/nativebrowser. Actual WXT ViteNode evaluates document-get-scope.ts with window present but location undefined; top-level window.location.protocol throws. All7748consumedinputs/originalbuilds/publiclink/protectedowners preserved. Actual callers inventoried: background-proxy,api-send,TldwApiClient; shared helper added for WebUI document GET retirement and must preserve HTTP pagehide/pageshow/BFCache renewal, unsafe-method exclusion and extension runtime exclusion. Add real-module import/lifecycle controls for windowless/WXT-stub/HTTP/extension scopes; establish causal red before minimal location-presence guard. No application import shim/fixture weakening or speculative transport change. Reviewer additionally requires absence/approved nonsecret inputs for eight WXT production Chrome dotenv filenames and ownership of .wxt/output/export; add preflight before retry, never print env contents. Native permission scope still awaits actual successful generated manifest; do not invoke existing E2E setup which patches host permissions. X04 remains open36/43.


## Shared initializer causal verification

X04 actual WXT locationless-window module regression reproduced1failed3passed0.899s on unchanged shared initializer. Minimal repair changes only HTTP predicate to window.location?.protocol ?? empty string; real HTTP pagehide/pageshow/BFCache renewal and GET/runtime exclusions unchanged. Four new environment/lifecycle controls plus20genuinely affected existing X02document GET cases PASS24/config, UI2.364s/WebUI2.507s;86unrelated refresh cases intentionally deselected in each config, no full-module claim. BothchangedTSfiles parse; proper apps-base ESLint0errors0warnings/noignored, inherited Pages-discovery advisory. Newtest normal formatting structurally AST-identical. Eight WXT productionChrome dotenv candidates all absent without contents read; .wxt/.output owned sameUID/noalias, build export absent. Source reviewer assessing; normalhooks/localcommit then unchangednormal productionbuild retry with additional preflight. TS-only BanditN/A. No native X04 acceptance yet,36/43; no browser/model/originalcycle/publication.
