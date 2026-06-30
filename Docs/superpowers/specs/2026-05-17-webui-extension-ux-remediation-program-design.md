# WebUI And Extension UX Remediation Program Design

Date: 2026-05-17
Owner: Codex collaboration session
Status: Ready for user review
Backlog: TASK-417
Source audit: `Docs/Reviews/WEBUI_EXTENSION_UX_HCI_AUDIT_2026_05_17.md`

## Summary

This spec turns the WebUI/extension UX/HCI audit into a coverage-driven
remediation program. It is not a broad visual redesign and it is not a set of
74 disconnected route patches. The plan uses shared work packages first, then
route-family remediation, with explicit coverage matrices proving that every
severity-ranked issue and every audited root page has an owner.

The core product problem is structural: the WebUI has many capable surfaces, but
route truth, user-facing labels, self-hosted capability states, setup state,
mobile layout behavior, headings, and command/navigation targets are not governed
by one contract. Fixing individual pages before fixing those shared causes would
repeat the same work and leave the product inconsistent.

## Goals

1. Address every P1/P2/P3 issue identified in the audit with a named remediation
   package and acceptance proof.
2. Address every discovered root/top-level route from the audit with at least one
   remediation owner.
3. Preserve self-hosted product intent, local-first capability variation, and
   expert density where those patterns are useful.
4. Prevent broad visual drift by prioritizing route contracts, capability states,
   setup flow, responsive behavior, page landmarks, and workflow clarity.
5. Create a spec that can be converted into staged implementation plans without
   needing another discovery pass.

## Non-Goals

- Do not redesign the entire app shell visually.
- Do not create a new design system.
- Do not rename routes without a compatibility and alias policy.
- Do not hide advanced tools merely because they are advanced.
- Do not change backend APIs unless a package explicitly proves the frontend
  cannot solve the UX issue responsibly without backend support.
- Do not combine all remediation into one implementation PR.
- Do not treat hosted-only, labs, debug, and self-hosted routes as the same
  product surface.

## Current Evidence

The remediation plan is based on:

- 74 audited root/top-level routes in the route-by-route audit section.
- 124-route connected desktop browser sweep from the smoke inventory.
- 9 extra page-file routes omitted from the smoke inventory.
- 11-route mobile core sweep at 390px width.
- Route ownership evidence from Next pages, shared route registries, extension
  route registries, command palette, mode selector, header shortcuts, and shared
  layouts.

Important source files and systems:

- `apps/tldw-frontend/pages`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- `apps/packages/ui/src/components/Layouts/ModeSelector.tsx`
- `apps/tldw-frontend/components/layout/WebLayout.tsx`
- `apps/tldw-frontend/components/layout/Header.tsx`

## Remediation Principles

### Coverage Before Polish

Every audit issue and every root route must map to a package. The plan can defer
implementation, but it cannot leave an issue ownerless.

### Shared Causes Before Page Fixes

Route drift, raw endpoint errors, missing headings, mobile overflow, command
palette mismatches, and setup-shell confusion are repeated defects. They should
be fixed through shared contracts before page-local polish.

### Preserve Power-User Density

Pages such as `/media`, `/knowledge`, `/mcp-hub`, `/evaluations`, and parts of
`/settings` already contain useful power-user affordances. The plan should make
those controls better organized, not remove them.

### Self-Hosted States Are Product States

Unavailable endpoints, missing workers, unsupported modules, hosted-only pages,
and local capability variation are expected in this product. They must be
represented in user language, with diagnostics available behind disclosure.

### Route Names Are UX

Aliases, redirects, hosted-only pages, labs routes, debug previews, extension
sidepanel routes, and canonical routes must be declared intentionally. Users
should not discover product structure by trial and error.

### Interaction Before Explanation

Use visible copy to label state, actions, and recovery paths. Do not treat
paragraph-level in-app explanation as the primary fix for unclear route
ownership, missing system state, overloaded controls, or weak hierarchy.

## Work Packages

### WP1: Canonical Route Contract And Visibility Policy

Goal:
Create one source of truth for user-facing route labels, route groups, canonical
paths, aliases, hosted/self-hosted visibility, extension sidepanel availability,
labs/debug classification, smoke inventory inclusion, and command palette
targets.

Audit findings addressed:
F1, F8, F12, F17, F18.

Routes covered:
All root routes, with special focus on aliases and visibility-sensitive routes:
`/`, `/chat`, `/search`, `/reading`, `/review`, `/audio`, `/prompt-studio`,
`/workspace-playground`, `/billing`, `/account`, `/signup`,
`/composer-variants-preview`, `/onboarding-test`, `/vn-assets`, `/vn-play`.

User outcomes:
First-time users see stable labels and do not stumble into debug/hosted/labs
surfaces unintentionally. Power users can trust command search, nav labels,
route aliases, and extension route availability.

Implementation scope:
Route metadata, route visibility fields, command palette route targets, nav
generation or validation, smoke inventory generation or validation, sidepanel
availability matrix, docs updates.

Dependencies:
None. This package should start first.

Out of scope:
Visual restyling and page-specific workflow redesign.

Acceptance criteria:
- Every audited root route has route metadata.
- Every route is classified as default self-hosted, advanced self-hosted,
  hosted-only, admin/operator, extension sidepanel, labs/beta, internal QA/debug,
  legacy alias, redirect, or deprecated.
- Command palette "Go to Chat" targets `/chat` or is renamed if it targets `/`.
- The smoke inventory is generated from, or checked against, the route contract.
- Extension sidepanel availability is explicit and tested.

Verification:
Route metadata tests, command palette target tests, sidepanel matrix tests, route
inventory tests, and browser smoke on canonical and alias routes.

### WP2: Shared Capability And Error State System

Goal:
Replace raw endpoint and capability failures with reusable, user-language states
for self-hosted deployments.

Audit findings addressed:
F4, F9, F5, F18.

Routes covered:
`/sources`, `/scheduled-tasks`, `/integrations`, `/admin`, `/agents`,
`/agent-tasks`, `/acp-playground`, `/settings/model`, `/evaluations`,
`/mcp-hub`, `/skills`, `/tts`, `/speech`, `/data-tables`.

User outcomes:
Users can distinguish no data, unavailable server capability, missing worker,
missing permission, not configured, degraded, unsupported version, and network
failure. Technical details remain available for operators.

Implementation scope:
Shared capability state component, copy vocabulary, error mapping helpers,
diagnostics disclosure, route adoption plan, and targeted backend capability map
dependency if frontend probing cannot distinguish states.

Dependencies:
WP1 helps determine which routes need user-facing capability states.

Out of scope:
Implementing missing backend modules.

Acceptance criteria:
- No primary route state displays raw text like `Not Found (GET <endpoint>)`.
- Each capability state has a user-language diagnosis and at least one next
  action.
- Raw endpoint details are behind disclosure or diagnostics.
- `/sources` and `/scheduled-tasks` use the shared state.

Verification:
Component tests for each state type, route tests for representative failures,
browser QA on `/sources`, `/scheduled-tasks`, `/integrations`, and model
settings.

### WP3: First-Run Setup And Connection Flow

Goal:
Make first-run and connection recovery focused, trustworthy, and separate from
chat-oriented chrome.

Audit findings addressed:
F3, F15, F1.

Routes covered:
`/`, `/setup`, `/login`, `/signup`, `/account`, `/profile`, `/privileges`,
`/config`, `/billing`, `/404`.

User outcomes:
New users can tell whether they should connect a self-hosted backend, try demo
mode, or log in. Operators can diagnose server URL, API key, auth mode, and
health state.

Implementation scope:
Setup-only shell, home resolver, connection summary, backend URL semantics,
auth-mode-specific login/signup/account states, placeholder route visibility,
404 recovery.

Dependencies:
WP1 for route visibility and alias policy. WP2 for unreachable/degraded states.

Out of scope:
Auth backend changes unless required to expose current auth mode or health.

Acceptance criteria:
- `/setup` has a single semantic `h1` and does not show chat-specific chrome.
- `/` routes or renders based on connection state.
- Frontend origin and backend API target are distinguishable.
- Hosted-only account/signup/billing pages are hidden or clearly classified in
  self-hosted mode.

Verification:
Browser QA for unconfigured, configured, demo, and degraded states; route tests
for `/`, `/setup`, `/login`, `/signup`, `/account`, `/billing`, and `/404`.

### WP4: Responsive App Shell And Page Landmark Contract

Goal:
Eliminate root-page horizontal overflow and establish page orientation rules for
desktop, mobile, and extension-sidepanel widths.

Audit findings addressed:
F2, F13, F15, F10, F11.

Routes covered:
`/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`,
`/workspace-playground`, `/setup`, `/sources`, `/mcp-hub`, `/stt`, `/tts`,
`/chat-workspace`.

User outcomes:
Users can complete core workflows at 390px and sidepanel widths. Screen-reader
and keyboard users get stable route landmarks.

Implementation scope:
Shared shell constraints, no-horizontal-overflow tests, semantic `h1` policy,
settings mobile nav, media master-detail behavior, chat composer safe area,
workspace/prompts container fixes.

Dependencies:
WP1 for user-facing route labels. WP3 may consume setup shell rules.

Out of scope:
Changing the visual language of all pages.

Acceptance criteria:
- User-facing root routes have one semantic `h1` or documented exception.
- Representative root routes have no page-level horizontal overflow at 390px.
- Settings nav, media list-detail, chat composer, prompts, and workspace panels
  have explicit narrow-width behavior.

Verification:
Playwright or equivalent assertions for scroll width, heading checks, touch
target follow-up audits, and before/after screenshots.

### WP5: Settings And Model Provider Configuration

Goal:
Make settings task-led and make model/provider setup start with configured,
usable options instead of a full catalog.

Audit findings addressed:
F5, F7, F11, F16, F15, F2.

Routes covered:
`/settings`, `/settings/model`, `/login`, `/privileges`, `/prompts`,
`/prompt-studio`, settings subroutes.

User outcomes:
First-time users can configure provider keys and default model without scanning
hundreds of unavailable models. Power users can search settings and safely manage
data/system actions.

Implementation scope:
Settings grouping, responsive settings nav, search synonyms, provider keys label
fix, model/provider health summary, configured-first model UI, full catalog
behind search/filter, destructive action separation.

Dependencies:
WP1 for route metadata. WP2 for provider/capability failures. WP4 for
responsive nav and headings.

Out of scope:
Provider API redesign unless health/status data cannot be represented with
existing APIs.

Acceptance criteria:
- `providerKeys.navTitle` is not visible.
- Model settings show configured/usable providers and default model before full
  catalog.
- Destructive/data management actions are separated from routine preferences.
- Mobile settings has no page-level horizontal overflow.

Verification:
Settings label tests, model settings state tests, browser QA on `/settings` and
`/settings/model`, mobile overflow checks.

### WP6: Chat, Composer, And Global Chrome

Goal:
Make chat composer-first, separate global app navigation from chat-context
controls, and restore trust in keyboard/command navigation.

Audit findings addressed:
F6, F8, F13, F2, F15.

Routes covered:
`/chat`, `/quick-chat-popout`, `/knowledge`, `/media`, `/sources`,
`/settings`, `/mcp-hub`, `/stt`, `/tts`.

User outcomes:
New users can send a first chat message without choosing among multiple modes.
Power users can rely on command palette, shortcuts, and context-specific chat
controls.

Implementation scope:
Chat empty state, composer safe layout, compact mode launcher, command palette
target fix, context-specific global actions, route-level chrome policy.

Dependencies:
WP1 for route/action metadata. WP4 for responsive safe-area behavior.

Out of scope:
Rebuilding chat runtime, changing LLM APIs, or redesigning all chat controls at
once.

Acceptance criteria:
- `/chat` primary screen foregrounds the composer and model readiness.
- Mode cards move into progressive disclosure.
- "Go to Chat" opens `/chat`.
- Non-chat pages do not foreground chat-only controls as their primary header
  actions.

Verification:
Command palette tests, chat route browser QA, 390px chat screenshot, keyboard
navigation smoke.

### WP7: Persona, Characters, Context Assets, Companion, And Agents

Goal:
Clarify the relationship among persona, character, companion, buddy/visual
fallback, and agents.

Audit findings addressed:
F1, F9, F15, F18.

Routes covered:
`/persona`, `/characters`, `/companion`, `/agents`, `/agent-tasks`,
`/acp-playground`, `/chat-workflows`, `/dictionaries`, `/world-books`.

User outcomes:
First-time users understand what persona, character, context asset, companion,
or agent they are managing and how to launch or attach it. Power users get fast
switching, launch, activation scope, setup status, and degraded capability
diagnostics.

Implementation scope:
Page headings, concept copy, launch-to-chat actions, context activation scope,
readiness/status summaries, agent capability states, links between
Persona/Characters/Dictionaries/World Books/Companion/Agents/ACP.

Dependencies:
WP1 for route taxonomy. WP2 for ACP/agent capability states. WP6 for chat
integration.

Out of scope:
New persona runtime features or visual-pack generation.

Acceptance criteria:
- Persona, Characters, Companion, and Agents each have a distinct primary job.
- Character and persona pages expose "use/start in chat" actions.
- Agent unavailable/degraded states use shared capability language.

Verification:
Route browser QA, heading checks, capability-state tests, and command/nav checks.

### WP8: Media, Library, Review, And Sharing

Goal:
Keep large-library power while making first selection, mobile browsing, review,
trash, reading, notes, sharing, and chatbook concepts understandable.

Audit findings addressed:
F10, F2, F1, F18, F15.

Routes covered:
`/media`, `/media-multi`, `/review`, `/media-trash`, `/items`, `/collections`,
`/reading`, `/notes`, `/shared`, `/chatbooks`, `/chatbooks-playground`,
`/notifications`.

User outcomes:
New users can pick or ingest a first item. Power users can filter, bulk act,
recover, restore, and manage library objects efficiently.

Implementation scope:
Media list-detail responsive behavior, first-selection empty state, review alias
policy, trash safeguards, library terminology, collections/reading alias policy,
notes heading, shared tabs, chatbooks import/export/jobs framing.

Dependencies:
WP1 for aliases and route labels. WP4 for media mobile layout. WP2 for
capability states where needed.

Out of scope:
Changing media ingestion backend behavior.

Acceptance criteria:
- `/media` has usable list-detail behavior at 390px.
- `/review` and `/reading` have canonical alias behavior.
- Trash destructive actions show policy and safeguards.
- Shared and Chatbooks roots explain object type, direction, and next action.

Verification:
Browser QA for `/media`, `/media-trash`, `/collections`, `/notes`, `/shared`,
and `/chatbooks`; mobile overflow checks; alias route tests.

### WP9: Knowledge, Research, Workspace, And Transform Tools

Goal:
Define a clear product ladder: Ask, Research, Workspace, Transform.

Audit findings addressed:
F14, F1, F2, F15.

Routes covered:
`/knowledge`, `/search`, `/research`, `/workspace-playground`,
`/chat-workspace`, `/document-workspace`, `/repo2txt`, `/model-playground`,
`/writing-playground`, `/presentation-studio`.

User outcomes:
Users can choose the right surface without memorizing product history. Experts
can recover research runs, workspaces, transforms, and generated artifacts.

Implementation scope:
Route labels, subtitles, ladder copy, aliases, workspace no-state headers,
research run framing, transform tool headings, mobile workspace layout, recent
items where existing data supports it.

Dependencies:
WP1 for route names and aliases. WP4 for responsive behavior. WP6 for chat
controls where routes include chat. WP11 owns `/audiobook-studio`, with WP9
providing only product-ladder and transform-label alignment.

Out of scope:
Full research engine redesign or new artifact systems.

Acceptance criteria:
- `/knowledge` remains direct cited Q&A.
- `/research` uses user-language research run framing.
- `/workspace-playground` gets canonical label policy.
- Transform tools explain inputs, outputs, persistence, and export.

Verification:
Browser QA on `/knowledge`, `/research`, `/workspace-playground`,
`/chat-workspace`, `/repo2txt`, and `/model-playground`; route label tests.

### WP10: Operations, Automation, And Integrations

Goal:
Make operator surfaces status-first and capability-aware.

Audit findings addressed:
F4, F9, F12, F17, F18.

Routes covered:
`/admin`, `/mcp-hub`, `/sources`, `/connectors`, `/integrations`,
`/scheduled-tasks`, `/watchlists`, `/workflow-editor`, `/skills`.

User outcomes:
Operators can see what is available, degraded, unavailable, or not configured,
then take the next action without reading raw endpoint text.

Implementation scope:
Admin landing, operations health/capability map, sources/integrations/scheduled
tasks capability states, watchlists dashboard improvements, workflow editor
heading/status, skills capability/empty states.

Dependencies:
WP1 and WP2. WP4 for headings and responsive behavior.

Out of scope:
Building missing integrations or scheduling backends.

Acceptance criteria:
- `/sources` and `/scheduled-tasks` no longer show raw endpoint errors as the
  main UI.
- `/admin` shows an operations overview before module drill-down.
- `/mcp-hub` keeps workflow-first setup and adds clearer status summary.
- Watchlists expose current monitor/feed health and repeat-user controls.

Verification:
Browser QA on `/admin`, `/mcp-hub`, `/sources`, `/integrations`,
`/scheduled-tasks`, `/watchlists`; capability-state fixtures.

### WP11: Audio, Study, Safety, And Specialized Tools

Goal:
Improve route identity, readiness, and classification for audio, study, safety,
and advanced/labs tools.

Audit findings addressed:
F2, F9, F15, F18, F19.

Routes covered:
`/speech`, `/audio`, `/stt`, `/tts`, `/audiobook-studio`, `/evaluations`,
`/flashcards`, `/quiz`, `/moderation-playground`, `/content-review`,
`/claims-review`, `/data-tables`, `/chunking-playground`, `/kanban`,
`/vn-assets`, `/vn-play`.

Implementation split:
WP11 is an umbrella package. Implementation plans must keep it split into
WP11A for audio routes (`/speech`, `/audio`, `/stt`, `/tts`,
`/audiobook-studio`) and WP11B for study, safety, review, data, chunking,
kanban, and VN routes. Route coverage rows below use that split directly.

User outcomes:
Users understand whether a route is a production tool, advanced tool, beta,
labs, hosted-only, or internal/debug surface. Power users get readiness and
recovery state for providers, workers, study data, and specialized workflows.

Implementation scope:
Audio route canonicalization, STT/TTS headings/readiness, evaluation presets,
flashcard/quiz mode framing, moderation/content review route identity,
data-table and chunking advanced tool framing, labs classification for VN/Kanban
if appropriate, deprecation cleanup only where it blocks UX work.

Dependencies:
WP1 for classification. WP2 for readiness/capability. WP4 for headings and
mobile behavior.

Out of scope:
New STT/TTS engines, new quiz algorithms, or VN runtime redesign.

Acceptance criteria:
- `/audio`, `/speech`, `/stt`, and `/tts` have one explicit audio model and
  canonical labels.
- Study routes expose Review, Author/Generate, Import/Export, and Results modes
  clearly where applicable.
- Claims/content review aliasing is intentional.
- Labs/specialized routes are classified and filtered appropriately.

Verification:
Route heading checks, browser QA on representative audio/study/safety routes,
route visibility tests, and targeted deprecation cleanup tests when changed.

### WP12: QA, Regression, And Route Governance

Goal:
Prevent the same UX failures from recurring.

Audit findings addressed:
All findings, especially F2, F15, F17, F18.

Routes covered:
All root routes and representative child routes used by nav, settings, admin,
sidepanel, and smoke tests.

User outcomes:
Future route additions are classified, tested, and visible only in appropriate
contexts. Regression risk drops for navigation, headings, mobile overflow,
capability states, and command targets.

Implementation scope:
Generated route inventory checks, route metadata lints, heading tests, no
horizontal overflow smoke, command palette target tests, sidepanel matrix tests,
capability-state fixtures, screenshot baseline protocol.

Dependencies:
WP1 defines route metadata. WP2 defines capability states. WP4 defines
responsive and heading gates.

Out of scope:
Full visual regression suite for every page unless later justified.

Acceptance criteria:
- Route inventory cannot silently omit user-facing page-file routes.
- Root pages without headings fail tests unless explicitly excepted.
- Core mobile routes fail tests on page-level horizontal overflow.
- Capability errors have route-level fixtures.
- Command palette route targets are validated.

Verification:
CI-compatible tests where feasible, local browser QA scripts, evidence artifact
updates, and diff checks for documentation.

## Finding Coverage Matrix

| ID | Finding | Severity | Primary package | Supporting packages | Acceptance proof |
|---|---|---:|---|---|---|
| F1 | Navigation and IA are fragmented across multiple source-of-truth systems | P1 | WP1 | WP6, WP8, WP9, WP12 | Route metadata drives or validates nav, command palette, sidepanel, and smoke inventory. |
| F2 | Mobile layouts for core root pages overflow instead of reflowing | P1 | WP4 | WP5, WP6, WP8, WP9, WP12 | 390px overflow checks pass for `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, `/workspace-playground`. |
| F3 | Setup and onboarding are mixed with chat-oriented global chrome | P1 | WP3 | WP1, WP4 | `/setup` uses setup shell; `/` resolves by connection state. |
| F4 | `/sources` exposes raw technical 404 as the main page state | P1 | WP2 | WP10, WP12 | `/sources` primary state uses capability copy and next actions, not raw endpoint text. |
| F5 | Model settings overwhelm users with full catalog before usable configuration | P1 | WP5 | WP2, WP12 | Configured-first provider/model view, health summary, full catalog collapsed or filtered. |
| F6 | Global chrome is chat-specific on non-chat pages | P2 | WP6 | WP1, WP4 | Route metadata controls chat-specific actions outside chat/workspace contexts. |
| F7 | Settings navigation leaks internal translation key | P2 | WP5 | WP12 | Settings nav has no dotted i18n keys and has a regression test. |
| F8 | Command palette contains route-label mismatch | P2 | WP1 | WP6, WP12 | "Go to Chat" opens `/chat` or is relabeled with separate Chat command. |
| F9 | Capability and unsupported-state handling is inconsistent | P2 | WP2 | WP10, WP11A, WP11B, WP12 | Shared capability state adopted by representative routes. |
| F10 | Media Inspector is powerful but not first-selection or mobile friendly | P2 | WP8 | WP4, WP12 | `/media` first-selection empty state and mobile list-detail pass browser QA. |
| F11 | Settings is too broad for one flat route experience | P2 | WP5 | WP4, WP12 | Settings groups are task-led and mobile-safe. |
| F12 | Extension sidepanel route availability is not aligned with shared intent | P2 | WP1 | WP12 | Sidepanel availability matrix exists and is tested. |
| F13 | Chat empty state and composer compete on narrow screens | P2 | WP6 | WP4, WP12 | Chat composer remains visible and mode choices do not collide at 390px. |
| F14 | Research, Knowledge, Chat, and Workspace do not form a clear product ladder | P2 | WP9 | WP1, WP6 | Ask, Research, Workspace, and Transform labels are reflected in route metadata and page framing. |
| F15 | Root pages have inconsistent heading landmarks | P2 | WP4 | WP12 | User-facing root routes have one `h1` or approved exception. |
| F16 | General settings mixes routine preferences with high-risk system actions | P2 | WP5 | WP4 | Data/destructive settings are separated from routine preferences. |
| F17 | Route inventory and smoke coverage are not authoritative | P2 | WP12 | WP1 | Smoke inventory is generated from or checked against route metadata. |
| F18 | Specialized, hosted, beta, and debug routes need a visibility policy | P2 | WP1 | WP11A, WP11B, WP12 | Route visibility classes gate nav, command palette, docs, and smoke suites. |
| F19 | Ant Design deprecation warnings signal maintenance risk | P3 | WP11A, WP11B | WP12 | Deprecated components are tracked separately and cleaned when they block a UX fix. |

## Route Coverage Matrix

| Route | Primary package | Secondary packages | Main issue addressed | Verification hook |
|---|---|---|---|---|
| `/` | WP3 | WP1, WP4 | Home/setup resolver and first-run state | Browser QA for configured and unconfigured states |
| `/setup` | WP3 | WP4 | Focused setup shell and semantic heading | Setup route screenshot and `h1` check |
| `/login` | WP3 | WP1 | Auth-mode-aware login behavior | Route test by auth mode |
| `/signup` | WP3 | WP1 | Hosted-only signup classification | Visibility route metadata test |
| `/account` | WP3 | WP1 | Hosted-only account classification | Visibility route metadata test |
| `/profile` | WP3 | WP1 | Placeholder/profile route policy | Route visibility or redirect test |
| `/privileges` | WP3 | WP5 | Privileges alias or permissions landing | Route target and settings grouping test |
| `/config` | WP3 | WP1 | Placeholder config route policy | Route visibility or redirect test |
| `/billing` | WP3 | WP1 | Hosted-only billing classification | Visibility route metadata test |
| `/404` | WP3 | WP1 | Recovery actions and route diagnostics | 404 browser QA |
| `/chat` | WP6 | WP4, WP1 | Composer-first chat and route target consistency | Chat browser QA and 390px overflow check |
| `/quick-chat-popout` | WP6 | WP1 | Utility chat persistence and return behavior | Route QA and command availability check |
| `/persona` | WP7 | WP6, WP4 | Persona concept and launch path | Heading and launch action QA |
| `/characters` | WP7 | WP6 | Character asset management and use in chat | Browser QA for launch/import actions |
| `/companion` | WP7 | WP2 | Companion setup/readiness and inbox | Capability/status QA |
| `/agents` | WP7 | WP2, WP10 | Agent health and concept boundary | Capability state and heading check |
| `/agent-tasks` | WP7 | WP2, WP10 | Agent task unavailable state | Capability fixture test |
| `/chat-workflows` | WP7 | WP9 | Available-now workflow framing | Route browser QA |
| `/chat-workspace` | WP9 | WP4, WP6 | Workspace cockpit orientation | Heading and no-workspace state QA |
| `/knowledge` | WP9 | WP1 | Direct cited-QA pattern and ladder placement | Browser QA preserving current strengths |
| `/search` | WP9 | WP1 | Search alias or broad search policy | Alias route test |
| `/research` | WP9 | WP1, WP2 | User-language research runs | Route browser QA |
| `/workspace-playground` | WP9 | WP1, WP4 | Research Workspace label and responsive layout | Alias/label test and 390px check |
| `/document-workspace` | WP9 | WP4 | Single-document workspace framing | Empty-state browser QA |
| `/repo2txt` | WP9 | WP4 | Technical route name and output clarity | Heading and output-flow QA |
| `/model-playground` | WP9 | WP5, WP6 | Model testing versus generic chat | Browser QA for compare/config layout |
| `/writing-playground` | WP9 | WP4 | Writing session framing and persistence | Heading and session QA |
| `/presentation-studio` | WP9 | WP1 | Deck creation and render recovery | Route metadata and browser QA |
| `/audiobook-studio` | WP11A | WP9, WP2 | Audio generation readiness and jobs | Readiness-state QA |
| `/media` | WP8 | WP4 | First selection and mobile master-detail | Desktop and 390px browser QA |
| `/media-multi` | WP8 | WP4 | Bulk selection purpose and recovery | Selection action QA |
| `/review` | WP8 | WP1 | Review alias to bulk media workflow | Alias route test |
| `/media-trash` | WP8 | WP4 | Restore/delete policy and safeguards | Trash browser QA |
| `/items` | WP8 | WP1 | Generic library terminology | Route label and object taxonomy check |
| `/collections` | WP8 | WP1 | Collections/Reading canonical label | Route metadata and browser QA |
| `/reading` | WP8 | WP1 | Reading alias to collections | Alias route test |
| `/notes` | WP8 | WP4 | Notes title and library/workspace linkage | Heading and empty-state QA |
| `/shared` | WP8 | WP2 | Incoming/outgoing shared content model | Tabs and empty-state QA |
| `/chatbooks` | WP8 | WP2 | Chatbook import/export/jobs framing | Browser QA |
| `/chatbooks-playground` | WP8 | WP1 | Playground route classification | Visibility or redirect test |
| `/sources` | WP10 | WP2, WP12 | Raw 404 replacement and source recovery | Capability fixture and browser QA |
| `/connectors` | WP10 | WP1, WP2 | Placeholder connector policy | Visibility or capability-state test |
| `/integrations` | WP10 | WP2 | Personal integration unavailable state | Capability fixture QA |
| `/scheduled-tasks` | WP10 | WP2 | Raw scheduled-task error replacement | Capability fixture QA |
| `/watchlists` | WP10 | WP2 | Monitor/feed health and repeat controls | Browser QA |
| `/workflow-editor` | WP10 | WP4 | Workflow editor identity and run linkage | Heading and empty-state QA |
| `/settings` | WP5 | WP4, WP12 | Settings IA, mobile nav, unsafe action separation | Settings browser QA and 390px check |
| `/admin` | WP10 | WP2, WP4 | Operations landing and capability map | Admin browser QA |
| `/mcp-hub` | WP10 | WP2, WP4 | MCP status summary and semantic heading | Browser QA and heading check |
| `/acp-playground` | WP7 | WP2, WP10 | ACP unavailable state and setup path | Capability fixture QA |
| `/prompts` | WP5 | WP4, WP1 | Prompt library/studio hierarchy and mobile overflow | Browser QA and 390px check |
| `/prompt-studio` | WP5 | WP1 | Prompt Studio alias/tab state | Alias route test |
| `/dictionaries` | WP7 | WP1 | Dictionary activation scope | Browser QA |
| `/world-books` | WP7 | WP2 | World Book context scope and errors | Capability and heading QA |
| `/speech` | WP11A | WP2, WP4 | Audio route canonicalization and readiness | Browser QA and heading check |
| `/stt` | WP11A | WP2, WP4 | STT provider readiness and heading | Browser QA and heading check |
| `/tts` | WP11A | WP2, WP4 | TTS provider readiness and voice state | Browser QA and heading check |
| `/audio` | WP11A | WP1 | Audio alias or hub policy | Alias route test |
| `/evaluations` | WP11B | WP2 | Worker unavailable and eval presets | Browser QA |
| `/flashcards` | WP11B | WP4 | Study mode framing and heading | Browser QA and heading check |
| `/quiz` | WP11B | WP2, WP4 | Quiz start state and degraded mode | Browser QA |
| `/moderation-playground` | WP11B | WP4 | Safety test versus setup framing | Heading and route QA |
| `/content-review` | WP11B | WP8 | Review queue identity | Empty-state browser QA |
| `/claims-review` | WP11B | WP1 | Claims alias or queue distinction | Alias route test |
| `/data-tables` | WP11B | WP2 | Backend readiness and schema/output flow | Capability and browser QA |
| `/chunking-playground` | WP11B | WP1 | Advanced RAG tuning classification | Visibility and browser QA |
| `/kanban` | WP11B | WP1 | Labs versus production planning board | Visibility or persistence QA |
| `/skills` | WP10 | WP2, WP11B | Skill capability and empty state | Capability fixture QA |
| `/vn-assets` | WP11B | WP1 | VN asset lab classification and readiness | Visibility and route QA |
| `/vn-play` | WP11B | WP1 | VN play versus runtime inspector | Browser QA |
| `/documentation` | WP1 | WP10 | Docs root index and route classification | Docs route browser QA |
| `/notifications` | WP8 | WP10 | Notification grouping and deep links | Browser QA |
| `/composer-variants-preview` | WP1 | WP12 | Internal QA route classification | Visibility metadata test |
| `/onboarding-test` | WP1 | WP12 | Internal onboarding harness classification | Visibility metadata test |

## Sequenced Implementation Slices

This spec is not the implementation plan. The later implementation plan should
split the work into reviewable slices. Recommended order:

1. Route metadata and visibility policy skeleton.
2. Command palette target fix and route inventory/smoke validation.
3. Shared capability state component and first adopters: `/sources`,
   `/scheduled-tasks`, `/integrations`.
4. Setup shell and `/` resolver.
5. Responsive/heading test harness.
6. Settings/model provider remediation.
7. Chat/composer and global chrome separation.
8. Media master-detail and library alias cleanup.
9. Knowledge/research/workspace ladder cleanup.
10. Operations/admin/MCP/agents/scheduling state cleanup.
11. Audio/study/safety/specialized route cleanup.
12. Final route coverage and browser QA sweep.

Each implementation slice should create or update a Backlog task before code
edits begin.

## Verification Gates

### Route Contract Gate

- All audited root routes have metadata.
- Nav, command palette, sidepanel, and smoke inventory consume or validate
  against the contract.
- Alias, hosted-only, labs, debug, internal, and deprecated routes are
  explicitly classified.

### Accessibility And Orientation Gate

- Every user-facing root route has one semantic `h1` or documented exception.
- Page title, visible route label, and command palette label agree.
- Primary action and system state are visible in user language.

### Capability And Error-State Gate

- Primary UI states do not show raw endpoint text.
- Capability failures map to user-language categories.
- Diagnostics remain available behind disclosure.

### Responsive Gate

- Representative core routes pass at desktop, 390px mobile, and extension
  sidepanel widths.
- There is no page-level horizontal overflow on core routes.
- Chat composer, media master-detail, settings nav, prompts, and workspace
  panels have explicit narrow-width behavior.

### Power-User Gate

- Useful dense controls remain available.
- Repeated workflows expose shortcuts, recent items, saved state, bulk actions,
  or direct routes where applicable.
- Advanced controls are discoverable without dominating first-run states.

### Browser QA Gate

- Each package captures before/after browser observations for affected pages.
- Route sweep confirms no route-level error boundary regressions.
- Console/request errors are categorized as expected capability failures or
  regressions.

### No-Drift Gate

- Each package states out-of-scope boundaries.
- No broad visual redesign, new design system, backend API change, or route
  rename lands without explicit package justification.
- Backend dependencies are tracked separately from frontend-only work.

## Implementation Planning Rules

When this spec is converted into an implementation plan:

- Do not create one giant remediation PR.
- Use Backlog tasks per reviewable slice.
- Start with WP1/WP2/WP4 foundations before broad page-local remediation.
- Convert this program design into multiple implementation plans or a parent
  plan with child plans; do not force all 12 work packages into one execution
  plan.
- Each slice must list the finding IDs and route matrix rows it intends to
  close, plus the rows intentionally left open.
- Distinguish foundation-package adoption from route-family primary ownership.
  For example, `/sources` may be owned by the Operations package while adopting
  the shared capability-state package.
- Split overloaded route-family packages before implementation when one package
  spans unrelated workflows. Keep WP11 split into WP11A audio and WP11B
  study/safety/specialized-tool slices unless the actual change is only a
  shared classification or route-contract update.
- Do not use explanatory text as a substitute for structural UX fixes. Prefer
  clearer controls, status states, progressive disclosure, route ownership, and
  recovery affordances.
- Keep tests scoped to the changed surfaces and shared contracts.
- Browser QA is mandatory for changed visual routes.
- Bandit is required only for touched Python/backend scope.
- Preserve existing user data, route aliases, and persisted settings unless a
  migration is explicitly approved.

## Spec Approval Status

Approved design direction from the user:

- Use the hybrid plan shape: work packages as the main plan plus coverage
  matrices proving every route and finding is addressed.
- Keep the plan text-only.
- Avoid a broad visual redesign or generic SaaS rewrite.
- Treat the remediation as UX/HCI program design before any implementation work.
