# WebUI and Extension UX/HCI Audit

Date: 2026-05-17
Task: TASK-410
Repository: `/Users/macbook-dev/Documents/GitHub/tldw_server2`
Scope: WebUI and browser-extension user-facing route surfaces, with emphasis on top-level/root pages and route families
Status: Current baseline audit, report-only

## Operating Constraints

- This is a UX/HCI audit report, not an implementation pass.
- No application code, test code, backend code, frontend implementation, route implementation, config, or build files were modified by this audit.
- The only intended repository changes are this Markdown report, Backlog task metadata, and browser evidence artifacts under `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/`.
- Findings are grounded in route/component ownership, browser-observed evidence where available, and NN/g-style heuristics.
- Browser evidence comes from the currently running local dev app. The current dev server is configured as a connected self-hosted environment, so a truly clean unauthenticated first-run profile was not fully verified in this pass.

## Executive Summary

The WebUI/extension has a broad and unusually capable surface, but the current root-page experience is harder to learn than it needs to be because the product lacks a stable user-facing information architecture. The largest UX problems are not color polish or isolated empty states. They are structural: multiple navigation systems disagree, route names and aliases drift, chat-specific chrome appears everywhere, several capability failures expose implementation details, and mobile layouts for key root pages overflow instead of reflowing.

There are also strong foundations worth preserving. The connected desktop sweep rendered 124 routes without a route-level error boundary failure. `/knowledge`, `/mcp-hub`, `/evaluations`, and parts of `/media` show useful workflow-first patterns: explicit source state, guided setup, capability warnings, and dense controls for repeat users. The problem is consistency and product shape, not a total absence of good UI patterns.

Highest-priority remediation should focus on:

1. A canonical route taxonomy and navigation model across WebUI and extension.
2. Setup/onboarding separation from chat chrome and developer-heavy state.
3. Capability-aware error and empty states, starting with `/sources` and model/settings surfaces.
4. Mobile structural fixes for the shared shell and the most-used root pages.
5. Settings/model management simplification around configured, usable providers first.

## Evidence Base

### Code Evidence

Route and ownership evidence was gathered from:

- `apps/tldw-frontend/pages`
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx`
- `apps/packages/ui/src/components/Layouts/header-shortcut-items.ts`
- `apps/packages/ui/src/components/Layouts/ModeSelector.tsx`
- `apps/packages/ui/src/components/Common/CommandPalette.tsx`
- `apps/tldw-frontend/components/layout/WebLayout.tsx`
- `apps/tldw-frontend/components/layout/Header.tsx`

### Browser Evidence

Connected desktop sweep:

- Base URL: `http://127.0.0.1:3000`
- Routes visited: 124 from the current smoke inventory
- Route loads without error boundary: 124
- Routes with console/request errors or warnings: 28
- Evidence JSON: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/route-sweep-connected-desktop.json`

Extra page-file sweep:

- Page-file routes found outside the current smoke inventory: 10
- User-facing extra routes visited: 9
- Route loads without error boundary: 9
- Routes with console/request errors or warnings: 4
- Evidence JSON: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/route-sweep-connected-extra-pages.json`
- Not visited: `/__debug__/authz.spec`, treated as a debug/spec route rather than a user-facing root page.

Representative desktop screenshots:

- `/`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-home.png`
- `/setup`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-setup.png`
- `/chat`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-chat.png`
- `/media`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-media.png`
- `/knowledge`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-knowledge.png`
- `/sources`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-sources.png`
- `/settings`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-settings.png`
- `/settings/model`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-settings-model.png`
- `/mcp-hub`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/desktop-mcp-hub.png`

Connected mobile core sweep:

- Viewport: 390 x 844
- Routes visited: 11
- Route loads without error boundary: 11
- Routes with horizontal overflow: `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, `/workspace-playground`
- Evidence JSON: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/route-sweep-connected-mobile-core.json`

Representative mobile screenshots:

- `/chat`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/mobile-chat.png`
- `/media`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/mobile-media.png`
- `/settings`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/mobile-settings.png`
- `/sources`: `Docs/Reviews/assets/2026-05-17-webui-extension-ux-hci-audit/mobile-sources.png`

## Evidence Limitations

- The browser run used the currently running local development server and existing local data. It is valid connected-state evidence, but not a clean first-install lab.
- The WebUI server was already running on port 3000 and backend on port 8000. The audit did not stop or reconfigure those processes.
- The browser extension runtime was not built and installed as an actual extension during this pass. Extension findings are therefore based on shared route code, extension route wrappers, and WebUI route behavior, with only WebUI screenshots.
- The smoke inventory is not the full `pages/` tree. A targeted extra sweep covered user-facing page-file routes omitted from the smoke inventory, including `/research`, `/integrations`, `/scheduled-tasks`, and `/vn-play`.
- The audit did not attempt destructive, state-changing, or account-changing actions.
- No code tests were required because no product code was changed. Verification is limited to route sweeps, JSON evidence parsing, report lint/diff checks, and worktree scoping.

## UX/HCI Review Lenses

Each route family was reviewed through:

- First-time user journey: purpose clarity, primary action, empty states, onboarding, labels, navigation, error states, and system state visibility.
- Experienced/power-user journey: repeat workflow speed, density, shortcuts, bulk actions, advanced controls, saved state, recovery paths, and discoverability.
- Heuristics: visibility of system status, match between system and user language, user control and freedom, consistency and standards, error prevention and recovery, recognition over recall, flexibility and efficiency, aesthetic/minimalist design, accessibility, responsive behavior, and cognitive load.

## Severity Rubric

- P0: Blocks key workflow completion, causes unsafe/destructive confusion, or makes a critical page unusable.
- P1: Seriously degrades a primary first-time or power-user workflow.
- P2: Creates significant friction, ambiguity, inconsistency, accessibility risk, or responsive weakness with workarounds.
- P3: Polish, hierarchy, copy, consistency, or efficiency issue with limited direct task impact.

## Baseline Usability Score

This score is for the overall root-route surface, not for every individual page. Several pages are much better than the aggregate, especially `/knowledge`, `/mcp-hub`, `/evaluations`, and parts of `/media`. The aggregate score is pulled down by inconsistent IA, narrow-viewport failures, and inconsistent self-hosted capability/error states.

| # | Heuristic | Score | Key evidence |
|---|---:|---:|---|
| 1 | Visibility of system status | 2/4 | Strong status on `/knowledge`, `/mcp-hub`, `/evaluations`; raw or ambiguous status on `/sources`, `/admin/monitoring`, `/settings/model`. |
| 2 | Match between system and user language | 1/4 | Raw endpoint messages, `providerKeys.navTitle`, route names like Workspace Playground, Prompt Studio, Speech/Audio aliases, and implementation-heavy model/provider surfaces. |
| 3 | User control and freedom | 2/4 | Many actions and shortcuts exist, but route recovery, setup state, mobile back paths, and settings exits are inconsistent. |
| 4 | Consistency and standards | 1/4 | Navigation is split across route registry, header shortcuts, mode selector, command palette, Next header, settings nav, and sidepanel registries. |
| 5 | Error prevention | 2/4 | Some beta/unsupported states prevent confusion, but setup can validate the frontend origin as server URL and model/provider setup exposes too much unusable choice. |
| 6 | Recognition rather than recall | 1/4 | Users must remember which of several similar concepts maps to their intent: Chat, Knowledge, Research, Workspace, Sources, Collections, Reading, Audio, Speech. |
| 7 | Flexibility and efficiency | 3/4 | Command palette, shortcuts, dense media filters, bulk mode, workspace controls, and model/provider options show strong power-user potential. |
| 8 | Aesthetic and minimalist design | 1/4 | Many roots expose global chrome and dense advanced controls before the route's primary job is established. |
| 9 | Error recognition and recovery | 1/4 | Good recovery patterns exist, but raw endpoint errors and 403/404 states often lack plain-language next steps. |
| 10 | Help and documentation | 2/4 | MCP Hub, Knowledge, Watchlists, and Evaluations include useful help; setup, sources, admin, and model settings need contextual recovery help. |
| **Total** |  | **16/40** | **Poor aggregate usability, with recoverable foundations.** |

## Product Register Assessment

This is product UI, not a marketing surface. The bar for quality is earned familiarity, predictable controls, fast recovery, and task-focused density. The current interface generally avoids the obvious "generic SaaS landing page" failure, but it has a different product-UI failure mode: it exposes too many internal product eras and implementation concepts simultaneously.

Current dominant pattern:

- The product is dense and capable, which fits a self-hosted research/media assistant.
- The top-level shell makes almost every route feel like a chat mode, even when the page job is source management, media inspection, admin, or settings.
- First-time users are asked to understand server connection, demo mode, auth mode, chat modes, ingestion, model choice, global search, notifications, extension state, and route icons before they have one successful loop.
- Power users can see many controls, but the inconsistency of names and route targets weakens the shortcuts that should make the product fast.

The visual problem is therefore not "make it prettier." It is: make each route declare one job, one status, one primary action, and one expert acceleration path.

## Persona Stress Tests

### Jordan, First-Time Self-Hosted User

Primary task: connect a local server and ask one grounded question about personal content.

Observed failure points:

- `/` and `/setup` both present setup, but both also show the full app shell. Jordan sees Search, Temp, Character, Quick Ingest, settings, route icons, and notifications before knowing if the backend is connected.
- The setup form in the observed state accepts `http://127.0.0.1:3000` as the server URL, which is the frontend origin in this dev run. The copy says "connect your server", so Jordan may believe the frontend app is the server.
- After setup, Jordan has several plausible next routes: Chat, Quick Ingest, Media, Knowledge QA, Sources, Research Studio, Chat Workspace, Collections. The UI does not name the shortest path to first value.

Concrete product implication: first-run should be designed around a minimum viable loop: connect server, ingest one thing or use demo data, ask one question, see a cited answer, then reveal the broader route map.

### Alex, Experienced Research/Automation User

Primary task: import or select many sources, run research, inspect results, reuse the setup later.

Observed failure points:

- Alex has power controls, but concepts are split: `/knowledge` asks over selected sources, `/workspace-playground` starts research with sources, `/chat-workspace` has runtime and approvals, `/research` exists as an extra page-file route, `/sources` is a source-management root, and `/media` holds ingested items.
- Command palette and route labels are not reliable enough for expert muscle memory. The concrete "Go to Chat" command targets `/`, not `/chat`.
- Unsupported states vary by page. Alex can infer a backend/API mismatch, but the UI should not require that inference.

Concrete product implication: the product needs a route contract and a workflow map for "capture -> organize -> retrieve -> research -> export/automate", with command palette and sidepanel routes generated from the same contract.

### Sam, Keyboard/Screen-Reader Or Low-Vision User

Primary task: navigate to a root page, understand state, and complete the primary action without visual guessing.

Observed failure points:

- Many root pages lack an extracted `h1`, including `/setup`, `/chat`, `/sources`, `/settings`, `/settings/model`, `/mcp-hub`, `/stt`, and `/tts`.
- The left rail uses many icon-only buttons. They have ARIA labels in the sweep, which is good, but visually collapsed icons increase recognition burden for low-vision and cognitive-load users.
- Mobile/narrow screenshots show page-level horizontal overflow, especially settings and media. At 200 percent zoom or sidepanel width, this becomes an accessibility failure, not merely a mobile polish issue.
- Error states such as `Not Found (GET /api/v1/ingestion-sources)` force technical interpretation and are not recoverable by screen-reader users unless the next action is explicit.

Concrete product implication: root pages need stable heading structure, visible route labels where possible, no horizontal page overflow at sidepanel widths, and standardized live-region/error copy.

### Casey, Distracted Extension/Sidepanel User

Primary task: capture or ask about a page quickly from a constrained viewport.

Observed failure points:

- Shared sidepanel routes include clipper, companion, flashcards, and `/chat`, while the extension wrapper sidepanel exposes only `/`, `/agent`, `/persona`, `/settings`, and error-boundary test in code.
- Mobile-size WebUI evidence shows `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, and `/workspace-playground` overflow horizontally. Those are the same constraints sidepanel users are likely to encounter.
- Chat empty state and composer compete for the lower viewport. The primary message field should be the stable anchor; mode cards should not be partially hidden by the composer.

Concrete product implication: the extension needs its own declared task model, not a compressed desktop route map. The sidepanel should optimize capture, ask, summarize, save, and recover, then deep-link to desktop routes for heavy management.

## Core Workflow Audit

### Workflow 1: First Run And Connection

User goal: decide whether to use demo mode, connect a self-hosted backend, or log into a multi-user server.

Observed route evidence:

- `/` shows `Home Onboarding` and a card titled "Welcome to tldw Browser Assistant- Let's get you connected."
- `/setup` shows `Setup Wizard` plus the same welcome card, but no extracted `h1`.
- Both pages show global app chrome and route rail controls before connection is complete.
- The setup form shows "Server URL" and the observed value `http://127.0.0.1:3000`.

First-time assessment:

- Purpose is partially clear: the user sees setup is required.
- Primary action is visible: Connect, Try Demo.
- The route hierarchy is not clear: `/` and `/setup` compete.
- The app exposes too much unavailable product surface before connection state is resolved.
- The server URL field risks teaching the wrong mental model if the frontend origin is treated as the server.

Power-user assessment:

- Experienced self-hosters need fast diagnostics: detected frontend URL, backend URL, auth mode, health, API key scope, and CORS/proxy mode.
- The current wizard has some diagnostics, but they are embedded in onboarding rather than presented as operator-grade status.

Recommended improvement:

- Make `/setup` the canonical connection task.
- Let `/` resolve to setup, chat/dashboard, or demo based on state.
- Use a setup-only shell with no chat controls.
- Show explicit connection status: frontend origin, API backend target, auth mode, health result, and what will be stored locally.
- Provide expert affordances: paste API key, test backend, open health diagnostics, clear saved connection, skip to demo.

### Workflow 2: Start A Chat

User goal: send a first message, optionally with model/persona/context choices.

Observed route evidence:

- `/chat` has no extracted `h1`; the main heading is `Start a new chat` as `h2`.
- The visible page shows two immediate CTAs plus five equal mode cards: General chat, Compare AI models side-by-side, Chat as a character, Search your documents, Deep research.
- The composer has many controls at once: modes, MCP, search/context, prompt, persona, mic, headphones, attachment, control sliders, context gauge, saved status, advanced controls.
- Mobile `/chat` overflows horizontally: client width 390, scroll width 591.

First-time assessment:

- The page explains that chat exists, but it asks the user to choose among product modes before they have sent a message.
- "All five stay equally available" is not a substitute for hierarchy. A first-time user needs the product to recommend the default path.
- The composer is the most important control, but the mode cards and bottom chrome visually compete with it.

Power-user assessment:

- The composer has excellent potential for expert use: MCP, context, prompt, persona, token budget, model health, saved/temp mode.
- The issue is not missing controls. It is that expert controls are visible at the same level as first-message input.
- Keyboard command hints are present, but command palette route mismatch weakens trust.

Recommended improvement:

- Make the composer and current model/provider readiness the first screen's anchor.
- Collapse mode choices into a compact mode switcher with a recommended default.
- Show advanced controls after the first message or through a persistent but secondary control.
- Preserve power-user density inside the composer once a conversation is active.
- Add no-overflow checks at 390px and sidepanel widths.

### Workflow 3: Ask A Cited Question Over The Library

User goal: select sources, ask a question, understand citations and retrieval settings.

Observed route evidence:

- `/knowledge` has `h1` "Ask Your Library".
- The top control row shows selected sources, retrieval balance, web toggle, AI model, and settings.
- The empty state gives example questions and recent searches.
- Primary input is a large bottom prompt: "What are the key findings from the research?"

First-time assessment:

- This is one of the strongest pages. It has a clear job, clear source state, examples, recent history, and a primary action.
- Remaining risk: the model label "AI: Default default" is system language, not user language.
- The relationship to `/chat` "Search your documents" and `/workspace-playground` "Start your research" is not clear.

Power-user assessment:

- Power users need saved source sets, last-used retrieval recipes, and a way to jump from answer to source/media/workspace without losing context.
- The page already has the right scaffolding for this. It needs tighter integration with source management and research workspaces.

Recommended improvement:

- Preserve this page as the pattern for other roots.
- Rename technical model state to "Model: [configured default]" or "Model not configured".
- Add saved source-set and retrieval-preset controls.
- Clarify the route relationship: Knowledge QA is for direct cited answers; Research/Workspace is for multi-step investigation.

### Workflow 4: Browse And Inspect Media

User goal: find an ingested item, inspect content, act on it, recover deleted items, or create study packs.

Observed route evidence:

- `/media` has `h1` "Media Inspector".
- The left panel shows 20 of 918 items, filters, sort, date range, media types, excluded keywords, collection controls, pagination, jump-to, and library tools.
- The right pane is empty until selection: "No media item selected. Select a media item from the left sidebar to view its content and analyses here."
- Desktop screenshot shows a very tall dense left panel and a large unused right pane.
- Mobile `/media` overflows horizontally: client width 390, scroll width 683.

First-time assessment:

- The route purpose is understandable, but the first useful step is not visually prioritized. Search and filter controls dominate before selection.
- The detail empty state is far away from the list's first item and uses desktop positional language.
- Quick Ingest is available, but the relationship between ingest and media library is not foregrounded.

Power-user assessment:

- This page has the right density for large libraries: filters, sorting, favorites, bulk mode, pagination, and shortcuts.
- The risk is that density is front-loaded and not adaptive. Expert controls should stay available, but not at the cost of first selection and responsive behavior.

Recommended improvement:

- Desktop: keep the split view, but improve first-selection hierarchy: search, recent items, selected item preview, and bulk actions should form a clearer task lane.
- Narrow viewport: use list-only until an item is selected, then a detail view with Back to results.
- Replace positional copy with responsive copy: "Choose an item to inspect its transcript, notes, and analyses."
- Add an explicit "Ingest new media" path in the empty/detail state, not just global Quick Ingest.

### Workflow 5: Manage Sources, Connectors, Integrations, And Scheduled Tasks

User goal: configure recurring or local content sources and understand what is connected.

Observed route evidence:

- `/sources` shows title/copy and "New source", but the main state is `Not Found (GET /api/v1/ingestion-sources)`.
- `/admin/sources` repeats the same raw endpoint state.
- `/integrations` shows "Personal integrations unavailable. This server does not expose the personal integrations control-plane yet."
- `/admin/integrations` shows multiple policy/actor loading failures, including raw endpoint text.
- `/scheduled-tasks` shows "Unable to load scheduled tasks. Not Found (GET /api/v1/scheduled-tasks)."

First-time assessment:

- This route family fails the user-language test. A non-developer cannot tell whether sources are empty, disabled, unsupported by this backend, unauthorized, or broken.
- "New source" remains visible next to a missing endpoint state, making the user's next action risky or unclear.

Power-user assessment:

- Operators can infer server/API mismatch, but they need a capability matrix and recovery path, not raw probes scattered across routes.
- Integrations, sources, scheduled tasks, watchlists, quick ingest, and connectors need a declared relationship. They are all "getting content into the system" workflows, but the UI treats them as separate product islands.

Recommended improvement:

- Create a shared capability state for this entire family: unavailable, disabled, not configured, no permission, unreachable, empty, beta.
- Add route-local next actions: use Quick Ingest, check server version, open health, enable source APIs, view docs.
- Make raw endpoint/method details available only inside diagnostics.
- Define the ingestion IA: Quick Ingest for one-off capture, Sources for local/archive sources, Integrations for service connections, Watchlists/Scheduled Tasks for recurring ingestion.

### Workflow 6: Configure Models And Providers

User goal: know which model will be used, configure provider credentials, choose defaults, and diagnose failures.

Observed route evidence:

- `/settings/model` has no extracted `h1` or `h2`.
- The visible full-page screenshot is a massive provider/model catalog, with hundreds of chips.
- Sweep captured a 403 resource error while the page continues to render the catalog.
- Settings navigation also shows `providerKeys.navTitle`.

First-time assessment:

- This is a high-risk setup route because model configuration directly controls whether chat works.
- The page answers "what models exist in the universe" before it answers "what can I use right now?"
- Missing heading structure and unfiltered catalog create extreme cognitive load.

Power-user assessment:

- Power users do need full catalog access, search, aliases, provider-specific controls, and favorites.
- The current catalog should be secondary to configured/healthy providers, defaults, recent models, and failure states.

Recommended improvement:

- Default to a "Usable now" view: default model, configured providers, missing credentials, health, and recent/favorite models.
- Provide full catalog as a searchable expert drawer/tab.
- Put OAuth/API-key failures into plain-language provider cards.
- Fix the internal nav label and add a settings-label smoke test.

### Workflow 7: Manage Prompts, Characters, World Books, Dictionaries, Notes, And Collections

User goal: build reusable context assets and use them in chat/research workflows.

Observed route evidence:

- `/prompts` has `h1` "Prompts" and empty state `No custom prompts yet`.
- `/characters` reports "2 characters found" in status text.
- Settings and route inventory include Prompts, Prompt Studio, Manage Prompts, Characters, Persona, Companion, World Books, Chat Dictionaries, Collections, Reading, Notes, Shared, and Chatbooks.
- `/reading` redirects to `/collections`; `/prompt-studio` redirects to `/prompts?tab=studio`.

First-time assessment:

- The pieces are individually understandable, but the product model is not. Users need to know whether a "character" is the same kind of thing as a persona or companion, whether Prompt Studio is a route or a tab, and whether Reading is Collections.
- Empty states help, but the route vocabulary makes the library feel like several overlapping systems.

Power-user assessment:

- This area has strong power-user potential: import/export, templates, versioning, character launch, lore assets, dictionaries, and reusable prompt workspaces.
- Experts need faster cross-links: prompt to chat, character to character chat, collection to Knowledge QA, note to workspace, chatbook import/export.

Recommended improvement:

- Group these under a user-facing "Library Assets" or "Context Library" concept, while preserving specialized routes.
- Keep route aliases for compatibility but stop exposing duplicate labels in primary navigation.
- Add launch paths from asset pages into the workflows where assets are used.
- Make empty states show the next useful workflow, not only create/import.

### Workflow 8: Research And Workspace Routes

User goal: assemble sources, conduct multi-step research, use chat with workspace context, and create outputs.

Observed route evidence:

- `/workspace-playground` has `h1` "New Research" and `h2` sections "Sources", "Start your research", and "Studio".
- `/chat-workspace` has `h2` sections "Workspace", "Workspace chat", "Scope", "Sources", "Model / Persona", "Approvals", "Task Progress", and "Runtime".
- `/research` exists as an extra page-file route.
- `/document-workspace`, `/repo2txt`, `/model-playground`, and `/writing-playground` are also root-level work tools.

First-time assessment:

- The product has a real research assistant ambition, but the route names do not map cleanly to user jobs.
- "Research Studio", "Workspace Playground", "Chat Workspace", and "Document Workspace" sound like internal evolution stages rather than clear product modes.

Power-user assessment:

- Expert users need persistence, source scoping, branch/retry/recovery, approvals, exports, and runtime visibility. `/chat-workspace` already exposes some of this.
- The risk is route scattering: users must know which workspace surface has the needed capability.

Recommended improvement:

- Define a research workflow ladder:
  1. Knowledge QA: ask one cited question.
  2. Research Workspace: multi-step investigation over selected sources.
  3. Document/Repo tools: transform source material into structured outputs.
  4. Chat Workspace: agentic/runtime-aware work with approvals.
- Rename or relabel user-facing entries to match those jobs, while keeping old paths as aliases.
- Add recovery and recent-workspace entry points to the root research surfaces.

### Workflow 9: MCP, Agents, Workflows, Admin, And Operations

User goal: connect tools, manage agents/workflows, and operate the server.

Observed route evidence:

- `/mcp-hub` has strong onboarding: page title, setup tabs, "Getting Started with MCP Hub", "New Managed Server", and a no-server empty state.
- `/agent-tasks` has a clear unsupported state and actions: "Open Agent Registry", "Open ACP Playground".
- `/acp-playground` reports "ACP backend is not configured or unreachable."
- `/admin/llamacpp` clearly says admin APIs are unavailable on this server.
- `/admin/monitoring` mixes "Sandbox diagnostics unavailable" with raw endpoint placeholder text and example alert rules.

First-time assessment:

- MCP Hub is a positive model: it explains what the page is for and what to do next.
- Admin/agent pages vary widely in quality. Some teach the user; others leak backend probes.

Power-user assessment:

- Operators expect status, capabilities, logs, permissions, and safe recovery. The product has those concepts, but they are distributed across MCP Hub, ACP Playground, Agents, Agent Tasks, Chat Workflows, Scheduled Tasks, Admin, Settings Health, and Integrations.
- The missing piece is not more pages. It is an operational map.

Recommended improvement:

- Use MCP Hub's structure as the model for advanced operational pages: overview, setup, access, workspaces, governance, audit.
- Introduce an "Operations" grouping for admin/capability/server health, separate from everyday research/library routes.
- Use one unsupported-state component everywhere an API/module is unavailable.

### Workflow 10: Audio, Evaluations, Watchlists, And Specialized Tools

User goal: use specialized capabilities without losing the broader product map.

Observed route evidence:

- `/evaluations` has `h1` "Evaluations playground" and a clear worker-unavailable state that still offers completed-run reuse.
- `/watchlists` has `h1` "Watchlists", beta state, docs/report links, and overview CTAs.
- `/tts`, `/speech`, and `/audio` show route aliasing and 404 console errors in the observed connected state.
- `/audiobook-studio`, `/presentation-studio/new`, `/vn-assets`, and `/vn-play` exist as specialized root-level tools.

First-time assessment:

- Evaluations and Watchlists are relatively clear because they state status and next actions.
- Audio routes are less clear due route aliases and backend/provider failures.
- Specialized tools should be discoverable, but not mixed into the primary route surface unless the deployment/persona expects them.

Power-user assessment:

- Power users need saved presets, history, batch actions, and provider readiness across audio/eval/watchlist tools.
- Specialized tool routes need consistent beta/readiness labeling and a way back to the core workflow they support.

Recommended improvement:

- Keep specialized tools, but classify them as Advanced Tools or Labs unless they are core to the selected persona/deployment.
- Standardize alias policy: `/audio` should visibly resolve to the canonical audio route and the nav should use one label.
- Reuse Evaluations' degraded-worker copy pattern for TTS/STT/provider capability failures.

## Route-Family Scores

These scores are heuristic summaries for prioritization. They are not implementation test results.

| Route family | First-time score | Power-user score | Main reason |
|---|---:|---:|---|
| `/`, `/setup` | 1/4 | 2/4 | Setup intent exists, but global chrome and URL semantics blur connection state. |
| `/chat` | 2/4 | 3/4 | Strong composer potential, but mode overload and mobile collision hurt first use. |
| `/knowledge` | 3/4 | 3/4 | Clear page job and source state; needs clearer relation to research/chat and saved presets. |
| `/media` | 2/4 | 4/4 | Powerful library controls; first selection and narrow layouts are weak. |
| `/sources`, `/integrations`, `/scheduled-tasks` | 1/4 | 2/4 | Raw capability/API failures block comprehension. |
| `/settings`, `/settings/model` | 1/4 | 3/4 | High configurability, but overwhelming and structurally broken on mobile. |
| Prompts/characters/library assets | 2/4 | 3/4 | Useful assets, but route vocabulary and launch paths are fragmented. |
| Research/workspace routes | 2/4 | 3/4 | Strong capabilities, unclear product ladder and route naming. |
| MCP/agents/admin | 2/4 | 3/4 | MCP and some unsupported states are good; operational IA is scattered. |
| Audio/evals/watchlists/specialized tools | 2/4 | 3/4 | Evaluations/Watchlists show good patterns; audio/specialized routes need readiness consistency. |

## Workflow Remediation Acceptance Criteria

Use these as future slice acceptance criteria. They are intentionally written as outcome checks, not implementation instructions.

| Slice | Acceptance criteria |
|---|---|
| First-run setup | A new user can identify Connect, Demo, and Login paths within 5 seconds; setup does not show chat/persona/global route controls before connection; backend URL status distinguishes frontend origin from API target. |
| Chat first message | At 390px and desktop width, the message input is visible, not overlapped, and sending a basic message does not require choosing among more than four visible options. |
| Knowledge/research IA | Navigation and page copy distinguish direct cited Q&A from multi-step research and workspace/agent workflows. Users can move from Knowledge QA to a research workspace without losing selected sources. |
| Media browsing | On mobile/sidepanel width, media uses list-to-detail navigation with no page-level horizontal scroll; on desktop, search/list/detail hierarchy makes the first item inspection obvious. |
| Sources/capabilities | Sources, integrations, scheduled tasks, admin sources, and related routes never expose raw endpoint text as the primary error. Each state provides a diagnosis category and next action. |
| Model configuration | The default model/settings view shows configured/usable providers before full catalog; provider failures are visible in user language; full catalog remains searchable for experts. |
| Settings IA | Settings has one responsive navigation model, no internal translation keys, stable `h1`/section headings, searchable synonyms, and safe separation of destructive reset/import/export controls. |
| Sidepanel | Extension sidepanel routes are declared in a capability matrix, tested against it, and optimized for capture/ask/save/recover workflows rather than compressed desktop management. |

## Root Page Inventory And Route-By-Route Audit

This replaces the earlier compressed route tables. Each record below is a root-page audit entry with the requested inventory fields and the page-specific UX/HCI findings: primary goal, observed components/evidence, main workflows, first-time user issues, power-user issues, and concrete recommended fixes.

### Start, Auth, Account, And Recovery

#### `/`

- Primary user goal: Start the app, connect a tldw server, or try demo mode.
- Evidence: Browser sweep rendered `Home Onboarding` with the card "Welcome to tldw Browser Assistant- Let's get you connected." Desktop and mobile screenshots show the full shared shell around setup. Mobile did not overflow.
- Main workflows: Try Demo, enter server URL, choose single-user API key or multi-user login, connect, skip.
- First-time issues:
  - P1: The first-run connection task is visually surrounded by chat/global chrome: Search, Temp, Character, route rail, notifications, settings, New Chat, and Quick Ingest. That makes the app look usable before the connection state is established.
  - P1: `/` and `/setup` present similar connection work, so the user cannot learn which route is canonical.
  - P2: The server URL semantics are fragile. A user can confuse the frontend origin with the backend/API origin.
- Power-user issues:
  - P2: Self-hosted operators need immediate server health, saved backend URL, auth mode, and API key status. Those are implied, not summarized.
  - P3: A returning user who is already configured should not have to pass through onboarding unless the connection is degraded.
- Recommended fixes:
  - Make `/` a resolver route: configured users go to the last primary workspace or dashboard; unconfigured users go to setup; demo users go to demo state.
  - Use a setup-only shell until backend state is known.
  - Show a compact connection summary with frontend origin, backend API target, auth mode, and last health check.

#### `/setup`

- Primary user goal: Complete guided setup for a self-hosted or demo connection.
- Evidence: Browser sweep rendered `Setup Wizard` plus the same setup card used on `/`; no extracted `h1`; desktop/mobile screenshots show global app controls in the setup state.
- Main workflows: Start setup, try demo, choose auth mode, enter URL and API key, connect, open startup help.
- First-time issues:
  - P1: The page is not focused enough for first-run. It still exposes route navigation and chat controls before the user has a valid backend.
  - P2: The visual hierarchy has both `Setup Wizard` and the welcome card, but no semantic `h1`.
  - P2: Setup does not visibly distinguish "URL format is valid" from "server is reachable and authenticated."
- Power-user issues:
  - P1: Operators cannot quickly diagnose CORS/proxy/API-key/auth-mode mismatch from the setup screen.
  - P2: There is no obvious "clear saved connection and re-run setup" recovery path.
- Recommended fixes:
  - Make `/setup` the canonical first-run route with a setup-only shell.
  - Add a single semantic `h1`, backend health details, auth mode details, retry, clear saved connection, and copy diagnostics.
  - Separate syntactic URL validation from live health/auth verification.

#### `/login`

- Primary user goal: Sign in or recover authentication when multi-user mode applies.
- Evidence: Connected sweep ended at `/settings/tldw` and rendered server configuration, not a login form.
- Main workflows: Login, switch account, recover auth state, return to server settings.
- First-time issues:
  - P1: A user selecting login lands on configuration, so the route violates the user's expectation and hides the auth model.
  - P2: The page does not explain whether single-user API-key mode disables login.
- Power-user issues:
  - P2: Multi-user admins cannot use `/login` as a reliable reauthentication or account-switch route.
  - P3: Redirect behavior weakens automated QA for auth mode differences.
- Recommended fixes:
  - Make `/login` auth-mode aware: show login form in multi-user mode, show "single-user uses API key in setup/settings" in single-user mode, and show current session state when already authenticated.
  - Keep any redirect explicit with a visible reason.

#### `/signup`

- Primary user goal: Create an account in a hosted/private distribution.
- Evidence: Page displays `Signup Is Not Part Of The OSS Web Surface`.
- Main workflows: Hosted signup, self-hosted account guidance.
- First-time issues:
  - P2: A self-hosted user hits a route that looks like a missing product surface instead of a deliberate OSS limitation.
  - P3: The next action is not as strong as setup/login guidance should be.
- Power-user issues:
  - P2: Hosted-only route exposure is not governed by a clear route visibility policy.
- Recommended fixes:
  - Hide from OSS/self-hosted navigation and command search.
  - If deep-linked, route to setup/login with explicit "hosted-only" state and self-hosted alternatives.

#### `/account`

- Primary user goal: Manage hosted account details.
- Evidence: Page displays `Hosted Account Pages Live In The Private Distribution`.
- Main workflows: Hosted account management or self-hosted account explanation.
- First-time issues:
  - P2: The page can read as broken or paywalled rather than intentionally absent in OSS.
- Power-user issues:
  - P2: Hosted/private route leakage makes QA and self-hosted docs ambiguous.
- Recommended fixes:
  - Mark the route hosted-only in route metadata.
  - Remove it from default self-hosted navigation, command palette, and smoke expectations unless testing hosted mode.

#### `/profile`

- Primary user goal: Manage profile or session-level identity.
- Evidence: Page displays `Profile Page Is Coming Soon`.
- Main workflows: View profile, session details, account identity.
- First-time issues:
  - P3: A placeholder root page wastes exploration during onboarding.
- Power-user issues:
  - P2: There is no reliable profile/session page for account troubleshooting.
- Recommended fixes:
  - Hide until implemented, or redirect to the actual account/auth settings with clear copy.
  - If retained, show current session/auth mode and route users to settings/admin as appropriate.

#### `/privileges`

- Primary user goal: Review permissions, privileges, or access rights.
- Evidence: Browser sweep landed on `/settings`; settings content appeared.
- Main workflows: Permission audit, access explanation, settings recovery.
- First-time issues:
  - P2: The route name promises permissions but delivers general settings.
- Power-user issues:
  - P1: Admins cannot quickly audit effective privileges, auth mode, RBAC status, or available actions.
- Recommended fixes:
  - Either implement a privileges landing page with effective user permissions, or remove the alias from user-facing surfaces.
  - If redirecting, show a transition banner that explains where privilege settings live.

#### `/config`

- Primary user goal: Configure the app/server.
- Evidence: Page displays `Configuration Center Is Coming Soon`.
- Main workflows: General configuration, settings redirect.
- First-time issues:
  - P2: It competes with `/settings` and dead-ends.
- Power-user issues:
  - P2: Experts must remember whether `config` or `settings` is the valid route.
- Recommended fixes:
  - Hide or redirect to `/settings`.
  - If a future Configuration Center is planned, classify it as roadmap/internal until usable.

#### `/billing`

- Primary user goal: Manage hosted subscription/invoices.
- Evidence: Page displays `Hosted Billing Lives In The Private Distribution`.
- Main workflows: Hosted billing, OSS billing explanation.
- First-time issues:
  - P3: Self-hosted users encounter an irrelevant commercial/account route.
- Power-user issues:
  - P2: Hosted-only pages can pollute self-hosted route inventory and smoke coverage.
- Recommended fixes:
  - Mark as hosted-only and filter by distribution.
  - Deep links should explain the distribution mismatch and point to self-hosted admin/setup docs.

#### `/404`

- Primary user goal: Recover from an invalid or removed route.
- Evidence: Page displays `We could not find that route`, with route suggestions and the current missing path.
- Main workflows: Return to valid page, recover from stale link, debug route.
- First-time issues:
  - P3: Recovery exists, but it still assumes users know which suggested route fits their task.
- Power-user issues:
  - P3: It lacks route diagnostics such as source route, alias/deprecation state, or command palette trigger.
- Recommended fixes:
  - Add search/command palette CTA, Home/Setup/Chat/Settings actions, and optional technical route diagnostics.

### Chat, Persona, Agents, And Companion

#### `/chat`

- Primary user goal: Send messages to an LLM, optionally with documents, web, persona, or research mode.
- Evidence: Browser sweep extracted no `h1`; `Start a new chat` is an `h2`; desktop screenshot shows five equal mode cards and dense composer controls; mobile sweep reported horizontal overflow 591/390 and 25 small touch targets.
- Main workflows: Send first message, select model, choose mode, quick ingest, attach files, configure prompt/context/persona, run document search or deep research.
- First-time issues:
  - P1: The first action is not singular. Users must parse General, Compare, Character, Search documents, and Deep research before doing the basic thing: type a message.
  - P1: On mobile/sidepanel width, the empty-state cards and sticky composer compete for the same visual space.
  - P2: The page has no semantic `h1`, which hurts orientation and accessibility.
- Power-user issues:
  - P1: The global command palette currently labels "Go to Chat" but routes to `/`, weakening keyboard trust.
  - P2: Expert controls are visible but not organized into a stable hierarchy of model, prompt, context, tools, and send behavior.
- Recommended fixes:
  - Make the composer and model readiness the primary first screen.
  - Convert mode cards into a compact mode switcher or expandable panel.
  - Add a semantic `h1`, fix mobile overflow, and make command palette target `/chat`.

#### `/quick-chat-popout`

- Primary user goal: Start a small side chat without disturbing the main thread.
- Evidence: Page displays `Quick Chat Helper`, model/provider controls, and copy: "Start a quick side chat to keep your main thread clean."
- Main workflows: Quick question, docs Q&A, browse guides, keep main thread separate.
- First-time issues:
  - P2: It is unclear whether this is a route, drawer, popout, extension mode, or separate chat history.
- Power-user issues:
  - P2: Without explicit history and return behavior, it can fragment conversations and keyboard shortcuts.
- Recommended fixes:
  - Define it as a utility surface with clear persistence rules.
  - Provide "return to main chat", "promote to saved chat", and "discard" actions.

#### `/persona`

- Primary user goal: Configure and use persona-based chat.
- Evidence: Body includes `Persona Garden`, assistant setup steps, persona picker, create persona, persona live, and a text fallback state; sweep extracted no headings.
- Main workflows: Choose existing persona, create persona, finish assistant setup, activate visual pack/fallback, start persona chat.
- First-time issues:
  - P1: Persona, Characters, Companion, and visual fallback are introduced together without a clear concept model.
  - P2: No extracted heading means screen-reader and route QA users lose the page identity.
- Power-user issues:
  - P2: Fast persona switching, active persona state, and launch-to-chat are not visible enough from the root.
- Recommended fixes:
  - Add a Persona landing page with selected persona, readiness checklist, create/import, and "start chat with this persona."
  - Clarify relationship: Characters are assets, Persona is active behavior/profile, Companion is persistent assistant.

#### `/characters`

- Primary user goal: Create, import, manage, and launch reusable characters.
- Evidence: `h1` Characters; 2 characters found; upload action; keyboard shortcut copy; desktop screenshot; console warnings/404s.
- Main workflows: Upload character, create character, search/list characters, manage details, use character in chat.
- First-time issues:
  - P2: Users may not know whether a character is a persona, a chat mode, a visual avatar, or a prompt preset.
  - P2: Data exists, but the route does not strongly foreground "use this character now."
- Power-user issues:
  - P2: Returning users need fast launch-to-chat, import/export, favorite/recent characters, and active-character state.
- Recommended fixes:
  - Add per-character primary actions: Use in chat, edit, duplicate, export.
  - Show active character/persona relationship and import status.

#### `/companion`

- Primary user goal: Review companion state, inbox, and work threads.
- Evidence: `h1` Companion; setup required; Inbox Preview; Needs Attention.
- Main workflows: Finish companion setup, review inbox, customize home, refresh, resume work.
- First-time issues:
  - P2: Companion is introduced as another top-level assistant concept without enough distinction from Chat/Persona.
  - P2: Setup requirement is visible, but dependency and permission requirements need clearer "why."
- Power-user issues:
  - P2: Inbox triage and resume workflows need denser controls, filters, and direct links into workspaces/chat.
- Recommended fixes:
  - Add a concise companion status header: setup, permissions, inbox count, last activity.
  - Tie each inbox item to a clear resume/dismiss/snooze action.

#### `/agents`

- Primary user goal: Inspect registered coding/automation agents and ACP health.
- Evidence: Body shows ACP System Health, runner binary, agent status degraded, API keys OK, 7 registered agents, 3/7 available; no extracted `h1`.
- Main workflows: Refresh health, inspect available agents, diagnose runner/API keys, open related ACP/task surfaces.
- First-time issues:
  - P2: The page lacks a semantic title and does not explain "agent" versus MCP tools or chat workflows.
- Power-user issues:
  - P1: Operators need capability, runner, credentials, and endpoint status in a concise dashboard.
- Recommended fixes:
  - Add `Agents` heading, health summary, degraded cause, and actions to fix runner/API keys.
  - Link explicitly to `/acp-playground`, `/agent-tasks`, MCP Hub, and settings.

#### `/agent-tasks`

- Primary user goal: Manage long-running agent orchestration tasks.
- Evidence: Shows "Agent orchestration unavailable" and "Agent task routes are missing"; 404 console errors; no heading.
- Main workflows: Create/review tasks when available, open Agent Registry, open ACP Playground.
- First-time issues:
  - P2: The unavailable state is better than raw errors, but missing `h1` makes the page feel less owned.
- Power-user issues:
  - P2: Endpoint details are still too close to the main user state and should be behind diagnostics.
- Recommended fixes:
  - Preserve the good unsupported-state copy.
  - Add stable route heading, capability status, backend version requirement, and diagnostics disclosure.

#### `/chat-workflows`

- Primary user goal: Author and run structured Q&A chat workflows.
- Evidence: `h1` Chat Workflows; beta status says template authoring, guided run playback, and handoff are ready; context pickers/resumable runs land next.
- Main workflows: Author template, run guided Q&A, preserve run state, hand off to free chat.
- First-time issues:
  - P2: Available capabilities and roadmap items are mixed in one beta statement.
- Power-user issues:
  - P2: Workflow authors need recent templates, run status, failures, and duplication before future roadmap copy.
- Recommended fixes:
  - Split "Ready now" from "Coming next."
  - Foreground create template, import, recent runs, failed runs, and duplicate/edit actions.

#### `/chat-workspace`

- Primary user goal: Work with sources, approvals, runtime state, and chat in one workspace.
- Evidence: Sections include Workspace, Workspace chat, Scope, Sources, Model/Persona, Approvals, Task Progress, Runtime; status "Loading workspace context"; no `h1`.
- Main workflows: Choose workspace, filter/add sources, chat, set scope/model/persona, approve tasks, track runtime.
- First-time issues:
  - P1: The page opens as a cockpit without first establishing selected workspace, no-workspace state, or primary action.
  - P2: No semantic `h1`.
- Power-user issues:
  - P1: Experts need workspace persistence, recent workspaces, source set reuse, approvals queue, and runtime recovery at the top.
- Recommended fixes:
  - Add a workspace status header: selected workspace, sources, runtime, approvals, last saved.
  - Add create/open workspace as the first no-workspace action.
  - Move advanced panels below a stable task header.

### Knowledge, Research, Workspace, And Artifact Creation

#### `/knowledge`

- Primary user goal: Ask cited questions over selected library sources.
- Evidence: `h1` Ask Your Library; 4 selected sources; retrieval mode Balanced; web toggle; examples; recent searches; desktop/mobile screenshots; no mobile overflow.
- Main workflows: Select sources, choose retrieval mode/model, ask cited question, inspect citations, reuse recent searches.
- First-time issues:
  - P2: The page is one of the strongest surfaces, but it still overlaps conceptually with Chat's "Search your documents" and Research.
  - P3: "AI: Default default" style state can read as implementation leakage.
- Power-user issues:
  - P2: Needs saved source sets, retrieval presets, query templates, and faster source-scope switching.
- Recommended fixes:
  - Preserve this page as a design pattern: clear job, source status, examples, and focused input.
  - Add saved source sets and clarify ladder: Chat for conversation, Knowledge for direct cited answers, Research for multi-step investigation.

#### `/search`

- Primary user goal: Search or ask over personal content.
- Evidence: Redirect/status to `/knowledge`; renders Ask Your Library.
- Main workflows: Open knowledge QA/search.
- First-time issues:
  - P2: Generic "Search" suggests search across media, notes, prompts, settings, and docs, but route resolves to Knowledge QA.
- Power-user issues:
  - P2: Alias reduces command palette and keyboard predictability.
- Recommended fixes:
  - Either make `/search` a true global search hub or expose it as "Knowledge Search" everywhere.
  - Keep redirect copy if alias is retained.

#### `/research`

- Primary user goal: Create and inspect long-running deep research runs.
- Evidence: Extra page-file sweep rendered `Run console`, Newly created runs, Selected run: No run selected; desktop screenshot.
- Main workflows: Enter research question, start run, inspect selected run.
- First-time issues:
  - P2: `Run console` is implementation/operator language, not the user's task language.
  - P2: No-run state does not explain what inputs/sources are needed.
- Power-user issues:
  - P2: Researchers need run history, status, logs, source scope, retry/cancel, and export.
- Recommended fixes:
  - Rename visible title to Research Runs or Deep Research.
  - Show question, source scope, status, elapsed time, outputs, and recovery controls.

#### `/workspace-playground`

- Primary user goal: Start a research workspace with sources and a studio panel.
- Evidence: `h1` New Research; sections Sources, Start your research, Studio; mobile overflow 449/390; desktop/mobile screenshots.
- Main workflows: Add sources, start research, use tour, manage studio.
- First-time issues:
  - P1: Route label says Playground, page says New Research. That signals prototype status for what appears to be a core workflow.
  - P2: Source and studio panels increase cognitive load before the first research question.
- Power-user issues:
  - P1: Mobile/sidepanel overflow makes the workspace hard to use in constrained contexts.
  - P2: Workspace persistence and recovery need to be more obvious.
- Recommended fixes:
  - Rename user-facing label to Research Workspace.
  - Use a no-source/no-question guided state first, then expose studio controls.
  - Fix responsive layout around source and studio panels.

#### `/document-workspace`

- Primary user goal: Open and work through a document.
- Evidence: `h1` Document Workspace; no document selected; actions include Upload and Open document.
- Main workflows: Upload/open document, read contents, inspect insights/info, use help.
- First-time issues:
  - P2: The page does not explain how Document Workspace differs from Media Inspector or Knowledge QA.
- Power-user issues:
  - P2: Heavy document users need recent documents, resume location, extracted notes, citations, and export paths.
- Recommended fixes:
  - Add a clear empty state: open from media library, upload new document, or resume recent.
  - Show relationship to Knowledge and Research: read one document here, ask across many in Knowledge.

#### `/repo2txt`

- Primary user goal: Convert repository files into a text artifact.
- Evidence: Sections Source Provider and Output; GitHub/Local selector; filters; "No files loaded"; Generate Output/Copy/Download.
- Main workflows: Choose GitHub/local source, select files, generate output, copy/download.
- First-time issues:
  - P2: Route name and missing `h1` require technical inference.
  - P2: Privacy and local file access implications are not visible enough.
- Power-user issues:
  - P2: Repeat users need presets, include/exclude patterns, token estimates, and saved source configurations.
- Recommended fixes:
  - Use visible title "Repository to Text" or "Repository Export."
  - Add privacy note, token/file count preview, presets, and output format controls.

#### `/model-playground`

- Primary user goal: Test, compare, and tune model behavior.
- Evidence: `h1` Model Playground; also renders the generic chat empty state and Configuration panel.
- Main workflows: Simple chat, compare mode, configure model/prompt/context.
- First-time issues:
  - P2: The route promises model testing but initially looks like ordinary chat.
- Power-user issues:
  - P1: Model testers need side-by-side outputs, provider health, cost/latency, prompt reuse, and result comparison history.
- Recommended fixes:
  - Make comparison/configuration the primary layout.
  - Use chat composer only as the prompt input, not as the page's mental model.
  - Add recent experiments and provider health.

#### `/writing-playground`

- Primary user goal: Generate or manage writing sessions.
- Evidence: Sessions list, "Manuscript Draft test", "No session", Generate Ready, shortcut status; no extracted headings.
- Main workflows: Create/open writing session, generate, stop, edit draft.
- First-time issues:
  - P2: The page does not clearly state what kind of writing task it supports or where output is saved.
- Power-user issues:
  - P2: Writers need version history, templates, prompt settings, export, and diff/restore.
- Recommended fixes:
  - Add semantic title and a two-pane writing layout: brief/input, draft/output.
  - Show session persistence, templates, version list, and export.

#### `/presentation-studio`

- Primary user goal: Create structured narrated slide decks.
- Evidence: `h1` Presentation Studio; copy says create structured narrated slide decks, stage media, publish rendered presentation video.
- Main workflows: Create deck, choose style, stage media, render/export.
- First-time issues:
  - P2: Root page tells the purpose, but creation, recent projects, and render readiness are not all visible at the root.
- Power-user issues:
  - P2: Deck creators need recent projects, render job recovery, style presets, export status, and media staging state.
- Recommended fixes:
  - Add root dashboard: New deck, recent decks, render queue, style presets, failed jobs.
  - Include `/presentation-studio/new` in canonical route metadata.

#### `/audiobook-studio`

- Primary user goal: Convert text into an audiobook project.
- Evidence: Audiobook Studio copy, My Projects/New/Save, beta feature status, no extracted `h1`.
- Main workflows: Create project, add text, choose voice, save/generate.
- First-time issues:
  - P2: It does not clearly state prerequisites such as TTS provider readiness or source text requirements.
- Power-user issues:
  - P2: Batch generation, voice presets, chapter management, recent jobs, and retry/recovery are not prominent.
- Recommended fixes:
  - Add `h1`, readiness checks, source selection, voice readiness, recent jobs, and relation to TTS.

### Media, Library, Notes, Collections, And Sharing

#### `/media`

- Primary user goal: Browse, filter, and inspect ingested media.
- Evidence: `h1` Media Inspector; 918 items; filters; no item selected; mobile overflow 683/390; desktop/mobile screenshots.
- Main workflows: Search/filter, select item, inspect content/analyses, favorite, bulk select, trash, quick ingest.
- First-time issues:
  - P1: The default state is a dense inspector with no selected item. Users see many filters before a clear first task.
  - P2: Empty detail copy references a left sidebar, which fails on narrow layouts.
- Power-user issues:
  - P1: Mobile/sidepanel split-pane layout overflows, making large-library review hard outside desktop.
  - P2: Dense filters are useful, but saved views and selection recovery would make repeat work faster.
- Recommended fixes:
  - On desktop, keep density but make first selection and Quick Ingest more prominent.
  - On narrow viewports, use list-to-detail navigation with a back button.
  - Replace positional copy with responsive copy.

#### `/media-multi`

- Primary user goal: Select and bulk-review media items.
- Evidence: Status shows 0 items selected, 30 remaining; selection status safe; no extracted heading.
- Main workflows: Search/filter, select items, apply bulk/review actions, inspect shortcuts.
- First-time issues:
  - P2: Selection mechanics appear before route purpose or available actions.
- Power-user issues:
  - P2: Bulk users need persistent selection, undo/recovery, safe destructive action separation, and keyboard flow.
- Recommended fixes:
  - Add route title, selected count, action bar, undo, and keyboard help.
  - Clarify relationship to `/media` and `/review`.

#### `/review`

- Primary user goal: Review selected media/content.
- Evidence: Redirect/status to `/media-multi`; selection status appears.
- Main workflows: Enter media multi-review.
- First-time issues:
  - P2: Review appears distinct but behaves as Media Multi.
- Power-user issues:
  - P2: Duplicate naming hurts command palette, docs, and muscle memory.
- Recommended fixes:
  - Pick one canonical public label.
  - If alias remains, show "Review opens Media Multi" transition copy.

#### `/media-trash`

- Primary user goal: Restore or permanently delete trashed media.
- Evidence: `h1` Trash; auto-purge policy not configured; refresh and empty trash actions.
- Main workflows: Inspect trashed items, restore, select, empty trash.
- First-time issues:
  - P2: Retention and permanent deletion policy are not visible enough for a destructive area.
- Power-user issues:
  - P2: Operators need filters, bulk restore, audit trail, and clear empty trash safeguards.
- Recommended fixes:
  - Add retention policy, restore-all selected, permanent delete confirmation, and audit/recovery copy.

#### `/items`

- Primary user goal: Manage shared or generated items.
- Evidence: `h1` Items; filters; list includes workspace output artifacts.
- Main workflows: Refresh, filter, select, generate output from items.
- First-time issues:
  - P2: "Items" is too generic beside Media, Collections, Notes, Shared, and Chatbooks.
- Power-user issues:
  - P2: Library mental model remains unstable without a defined item taxonomy.
- Recommended fixes:
  - Rename around the actual object type or fold into a canonical Library/Artifacts route.
  - Add object categories and clear source relationships.

#### `/collections`

- Primary user goal: Manage saved articles, highlights, reading lists, and templates.
- Evidence: `h1` Collections; beta feature status; desktop screenshot.
- Main workflows: Save articles, create highlights, manage templates, import/export reading list.
- First-time issues:
  - P2: Reading and Collections are both exposed, creating label mismatch.
  - P3: Beta feature state can make a core library concept feel unstable.
- Power-user issues:
  - P2: Needs saved views, bulk actions, and direct routing from Media/Knowledge.
- Recommended fixes:
  - Pick one public label: Collections or Reading.
  - Add saved views, bulk edit, import/export status, and links from Media and Knowledge.

#### `/reading`

- Primary user goal: Open reading-list content.
- Evidence: Redirect/status to `/collections`; renders Collections.
- Main workflows: Reading list management through Collections.
- First-time issues:
  - P2: User chooses Reading but sees Collections.
- Power-user issues:
  - P2: Alias weakens repeat navigation and docs.
- Recommended fixes:
  - Canonicalize the public label.
  - If migration is ongoing, show "Reading is now Collections" only once and then route consistently.

#### `/notes`

- Primary user goal: Create, edit, and link notes.
- Evidence: Empty note state; Create note; backlink tip; no extracted `h1`.
- Main workflows: Create note, edit note, link notes, create study pack.
- First-time issues:
  - P2: No stable route title in heading extraction and weak explanation of how notes relate to media/research.
- Power-user issues:
  - P2: Note-heavy workflows need search, backlinks, recents, graph/context, export, and workspace linkage.
- Recommended fixes:
  - Add `h1` Notes, recent notes, create/import, search, backlinks, and "use in Knowledge/Workspace" links.

#### `/shared`

- Primary user goal: View or manage shared workspaces/content.
- Evidence: Body says `No shared workspaces available yet`; no headings.
- Main workflows: Open shared workspaces/content, manage incoming/outgoing shares.
- First-time issues:
  - P2: "Shared" does not say whether it means shared by me, shared with me, public links, or tokens.
- Power-user issues:
  - P1: Sharing needs revocation, permissions, link/token recovery, and audit state.
- Recommended fixes:
  - Add Shared landing with tabs: Shared with me, Shared by me, Links/tokens, Permissions.
  - Add empty states and recovery for expired/invalid share tokens.

#### `/chatbooks`

- Primary user goal: Export/import chatbooks and inspect related jobs.
- Evidence: Body says Chatbooks Playground; Export/Import/Jobs tabs; no extracted headings.
- Main workflows: Export bundle, import bundle, inspect jobs.
- First-time issues:
  - P2: Chatbook as a concept is not explained, and "Playground" makes it feel experimental.
- Power-user issues:
  - P2: Import/export users need recent bundles, conflict handling, background job progress, retry, and validation.
- Recommended fixes:
  - Rename visible page to Chatbooks, not Playground.
  - Add import/export entry cards, recent bundles, job failures, conflicts, and docs link.

#### `/chatbooks-playground`

- Primary user goal: Test or run chatbook export/import controls.
- Evidence: Same Chatbooks Playground surface as `/chatbooks`; no headings.
- Main workflows: Export/import/jobs.
- First-time issues:
  - P2: A playground route is exposed as a product root.
- Power-user issues:
  - P2: Duplicate route can pollute command palette, docs, and smoke tests.
- Recommended fixes:
  - Classify as Labs/internal or redirect to `/chatbooks`.
  - Keep test-only states out of default navigation.

### Sources, Connectors, Integrations, Watchlists, And Scheduling

#### `/sources`

- Primary user goal: Add and manage ingestion sources.
- Evidence: Page shows Sources, New source, and raw `Not Found (GET /api/v1/ingestion-sources)`; desktop/mobile screenshots; 404 console errors.
- Main workflows: Add source, inspect source, sync/archive, troubleshoot source endpoint.
- First-time issues:
  - P1: The main page state is a raw endpoint error. Users cannot distinguish empty, unavailable, unauthorized, old server, or broken route.
- Power-user issues:
  - P1: Operators need capability/version diagnostics and retry path, not raw route text.
- Recommended fixes:
  - Replace raw error with a shared capability state: unavailable, disabled, unauthorized, no sources, unreachable.
  - Include Retry, Quick Ingest, Health & Diagnostics, and docs actions.
  - Put endpoint details behind a collapsible diagnostics disclosure.

#### `/connectors`

- Primary user goal: Manage connector onboarding and configuration.
- Evidence: Page displays `Connectors Hub Is Coming Soon`.
- Main workflows: Connector setup or routing to current sources/integrations.
- First-time issues:
  - P2: Placeholder overlaps Sources and Integrations without telling users where real connector work lives.
- Power-user issues:
  - P2: Route visibility is not tied to capability readiness.
- Recommended fixes:
  - Hide until implemented or route to current Sources/Integrations surfaces.
  - If visible, show exact supported connectors and next setup actions.

#### `/integrations`

- Primary user goal: Review personal Slack/Discord integrations.
- Evidence: Extra sweep rendered `Personal integrations`; unavailable state says the server does not expose the personal integrations control-plane; 404s; screenshot.
- Main workflows: Refresh integrations, inspect Slack/Discord connections.
- First-time issues:
  - P2: Better than `/sources`, but the page still does not provide alternatives or setup path.
- Power-user issues:
  - P2: Integration admins need provider policy, install inventory, capability map, and permission state.
- Recommended fixes:
  - Use shared capability component and link to available admin/workspace integration surfaces.
  - Show personal vs workspace integration distinction.

#### `/scheduled-tasks`

- Primary user goal: Review reminder/scheduled tasks.
- Evidence: Extra sweep rendered `Scheduled tasks`; raw `Not Found (GET /api/v1/scheduled-tasks)`; screenshot.
- Main workflows: Review reminders, manage scheduled tasks, distinguish watchlist jobs.
- First-time issues:
  - P1: Main state is raw backend error, so users cannot understand what scheduling does.
- Power-user issues:
  - P1: Operators cannot distinguish missing module, wrong server version, auth issue, or empty task list.
- Recommended fixes:
  - Replace with shared capability state.
  - Provide next actions: open Watchlists for watchlist jobs, open server health, retry, docs.

#### `/watchlists`

- Primary user goal: Monitor RSS feeds, sites, and forums with scheduled ingestion.
- Evidence: `h1` Watchlists; overview guidance, docs, beta state, Open Feeds/Open Monitors; desktop screenshot.
- Main workflows: Review health, open feeds/monitors, add/refine feeds, review reports.
- First-time issues:
  - P2: The page is clearer than most, but beta/docs/status chrome competes with the primary monitor setup.
- Power-user issues:
  - P2: Repeat users need saved views, dense feed/monitor controls, failures, and schedule state.
- Recommended fixes:
  - Keep overview pattern, but make beta state dismissible.
  - Add a monitoring dashboard: failing monitors, due soon, last run, unread/new items.

#### `/workflow-editor`

- Primary user goal: Build and edit workflow graphs.
- Evidence: Body shows Untitled Workflow, Save, Nodes, Config, Run, Node Library; no extracted headings.
- Main workflows: Add nodes, configure workflow, save, run.
- First-time issues:
  - P2: Users cannot tell what type of workflows this editor creates or how it relates to Chat Workflows.
- Power-user issues:
  - P2: Workflow authors need create/import/recent templates, validation, run history, and failure diagnostics.
- Recommended fixes:
  - Add `h1` Workflow Editor, workflow type, selected workflow status, validation state, and recent templates.
  - Link explicitly to Chat Workflows where relevant.

### Settings, Admin, MCP, And Operations

#### `/settings`

- Primary user goal: Configure server, auth, models, chat, UI, data, and diagnostics.
- Evidence: No extracted `h1`; settings nav includes many groups and leaks `providerKeys.navTitle`; mobile overflow 2716/390; desktop/mobile screenshots.
- Main workflows: Search settings, change preferences, configure server/providers/models, import/export, reset.
- First-time issues:
  - P1: Too many unrelated settings appear in one broad surface: server/auth, provider keys, UI, chat, RAG, speech, import/export, reset.
  - P1: Mobile settings nav creates extreme horizontal scroll.
  - P2: Internal translation key leakage breaks trust.
- Power-user issues:
  - P1: Experts need fast, searchable, task-led settings groups and safe separation of destructive/data actions.
- Recommended fixes:
  - Reorganize into Server & Auth, Models & Providers, Chat & Persona, Knowledge/RAG, Media/Ingest, Extension/UI, Data Management, Admin/Diagnostics.
  - Fix responsive nav with drawer/accordion.
  - Remove internal i18n keys and add stable headings.

#### `/admin`

- Primary user goal: Operate and monitor server/admin functions.
- Evidence: Redirect/status `/admin/server`; page shows Server Admin, no system statistics, audio installer, and admin modules.
- Main workflows: Monitor server, install STT/TTS bundles, review admin modules and diagnostics.
- First-time issues:
  - P2: Admin root does not first explain what administration includes or whether server/admin APIs are available.
- Power-user issues:
  - P1: Operators need a single health/capability overview before drilling into modules.
- Recommended fixes:
  - Make `/admin` an operations landing page: server health, auth mode, capability map, workers, storage, recent failures, admin modules.
  - Use consistent unavailable-state components across admin subroutes.

#### `/mcp-hub`

- Primary user goal: Manage MCP servers, permissions, workspaces, governance, and audit.
- Evidence: MCP Hub copy, Getting Started panel, setup/access/workspaces/governance/audit tabs; no extracted `h1`; desktop/mobile screenshots; deprecated List warning.
- Main workflows: Add managed server, configure access, govern tools, inspect audit.
- First-time issues:
  - P2: Strong workflow copy, but missing semantic `h1` and duplicate help/empty panels reduce scan clarity.
- Power-user issues:
  - P2: Tool admins need fast status of servers, tool catalog, credentials, approvals, recent denials, and audit events.
- Recommended fixes:
  - Preserve MCP Hub as a positive pattern.
  - Add semantic title, server/tool status summary, and compress repeated help copy.

#### `/acp-playground`

- Primary user goal: Interact with ACP coding agents and inspect sessions/tools.
- Evidence: `Agent Playground`; status says ACP backend is not configured or unreachable; 500 error in sweep.
- Main workflows: Configure ACP backend, inspect sessions, inspect tools/capabilities.
- First-time issues:
  - P2: Error state and title blur together, so ACP can read as broken instead of unavailable.
- Power-user issues:
  - P2: Agent/tool developers need backend URL/status, setup docs, session list, and tool capability diagnostics.
- Recommended fixes:
  - Keep stable `ACP Playground` title.
  - Put backend unavailable state below with setup, diagnostics, and links to Agents/MCP.

### Context Assets: Prompts, Dictionaries, And World Books

#### `/prompts`

- Primary user goal: Manage reusable prompts and Prompt Studio workflows.
- Evidence: `h1` Prompts; no custom prompts; tabs for prompts/workspaces/custom copilot/studio/trash; mobile overflow 529/390; desktop screenshot.
- Main workflows: Create, import/export, search/filter, manage workspaces, test/evaluate prompts.
- First-time issues:
  - P2: The empty state is useful, but the first screen exposes too many concepts at once: collections, recents, density, sync, import/export, templates, studio.
- Power-user issues:
  - P2: Strong density, but prompt library and studio concepts need clearer hierarchy.
- Recommended fixes:
  - Make Prompts canonical.
  - Use a clear tab model: Library, Studio, Templates, Import/Export, Trash.
  - Fix mobile overflow and show primary Create Prompt action.

#### `/prompt-studio`

- Primary user goal: Open prompt-studio project/test/eval tooling.
- Evidence: Redirect/status to `/prompts?tab=studio&subtab=projects`; renders Prompts.
- Main workflows: Open studio projects, prompts, test cases, evaluations, optimization.
- First-time issues:
  - P2: User expects a distinct Prompt Studio page but lands inside Prompts.
- Power-user issues:
  - P2: Alias fragments docs, command palette, and route memory.
- Recommended fixes:
  - Keep compatibility redirect, but expose one public label and a visible "Studio" tab state.

#### `/dictionaries`

- Primary user goal: Manage reusable terminology substitutions for chat.
- Evidence: `h1` Chat Dictionaries; no dictionaries yet; learn-more link.
- Main workflows: Create dictionary, manage substitutions, attach/use in chat.
- First-time issues:
  - P2: Path says Dictionaries, page says Chat Dictionaries, and the activation context is not obvious.
- Power-user issues:
  - P2: Users need to know which chats/personas/workspaces a dictionary affects.
- Recommended fixes:
  - Group under Context Library.
  - Show activation scope, attached conversations/personas, and import/export.

#### `/world-books`

- Primary user goal: Manage structured lore/context books for chats and characters.
- Evidence: `h1` World Books; duplicate World Books heading; console 404.
- Main workflows: Create/manage world book, attach files/tools, use in characters/chats.
- First-time issues:
  - P2: Relationship to Characters, Persona, and Dictionaries is unclear.
- Power-user issues:
  - P2: Lore-heavy users need import/export, activation scope, broken-backend recovery, and search.
- Recommended fixes:
  - Put World Books, Dictionaries, and character context under a Context Library model.
  - Show where each world book is active and provide capability-aware errors.

### Audio, Study, Evaluation, Safety, And Specialized Tools

#### `/speech`

- Primary user goal: Use a combined speech workflow for record, transcribe, and synthesize.
- Evidence: Speech Playground copy; modes Round-trip, Speak, Listen; no extracted headings; console 404.
- Main workflows: Record, edit transcript, synthesize audio, switch mode.
- First-time issues:
  - P2: It overlaps with `/audio`, `/stt`, and `/tts` without explaining the audio model.
- Power-user issues:
  - P2: Audio users need readiness, recent transcripts/audio, provider health, and batch/history.
- Recommended fixes:
  - Choose a canonical audio hub.
  - Show STT/TTS readiness, provider status, recent jobs, and explicit relationship to STT/TTS pages.

#### `/stt`

- Primary user goal: Transcribe speech/audio into text.
- Evidence: STT Playground copy, configured-engine requirement, desktop screenshot; no extracted headings.
- Main workflows: Upload/record, select transcription model, run dictation, save transcript to Notes.
- First-time issues:
  - P2: Missing semantic title weakens accessibility and route orientation.
- Power-user issues:
  - P2: Repeat users need recent transcripts, batch upload, provider status, failures, and retry.
- Recommended fixes:
  - Add `h1` Speech to Text.
  - Show provider readiness, upload/record primary action, recent transcripts, and failed jobs.

#### `/tts`

- Primary user goal: Generate audio from text.
- Evidence: TTS Playground copy, voice/model/format controls, desktop screenshot; no extracted headings; console 404.
- Main workflows: Enter text, choose voice/model/format, generate audio.
- First-time issues:
  - P2: Voice/provider availability and 404 failures are not framed in plain user language.
- Power-user issues:
  - P2: Needs voice presets, favorites, batch jobs, recent generations, and provider health.
- Recommended fixes:
  - Add `h1` Text to Speech.
  - Show provider readiness, voice catalog state, generation history, and unavailable-state recovery.

#### `/audio`

- Primary user goal: Enter the audio workflow.
- Evidence: Redirect/status to `/speech`; renders Speech Playground; console 404.
- Main workflows: Audio hub or alias to Speech.
- First-time issues:
  - P2: Generic Audio silently becomes Speech, adding alias confusion.
- Power-user issues:
  - P2: Alias slows route memory and command search.
- Recommended fixes:
  - Make `/audio` the canonical hub or remove it from visible navigation.
  - If retained as alias, show route transition and canonical label.

#### `/evaluations`

- Primary user goal: Define, run, and inspect evaluation workflows.
- Evidence: `h1` Evaluations playground; beta state; clear worker-unavailable message that still offers completed-run reuse; screenshot.
- Main workflows: Configure eval recipe, run evals, reuse completed runs, inspect results.
- First-time issues:
  - P2: "Playground" and advanced JSON/config options may intimidate first-time users.
- Power-user issues:
  - P2: Evaluators need presets, run history, worker status, reproducible configs, and compare/retry.
- Recommended fixes:
  - Preserve the worker-unavailable state pattern.
  - Add simple/advanced split, task presets, run history, and worker diagnostics.

#### `/flashcards`

- Primary user goal: Create, import/export, and review flashcards/study packs.
- Evidence: Study tabs, export preview, image occlusion status, no extracted `h1`.
- Main workflows: Manage study packs, import/export, create from media/notes, test with quiz, image occlusion.
- First-time issues:
  - P2: Many modes are visible before the page establishes whether the user should study, author, import, or generate.
- Power-user issues:
  - P2: Study users need deck selection, due counts, keyboard review flow, import/export, and error recovery.
- Recommended fixes:
  - Add `h1` Study or Flashcards.
  - Use a primary mode selector: Review, Author, Generate, Import/Export, Image Occlusion.

#### `/quiz`

- Primary user goal: Generate and take quizzes from sources.
- Evidence: Beta/degraded state, Search/Take Quiz/Generate/Create/Manage/Results tabs; no extracted headings; AntD warnings.
- Main workflows: Select media/notes sources, generate quiz, take quiz, view results.
- First-time issues:
  - P2: The start state is not semantically clear, and "Beta Degraded" appears before the task model is understood.
- Power-user issues:
  - P2: Quiz users need source sets, deck linkage, recent attempts, result history, and retry.
- Recommended fixes:
  - Add stable title, source/deck selector, Start Quiz primary action, recent results, and degraded-state explanation.

#### `/moderation-playground`

- Primary user goal: Test moderation/content controls.
- Evidence: Moderation Playground; family guardrail setup CTA; no extracted `h1`.
- Main workflows: Set up family guardrails, test moderation input, inspect result.
- First-time issues:
  - P2: Safety, family guardrails, and playground testing are mixed into one route.
- Power-user issues:
  - P2: Safety reviewers need presets, audit trail, provider/capability status, saved test cases.
- Recommended fixes:
  - Add `h1` Moderation Playground.
  - Separate "set up safety rules" from "test moderation behavior."
  - Add supported checks and saved test cases.

#### `/content-review`

- Primary user goal: Review drafts before committing or saving.
- Evidence: Content Review with 0 ready, 0 committed, 0 total; Commit All, Clear drafts; no extracted headings.
- Main workflows: Review Quick Ingest drafts, commit all, clear drafts, open Quick Ingest.
- First-time issues:
  - P2: Users cannot tell whether this is moderation review, claims review, or ingest pre-save review.
- Power-user issues:
  - P2: Reviewers need queue counts, filters, item ownership, failed/blocked drafts, and undo.
- Recommended fixes:
  - Add title and queue taxonomy.
  - Show empty state tied to Quick Ingest and draft lifecycle.

#### `/claims-review`

- Primary user goal: Review claims.
- Evidence: Redirect/status to `/content-review`.
- Main workflows: Enter content review queue.
- First-time issues:
  - P2: Claims Review appears distinct but lands in Content Review.
- Power-user issues:
  - P2: Review workflow names split without distinct behavior.
- Recommended fixes:
  - Either implement a claims-specific queue or rename/remove the alias.
  - If alias remains, show the canonical route transition.

#### `/data-tables`

- Primary user goal: Generate structured tables from chats/documents/knowledge.
- Evidence: `h1` Data Tables Studio; beta feature says backend support required.
- Main workflows: Prompt for a table, select source, generate structured table, export.
- First-time issues:
  - P2: Backend requirement is mentioned but not translated into setup or capability status.
- Power-user issues:
  - P2: Table users need schema preview, source selection, export formats, job history, and validation errors.
- Recommended fixes:
  - Add backend readiness, examples, source selector, schema preview, and export/retry state.

#### `/chunking-playground`

- Primary user goal: Test chunking settings for RAG or ingestion.
- Evidence: Chunking Playground; Single, Compare, Templates, Capabilities; no extracted headings.
- Main workflows: Paste text, select chunking settings, compare outputs, use templates.
- First-time issues:
  - P2: Technical tool is not framed around "improve retrieval quality" or another user task.
- Power-user issues:
  - P2: RAG tuners need saved configs, output metrics, before/after comparisons, and export-to-settings.
- Recommended fixes:
  - Add stable title and use case framing.
  - Show presets, token/chunk metrics, comparison output, and "apply/copy config" path.

#### `/kanban`

- Primary user goal: Manage boards and tasks.
- Evidence: `h1` Kanban Playground; board list; deprecated Drawer warning.
- Main workflows: Select board, create board, inspect/manage tasks.
- First-time issues:
  - P2: "Playground" label undermines confidence in persistence.
- Power-user issues:
  - P2: Planning users need saved board state, workflow links, filters, and recovery.
- Recommended fixes:
  - Classify as Labs if experimental, or rename as Kanban Boards if production.
  - Show persistence and recent board state.

#### `/skills`

- Primary user goal: Import, seed, create, and manage skills.
- Evidence: Import, Seed Built-ins, New Skill form, empty table; no headings; console warnings/errors.
- Main workflows: Import skill, seed built-ins, create skill, inspect actions.
- First-time issues:
  - P2: Page appears underspecified because no title/state explains what a skill is or why no data appears.
- Power-user issues:
  - P2: Tool admins need installed/available grouping, capability state, endpoint diagnostics, import validation.
- Recommended fixes:
  - Add `h1` Skills, empty state, installed/built-in/imported tabs, and capability-aware errors.

#### `/vn-assets`

- Primary user goal: Prepare visual-novel asset packs.
- Evidence: `h1` VN asset packs; sections Packs, Setup, Selected pack.
- Main workflows: Create/select pack, setup matrix generation, review, portability.
- First-time issues:
  - P2: Specialized VN route is unclassified relative to personas, media, and labs.
- Power-user issues:
  - P2: VN users need asset readiness, validation errors, and launch-to-play path.
- Recommended fixes:
  - Classify under Advanced Tools or Labs.
  - Show selected pack readiness and direct open in `/vn-play`.

#### `/vn-play`

- Primary user goal: Run visual-novel play sessions.
- Evidence: Extra sweep rendered `VN play`; Sessions, Scene, Runtime inspector; desktop screenshot.
- Main workflows: Start freeform/story session, select scene, inspect runtime metadata.
- First-time issues:
  - P2: Play controls and runtime inspector/debug language are mixed on the root.
- Power-user issues:
  - P2: Session recovery, save state, branching, and inspector controls need clearer priority.
- Recommended fixes:
  - Separate Play from Runtime Inspector or mark inspector advanced.
  - Add recent sessions and resume path.

### Documentation, Notifications, Debug, And Internal Preview

#### `/documentation`

- Primary user goal: Browse product and API documentation.
- Evidence: `h1` Documentation; body opens Admin Organizations and Teams API content; doc sources listed.
- Main workflows: Browse docs, search docs, read setup/API/module documentation.
- First-time issues:
  - P2: Docs root opens deep admin API content instead of a task-oriented index.
- Power-user issues:
  - P2: Operators need search, module/version filtering, and quick access to self-hosted setup/API docs.
- Recommended fixes:
  - Make root an index: Getting Started, Self-hosting, API, Media, RAG, Audio, Extension, Admin.
  - Remember last doc only after the index is accessible.

#### `/notifications`

- Primary user goal: Review job and app notifications.
- Evidence: `h1` Notifications; unread count 7; repeated Job completed cards; actions Mark read, Snooze, Dismiss.
- Main workflows: Refresh, change preferences, mark read, snooze, dismiss, open related job.
- First-time issues:
  - P2: Event list does not explain grouping, severity, source, or next action.
- Power-user issues:
  - P2: Users need filters by type/status, mark all read, clear completed, and deep links to jobs/items.
- Recommended fixes:
  - Add type/status filters, grouping, bulk actions, and source deep links.

#### `/composer-variants-preview`

- Primary user goal: Preview composer redesign variants.
- Evidence: Extra sweep rendered `Primer composer variants`; dev harness; mocked state; desktop/sidepanel variants.
- Main workflows: QA visual variants, compare composer designs.
- First-time issues:
  - P2: Internal design preview is exposed as a root route and can look like product UI.
- Power-user issues:
  - P2: QA routes can pollute navigation, command palette, docs, and smoke inventory.
- Recommended fixes:
  - Mark as internal QA/debug route.
  - Exclude from user navigation and self-hosted docs.

#### `/onboarding-test`

- Primary user goal: Preview onboarding changes without changing production flow.
- Evidence: `h1` Onboarding Test Harness; setup card preview.
- Main workflows: Validate onboarding UX, jump to setup.
- First-time issues:
  - P2: QA harness can be mistaken for real onboarding.
- Power-user issues:
  - P2: Test route should not appear in user-facing inventory without explicit classification.
- Recommended fixes:
  - Mark internal and keep out of navigation/command palette.
  - Keep smoke coverage under QA/debug category only.

## Extension Sidepanel Inventory

There are two sidepanel registries, and they do not currently declare the same route set.

| Source | Declared sidepanel routes | UX risk |
|---|---|---|
| Shared registry: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx` | `/`, `/chat`, `/agent`, `/companion`, `/companion/conversation`, `/clipper`, `/persona`, `/flashcards`, `/settings`, `/error-boundary-test` | Shared product intent includes companion, clipper, flashcards, and conversation routes. |
| Extension wrapper: `apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx` | `/`, `/agent`, `/persona`, `/settings`, `/error-boundary-test` | Extension runtime wrapper omits shared sidepanel routes and maps `/` directly to chat, so user-facing extension behavior may diverge from shared WebUI expectations. |

This is not automatically a bug, but it is a product risk: route availability should be intentional and documented in one capability matrix rather than discovered by comparing registries.

## Severity-Ranked UX Findings

### P1: Navigation And IA Are Fragmented Across Multiple Source-Of-Truth Systems

Page/route: Cross-page WebUI and extension, especially `/`, `/chat`, `/media`, `/knowledge`, `/settings`, sidepanel routes
Evidence:

- `page-inventory.ts` lists 124 active routes, with comments such as "missing" for routes that now exist.
- `route-registry.tsx` defines shared option routes and redirects such as `/prompt-studio` -> `/prompts?tab=studio`, `/settings/image-gen` -> `/settings/image-generation`.
- `CommandPalette.tsx` labels a command "Go to Chat" but navigates to `/` with `targetPath: "/"`.
- `header-shortcut-items.ts` uses grouped shortcuts like Chat & Persona, Research, Library, Safety, Creation, Planning & Learning, Automation & Agents, Tools, Admin & Help.
- `ModeSelector.tsx` uses a different grouping: primary modes plus a "More" menu.
- `apps/tldw-frontend/components/layout/Header.tsx` still defines another top nav with labels such as Home, Items, Reading, Research, Search, Audio, Evals, Config.

Why it matters: First-time users must infer the product model from inconsistent labels and route aliases. Returning users cannot build reliable muscle memory if the same destination appears as Reading, Collections, Search, Knowledge QA, Prompt Studio, Prompts, Audio, or Speech depending on entry point.

Recommended fix: Establish a canonical IA contract with one user-facing route taxonomy, route aliases marked as legacy/internal/hosted-only, and one shared navigation manifest consumed by header shortcuts, mode selector, command palette, sidepanel, smoke inventory, and documentation. Keep aliases for compatibility, but make user-facing labels stable.

Implementation scope: Shared route metadata, command palette route targets, header shortcut group definitions, smoke inventory generation, docs. No backend API change required.

Expected user impact: Lower navigation recall burden, more predictable extension/WebUI parity, fewer false starts from command search and shortcut navigation.

### P1: Mobile Layouts For Core Root Pages Overflow Instead Of Reflowing

Page/route: `/chat`, `/media`, `/settings`, `/settings/model`, `/prompts`, `/workspace-playground`
Evidence:

- Mobile sweep at 390 x 844 reported horizontal overflow on six of eleven representative pages:
  - `/chat`: client width 390, scroll width 591
  - `/media`: client width 390, scroll width 683
  - `/settings`: client width 390, scroll width 2716
  - `/settings/model`: client width 390, scroll width 2734
  - `/prompts`: client width 390, scroll width 529
  - `/workspace-playground`: client width 390, scroll width 449
- `mobile-settings.png` shows settings navigation laid out as a very wide horizontal strip before the content, forcing page-level horizontal scroll.
- `mobile-media.png` shows the media list and detail panel side by side, with the empty detail state off to the right while the copy still says "left sidebar".
- `mobile-chat.png` shows the empty-state mode card area visually colliding with the sticky composer region.

Why it matters: Mobile users cannot reliably complete primary workflows if route content escapes the viewport. Even for desktop-first self-hosted users, extension sidepanel constraints make narrow-layout resilience a core requirement.

Recommended fix: Define shared responsive contracts for root-route shells: one-column mode for settings, drawer or segmented route navigation for settings groups, collapsible master-detail for media, composer-safe viewport constraints for chat, and no page-level horizontal scroll at 390px. Add responsive smoke assertions for scroll width and touch target minimums on representative root routes.

Implementation scope: Shared layout shell, settings layout, media inspector layout, chat empty/composer layout, prompts/workspace route containers, Playwright assertions. No backend API change required.

Expected user impact: Core workflows become usable in extension-sized and phone-sized viewports, with fewer accidental offscreen controls.

### P1: Setup And Onboarding Are Mixed With Chat-Oriented Global Chrome

Page/route: `/`, `/setup`
Evidence:

- Desktop `/` screenshot shows "Home Onboarding" and "Welcome to tldw Browser Assistant. Let's get you connected", but the full app header and sidebar are already present, including Search, New Chat, Quick Ingest, Temp, Character, notifications, settings, and many route icons.
- `/setup` has no `h1` in the sweep and shows both "Setup Wizard" and the same onboarding card.
- The observed setup form shows "Server URL" as `http://127.0.0.1:3000`, which is the frontend dev server origin in this local run, while the user-facing copy says to connect the server.
- First-run setup exists both at `/` and `/setup`, with unclear hierarchy.

Why it matters: The first-time user needs one job: connect or choose demo mode. Exposing chat, persona, notifications, route icons, settings, and global command affordances before connection increases cognitive load and blurs whether the app is ready.

Recommended fix: Use a dedicated setup shell for setup-required states. Keep only the controls needed for connection, demo, docs, diagnostics, and theme/accessibility. Make `/setup` the canonical setup route and let `/` route to setup or dashboard based on connection state. Validate/display backend origin in user language and avoid showing a frontend-origin value as the server unless it is intentionally a proxy mode.

Implementation scope: Setup route shell, home resolver, connection state copy, server URL validation/status display. No backend API change likely required.

Expected user impact: Faster first successful connection, fewer wrong-server mistakes, cleaner mental model for hosted vs self-hosted vs demo modes.

### P1: `/sources` Exposes A Raw Technical 404 As The Main Page State

Page/route: `/sources`, `/admin/sources`
Evidence:

- Desktop `/sources` screenshot shows a large red alert: `Not Found (GET /api/v1/ingestion-sources)`.
- Mobile `/sources` shows the same raw endpoint error directly below the primary "New source" button.
- Connected desktop sweep captured two 404 console errors plus an Ant Design alert warning for `/sources`.
- `/admin/sources` repeats the same `Not Found (GET /api/v1/ingestion-sources)` state.

Why it matters: Users cannot tell whether they have no sources, lack permissions, are on an older server, misconfigured auth, or hit a missing feature. It violates match between system and user language and gives no recovery path beyond guessing.

Recommended fix: Replace raw endpoint text with a capability-aware state: "Sources are not available on this server" vs "No sources yet" vs "You do not have permission" vs "Could not reach server". Provide `Retry`, `Open Health & Diagnostics`, `Use Quick Ingest`, and `Learn source setup` actions as appropriate. Keep the endpoint details behind a collapsible diagnostics disclosure.

Implementation scope: Sources page error mapping, shared API error-state component, admin sources parity. Backend change only needed if the frontend cannot distinguish missing route from auth/capability state.

Expected user impact: Users understand the next step instead of seeing implementation internals.

### P1: Model Settings Overwhelm Users With A Full Catalog Before Usable Configuration

Page/route: `/settings/model`
Evidence:

- Desktop `/settings/model` screenshot is a very long provider/model catalog, with hundreds of model chips visible under providers.
- Sweep body text starts with "Set your defaults", then immediately exposes many provider groups and model names.
- Sweep captured a 403 console error while the visible page still says OpenAI OAuth is unavailable and continues into the full model catalog.
- The page has no visible `h1` or `h2` in the extracted heading structure.

Why it matters: Model/provider setup is central to successful chat. A first-time user needs to know which providers are configured and usable. A power user needs search, recents, configured-only filters, provider health, and scoped settings. A full unfiltered catalog makes both journeys slower.

Recommended fix: Make the default view "configured and usable first". Show default provider/model, provider health, missing credentials, OAuth/API-key status, and recent/favorite models before the full catalog. Collapse the full catalog behind search and provider filters. Convert background 403s into visible, plain-language capability states.

Implementation scope: Model settings page layout and state model, provider/model list presentation, error mapping. Backend API change only if provider health/status cannot be retrieved consistently.

Expected user impact: Faster model setup and fewer failed chats caused by selecting unavailable models.

### P2: Global Chrome Is Chat-Specific On Non-Chat Pages

Page/route: Cross-page, visible on `/media`, `/knowledge`, `/sources`, `/settings`, `/mcp-hub`, `/stt`, `/tts`
Evidence:

- Desktop screenshots for non-chat routes all show a top header branded "tldw Assistant" with chat controls such as Temp, Character, New Chat, and Quick Ingest.
- `WebLayout.tsx` mounts chat sidebar, current chat model settings, quick ingest host, notes dock, buddy shell, timeline, command palette, and keyboard shortcuts across the general option layout.
- `/media` and `/settings` screenshots show chat chrome even when the user's task is media inspection or configuration.

Why it matters: A global shell should orient users to the product, not make every route feel like a chat submode. This creates conceptual noise and makes root pages feel less purposeful.

Recommended fix: Separate app-level navigation from chat-level controls. Keep global search, settings, notifications, and route launcher globally available, but move Temp, Character, current model, chat sidebar, and chat-specific actions into chat/workspace surfaces or a context-specific action rail.

Implementation scope: Shared `Header`, `WebLayout`, route metadata for context-specific actions, command palette action availability. No backend API change required.

Expected user impact: Better page identity, lower cognitive load, clearer route purpose.

### P2: Settings Navigation Leaks An Internal Translation Key

Page/route: `/settings`, `/settings/model`, other settings routes
Evidence:

- Desktop and mobile settings screenshots show `providerKeys.navTitle` in the settings nav.
- Sweep body snippets for `/settings` and `/settings/model` include `providerKeys.navTitle`.

Why it matters: Internal i18n keys break trust and force users to interpret implementation language. It is small in code scope but high-signal as a product quality issue.

Recommended fix: Fix the settings navigation label fallback so the route displays a user-facing label such as "Provider Keys", "API Keys", or "Model Providers". Add a smoke assertion that settings nav labels do not contain dotted translation keys.

Implementation scope: Settings labels/i18n fallback and focused test. No backend API change required.

Expected user impact: Removes visible implementation leakage from a core configuration area.

### P2: Command Palette Contains A Concrete Route-Label Mismatch

Page/route: Global command palette
Evidence:

- `CommandPalette.tsx` defines `id: "nav-chat"`, label "Go to Chat", but `action: () => { navigate("/") }` and `targetPath: "/"`.
- The actual route inventory contains `/chat` as a distinct page and `/` as home/setup/onboarding.

Why it matters: Command palettes are power-user infrastructure. If search says "Go to Chat" and sends the user to setup/home, it breaks trust in keyboard navigation and worsens the already fragmented IA.

Recommended fix: Change the target to `/chat` if the command means chat. If `/` is intentionally the starting route, relabel it "Go Home" or "Go to Start" and add a separate "Go to Chat" command.

Implementation scope: Command palette route metadata and tests. No backend API change required.

Expected user impact: Restores predictable command navigation and supports expert workflows.

### P2: Capability And Unsupported-State Handling Is Inconsistent

Page/route: `/sources`, `/admin/*`, `/integrations`, `/scheduled-tasks`, `/acp-playground`, `/agent-tasks`, `/settings/family-guardrails`, `/settings/guardian`, `/evaluations`, `/mcp-hub`
Evidence:

- `/sources` exposes raw endpoint text.
- `/admin/monitoring` reports `Sandbox diagnostics unavailable Not Found (GET [admin-endpoint])`.
- `/admin/integrations` reports `Unable to load Telegram linked actors Not Found (GET /api/v1/integrations/workspace/telegram/linked-actors)`.
- `/scheduled-tasks` reports `Unable to load scheduled tasks Not Found (GET /api/v1/scheduled-tasks)`.
- `/admin/llamacpp` uses a clearer message: "Admin APIs are not available on this server".
- `/agent-tasks` uses a clearer message: "Agent orchestration unavailable... Open Agent Registry... Open ACP Playground".
- `/settings/guardian` uses a clear unsupported state.
- `/evaluations` clearly explains that new recipe runs are unavailable because the worker is not running and offers reuse of completed runs.

Why it matters: Unsupported states are expected in a self-hosted, modular app. The UX issue is inconsistency: some pages teach the user what changed, while others expose endpoint failures.

Recommended fix: Create a shared capability-state vocabulary and component: unavailable, disabled, not configured, missing permission, unreachable, degraded, no data, beta. Require each route to pick one, include a next action, and put raw endpoint details behind diagnostics.

Implementation scope: Shared state component and per-route adoption. Backend may need a lightweight capability map if route probing remains ambiguous.

Expected user impact: Users can diagnose self-hosted configuration gaps without reading implementation details.

### P2: Media Inspector Is Powerful But Not First-Selection Or Mobile Friendly

Page/route: `/media`
Evidence:

- Desktop `/media` starts with 918 media items, filter controls, bulk/trash actions, and an empty detail pane saying "No media item selected".
- Mobile `/media` overflows horizontally and keeps a split-pane layout, with the detail empty state offscreen to the right.
- The empty detail state says "Select a media item from the left sidebar", which is inaccurate in constrained/mobile contexts.

Why it matters: Media is likely a primary daily route. Experienced users need dense filters, but first-time users and mobile/sidepanel users need a clear selection flow. The current default state makes the first useful item/action harder to find.

Recommended fix: Use responsive master-detail behavior: list-only until item selected on narrow viewports, then detail view with a back affordance. On desktop, preserve density but make the empty state and primary ingest/search path more prominent. Replace positional copy with context-aware copy.

Implementation scope: Media route layout and empty-state copy, responsive tests. No backend API change required.

Expected user impact: Faster first item inspection and usable media browsing on narrow screens.

### P2: Settings Is Too Broad For One Flat Route Experience

Page/route: `/settings`, `/settings/*`
Evidence:

- Settings nav includes server/auth, provider keys, chat, UI, splash screens, quick ingest, RAG, speech, manage models, MCP, image generation, evaluations, prompt studio, moderation, health, knowledge tools, workspace, and about.
- Desktop `/settings` provides a useful filter and current-section indicator, but the page is long and scroll-heavy.
- Mobile `/settings` turns the nav into a massive horizontal strip with 2716px scroll width.
- Several settings pages show no extracted `h1`, which weakens heading hierarchy.

Why it matters: Settings is a power-user area, but this volume makes first-time setup harder and expert configuration slower. Settings should be findable, grouped, searchable, and responsive.

Recommended fix: Split settings into task-led groups with route-aware landing cards: Server & Auth, Models & Providers, Chat, Knowledge/RAG, Media/Ingest, Extension/UI, Admin/Diagnostics. Use a mobile drawer or accordion for sections. Keep the filter but make it search titles, descriptions, and synonyms.

Implementation scope: Settings IA, nav layout, heading structure, search index. No backend API change required.

Expected user impact: Faster setup and fewer configuration mistakes.

### P2: Extension Sidepanel Route Availability Is Not Clearly Aligned With Shared Product Intent

Page/route: Extension sidepanel
Evidence:

- Shared sidepanel registry includes `/chat`, `/companion`, `/companion/conversation`, `/clipper`, `/flashcards`, and `/settings`.
- Extension sidepanel wrapper includes `/`, `/agent`, `/persona`, `/settings`, and an error-boundary test route, with `/` mapped directly to `SidepanelChat`.

Why it matters: The extension is a constrained, high-frequency workflow surface. If shared and extension route registries diverge without an explicit capability matrix, users may get different behavior depending on entry point, and QA may miss missing sidepanel routes.

Recommended fix: Publish a sidepanel route availability matrix in code/docs and generate tests from it. If routes are intentionally omitted from extension builds, declare why and what replacement workflow is available.

Implementation scope: Route metadata/docs/tests, possibly extension route registry alignment. No backend API change required.

Expected user impact: More predictable extension behavior and fewer broken assumptions in shared components.

### P2: Chat Empty State And Composer Compete On Narrow Screens

Page/route: `/chat`
Evidence:

- Desktop `/chat` has a centered empty state with "Start a new chat", "Start chatting", "Quick Ingest", and five equal mode cards.
- Mobile `/chat` shows the sticky composer overlapping the empty-state mode area, and the screenshot captures a partially obscured "Choose a mode" section.
- Mobile sweep reports horizontal overflow for `/chat`.

Why it matters: The primary chat action should be immediate and stable. First-time users should not have to choose among five modes before sending a message, and mobile users should not see instructional content hidden behind the composer.

Recommended fix: Make the first message field the dominant first-run action. Treat modes as progressive disclosure, recent presets, or a compact segmented launcher that does not compete with the composer. Reserve detailed mode cards for an expanded "Choose mode" panel.

Implementation scope: Chat empty state, composer layout, mode launcher, mobile tests. No backend API change required.

Expected user impact: Faster first message and fewer layout collisions in extension-sized contexts.

### P2: Research, Knowledge, Chat, And Workspace Routes Do Not Form A Clear Product Ladder

Page/route: `/knowledge`, `/chat`, `/workspace-playground`, `/chat-workspace`, `/research`, `/document-workspace`, `/model-playground`, `/writing-playground`, `/repo2txt`
Evidence:

- `/knowledge` presents a strong direct-QA route: "Ask Your Library", selected sources, retrieval mode, web toggle, examples, recent searches, and one large question input.
- `/chat` includes a "Search your documents" mode card and a "Deep research" mode card.
- `/workspace-playground` presents `h1` "New Research" with sections "Sources", "Start your research", and "Studio".
- `/chat-workspace` presents sections "Workspace", "Workspace chat", "Scope", "Sources", "Model / Persona", "Approvals", "Task Progress", and "Runtime".
- `/research` exists as a separate page-file route outside the smoke inventory.
- Route names mix user-facing jobs and implementation/prototype terms: Research Studio, Workspace Playground, Chat Workspace, Document Workspace, Model Playground, Writing Playground.

Why it matters: These routes are central to the product vision, but users must infer which surface is for direct cited answers, which is for multi-step research, which is for agentic workspace execution, and which is for transformations. That creates avoidable route choice anxiety for first-time users and route-switching overhead for experienced users.

Recommended fix: Define a product ladder and reflect it consistently in route labels and entry points:

- Ask: one cited question over selected sources.
- Research: multi-step investigation over selected sources.
- Workspace: persistent work with sources, notes, approvals, and runtime state.
- Transform: repo/document/model/writing tools that create artifacts.

Implementation scope: Route labels, nav grouping, page subtitles, redirects/aliases, command palette, and route metadata. No backend API change required.

Expected user impact: Users can choose the right work surface without memorizing product history or route aliases.

### P2: Root Pages Have Inconsistent Heading Landmarks

Page/route: `/setup`, `/chat`, `/sources`, `/settings`, `/settings/model`, `/mcp-hub`, `/stt`, `/tts`, `/chat-workspace`
Evidence:

- Desktop sweep extracted no `h1` for `/setup`, `/chat`, `/sources`, `/settings`, `/settings/model`, `/mcp-hub`, `/stt`, `/tts`, and `/chat-workspace`.
- Several of those pages have visually prominent titles, but the semantic heading structure does not expose them as the primary page heading in the sweep.
- `/settings/model` had no extracted `h1` or `h2`, despite being one of the most important setup routes.

Why it matters: Heading structure is not cosmetic. It supports screen-reader navigation, keyboard scanning, browser/extension route context, and automated smoke checks. Missing or inconsistent `h1` structure makes complex pages harder to understand and harder to QA.

Recommended fix: Require each root page to expose one stable `h1` matching its user-facing route name, followed by task-scoped `h2` sections. Add smoke coverage that fails if a user-facing root route has neither `h1` nor an approved exception.

Implementation scope: Page shells, title components, settings layout wrappers, smoke tests. No backend API change required.

Expected user impact: Better accessibility, easier orientation, stronger route QA.

### P2: General Settings Mixes Routine Preferences With High-Risk System Actions

Page/route: `/settings`
Evidence:

- Mobile and desktop settings screenshots show routine preferences, connection status, language, theme, OCR, web search, extension actions, import/export, upload, and `System Reset` in one long settings flow.
- The settings route shows destructive `Reset All` below other configuration and data movement controls in the same scroll path.
- The route has no extracted `h1`, and mobile settings produces a 2716px scroll width.

Why it matters: Settings is where users recover and configure the product. Mixing low-risk UI preferences with data import/export and reset controls increases accidental-action anxiety and slows expert configuration. Even if a confirmation exists later, the page-level IA does not separate routine tuning from data/system operations.

Recommended fix: Separate Settings into stable task groups:

- Server & Auth
- Models & Providers
- Chat & Persona
- Knowledge & RAG
- Media & Ingest
- Extension & UI
- Data Management
- Admin & Diagnostics

Keep destructive actions in a clearly separated Data Management or Danger section with explicit recovery language.

Implementation scope: Settings grouping, route/sidebar IA, heading structure, responsive settings nav. No backend API change required.

Expected user impact: Safer configuration, lower anxiety, faster repeat settings changes.

### P2: Route Inventory And Smoke Coverage Are Not Authoritative Enough For A Surface This Large

Page/route: Cross-page QA and product governance
Evidence:

- `apps/tldw-frontend/pages` contains user-facing page-file routes not present in `apps/tldw-frontend/e2e/smoke/page-inventory.ts`, including `/research`, `/integrations`, `/scheduled-tasks`, `/sources/new`, `/vn-assets`, `/vn-play`, and `/presentation-studio/new`.
- The smoke inventory includes comments such as "missing" for routes that now exist.
- The shared route registry, extension route registry, Next page tree, settings nav, sidepanel registries, command palette, header shortcuts, and old Next header can all expose or imply different routes.

Why it matters: A large self-hosted product can tolerate many routes, but it cannot tolerate route truth being discovered by manual comparison. QA gaps become UX gaps when pages exist but are not classified, tested, hidden, or intentionally exposed.

Recommended fix: Generate the smoke inventory from canonical route metadata, with explicit fields for user-facing, debug, hosted-only, self-hosted, beta, admin-only, extension-sidepanel, alias, redirect, and deprecated.

Implementation scope: Route metadata and smoke test generation. No backend API change required.

Expected user impact: Fewer broken or accidental routes, clearer navigation policy, more reliable future audits.

### P2: Specialized, Hosted, Beta, And Debug Routes Need A Visibility Policy

Page/route: `/billing`, `/for/*`, `/vn-assets`, `/vn-play`, `/presentation-studio/new`, `/composer-variants-preview`, `/onboarding-test`, `/__debug__/*`
Evidence:

- The page tree includes hosted/commercial routes, role/marketing routes, VN tools, presentation studio routes, composer preview, onboarding test, and debug/spec routes.
- The smoke and extra sweeps treated `/__debug__/authz.spec` as non-user-facing, but several other specialized routes are user-facing enough to load and screenshot.
- The report scope is the self-hosted broader set, but the current route inventory does not encode which surfaces belong in default self-hosted navigation, which are advanced tools, and which are QA-only.

Why it matters: Unclassified route surfaces create product ambiguity. Users may encounter pages that are valid but unexplained, or QA may spend effort on routes that should not be user-facing.

Recommended fix: Add a route visibility policy:

- Default self-hosted
- Advanced self-hosted
- Hosted/commercial
- Admin/operator
- Extension sidepanel
- Labs/beta
- Internal QA/debug
- Legacy alias/redirect

Surface only the appropriate set in navigation, command palette, docs, and smoke suites.

Implementation scope: Route metadata, nav filtering, command palette filtering, smoke inventory categories. No backend API change required.

Expected user impact: Clearer product shape and less accidental exposure of half-productized surfaces.

### P3: Ant Design Deprecation Warnings Are Not User-Facing Yet, But They Signal UI Maintenance Risk

Page/route: Cross-page
Evidence:

- Sweep captured warnings including deprecated `Alert.message`, `Modal.destroyOnClose`, `Tabs.destroyInactiveTabPane`, `Drawer.width`, and deprecated `List`.

Why it matters: These are not direct UX failures, but stale component APIs can become future regressions, especially in shared shells and error states.

Recommended fix: Track deprecation cleanup separately from UX remediation. Do not mix it into route redesign work unless a deprecated API is directly blocking a UX fix.

Implementation scope: Maintenance task, separate from UX route improvements.

Expected user impact: Lower regression risk over time, but no immediate workflow gain.

## Cross-Page Design Issues

### Navigation Consistency

- There is no single visible app map. Users encounter header shortcuts, command palette, mode selector, old Next header links, settings nav, sidebars, route redirects, and sidepanel registries.
- Some labels name product concepts (`Research Studio`, `Knowledge QA`), some name implementation or legacy surfaces (`Workspace Playground`, `Prompt Studio`, `Speech`, `Audio`, `Items`).
- Recommendation: define canonical route names and group them by user jobs, not implementation lineage.

### Information Architecture

- The app tries to expose the full breadth of a research/media/AI platform from almost every root page.
- Self-hosted breadth is valid, but route exposure needs tiering: daily routes, setup/health routes, library routes, research/workspace routes, automation/admin routes, and advanced/beta routes.
- Recommendation: keep Explorer/All Features available, but make first-run and default navigation persona-aware without hiding recoverability.

### Visual Hierarchy

- Some pages have strong hierarchy (`/knowledge`, `/mcp-hub`, `/watchlists`).
- Some pages are dense before intent is established (`/media`, `/settings/model`, `/prompts`).
- Some pages have missing or weak heading structure in browser extraction (`/setup`, `/settings/model`, `/tts`, `/stt`, several settings/admin routes).
- Recommendation: each root page should have one visible page title, one primary action, one system-state summary, and advanced controls behind local disclosure.

### Shared Components

- The global shell is doing too much: chat, quick ingest, notes, buddy, model settings, command palette, notifications, shortcuts, backend recovery, and route transitions.
- This makes it easy to create broad functionality, but hard to make pages feel task-specific.
- Recommendation: introduce route metadata for context-specific global actions and make shared shell affordances conditional by route family.

### Empty, Loading, And Error States

- Good examples: `/knowledge` explains its job; `/mcp-hub` explains setup; `/evaluations` explains worker availability; `/agent-tasks` gives next actions.
- Weak examples: `/sources` and `/admin/sources` expose raw endpoint errors; `/settings/model` mixes an OAuth 403 with a full catalog; `/admin/monitoring` uses partially technical endpoint language.
- Recommendation: standardize self-hosted capability states with a next-action requirement.

### Accessibility And Responsive Behavior

- Mobile sweep found horizontal overflow on 6 of 11 representative routes.
- Mobile sweep found many visible interactive targets under 44px across all representative pages. The detection is broad and should be validated per component, but the volume strongly indicates touch-target risk.
- Header and settings layouts are especially fragile on narrow viewports.
- Recommendation: add responsive smoke checks for `documentElement.scrollWidth <= clientWidth + 4` and route-specific touch target audits for the core root pages.

### Cognitive Load

- The interface is often power-user dense before a user has chosen a task.
- The product has many valid advanced features, but they need progressive disclosure based on current workflow.
- Recommendation: preserve expert shortcuts and dense modes, but make default states guide users through one primary job at a time.

## Suggested Improvement Program

This is a report-only recommendation sequence. It is intentionally split into reviewable slices and avoids prescribing one broad visual redesign.

### Slice 1: Canonical IA And Route Contract

Goal: Establish one source of truth for user-facing routes, names, groups, aliases, hosted/self-hosted visibility, sidepanel availability, and command palette targets.

Deliverables:

- Route taxonomy document or route metadata file.
- Generated smoke inventory from the taxonomy.
- Command palette and nav labels aligned to canonical route names.
- Explicit legacy alias policy.

Why first: It reduces drift before visual or route-specific changes begin.

### Slice 2: Setup And Connection State

Goal: Make `/setup` and setup-required `/` states focused, trustworthy, and diagnostic.

Deliverables:

- Setup-only shell.
- Clear backend/server URL semantics.
- Demo, connect, login, health, and docs actions only.
- Clean first-run verification matrix.

Why second: Setup is the first activation gate and currently inherits too much app chrome.

### Slice 3: Shared Capability/Error State System

Goal: Stop raw endpoint failures from becoming the UI.

Deliverables:

- Shared capability-state component and copy model.
- Route mappings for no data, missing API, disabled feature, auth/permission failure, unreachable server, degraded worker, and beta.
- First adopters: `/sources`, `/admin/sources`, `/settings/model`, `/admin/monitoring`.

Why third: Self-hosted deployments will always have partial capability states; the app needs to make those states understandable.

### Slice 4: Responsive Shell And Core Root Pages

Goal: Eliminate page-level horizontal overflow and unusable narrow layouts.

Deliverables:

- Shared shell responsive contract.
- Settings mobile drawer/accordion.
- Media mobile master-detail pattern.
- Chat composer/empty-state collision fix.
- Responsive Playwright assertions.

Why fourth: The extension/sidepanel experience depends on constrained layouts.

### Slice 5: Models And Provider Settings

Goal: Make model/provider setup optimized for configured, usable models first, with full catalog access still available.

Deliverables:

- Provider health/status summary.
- Configured/recent/favorite/search views.
- Collapsible full catalog.
- OAuth/API-key failure states.

Why fifth: Model readiness directly controls whether chat succeeds.

### Slice 6: Route-Family Deep Dives

Goal: Run follow-up route-family audits after structural fixes.

Recommended order:

1. Chat and companion routes.
2. Media/library routes.
3. Sources/connectors/integrations.
4. Settings/admin.
5. Agent/MCP/workflows.
6. Audio and evaluation tools.

Why later: These audits will be clearer after IA, shell, error-state, and responsive foundations are aligned.

## Positive Findings To Preserve

- Connected desktop render health is strong: 124 routes loaded without route-level error boundary failures.
- `/knowledge` has clear job framing, source state, recent searches, suggestions, and a focused input.
- `/mcp-hub` uses a workflow-first setup model with tabs, empty state, and "New Managed Server" actions.
- `/evaluations` explains worker unavailability in user-facing terms and offers a fallback path.
- `/media` has strong power-user density, including filters, bulk mode, pagination, keyboard shortcuts, and quick ingest.
- `/settings` has useful ingredients: a filter, current-section indicator, grouped nav, and health diagnostics route.

## Verification Log

| Evidence item | Command/tool | Result | Notes |
|---|---|---|---|
| Route source inspection | `find`, `sed`, `rg` | Completed | Route sources confirmed across Next pages, shared registries, extension wrappers, and nav components. |
| Backlog task | Backlog MCP | Completed | `TASK-410` created for report-only audit. |
| Connected desktop route sweep | Playwright via temp script | Completed | 124 routes visited, 124 loaded without error boundary, 28 had console/request errors or warnings. |
| Extra page-file route sweep | Playwright via temp script | Completed | 9 user-facing routes omitted from smoke inventory visited, 9 loaded without error boundary, 4 had console/request errors or warnings. |
| Connected mobile core sweep | Playwright via temp script | Completed | 11 representative routes visited, 6 horizontal overflow findings, screenshots captured. |
| Screenshot review | Local image inspection | Completed | Reviewed representative desktop/mobile screenshots for home, setup, chat, media, sources, settings, model settings, knowledge, and MCP Hub. |
| Root page inventory and route-by-route audit | Markdown report section | Completed | 74 discovered root/top-level routes reviewed as individual page records with goal, observed evidence, workflows, first-time issues, power-user issues, and recommended fixes. |
| Product code modifications | Worktree review | Completed | No product code modifications were made by this audit. |
| Before/after state | Audit baseline only | Not applicable | This report captures the current baseline. No remediation code was implemented, so there is no after state. |

## Appendix A: Browser Artifact Index

Desktop:

- `desktop-home.png`
- `desktop-setup.png`
- `desktop-chat.png`
- `desktop-media.png`
- `desktop-knowledge.png`
- `desktop-sources.png`
- `desktop-prompts.png`
- `desktop-characters.png`
- `desktop-settings.png`
- `desktop-settings-tldw.png`
- `desktop-settings-model.png`
- `desktop-settings-health.png`
- `desktop-watchlists.png`
- `desktop-chat-workspace.png`
- `desktop-collections.png`
- `desktop-evaluations.png`
- `desktop-tts.png`
- `desktop-stt.png`
- `desktop-workspace-playground.png`
- `desktop-mcp-hub.png`
- `desktop-extra-integrations.png`
- `desktop-extra-research.png`
- `desktop-extra-scheduled-tasks.png`
- `desktop-extra-vn-play.png`

Mobile:

- `mobile-home.png`
- `mobile-setup.png`
- `mobile-chat.png`
- `mobile-media.png`
- `mobile-knowledge.png`
- `mobile-sources.png`
- `mobile-settings.png`
- `mobile-settings-model.png`
- `mobile-prompts.png`
- `mobile-workspace-playground.png`
- `mobile-mcp-hub.png`

JSON:

- `route-sweep-connected-desktop.json`
- `route-sweep-connected-extra-pages.json`
- `route-sweep-connected-mobile-core.json`

## Appendix B: Route Discovery Notes

- `apps/tldw-frontend/pages` contains more routes than the shared route registry alone.
- `apps/tldw-frontend/e2e/smoke/page-inventory.ts` is useful but not fully authoritative; it includes route entries marked "missing" even though route files now exist, and it omits page-file routes such as `/research`, `/integrations`, `/scheduled-tasks`, `/vn-assets`, and `/vn-play`.
- `apps/packages/ui/src/routes/route-registry.tsx` is the strongest source of shared WebUI/extension route intent.
- `apps/packages/ui/src/routes/option-route-visibility.ts` narrows hosted visible option paths to `/`, `/chat`, `/chat-workspace`, `/media`, `/knowledge`, `/collections`, `/stt`, and `/tts`.
- The broader self-hosted route surface includes admin, agent, MCP, audio, media, knowledge, prompt, character, settings, source, connector, and debug/specialized routes.

## Appendix C: Review Heuristic Summary

| Heuristic | Current assessment | Main evidence |
|---|---|---|
| Visibility of system status | Mixed | Strong in `/knowledge`, `/evaluations`, `/mcp-hub`; weak/raw in `/sources`, parts of admin/settings. |
| Match system/user language | Weak in several states | Raw endpoint errors, `providerKeys.navTitle`, route aliases, implementation-heavy labels. |
| User control and freedom | Mixed | Many controls and shortcuts exist, but route recovery and setup state are not always clear. |
| Consistency and standards | Weak | Multiple nav systems, route aliases, divergent sidepanel registries. |
| Error prevention/recovery | Mixed | Some unsupported states are good; raw 404/403 states remain. |
| Recognition over recall | Weak | Users must remember route aliases and product concept names. |
| Flexibility and efficiency | Strong potential, uneven execution | Dense media/settings/model controls and command palette exist, but some power shortcuts route incorrectly. |
| Aesthetic/minimalist design | Mixed | Utility-first density fits the self-hosted tool, but many pages expose too much before task intent is established. |
| Accessibility/responsive behavior | Significant risk | Mobile overflow and sub-44px touch-target signals across representative pages. |
| Help and documentation | Mixed | Good page-local help in MCP/Knowledge/Watchlists; setup/admin/source recovery needs stronger help paths. |
