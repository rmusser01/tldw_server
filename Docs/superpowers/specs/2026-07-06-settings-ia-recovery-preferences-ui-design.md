# Settings IA, Recovery, Preferences, And UI Split Design

Date: 2026-07-06
Status: Approved for implementation planning
Backlog: TASK-12160, TASK-12166

## Summary

Redesign the WebUI and browser-extension settings experience around a clearer
information architecture without adding a new shared route registry. The
first implementation slice makes `/settings` a Setup & Recovery home, adds
`/settings/preferences` for personal behavior defaults, sharpens
`/settings/ui` into interface customization, and fixes the main navigation,
mobile, layout, and consistency issues found in the visual/code review.

The design uses existing settings structures:

- `settings-nav-config.ts` remains the settings menu source.
- `settings-active-route.ts` owns canonical path and active-route matching.
- `SettingsOptionLayout.tsx` owns layout, not route logic.
- Existing connection, health, provider-key, model, and RAG pages remain the
  authoritative detailed surfaces.

This is a settings redesign, not an app-wide routing redesign.

## Current Problems

The current settings surface has several linked UX and code problems.

1. `/settings` currently means General Settings, but it also tries to explain
   connection, onboarding, theme, OCR, persona, search, and system controls.
   That makes the page a catch-all instead of a usable entry point.
2. Nested settings routes can mark both `/settings` and the child route active,
   causing duplicate active navigation and duplicate `aria-current="page"`.
3. `/settings/health` and `/settings/processed` bypass the shared settings
   layout, so they visually detach from settings navigation.
4. On mobile, the full grouped settings nav wraps into the first screen and
   pushes content below the fold.
5. Current groups are too granular for first-level navigation, but collapsing
   them into three hubs without sections would create new junk drawers.
6. General Settings contains controls that need separate owners: preferences,
   visual customization, setup recovery, processing defaults, RAG/search, and
   destructive admin actions.
7. The sidepanel settings surface behaves like a broad options page. It should
   be a compact operational shortcut panel.

## Goals

- Make `/settings` useful for returning users recovering a broken setup.
- Add `/settings/preferences` as the home for personal behavior defaults.
- Make `/settings/ui` the clear home for visual and navigation customization.
- Reduce first-level settings navigation to three hubs with lightweight
  sections, reusing the existing settings nav config.
- Fix duplicate active navigation, mobile nav overload, and layout drift.
- Keep full diagnostics, provider validation, model refresh, and detailed
  health probes on their specialist pages.
- Preserve power-user density while making the first visible layer answer:
  what is broken, what does it affect, and where do I fix it?

## Non-Goals

- Do not introduce a new route ownership abstraction or shared registry.
- Do not introduce a diagnostic contributor framework.
- Do not redesign the whole app shell.
- Do not move chat rendering controls out of `/settings/chat` in this slice.
- Do not validate every provider/model on `/settings` page load.
- Do not expose the full settings IA inside the sidepanel.
- Do not combine the broader Phase C settings-system reset into this first
  implementation slice.

## IA Model

Settings should use three hubs with sections inside each hub.

### Setup & Recovery

Purpose: setup, broken-state recovery, and readiness triage.

Sections:

- Connection: `tldw server`, provider keys.
- Models: model settings, default chat model, embedding readiness.
- Health: health summary and diagnostics.

Primary routes:

- `/settings`
- `/settings/tldw`
- `/settings/provider-keys`
- `/settings/model`
- `/settings/health`

`/settings` becomes the Setup & Recovery landing page.

Setup & Recovery may deep-link to `/settings/rag` when embedding or retrieval
defaults need attention, but `/settings/rag` is canonically owned by
Preferences & Workflow so it appears once in settings navigation.

### Preferences & Workflow

Purpose: repeated work defaults and task-specific configuration.

Sections:

- Personal preferences.
- Interface customization.
- Chat behavior.
- Knowledge and RAG defaults.
- Prompts, characters, chatbooks, dictionaries, quick ingest.
- Speech and image generation where they support daily work.

Primary routes include `/settings/preferences`, `/settings/ui`,
`/settings/splash`, `/settings/chat`, `/settings/rag`, `/settings/knowledge`,
`/settings/prompt`, `/settings/quick-ingest`, `/settings/speech`,
`/settings/image-generation`, and related workspace configuration pages.

`/settings/rag` is the canonical home for RAG and retrieval defaults. Other
hubs may link to it as a recovery action, but they must not duplicate it in
their navigation lists.

### Admin & Diagnostics

Purpose: high-risk, operational, data, safety, and advanced controls.

Sections:

- Full diagnostics.
- Data and processed content.
- Safety, guardian, moderation, evaluations.
- MCP and admin-heavy controls.
- About and system-level information.

Primary routes include `/settings/data`, `/settings/processed`,
`/settings/guardian`, `/settings/family-guardrails`, moderation routes,
`/settings/evaluations`, `/settings/mcp-hub`, and `/settings/about`.

## Page Ownership

`/settings/preferences` and `/settings/ui` are canonically owned by the
Preferences & Workflow hub. They appear in separate sections so personal
behavior defaults and interface customization do not collapse back into one
General Settings page.

### `/settings`: Setup & Recovery

`/settings` answers: what is broken and where do I fix it?

It shows status rows for:

- Server reachable.
- Authentication valid, API key missing, or authentication failed.
- Provider keys available, server-managed, or BYOK unavailable.
- Default chat model selected.
- Embedding model selected.
- Health checks passing or needing attention.
- Extension/site access state when available.

Each row has:

- status
- short plain-language explanation
- one primary action
- optional diagnostics disclosure or link

The page links to detailed pages:

- `/settings/tldw` for server URL and auth edits
- `/settings/provider-keys` for BYOK/provider-key management
- `/settings/model` for model catalog and defaults
- `/settings/rag` for embedding/retrieval defaults
- `/settings/health` for full diagnostics

It must not run full diagnostics, provider validation, or model catalog refresh
on initial render.

### `/settings/preferences`: General Preferences

Preferences are personal behavior defaults. They are not setup, admin,
destructive, or visual-design tooling.

Move here from current General Settings:

- language
- notifications after indexing
- onboarding auto-finish
- reset tutorial progress
- persona / feature visibility profile, with copy framed as feature visibility
  rather than personality
- default UI mode if exposed as a personal behavior default
- optional background readiness preference such as "check Ollama status" if the
  label makes the background behavior clear

Do not put these here:

- server connection status
- provider/model readiness
- ThemePicker or theme editor
- OCR processing defaults unless explicitly temporary
- SearchModeSettings unless confirmed to be global app search behavior
- SystemSettings/destructive reset
- browser extension promotion

`Restart onboarding` belongs on `/settings` as a recovery action, not in
Preferences.

### `/settings/ui`: Interface Customization

UI customization is about visible app presentation and navigation shape.

Keep or move here:

- persona buddy shell
- header shortcuts
- sidebar shortcuts
- theme mode/picker/editor/token tooling
- density/layout appearance controls
- advanced visual controls behind disclosure

Splash settings can remain at `/settings/splash` for implementation safety, but
in the IA it belongs under the Interface Customization section. It should not be
mixed with preferences.

Do not move chat rendering controls here in this slice. They remain under
`/settings/chat` until the Preferences & Workflow redesign.

### Deferred Ownership Decisions

Some current General Settings controls should not be moved casually:

- OCR defaults should move to Quick Ingest or a future Processing Defaults page.
- SearchModeSettings should move to RAG/Knowledge if it controls retrieval
  behavior; otherwise it can remain in Preferences only if it is truly global
  app search behavior.
- SystemSettings/destructive reset should move to Data Management or
  Admin & Diagnostics with destructive styling and confirmation preserved.
- Browser extension promotion should move to About/help or be removed from the
  settings workflow.

## Navigation And Layout

Use existing settings navigation code. Extend it minimally.

`settings-nav-config.ts` should remain the only settings menu source and may add:

- `hub`
- `section`
- `aliases`

It should not add:

- route ownership metadata
- diagnostic contributor metadata
- backend capability descriptions beyond the existing capability gating
- app-wide route taxonomy

`settings-active-route.ts` should own canonicalization and active matching:

- `/settings` exact-matches only `/settings`
- aliases canonicalize before matching, for example `/settings/image-gen` to
  `/settings/image-generation`
- one and only one nav item is active for any settings path

`SettingsOptionLayout.tsx` should own only layout behavior:

- desktop hub rail
- selected hub page list grouped by section
- global settings search/filter
- mobile hub selector
- mobile page selector/search
- close action
- content frame

The redundant current-section banner should be removed. The page heading and
active navigation are sufficient.

### Desktop

Desktop layout:

- left rail shows three hubs
- secondary list shows pages for the selected hub, grouped into sections
- main content uses a compact page header and content sections
- settings search can find pages across hubs

### Mobile

Mobile layout:

- no full wrapped sidebar before content
- top control chooses hub
- compact page selector or search chooses the page
- content begins above the fold
- touch targets remain keyboard and screen-reader accessible

## Setup & Recovery Data Flow

The recovery home should reuse existing state and cheap hooks.

Allowed sources:

- existing connection state and connection UX state
- current server URL/auth mode from existing config paths where already loaded
- existing health summary concepts for server and knowledge status
- cached model/provider/default selection state if already available without a
  heavy refresh

Disallowed on initial render:

- full health diagnostics run
- provider key validation sweep
- model catalog refresh
- embedding catalog refresh
- long-running server probes

Rows should degrade honestly. Unknown state is allowed and should link to the
detailed page.

Provider key wording must handle BYOK disabled. BYOK disabled is not a broken
state; it means server-managed keys only.

## Sidepanel Settings

The sidepanel should not inherit the full settings IA.

Replace the broad sidepanel settings page with a compact operational panel:

- connection status
- default interaction mode
- RAG/page-context toggle
- open full settings
- health/diagnostics link

Sidepanel links should deep-link to full settings pages for detailed edits.

## Route And Wrapper Changes

Required route work:

- add `/settings/preferences` to shared option routes
- add a Next wrapper page at `apps/tldw-frontend/pages/settings/preferences.tsx`
- make `/settings` render `SetupRecoverySettings`
- keep `/settings/preferences` rendering `PreferencesSettings`
- ensure extension route mirrors or parity-test fixtures are updated when tests
  require it
- wrap `/settings/health` and `/settings/processed` in `SettingsRoute`
- verify wrapped pages do not render conflicting page-level `h1`s

Do not redirect `/settings` to `/settings/preferences`.

Existing links to `/settings` remain valid and now land on Setup & Recovery.
Search, command palette, docs, and tests that specifically intend personal
preferences must update to `/settings/preferences`.

## Implementation Stages

### Stage 1: Route And Page Split

Goal: establish stable page ownership before changing layout.

Tasks:

- add `/settings/preferences`
- create `SetupRecoverySettings`
- create `PreferencesSettings`
- split current General Settings into owned pieces
- move or defer ambiguous controls according to this spec
- update search index entries for preferences and theme/UI controls

### Stage 2: Nav, Aliases, And Layout Fixes

Goal: fix wayfinding and responsive structure.

Tasks:

- reshape settings nav config into three hubs with sections
- add alias canonicalization in `settings-active-route.ts`
- exact-match `/settings`
- render desktop hub + sectioned page nav
- render mobile hub/page selectors
- remove the current-section banner

### Stage 3: Setup & Recovery Home

Goal: make `/settings` a lightweight triage page.

Tasks:

- build status rows from existing lightweight state
- add plain-language copy and one primary action per row
- use diagnostics disclosure or links, not default raw payloads
- preserve broken-state access to core recovery pages

### Stage 4: Layout Consistency

Goal: ensure all settings routes share the same frame.

Tasks:

- wrap health and processed routes
- verify heading semantics
- verify one `h1` and one active nav item
- update affected tests

### Stage 5: Sidepanel Cleanup

Goal: keep sidepanel settings short and operational.

Tasks:

- replace broad sidepanel settings list with shortcut panel
- add full-settings deep links
- verify static extension and packaged extension behavior
- verify no translation key leakage

## Testing And QA

Minimum acceptance tests:

- exactly one `aria-current="page"` per settings route
- `/settings` exact matching does not activate child routes
- `/settings/preferences` exists in shared routes and Next wrappers
- `/settings/image-gen` canonicalizes to `/settings/image-generation`
- nav paths resolve and no navigable settings page appears twice
- core recovery routes stay visible when capabilities are missing or server is
  offline
- `/settings/health` and `/settings/processed` render inside settings layout
- mobile `/settings` shows content before large nav lists
- search results for language, theme, preferences, and UI customization route
  correctly
- preferences page contains only approved behavior defaults
- UI page contains visual/navigation customization controls
- sidepanel settings exposes shortcuts and full-settings links, not the full hub
  nav

Accessibility checks:

- one `h1` per route
- one active page link
- keyboard reachable hub selector and page selector
- visible focus states
- diagnostics disclosures have accessible names and expanded/collapsed state
- status rows do not rely on color alone

Visual QA:

- desktop `/settings`
- desktop `/settings/preferences`
- desktop `/settings/ui`
- desktop `/settings/health`
- mobile `/settings`
- extension sidepanel settings

## Risks And Mitigations

Risk: `/settings` behavior change surprises users.

Mitigation: keep `/settings` useful for recovery and update command/search
entries that mean preferences to point to `/settings/preferences`.

Risk: Preferences becomes another junk drawer.

Mitigation: enforce the ownership rules above and test for allowed control
families.

Risk: Setup & Recovery becomes expensive on load.

Mitigation: only read existing or cheap state; link to detailed pages for full
checks.

Risk: capability gating hides recovery pages.

Mitigation: core recovery pages remain visible even when capabilities are
unknown or server is offline. Capability gating remains for optional modules.

Risk: sidepanel and WebUI drift.

Mitigation: sidepanel is intentionally a shortcut panel with full-settings deep
links, not a duplicate settings app.

Risk: extension mirror routes cause test failures.

Mitigation: account for shared route registry, Next wrappers, and parity-test
mirrors during implementation planning.

## Open Questions For Implementation Planning

- Confirm whether `SearchModeSettings` is global app search behavior or
  RAG/Knowledge retrieval behavior.
- Decide whether OCR defaults move in this slice or wait for Quick Ingest /
  Processing Defaults.
- Decide whether splash settings remains a separate page or becomes a section of
  `/settings/ui` in a later slice.
- Confirm which existing model settings hook can expose default chat and
  embedding selection without refreshing catalogs.
