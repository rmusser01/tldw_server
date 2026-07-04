# WebUI And Extension Documentation Design

## Goal

Create a user-facing documentation section that explains what the Next.js WebUI and browser extension make possible. The section should help users choose the right page or feature area, understand larger systems, and find deeper docs without reading source code or guessing from route names.

## Reviewed Scope

The WebUI and extension share a large route/component surface through `apps/packages/ui`, with additional Next.js pages in `apps/tldw-frontend/pages` and browser-extension entry points in `apps/extension`. The docs should cover capabilities at the page and feature-set level, not every component or implementation detail.

The source docs should live under `Docs/User_Guides/WebUI/` rather than a new root-level `Docs/WebUI/` folder. This keeps the work inside the existing docs publishing pipeline, because `Helper_Scripts/refresh_docs_published.sh` already syncs `Docs/User_Guides` into `Docs/Published`.

`Docs/Published` is generated output and must not be edited manually. The publishing process is responsible for regenerating published pages from the source docs.

## Deliverables

1. Add a new `Docs/User_Guides/WebUI/` section with a landing page, a page/feature index, and focused feature-set guides.
2. Update `Docs/User_Guides/index.md` so users can discover the new WebUI and extension section from the user guide map.
3. Update `Docs/mkdocs.yml` so the published site has a top-level `WebUI & Extension` navigation section.
4. Link existing stable WebUI and extension documentation where it adds useful depth, without copying WIP product notes or internal-only material.

## Section Structure

The section should use these source files:

- `Docs/User_Guides/WebUI/index.md`
- `Docs/User_Guides/WebUI/Page_Feature_Index.md`
- `Docs/User_Guides/WebUI/Start_Account_Settings.md`
- `Docs/User_Guides/WebUI/Chat_Characters_Assistants.md`
- `Docs/User_Guides/WebUI/Knowledge_Media_Sources.md`
- `Docs/User_Guides/WebUI/Audio_Speech_Audiobooks.md`
- `Docs/User_Guides/WebUI/Study_Writing_Artifacts.md`
- `Docs/User_Guides/WebUI/Automation_Admin_Operations.md`
- `Docs/User_Guides/WebUI/Extension_Sidepanel.md`
- `Docs/User_Guides/WebUI/Experimental_And_Specialized.md`

## User-Facing Organization

The landing page should explain how the surfaces fit together:

- WebUI: full browser application for day-to-day workflows.
- Extension options: browser-extension full-page options UI using shared route components.
- Extension sidepanel: compact browser-adjacent workflows such as page chat, clipper, persona, companion, and flashcard review.
- Shared UI: pages/components reused by WebUI and extension options.
- Admin/operator pages: routes intended for administrators or deployment operators.
- Hosted-only pages: routes that apply to hosted/multi-user commercial or account flows and may not appear in self-hosted setups.
- Experimental or specialized pages: labs, visual novel tools, ACP, prototype workspaces, model playgrounds, and internal QA surfaces.

The page/feature index should group routes by user goal, using columns for page or feature, surface/status, what it lets users do, common uses, and related docs.

## Feature-Set Pages

Each feature-set page should explain the capabilities available in that area and include a compact table of relevant pages/routes. The pages should cover:

- Start, account, setup, settings, and health.
- Chat, characters, persona, companion, agents, dictionaries, world books, chat workflows, and chat workspaces.
- Media, sources, connectors, collections, reading, notes, knowledge search, research workspace, document workspace, and sharing.
- Speech, transcription, TTS, audio aliases, and audiobook production.
- Study, writing, prompts, prompt studio, chatbooks, presentation studio, data tables, kanban, repo2txt, and content review.
- Automation, integrations, scheduled tasks, watchlists, MCP hub, ACP playground, admin pages, moderation, and claims review.
- Extension sidepanel and extension-specific behaviors.
- Experimental, specialized, hosted-only, legacy alias, and internal QA/debug pages.

## Link Policy

Prefer stable source docs under published source folders:

- `Docs/User_Guides/WebUI_Extension`
- `Docs/User_Guides/Server`
- `Docs/API-related`
- `Docs/API`
- `Docs/MCP`
- `Docs/Getting_Started`

Use links into `apps/extension/docs` only when a page is clearly user-facing and stable. Avoid linking to WIP PRDs, internal product plans, test-only notes, private/hosted operational material, or docs outside the public publishing contract unless clearly labeled as source-only reference material.

## Risk Controls

- Do not change WebUI or extension runtime code.
- Do not add generated files or edit `Docs/Published`.
- Do not promise that hosted-only, admin-only, or experimental routes are available in every deployment.
- Keep the page index broad but readable; individual feature pages should summarize capabilities rather than becoming endpoint references.
- Preserve existing `Docs/User_Guides/WebUI_Extension` pages and link to them instead of replacing them in this pass.
- Keep MkDocs navigation concise enough for the sidebar.

## Verification

Verification should include:

- Local Markdown link check for the new/edited source docs.
- MkDocs nav sanity check that every new nav target exists in `Docs/Published` after running the refresh script in a temporary or reverted context, or an equivalent source-target existence check against `Docs/User_Guides`.
- `git diff --quiet dev -- Docs/Published` to confirm generated Published docs are unchanged.
- `git diff --check`.
- Bandit is not applicable unless Python files are changed; record the skip if this remains a docs-only change.

## Non-Goals

- No WebUI or extension behavior changes.
- No component-level developer reference for every shared UI component.
- No generated route-doc tooling in this pass.
- No migration or deletion of existing `Docs/User_Guides/WebUI_Extension` pages.
- No publication of hosted/private commercial support docs.
