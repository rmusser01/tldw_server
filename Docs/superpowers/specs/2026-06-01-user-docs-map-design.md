# User-Facing Documentation Map Design

Backlog: TASK-585
Date: 2026-06-01
Status: Approved for implementation planning

## Problem

tldw_server has substantial documentation, but new users have no clear map. The current docs are spread across the repository README, `Docs/Getting_Started`, `Docs/User_Guides`, `Docs/API-related`, `Docs/Code_Documentation`, the WebUI in-app documentation browser, and the extension VitePress docs. Much of the content is useful, but the entry points are inconsistent and the public MkDocs navigation exposes only a small part of the available user-facing material.

The main failure mode is discoverability, not the absence of documentation. A newcomer needs one canonical place that answers:

- Which setup path should I choose?
- What can I do with the server API, WebUI, and browser extension?
- Which UI surface or API endpoint family should I use for a task?
- Where is the best deeper guide for each workflow?
- Where should I go when setup, auth, providers, media, or audio fails?

## Goals

- Make `Docs/User_Guides/index.md` the canonical public user documentation hub.
- Give new users a clear path from setup to first successful value:
  1. choose a setup profile,
  2. open the WebUI,
  3. complete first chat,
  4. add a first source,
  5. learn where the rest of the product surfaces live.
- Add an overarching feature map organized by user intent instead of internal module names.
- Link the hub from the README, MkDocs navigation, and extension docs so users do not have to discover it by browsing the repository tree.
- Keep the first implementation slice low-risk by linking to existing deep pages instead of moving large numbers of files.

## Non-Goals

- Do not reorganize the full `Docs/` tree in the first slice.
- Do not rename or move already published deep docs in this slice.
- Do not rewrite every feature guide.
- Do not change API behavior, WebUI behavior, or extension runtime behavior.
- Do not make the WebUI `/documentation` viewer the canonical documentation information architecture in this slice.

## Source Of Truth

The canonical user-facing docs map will live at:

- `Docs/User_Guides/index.md`

The generated public copy under `Docs/Published/User_Guides/index.md` remains generated output. It should be refreshed through `Helper_Scripts/refresh_docs_published.sh`, not edited by hand.

Supporting entry points should point to the canonical hub:

- `README.md`
- `Docs/mkdocs.yml`
- `apps/extension/docs/index.md`

The WebUI `/documentation` route remains a convenience browser for documentation files. It is not the source of truth for the first documentation IA cleanup.

## Information Architecture

The user guide hub should be written as a map, not a flat list. It should start with a short statement explaining that this is the place to orient around the server API, WebUI, browser extension, and admin/operator docs.

Recommended structure:

1. **Start Here**
   - Link to self-hosting profiles.
   - Explain the recommended Docker single-user path for most first-time users.
   - Explain local single-user and Docker multi-user paths.
   - State the first-value path: WebUI reachable, first chat completed, first source added.

2. **Choose Your Surface**
   - WebUI: primary self-hosted product surface.
   - Browser extension: capture and sidepanel workflows that talk to the server.
   - Server API: OpenAI-compatible and tldw-specific APIs.
   - Admin/operator docs: multi-user, hardening, usage, monitoring, backups.

3. **What Can I Do?**
   Group capabilities by intent:
   - Chat with models and characters.
   - Add sources and media.
   - Search and ask questions over knowledge.
   - Transcribe audio and generate speech.
   - Study, evaluate, and review outputs.
   - Create, write, and manage knowledge artifacts.
   - Automate workflows and integrate external tools.
   - Administer a shared server.

   Each group should include:
   - plain-English summary,
   - primary surface, such as WebUI, extension, API, or admin,
   - best next guide,
   - key API docs when relevant.

4. **Troubleshooting**
   - Authentication and connection issues.
   - Provider and model setup.
   - Media, ffmpeg, and ingestion issues.
   - Audio/STT/TTS issues.
   - Deployment, Postgres, and operations issues.

5. **For Builders**
   - API docs.
   - OpenAPI docs at `/docs`.
   - developer/code docs.
   - SDK docs where relevant.

## Feature Map Page

If `Docs/User_Guides/index.md` becomes too dense, create a second page:

- `Docs/User_Guides/Feature_Map.md`

The hub should remain concise and link to the feature map for the full matrix. The feature map should be user-facing and task-oriented. It should not become a dump of internal routers or modules.

The first implementation plan should decide whether the separate page is needed after drafting the hub. If the hub stays readable, the feature map can be folded into the hub for the first slice.

## MkDocs Navigation

`Docs/mkdocs.yml` should expose the canonical user docs map clearly. The current nav has a `User Guides` section with only `Authentication Setup`, even though many user guides are published. The first slice should add visible entries for:

- User Guides index.
- Feature map if created.
- WebUI and extension user guide.
- Server setup and authentication.
- RAG/search, media ingestion, audio, evaluations, and admin/operator guides at a practical level.

The nav should be selective. It should not list every existing guide.

## README Linkage

`README.md` should keep its quickstart role, but it needs a prominent pointer to the canonical documentation map near the existing Start Here or Documentation sections.

The README should not duplicate the full feature map. It should tell users where to go when they need the map.

## Extension Linkage

`apps/extension/docs/index.md` should make it clear that tldw Assistant is one client surface for tldw_server and link users back to the canonical server/WebUI/extension documentation map for full workflow discovery.

The extension docs can still describe extension-specific setup and browser behavior. They should not maintain a competing overall product map.

## Verification

The implementation slice should verify:

- `bash Helper_Scripts/refresh_docs_published.sh` completes.
- `mkdocs build` completes from the docs configuration used by the repository.
- Changed Markdown links are checked where practical.
- No manual edits are made under `Docs/Published/` except generated refresh output.
- Bandit is recorded as not applicable for docs-only edits unless code is touched.

## Open Questions Resolved During Brainstorming

- Canonical docs map location: `Docs/User_Guides/index.md`.
- First priority: new-user onboarding and discoverability.
- Primary problem: documentation sprawl with no overarching map.
- First slice approach: low-risk IA cleanup and linking, not a full docs tree reorganization.

## Spec Self-Review

- Placeholder scan: no placeholder sections remain.
- Internal consistency: the source of truth, linkage, and verification sections all point to the same first-slice scope.
- Scope check: this is small enough for one implementation plan because it touches a docs hub, optional feature map, nav, README pointer, and extension pointer.
- Ambiguity check: WebUI `/documentation` is explicitly excluded as the canonical source for this slice.
