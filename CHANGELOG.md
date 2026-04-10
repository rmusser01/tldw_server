# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to Some kind of Versioning


## [Unreleased]

### Added

- **Writing Suite Phases 2-4** — Expanded manuscript tooling with characters, world info, plot and research management, AI analysis and visualization, agent chat, and live feedback via mood detection and Echo Chamber. (PRs #999, #1001, #1002)
- **Persona Buddy Track-a-Plan Backend** — Completed the backend flow for persona buddy planning/tracking so the companion workflow can persist and coordinate plan progress. (PR #949)
- **Persona-Routed Onboarding** — Added persona-guided onboarding with filtered navigation, guided task flows, and extension promotion hooks. (PR #1000)
- **Mission Control Home and Storage Quotas** — Added progressive-unlock Mission Control home flows plus storage quota warnings. (PR #1004)
- **Browser Web Clipper** — Added browser-based web clipping support for capturing pages into the research workflow. (PR #1005)
- **Study Suggestions Engine** — Added study suggestions for quiz and flashcard workflows. (PR #1014)
- **Sandbox and ACP Improvements** — Expanded the sandbox and ACP stack with a broader competitive-improvements pass across execution ergonomics and runtime polish. (PR #1023)
- **OCR Backend Expansion** — Added `llama.cpp` and `chatllm` OCR backends. (PR #1038)

### Changed

- **Theme System Expansion** — Extended the shared theme system with adjustable typography, shape, layout, and component variants. (PR #1021)
- **GHCR Release Documentation** — Updated CI gate and release-checklist documentation for the GHCR publishing flow. (PR #1003)

### Fixed

- **FTUE Follow-Through** — Addressed additional first-time-user experience issues across `/chat`, `/monitoring`, and `/watchlists`. (PR #1009)
- **UX Audit Remediation** — Fixed Nielsen-style heuristic audit issues across chat, knowledge, workspace, and character flows. (PR #1039)

### Removed


## [0.1.30] - 2026-04-04

### Added

- **Evaluations Recipe Framework** — A recipe-driven evaluation system with manifests, run persistence, job worker integration, and reporting APIs. Includes guided retrieval tuning and RAG answer quality recipes, recipe-first UI flows with a guided launcher, and legacy evaluations tab fixes. Synthetic eval draft generation and shared review workflow added alongside browser and unit test coverage. (PR #942)
- **MCP Virtual CLI** — Phase-1 virtual CLI command runtime with workspace-bounded filesystem tools, governed `run` MCP tool, parser/registry/execution/presentation layers, approval-gated execution for governed chains, policy-aware discovery, nested idempotency propagation, spill-safe presentation, and integration test coverage across authz and path scoping. (PRs #939, #941)
- **Reference Manager Import/Sync** — Provider-neutral reference-manager connector contracts, storage, and Zotero integration. Zotero collection sources are exposed through the connectors API with import-mode scheduler and worker sync including dedupe, metadata-only tracking. (PR #940)
- **NotebookLM Presentation Style Catalog** — Built-in visual style catalog with resolver-backed metadata, prompt profiles, and reusable style packs. Reveal export rendering extended with namespaced style CSS and richer visual-block HTML. Presentation Studio gained a visual style picker, client metadata support, and built-in theme synchronization. (PR #935)
- **FTUE Audit** — Comprehensive first-time user experience audit addressing 41 issues across documentation, tooling, configuration, and frontend UX. Includes unified README entry points, `make help`/`make show-api-key` targets, restructured `.env.example`, improved onboarding error messages, Docker entrypoint auth-init failure handling, DATABASE_URL preflight checks, demo mode exit banners, and multi-user JWT setup guidance. (PRs #938, #944)
- **Evaluations Recipe Framework** — A recipe-driven evaluation system with manifests, run persistence, job worker integration, and reporting APIs. Includes guided retrieval tuning and RAG answer quality recipes, recipe-first UI flows with a guided launcher, and legacy evaluations tab fixes. Synthetic eval draft generation and shared review workflow added alongside browser and unit test coverage. (PR #942)
- **MCP Virtual CLI** — Phase-1 virtual CLI command runtime with workspace-bounded filesystem tools, governed `run` MCP tool, parser/registry/execution/presentation layers, approval-gated execution for governed chains, policy-aware discovery, nested idempotency propagation, spill-safe presentation, and integration test coverage across authz and path scoping. (PRs #939, #941)
- **Reference Manager Import/Sync** — Provider-neutral reference-manager connector contracts, storage, and Zotero integration. Zotero collection sources are exposed through the connectors API with import-mode scheduler and worker sync including dedupe, metadata-only tracking. (PR #940)
- **NotebookLM Presentation Style Catalog** — Built-in visual style catalog with resolver-backed metadata, prompt profiles, and reusable style packs. Reveal export rendering extended with namespaced style CSS and richer visual-block HTML. Presentation Studio gained a visual style picker, client metadata support, and built-in theme synchronization. (PR #935)
- **FTUE Audit** — Comprehensive first-time user experience audit addressing 41 issues across documentation, tooling, configuration, and frontend UX. Includes unified README entry points, `make help`/`make show-api-key` targets, restructured `.env.example`, improved onboarding error messages, Docker entrypoint auth-init failure handling, DATABASE_URL preflight checks, demo mode exit banners, and multi-user JWT setup guidance. (PRs #938, #944)
- **Speech / Audio FTUE Follow-through** — Added a TTS/STT first-time-user audit pass with broader onboarding, navigation, and setup-guidance improvements across the speech entry points. (PR #983)
- **Deferred FTUE and Workspace Handoffs** — Added follow-up tutorials, JSON export affordances, Quick Ingest result CTAs into Knowledge and Workspace, recently-ingested document auto-open behavior, and "Open in Workspace" handoffs from knowledge sources. (PRs #984, #1007)
- **MCP Hub / Media / Quiz Phase 2 UX** — Added Joyride-guided onboarding for MCP Hub and quizzes, richer empty states and retry guidance for MCP Hub and Media Library, connected quiz orientation improvements, and extension sidepanel deep links into Media Library and Quizzes. (PRs #993, #997, #998)
- **Writing Suite Phase 1** — Added manuscript database tables, CRUD/FTS helpers, REST API endpoints and schemas, TipTap-backed writing editor surfaces, manuscript tree and focus mode, drag-and-drop reorder, and review-driven hardening across the writing stack. (PR #995)
- **Container Snapshot Publishing and Lifecycle Docs** — Added GHCR main snapshot publishing for the app and UI images, a single roll-up `container-build-check` summary job, and first-class container image lifecycle documentation. (PRs #996, #1006)
- **FTUE / FTUX Expansion** — First-time user improvements broadened across onboarding, LLM connection, chat and watchlist journeys, realtime and BYOK setup, extension notification flows, media and review surfaces, MCP Hub, moderation, quiz, and a larger flashcards-first experience with new tutorials, empty states, and UX polish. (PRs #948, #950, #951, #953, #954, #963, #965, #969, #973, #977, #981, #982)
- **Study Packs Phase 1** — Added a new study-pack pipeline spanning backend generation, provenance, source resolution, jobs-worker execution, API schemas/endpoints, and shared UI creation, remediation, and handoff flows with broad automated coverage. (PR #978)
- **Flashcards Workflow Expansion** — Added workspace deck preview support, global tag suggestions, deck references in the create drawer, richer create/review/import flows, and related study-assistant follow-through across backend and shared UI surfaces. (PRs #974, #975, #979)
- **Watchlist Alert Rules** — Added a watchlist alert-rules engine, database layer, and API support for user-defined run and feed notifications. (PR #970)
- **Coordinated App Shutdown** — Added coordinated application shutdown flow support to improve lifecycle cleanup and service teardown behavior. (PR #945)
- **MCP Virtual CLI Phase 2** — Extended the governed MCP virtual CLI beyond the initial runtime with deeper policy/runtime integration, safer command execution flows, and additional UX and test hardening for workspace-bounded command handling. (PRs #946, #962)

### Changed

- Chat mood badge hidden by default across shared WebUI and extension chat surfaces, with one-time legacy preference migration and preserved user opt-in. (PR #943)
- Workspace scope now forwarded when saving chat knowledge; voice-unavailable reason propagated through PlaygroundForm. (PR #943)
- Chat mood badge hidden by default across shared WebUI and extension chat surfaces, with one-time legacy preference migration and preserved user opt-in. (PR #943)
- Workspace scope now forwarded when saving chat knowledge; voice-unavailable reason propagated through PlaygroundForm. (PR #943)
- Settings internals were further decomposed by extracting the growing Settings page into smaller components, reducing page-level coupling while keeping the version line at `0.1.30`. (PR #994)
- Settings and notifications management moved further toward task-oriented configuration with tabbed Settings navigation, stronger extension notification subscription wiring, cleaner notification count/preference handling, and improved route parity across shared UI entry points. (PRs #968, #969)
- AuthNZ admin CLI maintenance was tightened with clearer module-level documentation expectations and dedicated regression coverage for create/reset-admin command docs. (PR #976)
- Admin, operator, and production-readiness surfaces continued to harden through follow-up infrastructure work in the admin UI stack. (PR #934)

### Fixed

- Evaluations recipe framework hardened with 50 review-feedback fixes: Pydantic-typed recipe endpoints, worker readiness gating before enqueue, custom `RecipeEnqueueError` exception, sanitized error metadata, `managed_media_database` context manager for worker sessions, `owner_user_id` enforcement on `get_run()`, pre-validated `build_reuse_hash`, DB-level reuse-hash lookup, `ConfidenceSummary.model_validate` fallback, field-specific weight validation errors, CLI JSON shape validation, non-retryable `fts` mode error, `RecipeNotFoundError` custom exception, dict-backed document normalization in RAG pipeline, parameterized Loguru logging, and comprehensive recipe logic fixes across retrieval tuning, answer quality, embeddings, and summarization recipes. (PR #942 review follow-ups)
- FTUE v2 residual issues: Docker entrypoint now exits on auth init failure, DATABASE_URL TCP connectivity preflight added, demo mode crash guard for missing `DemoModeProvider`, and clearer configuration guidance across Docker and local installs. (PR #944)
- MCP virtual CLI review feedback addressed: preflight validation, filesystem module registration, command runtime path scope tests, and idempotency propagation fixes. (PR #941 review follow-ups)
- Addressed post-review regressions across notifications, snoozed-state handling, persona buddy follow-ups, flashcard onboarding, and FTUE onboarding/realtime/watchlist flows so the new user journey and notification surfaces behave consistently across the web app and extension. (PRs #965, #966, #972, #973, #982)

## [0.1.29] 2026-03-29
### Added

- Shared workspace cloning and "Shared With Me" flows, backed by new sharing APIs, privilege-aware workspace sharing rules, and clone-service support. (PR #903)
- ACP workspaces gained workspace discovery and health services for workspace-entity orchestration surfaces. (PR #912)
- Deep Research now includes jobs-backed run persistence, provider-backed collection/synthesis, replayable SSE progress, checkpoint review, chat handoff and follow-up flows, and workflow launch/wait/bundle steps. (PR #831)
- MCP Hub governance pack management expanded with source distribution, trust policy and signer provenance/diagnostics, hosted staging/production overlays, and route parity across shared UI, the web app, and extension settings. (PRs #917, #922, #933)
- Companion/Home and integrations management grew with a customizable Companion Home dashboard, notifications surfaces, shared integrations and scheduled-task management, and workspace installation registry syncing. (PRs #920, #925, #926)
- The audio stack gained KittenTTS provider support, curated `kitten_tts` bundle profiles, shared admin audio installer routes/panel, and `pocket_tts_cpp` adapter/runtime/installer support. Audio setup docs and scripts were also migrated from `huggingface-cli` to `hf`. (PRs #918, #919, #921, #923, #931)
- Telegram bot AuthNZ/governance support now includes permission foundations, admin configuration APIs, webhook intake/deduplication, linked-actor mapping, scoped execution identities, and exact-scope approval callbacks. (PR #924)
- Reference-guided image generation for `POST /api/v1/files/create` image artifacts via `payload.reference_file_id`, including managed reference-image discovery through `GET /api/v1/files/reference-images`, backend `image_reference_input` capability metadata, documented Model Studio support for eligible reference-image model families, and Playground modal support for selecting one managed reference image as the request source.

### Changed

- Media DB v2 advanced through the phase-1 and stage-1 caller-first migrations, splitting the monolith into repository/runtime/schema modules, rebinding callers to the new API surface, tightening request-scope isolation, and expanding read-contract coverage. (PRs #911, #930, #925)
- Quick Ingest audio configuration now uses structured language selection and backend-driven transcription model dropdowns, and the web UI also gained audio input source switching. (direct dev merges)
- Knowledge QA and related workspace/chat flows were hardened with a wider desktop workspace, media retrieval fallback coverage, workspace study material ownership, flashcard visibility enforcement, and stricter voice-conversation capability contracts. (direct dev merges and PR #933 follow-ups)
- Admin and operator surfaces were overhauled with production-readiness and incident-management follow-through across dashboard, AI overview, budgets, compliance, monitoring, invitations, registration codes, webhooks, usage, and voice-command tooling. (PR #932)

### Fixed

- Embeddings jobs workers now normalize legacy chunk types instead of failing queued work. (PR #914)
- Sharing, research, governance-pack trust, Telegram approvals/webhooks, companion routing, MCP Hub navigation, and admin audio installer follow-ups received post-review stability fixes. (PRs #903, #831, #922, #924, #926, #933)
- `pocket_tts_cpp`, KittenTTS bundle verification, quick ingest language-state handling, quickstart same-origin hydration/browser config, and Media DB v2 delete/review regressions were addressed in post-merge hardening. (PRs #919, #921, #923, #930, #931 and direct follow-ups)
- CI and repo hygiene were tightened with frontend/browser gate fixes, isolated workspace installation repo tests, OSS/private boundary enforcement, and GitHub Actions dependency bumps for `docker/login-action`, `docker/setup-buildx-action`, and `dependency-review-action`. (PRs #843, #900, #901, #925 and related follow-ups)
- Resolved FTUX follow-up issues across MCP Hub routing/confirmation flows, media-library dead ends and search help, quiz mobile labels and empty-reset behavior, companion deep links, `window.open` safety, notification duration, and tutorial dispatch/timing behavior. (PRs #993, #997, #998)
- Hardened document-ingest and workspace handoff flows by fixing `keep_original_file` behavior for HTML and document types, centralizing doc-type constants, preventing process-only temp-file leaks, and improving missing-original-file recovery messaging. (PR #1007)
- Hardened manuscripts and writing review follow-ups with optimistic locking for reorder operations, corrected word-count propagation metadata, safer state selection, and broader security/sync/null-guard fixes across the new writing stack. (PR #995)
- Completed the GHCR/container follow-through with workflow safety fixes, provenance pinning, admin-monitoring null-safety cleanup, and clearer container build roll-up behavior. (PRs #996, #1006)
- Addressed post-review regressions across notifications, snoozed-state handling, persona buddy follow-ups, flashcard onboarding, and FTUE onboarding/realtime/watchlist flows so the new user journey and notification surfaces behave consistently across the web app and extension. (PRs #965, #966, #972, #973, #982)

## Backfill: merged on `main` before `dev` sync

> Changelog sync note: these entries document scope already merged on `main` via PR #910. They are retained here for history until `dev` is synced, but they are not unreleased work.

### Changed

- Backfilled the `dev` changelog for the `main`-merged PR #910 knowledge/workspace beta readiness audit so the merged review-remediation scope stays visible before those code paths are forward-ported into `dev`.

### Added

- Documented the merged setup-audio rollout on `main`, including bundle-and-profile recommendations in `/setup`, offline audio-pack manifest export/import, persisted audio readiness state, and richer setup verification/remediation reporting.
- Documented merged Knowledge QA and Workspace Playground hardening on `main`, including note-id normalization, shared-link hydration fixes, stronger study/remediation and live-workspace coverage, and tighter workspace probe/runtime assertions.

### Fixed

- Documented merged reliability fixes on `main` across setup install-status typing and health collection, audio-pack path and compatibility handling, RAG sensitivity/vector-store metadata handling, quiz test-mode and provenance behavior, and assorted slides/workspace retry and Chroma state-tracking edge cases.


## [0.1.28] 2026-03-15

### Added

- Profile-aware curated audio bundles in `/setup`, with `light`, `balanced`, and `performance` resource tiers for the local hardware families and conservative recommendation scoring across bundle/profile pairs.
- A v1 setup audio pack service for manifest export/import, including selection identity capture, checksum validation, and compatibility checks for platform, architecture, and Python minor version.
- Setup readiness persistence for imported audio packs so pack metadata, compatibility status, and asset manifests are visible alongside the selected bundle/profile state.

### Changed

- The `/setup` audio UI now recommends both a hardware bundle and a resource profile, shows profile-specific disk/tier metadata, and sends `resource_profile` through provisioning and verification flows.
- Audio bundle documentation and speech onboarding guides now describe per-profile footprints and the distinction between online provisioning and v1 offline-pack import.

### Fixed

- Safe rerun and verification flows now stay aligned with the selected bundle/profile identity instead of treating every bundle as a single undifferentiated install target.
- Setup documentation no longer implies a fully offline dependency installer; the v1 offline-pack path is documented as manifest-and-model portability only.


## [0.1.27] 2026-03-15

### Added

- Bundle-first audio setup provisioning in `/setup` with a curated audio bundle catalog for `cpu_local`, `apple_local`, `nvidia_local`, and `hosted_plus_local_backup`, plus backend coverage for bundle selection and expansion.
- Hardware-aware audio bundle recommendations driven by detected setup machine profile data, with new setup APIs for recommendation retrieval and bundle provisioning.
- A persisted `audio_readiness` state model and setup endpoints for readiness inspection and reset so audio provisioning status no longer depends on the global setup completion flag.
- End-to-end audio verification and readiness classification for primary STT/TTS paths, including setup-facing health collection, remediation reporting, and integration coverage for recommendation, readiness, provisioning, and verification flows.
- A generator-backed audio bundle documentation path via `Helper_Scripts/generate_audio_bundle_docs.py`, along with design and implementation plan docs for the bundle rollout.

### Changed

- The `/setup` audio experience now defaults to curated bundle selection instead of exposing provider-level STT/TTS choices first, with detected hardware, prerequisite visibility, provisioning progress, safe rerun, verification, and readiness-report UX.
- Setup and speech onboarding guides were rewritten around the new bundle-first flow, including clearer separation between online provisioning, pre-seeded offline provisioning, and offline runtime after provisioning.

### Fixed

- Setup audio completion is now classified by real verification outcomes (`ready`, `ready_with_warnings`, `partial`, `failed`) instead of treating install completion as proof that the speech stack is usable.
- Final setup-scope verification and security checks were tightened for the touched installer paths, including Bandit-clean subprocess annotations and docs/test alignment for the new audio bundle workflow.


## [0.1.26] 2026-03-15

### Consolidated Release Summary (Merged PRs #790-#897)

- Added: Repo2Txt shipped across shared UI, web, and extension surfaces with GitHub and local sources, file-tree filtering, preview/copy/download flows, and better launcher discoverability. (PR #790)
- Added: Safety and admin surfaces expanded with the Family Guardrails Wizard, per-user moderation phrase lists, and router analytics usage tabs plus aggregate APIs for operators. (PRs #804, #810, #797)
- Added: Reading and research workflows grew with pinboard-style Collections enhancements, switchable unified RAG profiles, and multi-source quiz generation. (PRs #805, #796, #807)
- Added: MCP Hub management shipped across AuthNZ storage, API, WebUI, and extension settings for ACP profiles and external server lifecycle management. (PR #806)
- Added: Workflow capabilities expanded with kanban orchestration primitives, strict structured-output generation, structured QA chat workflows, unified queued-request handling, and connector-backed external file sync orchestration. (PRs #809, #818, #820, #823, #821)
- Changed: Audio and ingestion experiences were redesigned: the Speech Playground gained richer TTS and STT comparison workflows, while Quick Ingest became a 5-step wizard with live progress, retry/error classification, and minimize-to-background controls. (PRs #816, #827)
- Changed: Knowledge and watchlists moved to more progressive, task-oriented layouts with clearer defaults, navigation, empty states, and reliability feedback. (PRs #812, #813)
- Changed: Prompt and writing workflows were reorganized with a fuller prompt workspace, stronger filtering/discoverability, and a restructured Writing Playground inspector. (PRs #817, #815)
- Changed: Notes management was overhauled around decomposed editor/sidebar panes, progressive disclosure, and clearer save-status handling. (PR #826)
- Changed: Media compatibility cleanup reduced legacy shim behavior and made deprecation signaling more explicit across ingestion flows. (PR #794)
- Fixed: Platform reliability hardened across monitoring, benchmark recovery, shim cleanup, and core runtime paths, including sandbox admission/state handling, monitoring range interactions, and broader cleanup bundles that had previously been scattered across draft entries. (PRs #792, #793, #795, #798)
- Fixed: Chat reliability issues were addressed for tool routing parity, OpenRouter freeze cases, selected-chat failure states, and conflict/retry behavior. (PRs #811, #814)
- Fixed: ACP and admin surfaces received remediation for control auth, persistence, forks, login/auth proxy paths, privileged actions, and users views. (PRs #825, #829)
- Fixed: Repo2Txt was hardened for local source selection, GitHub provider permissions, worker recovery, and deterministic compile behavior. (PR #790)
- Added: Admin and operator surfaces became more authoritative across billing and feature gating, monitoring, DSR intake, backup scheduling, incident management, maintenance rotation, BYOK retention previews, and privileged-admin backend verification. (PRs #834, #862, #864, #868, #879, #880, #883, #877)
- Added: ACP and MCP Hub governance expanded with production-readiness hardening, persisted session and registry state, tool permissions and persona approvals, path-policy controls, OPA governance packs, capability adapter registries, runtime policy integration, and governance pack upgrade workflows. (PRs #838, #847, #848, #853, #875, #891, #893, #895, #897)
- Added: Persona, assistant, and flashcard workflows grew with Persona Garden chat integration, live exemplars, structured Q&A preview import, study assistant and scheduler follow-ups, per-deck FSRS, voice assistant building, and tighter persona setup targeting and handoff analytics. (PRs #833, #841, #845, #878, #889, #890, #892, #894, #896)
- Added: Workspace, ingestion, and companion capture capabilities expanded with ingestion source sync, explicit and bulk companion capture, source folders and collections, document workspace upgrades, MinerU PDF OCR, and git repository source sync for notes. (PRs #846, #860, #866, #873, #874, #882, #887)
- Added: Prompting, presentation, and workflow tooling expanded with structured prompts across Prompt Studio, llama.cpp grammar and thinking controls, Presentation Studio, and workflow diagnostics with investigation APIs and run inspection. (PRs #858, #850, #881, #888, #869)
- Added: Sandbox and model runtime capabilities expanded with macOS sandbox runtimes, admin diagnostics, real seatbelt execution, and runtime-aware Qwen3 TTS backend support. (PRs #852, #856, #859, #870)
- Changed: Major interface redesigns landed across the Writing Playground, Speech Playground Compose & Compare flow, Workspace Playground, Media Review, Moderation Playground, and the shared `/tts` speech route. (PRs #835, #836, #837, #840, #865, #871)
- Changed: Companion and chat coordination were tightened with quality controls, proactive follow-ups, workspace chat isolation and full sync, and reduced request bursts from lazy hydration. (PRs #867, #876, #872)
- Fixed: Reliability fixes addressed Quick Ingest session wiring, chat-workflows route exposure, sandbox workspace locking, media-review preview rendering, and admin UI prod-readiness fallbacks for local-only flows. (PRs #830, #832, #839, #855, #857)

### Added

- **Document Workspace UX/UI Audit** — Comprehensive improvements across accessibility, keyboard navigation, mobile responsiveness, and user control, based on Nielsen's 10 Usability Heuristics:
  - **Accessibility (WCAG)**: `aria-label` on all 5 TextSelectionPopover buttons (Copy, Highlight, Translate, Ask AI, Listen), `group-focus-within:opacity-100` on annotation edit/delete actions for keyboard discoverability, `aria-label` on chat textarea
  - **Keyboard shortcuts**: Cmd+G (go to page), Cmd+[/] (toggle panels), Cmd+/ (focus chat), Cmd+=/- /0 (zoom in/out/reset), F (fullscreen toggle), Escape to dismiss text selection popover
  - **Mobile bottom navigation bar**: Fixed 48px bottom nav with Sidebar/Document/Chat icons replacing Ant Design top tabs, proper `role="tablist"` and `aria-selected` semantics
  - **Mobile text selection bottom sheet**: Touch-friendly slide-up sheet with backdrop, safe area inset padding, drag handle, and text preview replacing floating popover on mobile
  - **Ask AI prompt templates**: Dropdown submenu with Explain, Summarize, Define terms, and Simplify options replacing single hardcoded prompt
  - **Drag-and-drop document upload**: `onDragOver`/`onDragLeave`/`onDrop` handlers with visual feedback and client-side file type validation (PDF/EPUB only)
  - **Recent documents in picker**: Last 5 opened documents shown above search results, persisted in localStorage
  - **Per-tab reading progress bar**: Thin progress indicator under each document tab driven by page/percentage
  - **Onboarding feature cards**: 4 dismissible cards (Highlight, Chat, Insights, Quiz) in empty viewer state, localStorage-gated
  - **Chat clear confirmation**: Popconfirm dialog before clearing chat history
  - **Sync recovery animation**: Pulse animation on sync indicator for 2 seconds after error-to-synced recovery
  - **Quiz preference persistence**: Question count, type, and difficulty saved to localStorage across sessions
  - **Expanded server-required guidance**: Setup instructions in server-offline state
  - **EPUB chapter title in toolbar**: Shows current chapter name on left side of viewer toolbar
  - **Tablet drawer decoupling**: Toggle buttons always visible on tablet; drawer state independent from desktop pane state
  - **Responsive drawer width**: `Math.min(360, window.innerWidth * 0.85)` instead of fixed pixel width
  - **Text2SQL security hardening**:
    - Added dedicated `sql.read` RBAC permission constant and seeded baseline grants for default `user`/`admin` roles
    - Added connector ACL enforcement for `POST /api/v1/text2sql/query` with explicit `403 unauthorized_target` response on denied targets
    - Added security regression tests for RBAC and ACL behavior (`test_text2sql_rbac_and_acl.py`)
    - Updated API docs for standalone Text2SQL and unified RAG SQL source usage (`sources=["sql"]`, `sql_target_id`)
- repo2txt V1 integration across shared UI, web, and extension options surfaces:
  - Added shared options route scaffold and route wiring for `/repo2txt`.
  - Added repo2txt page shell and interaction flow in shared UI.
  - Added GitHub provider support for repo2txt generation.
  - Added local file/folder provider support for repo2txt generation.
  - Added repo2txt file-tree state slice for selection/exclusion behavior.
  - Added repo2txt formatter and tokenizer worker pipeline.
  - Added locale keys and parity guard coverage for repo2txt copy.
  - Added Next.js wrapper route at `apps/tldw-frontend/pages/repo2txt.tsx`.
  - Added extension E2E coverage for options route loading and sidepanel link-out behavior:
    - `apps/extension/tests/e2e/repo2txt-options.spec.ts`
    - `apps/extension/tests/e2e/repo2txt-sidepanel-linkout.spec.ts`
- Added repo2txt discoverability in the launcher/shortcuts modal.
- Added docs coverage for repo2txt route behavior in:
  - `apps/DEVELOPMENT.md`
  - `apps/tldw-frontend/README.md`
  - `apps/extension/README.md`
- Added third-party notice attribution for upstream `repo2txt` (project + MIT license) in:
  - `THIRD_PARTY_NOTICES.txt`
- Added extension compile tsconfig and entrypoint module declarations:
  - `apps/extension/tsconfig.compile.json`
  - `apps/extension/types/tldw-ui-entries.d.ts`
- Pinboard-style Collections enhancements for Reading:
  - Added saved-search CRUD API surfaces:
    - `POST /api/v1/reading/saved-searches`
    - `GET /api/v1/reading/saved-searches`
    - `PATCH /api/v1/reading/saved-searches/{search_id}`
    - `DELETE /api/v1/reading/saved-searches/{search_id}`
  - Added Reading item-note link API surfaces:
    - `POST /api/v1/reading/items/{item_id}/links/note`
    - `GET /api/v1/reading/items/{item_id}/links`
    - `DELETE /api/v1/reading/items/{item_id}/links/note/{note_id}`
  - Added strict same-user note existence enforcement for link creation while keeping note-content lifecycle under `/api/v1/notes`.
  - Added archive-mode controls (`use_default|always|never`) and persisted archive metadata fields (`archive_requested`, `has_archive_copy`, `last_fetch_error`) in Reading responses.
  - Added backend and frontend coverage for saved searches, note links, archive controls, and feature-flag gating.
- Added changelog coverage for this session’s Collections strict-boundary delivery and review remediation pass.
- Family Guardrails Wizard rollout across WebUI + extension settings surfaces:
  - Added dedicated settings route `/settings/family-guardrails` with capability-aware gating and navigation metadata.
  - Added WebUI page wrapper at `apps/tldw-frontend/pages/settings/family-guardrails.tsx`.
  - Added extension route module and registry wiring parity:
    - `apps/tldw-frontend/extension/routes/option-family-guardrails-wizard.tsx`
    - `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Family wizard backend contract and persistence surface:
  - Added family wizard request/response schemas in `tldw_Server_API/app/api/v1/schemas/family_wizard_schemas.py`.
  - Added family wizard draft/snapshot endpoints in `tldw_Server_API/app/api/v1/endpoints/family_wizard.py`.
  - Added guardian DB draft/member/relationship/plan persistence support in `tldw_Server_API/app/core/DB_Management/Guardian_DB.py`.
- Guardian moderation orchestration improvements:
  - Added queued family plan materialization on guardian acceptance.
  - Added strictest-wins shared-dependent policy conflict resolution path.
- Coverage and docs for the new workflow:
  - Added comprehensive Family Guardrails wizard unit/integration coverage across UI services, route parity, DB, schema, endpoint, and materialization flows.
  - Added Playwright workflow coverage for family setup, guardian mapping, templates, tracker blockers, and draft resume.
  - Added user guides:
    - `Docs/User_Guides/WebUI_Extension/Family_Guardian_Setup.md`
    - `Docs/User_Guides/WebUI_Extension/Family_Guardrails_Wizard_Guide.md`
- Quiz critical-path E2E coverage is now split into focused specs with shared strict helpers:
  - `apps/extension/tests/e2e/quiz-critical-edit.spec.ts`
  - `apps/extension/tests/e2e/quiz-critical-create.spec.ts`
  - `apps/extension/tests/e2e/quiz-critical-take-results.spec.ts`
  - `apps/extension/tests/e2e/utils/quiz-critical-helpers.ts`
- Added backend regression coverage to ensure the attempts listing route is not shadowed by dynamic quiz routes:
  - `tldw_Server_API/tests/Quizzes/test_quizzes_endpoint_integration.py`
- Moderation per-user phrase rule support across schema, runtime, and UX:
  - Added typed per-user override `rules` model (`id`, `pattern`, `is_regex`, `action`, `phase`) to moderation schemas.
  - Added runtime compilation of per-user rules with literal/regex handling and safety checks, then merged compiled rules into effective moderation policy resolution.
  - Added non-advanced Moderation Playground User-scope quick composer for phrase-list management:
    - `Banlist` entries map to `block`
    - `Notify list` entries map to `warn`
    - Optional regex toggle, per-item removal, and list views for `Banned phrases` and `Notify phrases`.
- Added moderation API and service test coverage for per-user rules:
  - Added contract coverage for `GET/PUT /api/v1/moderation/users/{user_id}` with `rules` payloads.
  - Added unit coverage for per-user rule compilation, dangerous-regex rejection, effective-policy merge behavior, and phase-aware action behavior.
- Added frontend service contract and component coverage for phrase-list workflows:
  - Added moderation service contract tests to verify rules roundtrip in user override requests/responses.
  - Added Moderation Playground quick-list tests for rendering, add/remove flows, duplicate and invalid-regex validation, and save-payload assertions.
- Watchlists UX redesign — progressive disclosure layout (PR #813):
  - Restructured watchlists from 8 horizontal tabs to 3 primary tabs (Feeds, Articles, Reports) with inline expandable secondary views (Monitors, Activity, Templates).
  - Added persistent collapsible health bar replacing the Overview tab, with 30s auto-refresh, health cards, and attention badges.
  - Added settings drawer (gear icon) replacing the Settings tab.
  - Added Cmd/Ctrl+K command palette with categorized commands (navigate, create, action) and fuzzy search.
  - Added keyboard shortcuts: 1/2/3 (tab switch), N (new entity), R (refresh), / (focus search), ? (help panel).
  - Added rich per-entity empty states with contextual descriptions and CTAs for feeds, monitors, activity, articles, reports, and templates.
  - Added "Show all views" toggle to restore original 8-tab layout for power users (persisted to localStorage).
  - Added mobile responsive layout: Select dropdown for tabs at <768px, full-width settings drawer on mobile.
  - Added run failure "common causes" section in RunDetailDrawer, pattern-matched by failure kind (auth, rate limit, timeout, network, TLS).
  - Added retry button to failed run notification toasts.
  - Added deep-link backward compatibility: old URL params (`?tab=sources`, `?tab=items`) map to new equivalents (`?tab=feeds`, `?tab=articles`).
  - Added 89 new i18n locale keys for all new UI elements.
- **Quick Ingest Wizard Redesign** — Replace 3-tab modal with 5-step progressive wizard:
  - **Step 1 — Add Content**: Combined file drop zone + multi-line URL paste, auto-detect content type with icons, inline validation (malformed URLs, duplicates, size limits), per-item remove, "Use defaults & process" quick mode for single items
  - **Step 2 — Configure**: Preset card selector (Quick/Standard/Deep) with per-queue time estimates, content-type-filtered options (audio language/diarization, document OCR, video captions shown only when relevant), storage mode (Server/Local), collapsible advanced options
  - **Step 3 — Review**: Confirmation checkpoint with per-item operation summary, contextual warnings (large files >50 MB, long estimated time >15 min, large batches >5 items), total time estimate
  - **Step 4 — Processing**: Per-item live progress dashboard with multi-stage indicators, current stage label + percentage, per-item cancel, overall summary bar (completed/processing/queued + elapsed/remaining), Cancel All and Minimize to Background buttons
  - **Step 5 — Results**: Grouped results (successes collapsed, errors expanded), error classification badges (Network/Format/Server + Retryable/Permanent), plain-language error descriptions, per-error Retry button, per-success Open/Chat/View actions, Retry All Errors batch button, Ingest More to restart
  - **Minimize-to-background floating widget**: Portal-rendered fixed-position widget (bottom-right) showing progress percentage, item count, estimated time, expand/restore button; auto-dismisses 10s after completion
  - **SSE real-time updates** via `useIngestSSE` hook: Connects to `GET /api/v1/media/ingest/jobs/events/stream`, handles `snapshot` and `job` events, maps backend statuses to wizard progress, exponential backoff reconnection (1s→10s)
  - **`IngestWizardContext`**: React context + `useReducer` managing wizard state (current step, queue items, preset config, processing state, minimized flag) with navigation guards
  - **`IngestWizardStepper`**: Horizontal breadcrumb stepper with accumulated context labels, click-to-revisit completed steps
  - **`timeEstimation.ts`**: Client-side heuristic time estimates by media type and preset (audio 10s+2s/MB, video 15s+3s/MB, document 3s+0.5s/MB, web ~8s) with preset multipliers
  - **`ErrorClassification.ts`**: Pattern-based error categorization with priority ordering and retryable/permanent flags
  - **`PresetSelector.tsx`**: Rewritten from dropdown to card layout with plain-language descriptions and dynamic time estimates
  - **Direct modal swap**: `QuickIngestButton.tsx` and `Sidepanel/Chat/form.tsx` now import `QuickIngestWizardModal` (no feature flag)
  - **`autoProcessQueued` backward compat**: New modal supports auto-skip to processing step when items are pre-queued (used by sidepanel chat form)
  - 70 new tests: `ErrorClassification.test.ts` (22), `timeEstimation.test.ts` (21), `IngestWizardContext.test.tsx` (13), `IngestWizardStepper.test.tsx` (7), `QuickIngestWizardModal.integration.test.tsx` (7 — full Steps 1→5 flow)
- Docker runner integration test (`test_docker_runner_integration.py`) for full container lifecycle validation (session → upload → run → artifacts → cleanup), skipped without Docker daemon.
- **TTS Listen Tab UX Redesign** — Two-zone layout (Workspace + Inspector Panel):
  - `CharacterProgressBar` component with color-coded thresholds and ARIA progressbar role
  - `VoicePreviewButton` component for inline voice previews via TTS API
  - `TtsStickyActionBar` with Play/Stop/Download controls, stream status indicator, and inspector toggle
  - `TtsProviderStrip` compact config summary strip with clickable tags and preset switcher
  - `TtsInspectorPanel` with Voice/Output/Advanced tabs, responsive drawer mode on mobile
  - `TtsVoiceTab`, `TtsOutputTab`, `TtsAdvancedTab` inspector tab components
  - Provider-conditional field visibility (browser shows voice only, openai shows model+voice, tldw shows full controls)
  - Multi-voice narration UI in Voice tab with role assignment cards (tldw provider only)
  - Inline `VoiceCloningManager` rendering in Advanced tab
  - Keyboard shortcuts: Ctrl/Cmd+Enter (play), Escape (stop), Ctrl/Cmd+. (toggle inspector)
  - `aria-live="polite"` on provider strip for screen reader announcements
  - 24 component tests across 6 test files
- **STT Playground Comparison-First Redesign** — Record once, compare across multiple models:
  - Rewrote `SttPlaygroundPage` from 736-line single-model tool into three-zone comparison architecture
  - **Zone 1 — RecordingStrip**: Record/stop with real-time duration timer, animated audio level meter (`role="meter"`), native audio playback, file upload (`accept="audio/*"`), collapsible settings gear toggle
  - **Zone 2 — ComparisonPanel**: Multi-select model picker, parallel transcription via `Promise.allSettled`, responsive card grid (1/2/3 cols), per-card skeleton loading, editable transcripts, latency + word count metrics, copy-to-clipboard, save to Notes, per-card retry on error
  - **Zone 3 — HistoryPanel**: Collapsible past recordings with re-compare (loads blob from IndexedDB), markdown export to clipboard, single-delete and confirmed clear-all via Modal
  - **Dexie/IndexedDB persistence** (`stt-recordings.ts`): Audio blobs stored with 20-recording cap and oldest-eviction, schema version bump in `schema.ts`
  - **`useAudioRecorder` hook**: MediaRecorder lifecycle, 200ms timer, blob retention, `loadBlob()` for re-compare, cleanup on unmount
  - **`useComparisonTranscribe` hook**: Parallel multi-model transcription, per-model status (pending/running/done/error), `performance.now()` latency, retry single model, multi-format response extraction
  - **`InlineSettingsPanel`**: Playground-local overrides for language, task, format, temperature, prompt, segmentation (progressive disclosure); "Reset to defaults" restores global settings
  - **Keyboard shortcuts**: Space to toggle record (when no text input focused), Cmd/Ctrl+Enter for Transcribe All, shortcut hints on buttons
  - **Accessibility**: Dynamic `aria-label` on record button with shortcut hint, `aria-live="polite"` on duration timer and transcript textareas, `role="region"` with model-specific labels on result cards, multi-state visual feedback (icon + color + text, not color alone)
  - 51 tests across 9 test files (Dexie store, both hooks, all 4 components, keyboard shortcuts, page integration)
- Text2SQL security hardening:
  - Added `sql.read` RBAC permission constant and seeded baseline grants for default `user` and `admin` roles.
  - Added SQL target ACL claims (`sql.target:*`, `sql.target:media_db`) and removed implicit default target allow.
  - Enforced connector ACL checks in `POST /api/v1/text2sql/query` with explicit `403 unauthorized_target` on denied targets.
  - Added RBAC/ACL regression tests in `tldw_Server_API/tests/Security/test_text2sql_rbac_and_acl.py`.
  - Added baseline seed consistency for `sql.read` in `rbac_seed` paths.
- API documentation updates for SQL retrieval:
  - Documented standalone Text2SQL endpoint in `Docs/API-related/API_README.md`.
  - Documented unified RAG SQL source usage (`sources=["sql"]`, `sql_target_id`) and required SQL ACL claims in `Docs/API-related/RAG_API_Documentation.md`.
- Chat Workflows web-shell regression coverage:
  - Added `apps/tldw-frontend/__tests__/pages/chat-workflows-route.test.tsx` to verify the Next.js page shim exists and still lazy-loads `@/routes/option-chat-workflows`.
  - Tightened `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx` to keep the `/chat-workflows` workspace navigation metadata aligned with the shared route registry.
- **TTS "Compose & Compare" Redesign** — New A/B comparison workflow for the Speech Playground TTS tab:
  - **`UnifiedAudioPlayer`** (`components/Common/UnifiedAudioPlayer.tsx`): Single audio player component for all providers — play/pause/seek slider, time display, waveform visualization (via `WaveformCanvas`), download button; accepts `audioUrl`, `audioBlob`, or streaming mode; replaces per-provider player inconsistency
  - **`RenderStrip`** (`components/Option/Speech/RenderStrip.tsx`): Self-contained generation card with five states (`idle`, `generating`, `ready`, `playing`, `error`); config tags (provider, voice, format, speed) as clickable chips; embeds `UnifiedAudioPlayer` for playback and `TtsJobProgress` for long-running generations; undo notification on remove with 4-second restore window
  - **`useMultiRenderState`** hook (`hooks/useMultiRenderState.ts`): Manages array of render strips with independent generation lifecycle; actions: `addRender`, `removeRender`, `generateRender`, `generateAll`, `clearAll`, `playAllSequentially`; play-one-at-a-time enforcement (starting one pauses others); object URL lifecycle management (create on generate, revoke on remove/unmount)
  - **`VoicePickerModal`** (`components/Option/Speech/VoicePickerModal.tsx`): Unified voice selection across all 19 backend TTS providers; search input, recent voices row (last 5 from localStorage), provider-grouped catalog (Local Engines / Cloud Providers) with collapsible sections; inline `[Play]` preview per voice; capability badges (Stream, Clone); selecting a voice returns `{ provider, voice, model }` and auto-selects provider
  - **Render strips zone** in `SpeechPlaygroundPage.tsx`: Between text editor and action bar — maps `useMultiRenderState` to `RenderStrip` components; "Add Render", "Pick Voice", "Generate All", "Play All", "Clear" controls; per-strip generate, edit (opens voice picker), retry, remove actions
  - **Action bar extensions** in `TtsStickyActionBar.tsx`: "Add Render" button with separator, "Play All" button (shown when multiple ready strips exist)
  - **Smart defaults**: Last-used voice config persisted to `localStorage` for render strip defaults; recent voices tracked in voice picker
  - **Accessibility**: `role="region"` + `aria-label` on all strips and players; `role="option"` + keyboard Enter on voice picker items; `aria-label` on all interactive buttons
  - 36 new tests: `UnifiedAudioPlayer.test.tsx` (8), `RenderStrip.test.tsx` (11), `useMultiRenderState.test.ts` (10), `VoicePickerModal.test.tsx` (7)
- **Writing Playground UX overhaul** — Editor-dominant layout with collapsible drawers, based on Nielsen heuristic audit:
  - **Editor-dominant layout (H8 fix):** Removed `PageShell` max-width constraint; editor now fills all available viewport space instead of competing with permanently-visible sidebars
  - **Drawer-based panels:** Library and Inspector panels converted to collapsible sidebars (pinned on desktop ≥1100px, overlay Drawers on mobile <1100px) via `WritingPlaygroundShell`; toggle buttons in new compact top bar
  - **Prominent Generate button (H1/H6 fix):** Generate button moved from buried inspector sidebar to always-visible top bar; disabled state shows tooltip explaining why (no session, no model, offline, no chat provider)
  - **Generation visual feedback (H1 fix):** Pulsing ring animation on editor during generation; live elapsed-time counter in status bar; tok/s display
  - **Compact top bar:** Session name, model input, Generate button, diagnostics tag, and panel toggles in a single row; replaces removed page title/subtitle
  - **Status bar:** Token count, generation speed, elapsed time, save status, and Ctrl+Enter shortcut hint
  - **Decluttered toolbar (H8 fix):** Toolbar reduced from 12+ always-visible buttons to icon-only Undo/Redo + view mode toggle + overflow dropdown (Read Aloud, Pause/Resume, View Chunks, Insert, Search)
  - **Flexible editor:** Replaced fixed `rows={18}` textarea with `autoSize={{ minRows: 12 }}` and resizable styling; fixed double-scroll caused by nested overflow containers
  - **Terminology cleanup (H2 fix):** "Basic stopping mode" → "Stop condition"; "Inspect" tab → "Analysis" (both default label and override)
  - Updated 4 test files (32 assertions) to match new layout structure

### Changed

- Extension compile script now targets explicit config:
  - `apps/extension/package.json` `compile` now uses `tsc --noEmit -p tsconfig.compile.json`.
- Frontend compile script now uses webpack build path for deterministic completion in this environment:
  - `apps/tldw-frontend/package.json` `compile` now uses `next build --webpack` before token-sync verification.
- Updated Product PRD acceptance and test criteria for Reading saved-search and item-note link surfaces, including strict notes-boundary expectations.
- Improved the Collections implementation plan’s final commit-step readability by consolidating long `git add` instructions into grouped multiline commands.
- Character import preview UX now shows a dedicated `OK` button after successful upload completion so users can dismiss with explicit confirmation instead of relying on the modal close icon.
- GitHub Advanced Security triage for PR #753 was completed end-to-end:
  - Investigated each remaining CodeQL finding by rule/path and separated true issues from false positives.
  - Closed residual `py/path-injection` findings as false positives only after validating trusted-root containment in MLX tokenizer artifact resolution.
  - Stepped through failing checks until the PR returned to all-green status.
- Family Guardrails wizard UX simplification and resilience:
  - Household setup now uses preset-driven onboarding (family/caregiver/institutional templates) as the primary mode selector path.
  - Wizard copy and validation feedback are mode-aware (`guardian` vs `caregiver`) across setup, mapping, review, and tracker steps.
  - Single-guardian flows skip relationship mapping and preserve back-navigation symmetry from templates/review.
  - Large-family workflows now support bulk entry/table interactions with keyboard-assisted selection/removal and status-aware tracker guidance.
  - Wizard continues now enforce stronger inline step validation, duplicate/collision messaging, and deterministic resume-to-latest-draft behavior.
- Route capability and parity guardrails now enforce consistent Family Guardrails visibility/registration across shared UI, web, and extension route registries.
- Refined MCP Hub external server update flow to use a dedicated repository-level `update_external_server` path (direct `UPDATE` pattern) for consistency with ACP profile updates and clearer audit semantics.
- Extended settings/navigation indexing so MCP Hub management is discoverable in both settings and workspace-oriented routes.
- Hardened quiz endpoint routing by typing dynamic route segments (`{...:int}`) for quiz/question/attempt IDs in:
  - `tldw_Server_API/app/api/v1/endpoints/quizzes.py`
- Upgraded strict quiz E2E critical flows to hard assertions for:
  - edit metadata save + persisted verification (UI + API),
  - manual create flow persistence verification,
  - take/submit/results flow with explicit attempts API assertion.
- MCP Hub management end-to-end surfaces across AuthNZ storage, API, and WebUI/extension:
  - Added AuthNZ migrations for MCP Hub ACP profile and external server tables in SQLite and PostgreSQL bootstrap paths.
  - Added `McpHubRepo` and `McpHubService` for ACP profile CRUD and external server lifecycle handling, including encrypted secret storage and masked secret status responses.
  - Added MCP Hub management API contracts and routes under `/api/v1/mcp/hub` for ACP profiles, external servers, and secret set operations.
  - Added WebUI + extension MCP Hub page wiring and tabs (`AcpProfilesTab`, `ExternalServersTab`, `ToolCatalogsTab`) with shared route-registry coverage (`/mcp-hub`, `/settings/mcp-hub`).
  - Added frontend MCP Hub API client module/tests and route parity tests for web/extension settings exposure.
  - Added docs for architecture and usage:
    - `Docs/MCP/mcp_hub_management.md`
    - `Docs/MCP/README.md`
- Added backend and frontend regression coverage for MCP Hub migrations, repo/service behavior, API authorization boundaries, and tab interactions/error states.
- Enhanced moderation policy snapshots to include per-rule `phase` metadata alongside existing pattern/action/replacement/category fields.
- Updated Moderation Playground override payload normalization/comparison so `rules` participate in dirty-state detection and save/reset lifecycle.
- `/api/v1/audio/chat/stream` now supports overlapped LLM->TTS streaming by incrementally chunking LLM deltas into phrase commits against realtime TTS sessions.
- `/api/v1/audio/chat/stream` turn finalization now runs as cancellable per-turn tasks, enabling barge-in interruption and stale-output guards.
- `/api/v1/audio/stream/tts/realtime` now supports in-session `interrupt` by finishing/cancelling the active synthesis window, reopening a session with the same config, and continuing on the same WebSocket.
- `apps/packages/ui/src/hooks/useVoiceChatStream.tsx` now sends `interrupt` on barge-in while speaking and transitions back to `listening` on server `interrupted` frames.
- Updated audio protocol/docs to document new frames and overlap behavior:
  - `Docs/API/Audio_Chat.md`
  - `Docs/Audio_Streaming_Protocol.md`
- Renamed "Knowledge QA" heading to "Ask Your Library" with updated subtitle across empty state and tests.
- Redesigned web search toggle from ambiguous text to visually distinct pill with globe icon, filled/outline states, and proper `aria-label`/`aria-pressed` attributes.
- `KnowledgeQALayout` now conditionally renders based on layout mode:
  - Simple mode: single centered column, `CompactToolbar`, no history sidebar.
  - Research mode: three-column layout with `HistoryPane`, full `KnowledgeContextBar`, `EvidenceRail`.
  - Mobile always forces Simple mode layout; promotion toast hidden on mobile.
- `KnowledgeReadyState` guide section auto-collapses for returning users (via `useEffect` on async history load).
- Evidence rail auto-open now respects user intent via `useRef` tracking — won't reopen after manual close.
- Settings panel and evidence rail are now mutually exclusive (opening one closes the other).
- Refactored `Knowledge/index.tsx` settings page status display from nested ternaries to a `STATUS_COPY` map.
- Updated golden layout tests for new conditional rendering paths and component structure.
- Renamed watchlist tabs in UI to user-task language: Sources→Feeds, Jobs→Monitors, Runs→Activity, Items→Articles, Outputs→Reports.
- Updated first-run copy contract test to match "monitor health" terminology (was "run health").
- Default sidebar tab changed from "toc" to "insights" for better first-impression content
- Tab label font size increased from 10px to 11px for readability
- "Fit width" button renamed to "Reset zoom" with `RotateCcw` icon for accuracy
- Keyboard shortcuts modal updated to list only implemented shortcuts
- Updated misleading docstrings in `DockerRunner`, `SandboxService`, `LinuxLimaEnforcer`, and `MacOSLimaEnforcer` to accurately reflect implemented functionality.

### Removed

- No removals in this session.

### Fixed

- Fixed stale/cancelled turn output leakage by dropping outdated audio/LLM emissions after interruption.
- Fixed realtime TTS interruption flow to allow continued text commits without requiring WebSocket reconnect.
- Fixed voice websocket test stability in local environments by stubbing heavyweight STT dependency imports in targeted WS test modules.
- Fixed user override loading/reset normalization so persisted `rules` are preserved in draft state and correctly re-applied.
- Fixed Moderation Playground User ID input suffix rendering to remain structurally stable, preventing focus-loss behavior from dynamic suffix mount/unmount.
- Fixed route matching ambiguity where `/api/v1/quizzes/attempts` could be interpreted as `/{quiz_id}`, causing attempts list failures.
- Fixed strict take/results E2E behavior to fail fast with diagnostic screenshot when the started-quiz question list does not render.
- Fixed unsaved-create tab navigation handling to accept only the expected unsaved-changes prompt copy.
- Fixed MCP Hub list endpoint access-control gaps by constraining ACP profile and external server visibility to principal-allowed scopes (`global`, own `user`, member `org/team`) and returning `403` for forbidden explicit filters.
- Fixed PostgreSQL timestamp persistence in MCP Hub repo by preserving timezone-aware UTC datetimes for `TIMESTAMPTZ` writes.
- Fixed silent UI failures in MCP Hub tabs by surfacing user-visible Ant Design error alerts for load/create/save-secret failures.
- Fixed MCP Hub security hardening follow-ups flagged during review and revalidated with targeted tests and Bandit checks for touched backend paths.
- Fixed web/extension family wizard route wiring parity regressions and added explicit route-parity tests to prevent drift.
- Fixed family wizard tracker/review action targeting so blocked dependents route guardians back to the correct mapping context.
- Fixed targeted route-parity test path resolution to be cwd/worktree agnostic during Vitest runs.
- Fixed extension compile command failure caused by missing local `tsconfig.json` in `apps/extension`.
- Fixed frontend compile gate stalling under Turbopack in this environment by switching compile verification to webpack build mode.
- Fixed archive artifact creation in async paths by offloading blocking filesystem operations (`mkdir`, `write_text`) with `asyncio.to_thread`.
- Fixed archive metadata update reliability by removing silent suppression and logging/reporting metadata patch failures via `archive_error`.
- Fixed import-time crash risk from invalid archive env vars by introducing guarded integer parsing with safe defaults.
- Fixed saved-search input validation gaps:
  - Reject unsupported query keys and malformed query value types.
  - Reject whitespace-only search names (create and update).
  - Normalize and validate `sort`/filter payloads before persistence.
- Fixed endpoint typing clarity by adding explicit row-type annotations for saved-search and note-link response helpers.
- Fixed remaining security findings in PR #753:
  - Hardened MLX tokenizer artifact candidate path normalization and trusted-root enforcement in `tokenizer_resolver.py` to prevent traversal/out-of-root resolution.
  - Strengthened Notes attachment markdown escaping to include backslashes and newline normalization.
  - Strengthened Notes export YAML escaping for backslashes and control characters (`\r`, `\n`, `\t`) in frontmatter values.
  - Removed admin UI API key cleartext persistence in `sessionStorage`; API key auth now remains in-memory only for single-user mode.
  - Added/updated regression coverage for MLX path handling, notes frontmatter escaping, and admin auth API key storage behavior.
- Fixed blank `/settings/knowledge` page (Critical, H1 violation) — added connection-aware error states.
- Fixed `contextChangedSinceLastRun` always showing "Scope changed" badge due to broken `normalizeIdentifierSet` comparisons against empty string.
- Removed dead `normalizeIdentifierSet` function and stale `include_media_ids`/`include_note_ids` from `useMemo` dependency array.
- Fixed `InlineRecentSessions` crash on invalid timestamp strings (added `NaN` guard in `formatRelativeTime`).
- Fixed `CompactToolbar` source summary using magic number — extracted `ALL_SOURCES_THRESHOLD` constant.
- Fixed evidence rail auto-open loop where `useEffect` would immediately reopen rail after user closed it.
- Voice-streaming interruption + overlap protocol additions:
  - Added additive client `interrupt` and server `interrupted` frame handling for `/api/v1/audio/chat/stream`.
  - Added active turn identifiers (`turn_id`) for interruption acknowledgements.
  - Added `interrupt` recovery flow for `/api/v1/audio/stream/tts/realtime` that rotates to a fresh realtime session without closing the socket.
- Added streaming phrase chunker utility:
  - `tldw_Server_API/app/core/Streaming/phrase_chunker.py`
- Added backend regression coverage for voice overlap/cancellation:
  - `tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py` now covers idle interrupt safety, overlap ordering (`tts_start`/audio before final `llm_message`), inflight cancellation, and stale-audio suppression after interrupt.
  - `tldw_Server_API/tests/Audio/test_ws_tts_realtime_endpoint.py` now covers realtime TTS interrupt handling and post-interrupt continuation.
- Added frontend regression coverage for barge-in interruption behavior:
  - `apps/packages/ui/src/hooks/__tests__/useVoiceChatStream.interrupt.test.tsx`
- Extracted `InlineSecondarySection` component outside render function to prevent unnecessary React remounts.
- Stabilized `useWatchlistsCommands` actions object with `useMemo` to prevent command list recomputation on every render.
- Removed redundant `useEffect` for `writeSecondaryExpanded` (already persisted in toggle callback).
- Knowledge QA "Adaptive Progressive" UX redesign (Stages 1 & 2):
  - New `useLayoutMode` hook with Simple/Research mode toggle, localStorage persistence, and auto-promotion toast after 3+ Q&A turns.
  - New `CompactToolbar` component: condensed pill bar (Sources, Preset, Web toggle, Settings gear) for Simple mode.
  - New `InlineRecentSessions` component: horizontal scrollable row of recent search session cards for returning users in Simple mode.
  - New `KnowledgeReadyState` empty state with "Ask Your Library" heading, collapsible "How it works" guide, suggested prompts, and source/session action buttons.
  - Added "Open workspace" / "Simplify view" toggle button (bottom-right corner) for manual mode switching.
  - Added "Scope changed" badge in compact toolbar when search context diverges from last run.
- Added 50 new tests across 3 new test files:
  - `useLayoutMode.test.ts` (25 tests): mode defaults, persistence, auto-promotion, dismiss persistence.
  - `CompactToolbar.test.tsx` (12 tests): source summaries, preset labels, web toggle, callbacks, scope badge.
  - `InlineRecentSessions.test.tsx` (13 tests): rendering, max-5 limit, click-to-restore, relative time formatting.
- **Annotation undo ID preservation**: Undo now restores the original annotation object with its server-synced ID via direct `setState`, instead of calling `addAnnotation()` which generated a new ID causing server/client state divergence and duplicates
- **Removed Cmd+W shortcut**: Conflicted with browser tab close; removed from both handler and shortcuts modal
- **Stable DOM selectors**: Fullscreen toggle and go-to-page shortcuts now use `data-testid` attributes instead of fragile CSS class substring matching and Ant Design internal class selectors
- **Quiz preferences render performance**: Moved `localStorage` read + `JSON.parse` into `useState` initializer to avoid re-parsing on every render
- Fixed `PostgresStore._coerce_created_at()` `AttributeError`: moved method from `SQLiteStore` to `SandboxStore` base class so both SQLite and Postgres backends inherit it. Admin list/count endpoints with date filters on Postgres would crash without this fix.
- Fixed event loop closure causing cascading "Event loop is closed" test failures in `RunStreamHub._schedule_dispatch()`: now clears stale `self._loop` reference and falls through to the threading.Timer fallback. Added `reset_loop()` method for test fixture cleanup.
- Fixed config cache leaking between sandbox tests: added `_sandbox_clear_config_cache` and `_sandbox_reset_stream_hub` autouse fixtures to sandbox `conftest.py` so environment variable changes and stale event loop references don't bleed across tests.
- Added debug logging to 8 silent `except ... pass` handlers in `network_policy.py` (`expand_allowlist_to_targets`, `pin_dns_map`, `apply_egress_rules_atomic`, `delete_rules_by_label`) so iptables/DNS failures are observable instead of silently swallowed.
- Fixed `Manager.first-use.test.tsx` selectors to match updated button labels (`Filters` instead of `Advanced filters`, text queries instead of role queries for onboarding cards)
- Fixed temporal dead zone bug in `AddContentStep` where `newItems` array was spread inside its own initialization callback
- Fixed `ReviewStep` "Start Processing" button not initializing processing state (was only calling `goNext`, now calls `startProcessing()` first)
- Fixed `ReviewStep` passing `PresetConfig` object instead of preset name string to `estimateTotalSeconds()`
- Fixed Ant Design `Radio.Group` onChange type mismatch in `ConfigureStep`
- **Chat UI reliability bug squash**:
  - Preserved server chat sidebar data on recoverable refresh failures instead of collapsing to an empty state during transient auth/config/rate-limit issues
  - Added explicit selected server chat load states (`idle`, `loading`, `loaded`, `failed`) so chat switching no longer leaves the conversation pane blank on failed loads
  - Rendered a dedicated selected-chat failure state in the playground instead of the generic empty-chat shell
  - Stopped no-op server chat settings sync writes when only sync metadata changed, preventing version churn during chat load
  - Retried chat delete, restore, and update mutations once after optimistic-lock conflicts by fetching the latest chat version first
  - Clarified sidebar reliability states with stale-data warnings, unavailable refresh copy, and conflict-specific mutation error messaging for chat actions
- Restored Chat Workflows route exposure in the Next.js web app:
  - Added `apps/tldw-frontend/pages/chat-workflows.tsx` so `/chat-workflows` resolves instead of falling through to a missing-page state.
  - Preserved discoverability through existing launcher coverage targeting `/chat-workflows`.
- Chat UI regression coverage for server-chat reliability:
  - Added recoverable refresh regression coverage in `useServerChatHistory.test.ts`
  - Added selected-chat load state and stale-request guard coverage in `useServerChatLoader.test.ts`
  - Added selected-chat failure rendering coverage in `PlaygroundChat.server-load-state.test.tsx`
  - Added settings sync no-op coverage in `chat-settings.sync.test.ts`
  - Added stale-version mutation retry coverage in `tldw-api-client.chat-trash.test.ts` and `tldw-api-client.chat-mutations.test.ts`
  - Added sidebar reliability and conflict-feedback coverage in `ServerChatList.reliability.test.tsx`
- Sandbox runs no longer remain stuck in `starting` when Docker, Firecracker, or Lima runner startup fails; runner exceptions now persist a terminal `failed` state.
- `sandbox_runs_started_total` now increments only after a sandbox run scaffold is accepted, so rejected requests are no longer counted as started runs.
- Sandbox runtime admission now uses shared runtime preflight results across service and policy layers instead of diverging ad hoc availability booleans.
- Sandbox websocket log streams now stop emitting heartbeats after a run ends, preventing terminal `end` frames from being buried in multi-subscriber stress scenarios.
- Sandbox snapshot creation now tolerates transient files disappearing during concurrent atomic workspace writes, avoiding archive failures on temporary rename targets.
- `RunStreamHub` subscriber registration now works in synchronous Python 3.11 contexts where no default main-thread event loop exists.

## [0.1.25] 2026-02-01

### Added

- Repo2Txt V1 integration across shared UI, web app, and extension options (PR #790):
  - Added new shared options route `/repo2txt` with web page wrapper (`apps/tldw-frontend/pages/repo2txt.tsx`) and extension/options route registration.
  - Added Repo2Txt providers and contracts for GitHub + Local sources, including repository tree/file retrieval and local directory/zip ingestion.
  - Added Repo2Txt formatter pipeline with worker-backed token counting and structured output generation (directory tree + file contents).
  - Added Repo2Txt UI surfaces for provider selection, file filtering/selection, output preview, copy, and download flows.
  - Added Repo2Txt state management slice (Zustand) plus focused route/component/provider/store/formatter test coverage.
  - Added route/navigation integration in shared options registry and header shortcuts for Repo2Txt discoverability.
  - Added locale key coverage and synchronized locale mirrors for Repo2Txt copy across supported option locales.
  - Added extension E2E coverage for Repo2Txt options route rendering and sidepanel link-out behavior.
  - Added upstream repo2txt attribution updates in `THIRD_PARTY_NOTICES.txt`.
- Strict LimaVM sandbox provider parity across REST, MCP, and ACP:
  - Added Lima runtime capability/preflight contracts and host enforcement probing (`runtime_capabilities.py`, `runners/lima_enforcer.py`, `runners/lima_runner.py`).
  - Added strict fail-closed Lima admission and execution-time revalidation in sandbox service flows.
  - Added structured Lima error contracts for `runtime_unavailable` and `policy_unsupported` responses.
  - Added Lima runtime support in MCP `sandbox.run` tool contract and ACP sandbox runner validation.
  - Added Lima strict capability fields in runtime discovery schemas and API docs updates for strict mode semantics/platform constraints.
  - Added regression coverage for Lima strict admission/preflight/runtime discovery/no-fallback/error-contract behavior across Sandbox, MCP, and ACP test suites.
- PostgreSQL AuthNZ bootstrap schema file for CI/runtime startup safety:
  - Added `tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql` with core bootstrap tables (`users`, `organizations`, `teams`) and supporting indexes.
  - Added CI guard test `tldw_Server_API/tests/CI/test_postgres_schema_file_presence.py` to enforce bootstrap schema file presence/content.
- Strict token counting phase-1 foundation for writing and provider metadata:
  - Added shared tokenizer resolver service at `tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py` to centralize provider/model tokenizer classification.
  - Added additive tokenizer metadata fields `count_accuracy` (`exact`/`unavailable`) and `strict_mode_effective` in writing tokenize/count/detokenize responses, writing capabilities payloads, and `/api/v1/llm/providers`.
  - Added exact tokenizer resolution paths for `ollama` (provider-native HTTP tokenize/detokenize probing) and `mlx` (active registry tokenizer with artifact fallback).
  - Added Bedrock `CountTokens` exact-count path (Anthropic-on-Bedrock model family) with model-level strict classification and mirrored metadata on `/api/v1/llm/providers`.
  - Added model-level exact classification for `groq` OpenAI-routed models (`openai/<model>`) via canonical `tiktoken` mapping.
  - Added explicit regression coverage that `deepseek` and `mistral` remain non-exact (`count_accuracy=unavailable`) until a verified provider-native exact tokenizer path is available.
  - Added strict writing token endpoint gate: when `STRICT_TOKEN_COUNTING=true`, non-exact tokenizer resolutions return HTTP `422`.
  - Added regression coverage for strict runtime propagation and tokenizer metadata mirroring across Writing and provider metadata tests.
- Alibaba Model Studio image backend support for `/api/v1/files/create` image generation via the new `modelstudio` backend.
- Model Studio adapter support for sync generation, async task submission/polling, and `auto` mode.
- Model Studio image configuration fields in `[Image-Generation]`:
  `modelstudio_image_base_url`, `modelstudio_image_api_key`, `modelstudio_image_default_model`, `modelstudio_image_region`, `modelstudio_image_mode`, `modelstudio_image_poll_interval_seconds`, `modelstudio_image_timeout_seconds`, `modelstudio_image_allowed_extra_params`.
- Qwen region-based endpoint presets for native HTTP chat routing (`sg`, `us`, `cn`) plus `qwen_api_region` config support.
- Curated Qwen model entries (`qwen-max`, `qwen-plus`, `qwen-turbo`) in model pricing/catalog metadata.
- Qwen provider mapping in LLM provider metadata endpoint wiring so Qwen models appear correctly.
- Regression coverage for Model Studio adapter behavior, config defaults, image allowlist behavior, image model listing, and Qwen base URL precedence/region routing.
- Workspace banners
  - Per-workspace custom header banner support in Research Workspace (`/workspace-playground`) with title, subtitle, and image.
  - New banner rendering surface in Workspace UI (`WorkspaceBanner`) with graceful no-image fallback styling.
  - “Customize banner” modal in Workspace header menu with upload, preview, save, remove-image, and reset flows.
  - Banner image normalization utility for local uploads (JPEG/PNG/WebP validation, resize, encoding, and byte-cap enforcement).
  - Extension parity route support for `/workspace-playground`.
  - Regression coverage for banner defaults, lifecycle persistence, bundle round-trip, header modal behavior, quota eviction, conflict labeling, and extension route parity.
- Skills authoring + seed UX
  - Added built-in `feynman-technique` skill under `tldw_Server_API/app/core/Skills/builtin/`.
  - Added importable `feynman-technique-template` skill under `Docs/Prompts/Skills/`.
  - Added Feynman prompt template at `Docs/Prompts/Academic-or-Studying/Feynman_Technique.md`.
  - Added Skills Manager built-ins seeding dropdown actions for both `Seed Missing Only` (`overwrite=false`) and `Seed and Overwrite Existing` (`overwrite=true`).
  - Added protocol-sync regression coverage for Feynman skill/prompt assets in `tldw_Server_API/tests/Skills/unit/test_feynman_assets_sync.py`.
- Chat slash Skills commands
  - Added backend `/skills` slash command to list available skills with optional filter support.
  - Added backend `/skill <name> [args]` slash command to execute a specific skill from chat.
  - Added privilege catalog scopes for `/skills` and `/skill` command execution.
  - Added command injection coverage for `/skill` and `/skills` system-mode behavior in `tldw_Server_API/tests/Chat_NEW/integration/test_chat_skill_commands_injection.py`.
- Added regression coverage ensuring Watchlists pipeline setup propagates HTML template format into generated job and output payloads.
- Watchlists guardrails and release-gate coverage:
  - Added static duplicate guard coverage for Watchlists source files (`useWatchlistsStore` selector reuse, duplicate top-level identifiers, duplicate interface keys).
  - Added `test:watchlists:typecheck` package script and wired it into the Watchlists scale gate workflow to prevent silent duplicate/type-regression drift.
  - Added/updated regression coverage for quick-setup candidate preview mocking, accessibility toggle labeling contracts, and run-notification polling harness consistency.
- RAG pipeline parity improvements:
  - Added pre-retrieval clarification gating (`clarification_gate.py`) with heuristic-first ambiguity detection and timeout-bounded fallback behavior.
  - Added unified request fields for clarification/dedup controls: `enable_pre_retrieval_clarification`, `clarification_timeout_sec`, and `enable_research_action_dedup`.
  - Added unified-pipeline clarification short-circuit for ambiguous generation requests (`200 OK` with clarifying prompt in `generated_answer` plus clarification metadata).
  - Added research-loop action-signature dedup (web/academic/discussion/local DB) with result reuse and `action_dedup` metadata reporting.
  - Added regression coverage for schema acceptance, clarification short-circuit behavior, clarification metric emission, action-signature dedup (unit + integration), and `/api/v1/rag/features` contract updates.
- Notes moodboard remediation coverage:
  - Added backend regression assertions for paged moodboard note listing and true `total` reporting in:
    - `tldw_Server_API/tests/Notes_NEW/unit/test_notes_moodboard_db.py`
    - `tldw_Server_API/tests/Notes_NEW/integration/test_moodboards_api.py`
  - Added/expanded frontend stage42 moodboard tests covering lazy moodboard fetch behavior and moodboard pagination controls/navigation.
- Slack/Discord endpoint modularization follow-up:
  - Added dedicated OAuth/admin handler modules:
    - `tldw_Server_API/app/api/v1/endpoints/slack_oauth_admin.py`
    - `tldw_Server_API/app/api/v1/endpoints/discord_oauth_admin.py`
  - Added delegated route wiring in primary endpoint modules while preserving existing API paths/function names and test monkeypatch compatibility.
- Jobs notifications abstraction hardening:
  - Added `JobManager.list_job_events_after(...)` to centralize event-stream retrieval behind Jobs core abstractions.
  - Added/updated bridge regression coverage to validate abstraction-backed notifications event processing behavior.
- Reminders/notifications review-remediation coverage:
  - Added API/DB regressions for scheduler-managed PATCH field rejection, dismissed-notification list filtering, reminders scheduler failure logging, and snooze reconciliation behavior.
- Writing Playground Phase-1 UI and diagnostics experience:
  - Added modular Writing Playground structure components (`WritingPlaygroundShell`, `WritingPlaygroundLibraryPanel`, `WritingPlaygroundEditorPanel`, `WritingPlaygroundInspectorPanel`) with tabbed inspector routing for Generation, Planning, and Diagnostics.
  - Added dedicated diagnostics UI components (`WritingPlaygroundDiagnosticsPanel`, `WritingPlaygroundResponseInspectorCard`, `WritingPlaygroundTokenInspectorCard`, `WritingPlaygroundWordcloudCard`) with shared diagnostics prop contracts.
  - Added utility helpers and coverage for diagnostics state summarization and responsive layout classification.
  - Added extension E2E coverage for inspector tab keyboard navigation and editor-content persistence across tab switches in `apps/extension/tests/e2e/writing-playground-themes-templates.spec.ts`.

### Changed
- Repo2Txt implementation hardening updates after review:
  - Repo2Txt page state now subscribes to the vanilla Zustand store via `useStore` instead of mirroring store state with local React state copies.
  - Token counting now uses `gpt-tokenizer` in Repo2Txt tokenizer worker paths (with guarded fallback behavior).
  - Repo2Txt user-facing strings now resolve through `useTranslation` + `option:repo2txt.*` keys instead of hardcoded English copy.
  - Repo2Txt output file fetching now uses bounded concurrency with progress status updates instead of unbounded `Promise.all` fanout.
- CI gate classification now computes `coverage_required` via dedicated coverage globs instead of mirroring `backend_changed`, preserving backend gate behavior while allowing workflow-only exclusions.
- Media ingestion compatibility reduction (phase 1):
  - Added shared endpoint helpers for compatibility patchpoints and input contracts (`compat_patchpoints.py`, `input_contracts.py`).
  - Added explicit media deprecation signaling helper (`deprecation_signals.py`) and wired additive deprecation headers for legacy `urls=[""]` sentinel compatibility flows on process endpoints.
  - Marked media legacy shim surface as adapter-only (`LEGACY_MEDIA_SHIM_MODE = "adapter_only"`).
  - Added regression tests for deprecation signals, compatibility patchpoints, input contract matrix, and shim adapter-only contract.
- Writing Playground settings parsing now validates `basic_stopping_mode_type` via a typed supported-mode allowlist in payload normalization.
- Workspace snapshot lifecycle now fully persists banner state across create/switch/duplicate/archive/restore/import/export pathways.
- Workspace bundle schema now includes `workspaceBanner` and preserves banner state on zip/json import/export.
- Cross-tab conflict detection now tracks `workspaceBanner` as an explicit conflict field.
- Storage recovery logic now evicts archived banner images before more destructive workspace eviction steps.
- Persistence diagnostics now surface a dedicated `workspaceBanner` byte section.
- ZIP import parsing now supports environments without `File.arrayBuffer()` via a safe fallback reader path.
- Qwen base URL resolution precedence is now explicitly ordered as:
  request `base_url` -> `QWEN_BASE_URL` -> config `qwen_api.api_base_url` -> region preset.
- Model Studio base URL resolution now uses region presets when explicit base URL overrides are unset.
- Model Studio payload construction was refactored to remove duplicate model/extra-parameter logic across sync and async builders.
- Env-var and image setup docs were updated for Model Studio and Qwen routing, including grouped readability improvements for `[Image-Generation]` key listings.
- Feynman skill/prompt assets now use a shared `protocol_version` marker key (replacing mixed legacy markers).
- Importable `feynman-technique-template` now defaults to `disable-model-invocation: true` to avoid duplicate auto-invocable behavior after import.
- Watchlists pipeline contract now carries selected template format (`md` or `html`) through job defaults and output-create payload construction instead of forcing Markdown.
- Watchlists source delete confirmation copy now reuses a single locale key across standard and in-use delete flows to reduce translation-key duplication.
- Watchlists guided quick setup now uses a single audio preference field (`includeAudioBriefing`) across telemetry, preview copy, and submission flow instead of dual field state.
- RAG features/capabilities surfaces now advertise pre-retrieval clarification and research action-dedup support (`/api/v1/rag/features`, `/api/v1/rag/capabilities`, API guide, and capabilities docs).
- Chat command routing now passes Skills command context from the chat endpoint into the command router so `/skill` execution can resolve and run skills consistently.
- Route-toggle policy now allows environment override application during test mode (`is_test_mode()`), preventing collection-order drift between explicit pytest runtime markers and test-mode configuration.
- Notes moodboard API/DB behavior:
  - Moodboard note listing now performs SQL-level union/collapse/pagination for manual + smart-rule membership instead of full in-memory merge/sort/slice.
  - Moodboard notes endpoint now reports a true `total` via `count_moodboard_notes(...)` while preserving `count` for the current page.
  - Moodboard endpoint signatures now include explicit return type hints, helper docstrings, and normalized function signature formatting.
  - Async moodboard endpoints now offload synchronous DB operations via `asyncio.to_thread` to avoid event-loop blocking.
- Notes moodboard WebUI behavior:
  - Moodboard query fetching is now gated to active moodboard view (`listMode=active` + `listViewMode=moodboard`) instead of eager active-mode fetches.
  - Moodboard list retrieval now iterates paged API requests and deduplicates IDs, removing the practical fixed-size fetch cap behavior.
  - Moodboard view now includes local pagination controls (page-size selector, prev/next, summary/index) backed by paged API requests.
  - Moodboard rename/delete expected-version handling now uses the simplified `selectedMoodboard.version ?? 1` path.
- Slack/Discord endpoint organization:
  - `slack.py` and `discord.py` now delegate OAuth/admin routes to focused modules, reducing endpoint-file size and separating routing from lifecycle/policy internals.
- Notifications/reminders endpoint behavior:
  - Notifications snooze now delegates task creation to `RemindersService` and performs immediate best-effort scheduler reconciliation.
  - Reminders create/update/delete endpoints now keep best-effort scheduling semantics but log reconciliation/unschedule failures instead of silently suppressing them.
  - Notifications SSE/list-window query helpers now align on active-inbox semantics by excluding dismissed rows from non-archived query paths.
  - Notifications endpoint handlers and reminders/notifications schemas now include expanded docstrings and explicit SSE generator return typing for stronger API-contract readability.
- Jobs notifications bridge:
  - `jobs_notifications_service` now consumes Jobs manager event-list APIs instead of in-service raw `job_events` SQL queries.
- Frontend/watchlists CI gate execution paths were re-baselined for deterministic branch runs:
  - Frontend UX gates now use a stable Bun-based dependency/install + Playwright invocation flow with a single all-pages smoke gate entrypoint.
  - Watchlists extension strict gate flow now preserves explicit launch/target wait timeout controls and aligns with the stable extension-launch helper contract.
  - All-pages smoke gate behavior now follows the stabilized route traversal baseline used by the current release-gate suite.
- Writing Playground UI interaction behavior:
  - Moved template/theme/chat-mode and context controls from the Generation inspector view into Planning for clearer IA separation.
  - Added compact-mode shell grid overrides plus `data-testid` layout markers to improve narrow-layout behavior and regression observability.
- Improved inspector keyboard interaction to support Arrow/Home/End traversal with active-tab focus movement.
- RAG streaming/profile parity hardening (PR #796 follow-up):
  - Streamed agentic retrieval now resolves strategy and retrieval/generation knobs from profile-aware `effective_payload` defaults instead of raw request-only fields.
  - Async generation paths now warm RAG prompt templates via event-loop-safe thread offload before generator instantiation.
  - Two-tier reranker runtime degradation now logs the underlying exception prior to profile degradation fallback (`two_tier` -> `hybrid`).
  - Added/updated regression coverage for stream parity and prompt-loader warmup behavior, including typed test signatures/docstrings in touched RAG test modules.

### Removed
- No removals in this session.

### Fixed

- Fixed Repo2Txt local source selection reliability:
  - Replaced ambiguous local multi-file picker with explicit directory (`webkitdirectory`) and zip pickers.
  - Added local duplicate-filename collision guard when directory context is unavailable.
  - Reset local file input value after selection to allow repeat-selection workflows.
- Fixed Repo2Txt extension reliability for GitHub provider by declaring `https://api.github.com/*` in extension `host_permissions`.
- Fixed Repo2Txt tokenizer worker hang risk by adding worker `onerror`/`onmessageerror` handling, per-request timeouts, pending-request cleanup, and worker recovery re-init.
- Fixed portability gaps in the Repo2Txt implementation plan by replacing machine-specific absolute paths with `<repo_root>` / `<repo_worktree>` placeholders.
- Fixed Lima strict-policy contract gaps:
  - Rejected unsupported `allowlist` strict mode until enforcement support exists, removing false-positive strict capability advertisement.
  - Added foreground execution-time preflight failure handling so Lima policy/preflight failures mark runs failed consistently (matching background behavior).
  - Removed unconditional Docker fallback suggestion behavior for Firecracker runtime-unavailable paths.
  - Prevented WSL/Windows readiness override bypass in Lima enforcer preflight checks.
- Fixed CI gate/classifier runtime issues by repairing `Helper_Scripts/ci/path_classifier.py` syntax and output wiring.
- Fixed frontend gate parsing failure in `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx` caused by malformed basic stopping mode expression.
- Fixed Full Suite multi-user startup regressions (`relation "organizations" does not exist`) by extending PostgreSQL bootstrap schema coverage to include organization/team dependencies.
- Fixed tokenizer resolution error precedence so non-native providers no longer surface misleading provider-native configuration errors when tokenizer lookup is unavailable.
- Fixed `modelstudio_image_mode=auto` to actually do sync-first fallback to async.
- Fixed validation so `payload.extra_params.mode` is accepted for Model Studio control flow without requiring passthrough allowlisting.
- Fixed user-facing error hygiene by sanitizing Model Studio transport exception details while logging internals.
- Fixed potential SSRF path by validating response-provided remote image URLs against egress policy/allowlist before fetch.
- Fixed `modelstudio_image_region` no-op behavior by wiring it into endpoint selection.
- Fixed missing docstring on `_coerce_choice` in image generation config helpers.
- Fixed bundle import compatibility in environments lacking `File.arrayBuffer()`.
- Fixed invalid imported banner image payload handling to fail soft (drop bad image, preserve banner text fields).
- Fixed a Watchlists UC2 mismatch where selecting an HTML template could still generate Markdown outputs due to hardcoded `default_format`/`format`.
- Fixed duplicate English locale maintenance drift for source delete undo-window messaging by consolidating copy to `sources.deleteConfirmDescription`.
- Fixed newly added Watchlists API audio-briefing tests to comply with typing/docstring standards by adding explicit fixture type hints, return annotations, and test docstrings.
- Fixed Watchlists static-guard false positives by distinguishing type/value namespaces and limiting interface-property scans to top-level declarations.
- Fixed Watchlists quick-setup review/test-flow regressions caused by stale CTA label assumptions and missing source-preview service mocks in the Overview test suite.
- Fixed Watchlists a11y/load-retry test drift for Sources/Monitors toggles by aligning assertions with row-context ARIA labels and current table-render behavior.
- Fixed redundant iterative research-loop calls for equivalent action/query signatures by reusing previously successful action outputs.
- Fixed `/skill` command error handling to convert unexpected skill-resolution/runtime exceptions into a structured command error response instead of bubbling unhandled exceptions.
- Fixed `tldw_Server_API/tests/Skills/integration/test_skills_api.py` startup abort risk by setting minimal test-route env toggles early and lazily importing `app.main` within the fixture, avoiding heavy module side effects at collection time.
- Fixed moodboard create endpoint robustness by handling unexpected `None` ID returns before integer conversion.
- Fixed moodboard docs path placeholder inconsistency by using `{moodboard_id}` for `GET /moodboards/{moodboard_id}/notes`.
- Fixed reminder task PATCH safety by removing scheduler-owned fields (`last_status`, `next_run_at`, `last_run_at`) from public update schema to prevent user-driven scheduler state corruption.
- Fixed dismissed-notification inbox consistency by excluding dismissed rows from non-archived list/stream query paths, aligning unread-count semantics with visible inbox rows.
- Fixed hidden scheduler-sync failures in reminders task endpoints by surfacing non-critical reconcile/unschedule exceptions in warning logs.
- Fixed strict watchlists no-skip gate regressions caused by stale/undefined local assertions in E2E specs and aligned those checks to deterministic harness behavior.
- Fixed extension E2E launch instability in CI by removing unsupported forced Playwright channel behavior and retaining explicit timeout override controls in workflow env.
- Fixed UX smoke gate branch instability by restoring the stabilized all-pages gate command path and traversal behavior expected by current frontend release gates.
- Fixed Writing Playground diagnostics prop leakage by removing `enabled` from card-prop spreads before passing props into inspector card components.
- Fixed monitoring metrics-history range interactions to avoid redundant API loads on manual range selection/apply and to avoid interval resets/reloads while editing draft custom-range inputs.
- Fixed monitoring custom-range UX by clearing range validation errors when users edit custom range start/end values.


## [0.1.24] 2026-02-22

### Added

- Persona memory namespace resilience improvements:
  - Added deterministic persistent fallback namespace keying when `scope_snapshot_id` is missing:
    - `persistent_fallback_sid_<sha256(session_id)>` in persona memory integration.
  - Added deterministic legacy persistent namespace mapping for older null-scope entries:
    - `persistent_legacy_pid_<sha256(user_id:persona_id)>`.
  - Added DB-level namespace backfill helper:
    - `backfill_persona_memory_scope_namespace(...)` on `CharactersRAGDB`.
- Persona regression coverage additions:
  - Added persistent fallback namespace isolation test for missing-scope persistent sessions.
  - Added retrieval-triggered legacy null-scope backfill test for persistent persona memory entries.
  - Added DB unit test verifying selective backfill behavior (only missing-scope rows, optional missing-session gate).
- Locale rollout completion for persona unsaved-draft prompts:
  - Added localized `persona.unsavedState*` prompt copy for remaining non-English sidepanel locales:
    - `ar`, `da`, `fa`, `it`, `ko`, `ml`, `no`, `pt-BR`, `ru`, `sv`, `uk`, `zh-TW`.
- Added implementation-plan docs for this session’s persona slices (stage 23 locale rollout, stage 24 namespace fallback, and stage 25 legacy namespace backfill).
- Watchlists UX IA/onboarding improvements:
  - Aligned user-facing Watchlists terminology for Activity/Reports across tabs, overview cards, and help labels.
  - Added always-visible Watchlists task shortcuts (`Set up feeds`, `Configure monitors`, `Check activity`, `Review articles`, `View reports`) for direct navigation.
  - Expanded guided quick setup with goal selection (`Generate briefing reports` vs `Feed review only`) and destination routing to Reports when briefing setup completes without run-now.
  - Refreshed guided-tour copy to explicitly explain feed -> monitor -> activity -> articles -> reports pipeline relationships.
  - Added onboarding telemetry capture for quick setup and guided tour funnels to measure step completion and drop-off.
- Watchlists briefing/audio delivery UX improvements:
  - Added audio briefing controls in monitor configuration for voice, speed, and target duration with safe range normalization.
  - Added monitor authoring modes (`Basic` vs `Advanced`) to reduce first-run form complexity while preserving advanced state in payloads.
  - Added template editor authoring modes (`Basic` vs `Advanced`) to separate focused editing from expert tools (snippets/docs/version tools).
  - Added Basic-mode template recipe builder with structured presets (`briefing_md`, `newsletter_html`, `mece_md`) and one-click recipe application.
  - Added explicit warning when advanced monitor settings are hidden by switching back to Basic mode.
  - Added advanced filter scaffolding with per-rule plain-language summaries, regex examples/flags, and inline regex validation feedback.
  - Added early `Preview impact` action in filter authoring to expose sample ingestion outcomes sooner in the workflow.
  - Added advanced cron scaffolding with five-field guidance, invalid-format hints, and one-click schedule examples (`Daily 09:00`, `Weekdays 08:00`, `Every 6 hours`).
  - Added monitor/template authoring telemetry for start, mode-switch, save, Basic-step completion, and recipe-application adoption tracking.
  - Added a unified Watchlists overview health model with cross-tab attention badges (Feeds/Activity/Reports) and direct "attention needed" deep links from Overview.
  - Extended watchlists output prefs typing to include audio briefing fields used by backend workflow orchestration.
  - Improved Outputs artifact visibility with explicit Markdown/HTML/Audio badges in table rows.
  - Added binary-safe audio output download support to prevent corrupted MP3 delivery.
  - Added in-drawer audio playback for generated audio outputs alongside existing text/HTML preview flows.
  - Hardened live RSS E2E briefing tests to current Watchlists response contracts (`run.stats.*`, source-preview `total/items`) and validated UC2 real-feed flow end-to-end.
  - Added regression coverage for audio monitor payload behavior, output artifact metadata classification, and drawer audio preview rendering.
- Watchlists reliability/remediation UX improvements:
  - Standardized monitor/source/run error remediation copy via shared `mapWatchlistsError` usage in source test preflight, run detail load failures, and monitor save fallback handling.
  - Added DNS/TLS-specific remediation mapping and locale keys to reduce ambiguous failure handling.
  - Added direct retry affordances on source preflight and run-detail load failure surfaces with regression coverage.
  - Added grouped run notification fan-in with dedupe and deep-link payloads to prevent repeated toast spam during multi-run failure bursts.
  - Added stalled-run detection in notification polling and a persistent Runs-tab reliability attention banner with direct run/filter actions.
- Watchlists article workflow and scale guardrail improvements:
  - Added explicit route query handoff for Watchlists tab/filter deep links (`tab`, source/status/smart/search, job/run/output IDs) so cross-tab context is restorable and testable.
  - Added regression coverage for article-to-monitor/run/reports handoff and include-in-next-briefing action paths.
  - Added shared watchlists scale guardrail metadata (`WATCHLISTS_SCALE_GUARDRAILS`, `WATCHLISTS_SCALE_SCENARIOS`) and wired perf/load tests to those constants for repeatable threshold tracking.
- Watchlists accessibility and inclusivity hardening (Plan 06 closeout):
  - Completed staged accessibility delivery for keyboard focus restoration, screen-reader live announcements, non-color status encoding, and plain-language onboarding/scheduling/template guidance.
  - Added release-gate checklist pass-rate tracking for Watchlists accessibility categories and recorded residual non-critical/manual review items.
  - Confirmed accessibility regression matrix remains green in touched flows with no critical keyboard/SR defects.
- Watchlists coordinated UX program closeout:
  - Added a cross-stream closeout report with verification summary and owned deferred backlog.
- Chat rich-text regression coverage additions:
  - Added `st_compat` LaTeX rendering regression test for inline and block math in:
    - `apps/packages/ui/src/utils/__tests__/chat-rich-text.test.ts`

### Changed

- Persona locale parity test enforcement:
  - Updated sidepanel persona locale test to require non-English localized copy for all non-English locale bundles (not only priority locales).
- Persona WS auth test harness/runtime stability:
  - Updated Persona WS auth tests to mount a local `FastAPI` app with persona router instead of importing `app.main`.
  - Updated API key manager test doubles to align with current auth validation call signature (`required_scope` support).
- Persona persistent-memory retrieval behavior:
  - Retrieval path now performs legacy namespace reconciliation/backfill for persistent mode when runtime scope is missing, while preserving explicit scope/session namespace behavior.
- Chat rich-text rendering pipeline:
  - Updated `st_compat` rendering to use a dedicated `Marked` instance with KaTeX extension wiring (matching existing markdown math support expectations).

### Removed

- No removals in this session.

### Fixed

- Fixed persistent-scoped persona memory retrieval blackhole cases when `scope_snapshot_id` is absent by introducing deterministic fallback namespace keying.
- Fixed backward compatibility for older persistent persona memory rows with null `scope_snapshot_id` by adding deterministic legacy namespace mapping and scoped backfill.
- Fixed Persona full-suite runtime instability caused by heavy app import side effects in auth tests.
- Fixed Persona auth test contract drift with API key validation scope handling.
- Fixed Watchlists OPML bulk import preflight handling for wrapped upload objects (`originFileObj`), restoring `SourcesBulkImport.preflight-commit` test coverage.
- Fixed Watchlists admin runs wrapper route contract by restoring redirect behavior to `/admin/server`.
- Fixed Watchlists suite runtime abort path in constrained environments by stubbing heavy STT imports (`torch`/`faster_whisper`/`transformers`) in Watchlists test setup.
- Fixed missing LaTeX rendering in `st_compat` chat rich-text mode by enabling KaTeX processing for the `marked`-based render path.

## [0.1.23] 2026-02-22

### Added

- OpenAI OAuth BYOK account-linking support (opencode-style) under existing user key routes:
  - Added `POST /api/v1/users/keys/openai/oauth/authorize`.
  - Added `GET /api/v1/users/keys/openai/oauth/callback`.
  - Added `GET /api/v1/users/keys/openai/oauth/status`.
  - Added `POST /api/v1/users/keys/openai/oauth/refresh`.
  - Added `DELETE /api/v1/users/keys/openai/oauth`.
  - Added `POST /api/v1/users/keys/openai/source` for active credential source switching.
- OpenAI OAuth UI linkage in settings:
  - Added OpenAI account-linking card actions for `Connect OpenAI`, `Check status`, `Refresh`, `Disconnect`, and `Use API Key Instead`.
  - Added frontend API client methods/types and OpenAPI guard entries for the new OpenAI OAuth routes.
- OAuth/credential lifecycle coverage additions:
  - Added SQLite state-repository cap enforcement coverage for outstanding OAuth state rows.
  - Added frontend service-level regression coverage for OpenAI OAuth endpoint contracts.
- Workspace account storage wiring in `/workspace-playground`:
  - Added authenticated client support for `GET /api/v1/users/storage`.
  - Added account quota fallback via `GET /api/v1/users/me/profile?sections=quotas`.
  - Added workspace header support for account usage/quota alongside local payload usage.
- Quick Chat pop-out context state support:
  - Added source-route serialization/restore for pop-out sessions so docs-mode retrieval and guide routing remain page-aware.
  - Added dedicated pop-out state normalization/validation helper for safer session payload parsing.
- Workspace persistence architecture upgrades:
  - Added split-key persistence (`index` + per-workspace `snapshot`/`chat`) with legacy monolith migration.
  - Added IndexedDB offload path for heavy chat/artifact payloads with localStorage pointer metadata.
  - Added rollout controls for split-key and IndexedDB offload (env + localStorage feature flags).
  - Added development diagnostics for workspace persistence payload size/write-count tracking.
  - Added design documentation: `Docs/Design/Workspace_Persistence_Architecture.md`.
- Expanded regression coverage:
  - Added API client tests for `/api/v1/users/storage`.
  - Added workspace persistence tests for split-key migration/fallback, IndexedDB flag-off behavior, chat-retention bounds, and server-backed artifact truncation.
  - Added backend user-profile quota tests for live storage override and fallback behavior.
  - Added Quick Chat regression tests for pop-out state parsing, sidepanel tutorial resolution behavior, and workflow-card parsing without explicit `routeLabel`.
- Added and completed implementation plans for workspace UI refresh and workspace storage efficiency rollout.
- Request-core security regression coverage:
  - Added tests for default-deny absolute URL behavior, explicit absolute-origin allowlist behavior, and no-auth-header behavior for allowlisted absolute requests.
- HTTP egress DNS pinning regression coverage:
  - Added tests for egress resolved-IP override behavior and resolved-IP propagation in policy evaluation.
  - Added HTTP client tests ensuring per-request host resolution is pinned across repeated egress checks and redirect hops.
- Watchlists selector safety hardening:
  - Added guardrails for user-controlled selector expressions so overly complex XPath/CSS rules are rejected during both selector validation and runtime selection.
  - Added environment-configurable selector limits:
    - `WATCHLIST_SELECTOR_MAX_EXPR_LEN`
    - `WATCHLIST_SELECTOR_MAX_XPATH_DESCENDANT_STEPS`
    - `WATCHLIST_SELECTOR_MAX_XPATH_PREDICATES`
    - `WATCHLIST_SELECTOR_MAX_XPATH_FUNCTION_CALLS`
  - Added regression coverage for complex-selector rejection and environment override behavior.
- Extension bundle compatibility hardening:
  - Added a bundle-contract test to ensure `TemplateCodeEditor` does not import `next/dynamic`, preventing extension build regressions in non-Next runtimes.
- Media search remediation regression coverage:
  - Added backend search-contract tests verifying `/api/v1/media/search` forwards `boost_fields` and relevance ordering changes under field-weight differences.
  - Added frontend integration coverage for metadata-mode constrained search path serialization and metadata-result snippet rendering.
  - Added frontend integration coverage for `ViewMediaPage` no-results recovery wiring (`clear search`, `clear filters`, `quick ingest` event dispatch).
  - Added/confirmed request-churn guard coverage for debounced rapid typing with active filters.

### Changed

- OpenAI OAuth runtime retry semantics in provider call paths:
  - Chat, embeddings, and audio OpenAI OAuth `401` handling now force-refreshes once and retries once.
  - If refresh/retry fails, handlers now propagate the original auth-class error (strict plan) instead of converting to reconnect-required payloads.
- OpenAI OAuth observability:
  - Added `byok_oauth_401_retry_total{provider,outcome}` outcome instrumentation for chat, embeddings, and audio retry flows.
- OpenAI OAuth endpoint contract hardening:
  - OAuth authorize tests now require and verify configured redirect URI propagation in the generated authorization URL.
- Workspace Playground layout and interaction model:
  - Refined shell hierarchy and status signaling.
  - Converted empty-state examples into actionable prompt chips that seed composer input.
  - Reduced composer control clutter while preserving mode/model/advanced controls.
  - Kept composer anchored at the viewport bottom with transcript scroll region above it.
  - Improved center-pane width reflow as sidebars collapse (`comfortable`/`expanded`/`full` behavior).
  - Removed residual max-width caps that left visible unused right-side space.
- Storage indicator clarity and policy:
  - Updated header copy to `Capacity Payload ... | Account ... | Browser ...` with clearer tooltip wording.
  - Replaced hardcoded 5 MB payload budget with env-configurable workspace payload budget:
    - `VITE_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB`
    - `NEXT_PUBLIC_WORKSPACE_STORAGE_PAYLOAD_BUDGET_MB`
  - Updated payload estimation to include both monolithic and split-key workspace storage entries.
- Workspace persistence efficiency/safety:
  - Persisted chat sessions now retain the most recent 250 messages per workspace.
  - Oversized server-backed artifact payloads are truncated/stripped in local persistence while preserving server-backed references.
- Characters manager table density styling:
  - Refactored Ant Design table density selectors to shared `.ant-table-cell` targeting and centralized common vertical alignment for easier maintenance.
- Backend quota sourcing:
  - User profile quota assembly now prefers live storage metrics from `calculate_user_storage(...)` for `storage_used_mb` and `storage_quota_mb`, with fallback to stored user values on failure.
- Quick Chat guide/tutorial behavior:
  - Tutorial registry lookups now support an explicit suppression bypass for call sites that need route-scoped tutorial discovery outside options-runtime defaults.
  - Browse Guides per-page tutorial lookup now opts into this explicit bypass for sidepanel-hosted Quick Chat.
  - Workflow-card parsing now treats `routeLabel` as optional and derives fallback labels from route paths when omitted.
  - Quick Chat workflow-card settings hint text now matches validation semantics (`title/question/answer/route` required; `id/routeLabel/tags` optional).
- Structured-output parser documentation:
  - Added module/function docstrings in `tldw_Server_API/app/core/LLM_Calls/structured_output.py` clarifying candidate parsing order and strict/lenient semantics for shared endpoint and claims parsing paths.
- Request core transport hardening:
  - Absolute `http(s)` request targets are now blocked by default unless their origin is explicitly included in `absoluteUrlAllowlist`.
  - Absolute URL requests now always skip auth injection at the core helper level (not only in background-proxy fallback paths).
  - Multi-user token refresh retry logic is now bypassed for forced no-auth absolute URL requests.
- HTTP client egress hardening:
  - Egress policy evaluation now supports passing pre-resolved IPs and returning resolved IPs for callers that need consistent host checks.
  - HTTP client retry/redirect loops now maintain a per-request DNS pin cache so subsequent checks for the same host reuse the initial resolved IP set.
- Media search backend relevance/scoping behavior:
  - `/api/v1/media/search` now threads `boost_fields` into DB search execution instead of silently ignoring weights.
  - SQLite relevance sort now applies weighted BM25 scoring when `boost_fields` are supplied.
  - PostgreSQL relevance sort now applies explicit weighted `ts_rank(...)` scoring when `boost_fields` are supplied.
  - Metadata search (`/api/v1/media/metadata-search`) now applies optional standard constraints server-side before pagination (`q`, `media_types`, include/exclude keywords, date range, sort).

### Removed

- No removals in this session.

### Fixed

- Fixed OpenAI OAuth retry behavior drift from design by enforcing strict original-auth-error propagation after failed refresh+retry in chat, embeddings, and audio paths.
- Fixed OpenAI OAuth endpoint test contract drift by aligning redirect URI requirements and assertions for both SQLite and PostgreSQL endpoint suites.
- Fixed missing frontend linkage for OpenAI OAuth account controls in model settings by wiring API client methods, route guards, and action buttons.
- Fixed workspace chat pane width reclaim issues when side panes were collapsed.
- Fixed storage usage ambiguity by separating workspace payload budget, account quota usage, and browser-origin storage usage in the UI.
- Fixed Quick Chat pop-out context loss where docs-mode and Browse Guides used `/quick-chat-popout` instead of the originating page route.
- Fixed Browse Guides `Tutorials for this page` empties in sidepanel-hosted Quick Chat by enabling targeted route lookup for tutorial cards.
- Fixed workflow-card validation/documentation mismatch by accepting valid cards that omit `routeLabel`.
- Fixed an SSRF-adjacent request flow in shared request-core by enforcing default-deny absolute URL policy and explicit allowlist gating.
- Fixed DNS rebinding exposure in server-side outbound HTTP by enforcing single-resolution IP pinning per host within a request lifecycle.
- Fixed a watchlists XPath selector DoS risk by enforcing complexity limits on user-controlled selectors before compile/evaluation.
- Fixed extension build/runtime compatibility for watchlist template editing by replacing `next/dynamic` usage in `TemplateCodeEditor` with `React.lazy` + `Suspense`, unblocking `quick-chat-guides-tutorials.spec.ts` Playwright runs.
- Fixed media-search advanced controls drift where `boost_fields` UI controls had no backend relevance effect.
- Fixed metadata-mode pagination/total-authority drift by enforcing server-side standard constraints before result pagination.
- Fixed `ViewMediaPage` no-results recovery contract by wiring clear-search, clear-filters, and quick-ingest action dispatch through the page-level flow.

## [0.1.22] 2026-02-22

### Added

- Quick Chat helper documentation-assistant expansion:
  - Added a docs-focused Q&A workflow path for project-documentation guidance.
  - Added/expanded pre-written workflow Q&A cards for guided “how do I do X?” discovery.
  - Added per-page `Tutorials for this page` section in Quick Chat Browse Guides (distinct from Q&A cards) with start/replay/locked states.
  - Added strict-scope docs profile defaults for RAG Q&A (`media_db` + `project_docs`) with scope controls in Settings (`quickChatStrictDocsOnly`, `quickChatDocsIndexNamespace`, `quickChatDocsProjectMediaIds`).
- Guided tutorial rollout (P1 routes) available from Help modal + Quick Chat Browse Guides:
  - `/prompts` (`prompts-basics`)
  - `/evaluations` (`evaluations-basics`)
  - `/notes` (`notes-basics`)
  - `/flashcards` (`flashcards-basics`)
  - `/world-books` (`world-books-basics`)
  - Legacy alias routing support for `/options/*` tutorial entry paths.
- New tutorial selector/test contracts:
  - Evaluations: stable tutorial target anchors for page title, tabs, create button, list/detail cards.
  - World Books: stable tutorial target anchors for shell, search/filters, table, new/import actions.
- Workspace Playground context controls:
  - Added option to include full file contents in chat context for tasks like synopsis/summarization questions.
- Validation/test assets:
  - Added dedicated Quick Chat tutorials validation spec and expanded tutorial/registry/unit coverage.
  - Added manual QA checklist and PR/release notes for the P1 tutorials rollout.
- Published docs IA and release-notes additions:
  - Added curated User Guides taxonomy under published docs (`Server`, `WebUI_Extension`, `Integrations_Experiments`) with updated guide index routing.
  - Added published release notes page (`Docs/Published/RELEASE_NOTES.md`) and wired it into docs nav.
  - Added targeted Playwright validation coverage updates for character selection/default bootstrap and options first-run onboarding contracts.

### Changed

- Workspace Playground conversation surface behavior:
  - Page now extends downward with conversation growth up to a max threshold, then switches to scrollable conversation behavior.
- Character preview UX:
  - Character preview modal now shows full description and full tag set rather than obscured/truncated content.
- Options first-run UX test contracts:
  - Updated Playwright coverage to align with onboarding-first shell and current landing-hub “Start Chatting” flows.
- Documentation/user-guide coverage:
  - Added/updated guidance on Quick Chat docs assistant usage, tutorial discovery, and editing pre-written workflow/Q&A cards.
- Documentation cross-link normalization for published docs:
  - Repointed published guide/API/code-documentation links from non-published paths to published equivalents.
  - Replaced source-file local path links that do not publish in MkDocs with stable GitHub source links where appropriate.
- MkDocs nav completeness updates:
  - Added previously orphaned published pages to nav (`Watchlists API`, `VoiceAssistant Module`, `Doc Researcher Features`, `Generated Files Storage Code Guide`, `Sidecar Workers.template`, `Firecracker Host Checklist`).
  - Added explicit “how to add/edit tutorials” developer workflow and “how to edit pre-written workflow cards” user workflow documentation.

### Removed

- No removals in this session.

### Fixed

- Fixed free-floating notes modal open behavior in WebUI/extension where the modal would open then shrink to minimum size.
- Fixed character-page pagination instability that could jump/skip back to page 1.
- Fixed character image rendering in both character grid cards and selected character detail views.
- Fixed character list response handling by normalizing list envelopes (`items`, `characters`, `results`, `data`) to prevent missing-character selector states.
- Fixed default-character bootstrap behavior in Options Playground so server defaults preselect correctly and manual override persists across reload.
- Fixed Playwright flow instability around modern onboarding/tutorial overlays and drawer interception in character-selection scenarios.
- Playwright validation hardening completed with targeted suite passing:
  - `playground-character-selection.spec.ts` (green)
  - `options-first-run.spec.ts` (green)
  - `quick-chat-guides-tutorials.spec.ts` + `page-review.spec.ts` + `options-first-run.spec.ts` combined run (`68 passed`)
- Media ingestion batch-processor shim resolution: `process_batch_media` now detects stale endpoint wrapper caches and prefers the live core audio/video processor callable when wrappers are out of sync.
- Restored order-independent behavior for MediaIngestion_NEW regression tests that monkeypatch core processors (claims toggle integration and metadata/pre-check contract paths) by preventing fallback to real network/file audio processing during those tests.
- Dictation reliability and diagnostics across WebUI `/chat` and extension sidepanel:
  - Hardened extension dictation fallback E2E by forcing taxonomy-classified server STT failures through the extension upload transport path.
  - Added `/chat` integration coverage for dictation mode routing (`server` vs `browser`) and transcript insertion behavior.
  - Added sanitized dictation diagnostics event schema (`tldw:dictation:diagnostics`) with explicit privacy guarantees (no transcript/prompt/audio payload content).
  - Fixed STT health false-negatives for non-Whisper providers by making model-status checks provider-aware (`parakeet`, `canary`, `external`, `qwen2audio`, `qwen3-asr`, `vibevoice` vs `whisper` local-model preflight).
  - Changed STT health payload semantics to include `usable` and `on_demand`, and updated Whisper `warm=true` behavior to mark warmed models as ready.
  - Fixed dictation gating in WebUI/extension (`useTldwAudioStatus`) to fail-open for non-Whisper provider readiness probes so dictation is not disabled when server transcription remains usable.
  - Added targeted regression coverage for STT health `usable` semantics and non-Whisper dictation health fail-open behavior (backend pytest + UI Vitest).
- MkDocs strict-build blockers for published documentation:
  - Fixed missing-nav-entry blockers by including all published pages reported as out-of-nav.
  - Fixed broken published links to removed/non-published `User_Guides` and `Development` paths.
  - Fixed stale table-of-contents anchors (`#...--...`) that no longer matched generated heading IDs.
  - Strict build now reports zero missing-nav and zero broken-anchor info warnings in the docs scope.

## [0.1.21] 2026-02-21

### Added

- New security and regression tests for:
  - Scheduler payload format/ref validation and legacy compatibility behavior.
  - Web dedupe JSON persistence and controlled legacy migration.
  - Placeholder-service guardrails and route isolation.
  - Loguru formatting guardrail.
- Operations runbook documenting secure defaults, compatibility flags, and migration steps.
- Added a startup/CI route guard that fails fast when duplicate `(path, method)` routes are registered.
- Added regression tests for duplicate-route detection, CORS guardrails, ULTRA_MINIMAL health routing, and OpenAPI CORS behavior.
- Skills
  - End-to-end Skills API workflow tests covering create, list/get, versioned update with supporting-file add/remove, execute preview, context payload, export, zip re-import, delete, and seed flows.
  - Seed endpoint integration test coverage for idempotency (`overwrite=false`) and overwrite restoration (`overwrite=true`).
  - Deterministic unit tests for built-in skill seeding: recursive directory copy (including nested files), no-overwrite preservation, and overwrite replacement behavior.
- Admin Backup Bundles
  - Added API contract coverage for admin bundle import error detail shapes:
  - `restore_failed` responses must return structured `detail` objects (`error_code`, `message`, optional `rollback_failures`).
  - Standard import validation failures (for example checksum mismatches) must continue returning string `detail`.
- Web UI/Extension UX audit gating and regression coverage:
  - Added Stage 2 route-contract smoke spec for audited navigation destinations.
  - Added Stage 3 rendering-resilience smoke spec for max-depth, template-leak, and retry/timeout assertions.
  - Added Stage 4 mobile-sidebar and accessibility smoke specs, plus Axe high-risk route matrix coverage.
  - Added Stage 5 audited-route release gate smoke spec and `e2e:smoke:stage5` script.
  - Added deterministic admin smoke fixture profile/route mocks to remove skip behavior and backend-state dependency.
- UX gate process updates:
  - Added PR checklist items for UX smoke gate, console-budget checks, and WebUI/extension parity validation.
  - Added shared sanitized server-error message utility with correlation-ID log-hint support for user-facing error states.
  - Added dedicated core route identity regression tests for `/`, `/setup`, and `/onboarding-test`.
  - Added media route guard/boundary coverage for `media-multi` and `media-trash` paths.
  - Added knowledge QA golden-layout and interaction guardrails to preserve high-quality search/history UX.
  - Added workspace/playground positive-pattern guardrail tests (Data Tables, Evaluations, Chunking, Workspace desktop layout).
  - Added explicit home theme-toggle control and associated unit/E2E coverage.
- Claims extraction portability hardening (LangExtract-aligned):
  - Centralized claims output parsing and response coercion with strict/lenient modes, fenced-JSON handling, wrapper-key normalization, and structured parse errors.
  - Shared claims runtime configuration and analyze-callback typing modules reused across extraction, ingestion, adjudication, and service paths.
  - New claims telemetry counters for structured output, parsing quality, and fallback/degradation:
    - `claims_response_format_selected_total`
    - `claims_output_parse_events_total`
    - `claims_fallback_total`
  - New claims monitoring dashboard JSON with parse/fallback ratio panels and response-format selection visibility.
  - New claims regression coverage for response-format contracts, strict/lenient API behavior, fallback resilience, parse-failure telemetry, and config precedence.

### Changed

- Normalized Loguru formatting across tldw_Server_API/app from %-style placeholders to {} style.
- Added CI/test guard to prevent reintroduction of %-style Loguru placeholders.
- Sync client defaults now align with server routes (`/api/v1/sync/send` and `/api/v1/sync/get`) and support configurable auth headers (`Authorization` bearer and `X-API-KEY`).
- Moved local LLM manager initialization out of import-time setup into lifespan startup, with lazy initialization behavior where applicable.
- ULTRA_MINIMAL mode now uses control-plane health endpoints only (`/health`, `/ready`, `/health/ready`) to avoid duplicate route ownership.
- Skills
  - `SkillsService.seed_builtin_skills()` now copies full built-in skill directories recursively instead of recreating skills from `SKILL.md` only.
  - Seed overwrite flow now force-syncs registry state and handles pre-existing soft-deleted rows before copy.
- Admin Backup Bundles
  - Admin bundle create/import concurrency control now uses an atomic non-blocking lock path to fail fast with `409 bundle_operation_in_progress` instead of race-prone check-then-acquire behavior.
  - Bundle import disk-space preflight now checks all real write targets (system temp, upload temp location, and live DB target directories from manifest datasets).
  - `retention_hours` is now enforced for bundle creation: expired bundles are pruned after create within the same bundle scope (`user_id` scoped, including global `None` scope).
  - Bundle import success/dry-run payloads now consistently include `rollback_failures` for schema stability.
- Circuit Breaker
  - Unified admin circuit breaker endpoint: `GET /api/v1/admin/circuit-breakers` with RBAC protection (`admin` role + `system.logs` permission), deterministic sorting, and filters (`state`, `category`, `service`, `name_prefix`).
  - Unified admin circuit breaker response contract with explicit `source` values: `memory`, `persistent`, and `mixed`.
  - Cross-worker HALF_OPEN probe lease coordination in shared registry storage, including lease TTL, cleanup, and release-on-completion behavior.
  - Optimistic-lock conflict observability metric: `circuit_breaker_persist_conflicts_total{category,service,operation,mutation}`.
  - New endpoint test coverage for admin circuit breaker listing, filtering, and authorization behavior.
  - Circuit breaker shared-state persistence now uses bounded merge/retry semantics on optimistic-lock conflicts to avoid dropping local mutations under contention.
  - Circuit breaker operator documentation now includes new env knobs and tuning guidance:
    - `CIRCUIT_BREAKER_REGISTRY_MODE`
    - `CIRCUIT_BREAKER_REGISTRY_DB_PATH`
    - `CIRCUIT_BREAKER_PERSIST_MAX_RETRIES`
    - `CIRCUIT_BREAKER_HALF_OPEN_LEASE_TTL_SECONDS`
  - Monitoring/admin docs now explicitly describe `source="mixed"` as expected when persistence is enabled for active in-process breakers.
  - Circuit breaker unification PRD status updated to reflect implemented state and delivered hardening follow-on work.
- Web UI/Extension parity:
  - Shared `WebLayout` now defaults to collapsed rail on mobile (`<768px`) and opens navigation via header drawer toggle.
  - Shared media-query initialization now reads `matchMedia` on first client render to prevent desktop-rail flash on mobile.
  - Section 2 audited wrong-content routes now resolve to intended content or explicit “Coming Soon” placeholders instead of silent misrouting.
  - Changed settings information architecture to be more navigable, searchable, and less badge-saturated, with tighter route-to-page mapping.
  - Changed settings routing to restore/cover `/settings/ui`, `/settings/image-generation`, and `/settings/image-gen` behavior via proper wrappers/alias handling.
  - Changed guardian settings behavior to show explicit unsupported-endpoint guidance instead of noisy repeated backend failures.
  - Changed admin route behavior so admin sub-routes render route-specific content/placeholder contracts instead of collapsing to Server Admin.
  - Changed admin diagnostics presentation to human-readable units for memory/byte and retry-window values.
  - Changed admin error handling to sanitize implementation details and gate controls when prerequisites are missing.
  - Changed chat disconnected-state messaging to remove redundancy and provide one clear user path.
  - Changed chat agent empty-state/workspace affordances to clarify prerequisites and next steps.
  - Changed chat mobile composer/toolbars to enforce 44x44 touch targets and improve control discoverability.
  - Changed chat/persona language from jargon (`k=n`) to user-facing memory-result wording.
  - Changed `/chat/settings` behavior to canonical redirect (`/chat/settings` -> `/settings/chat`) and aligned strict release-gate expectations.
  - Changed audio route identity so `/tts`, `/stt`, and `/speech` have distinct intended surfaces.
  - Changed TTS/STT/Speech loading lifecycle to include explicit timeout, actionable error states, and retry UX.
  - Changed workspace/playground mobile copy and layout behavior, including Sources tab wording, chunking mobile order, and workflow-editor responsive drawer behavior.
  - Changed workflow editor control semantics with improved labeling/aria clarity and consistent LLM casing.
  - Changed flashcards/quiz/kanban/watchlists/documentation empty/content states for intent clarity and first-use correctness.
  - Changed media/knowledge/characters/chatbooks error states to be recoverable, actionable, and sanitized.
  - Changed core onboarding/layout behavior to improve mobile readability, hide sidebar where appropriate, and clarify ambiguous connection-status wording.
- Accessibility and UX consistency:
  - Standardized dismissible beta-badge behavior in shared settings navigation with persisted hide state (`tldw:settings:hide-beta-badges`).
  - Added explicit labels/tooltips for previously icon-only controls in Document Workspace and Workflow Editor.
- AntD/Markdown modernization:
  - Migrated deprecated AntD usage in shared UI (`Drawer.width`, `Space.direction`, `Alert.message`, `Dropdown.Button`, and notification `message`) to current APIs.
  - Aligned shared markdown dependency stack to ReactMarkdown v10 parity across WebUI and extension (`react-markdown`, `remark-gfm`, `remark-math`, `rehype-katex`).
  - Updated shared markdown wrappers for ReactMarkdown v10 API compatibility (styling moved off direct `ReactMarkdown` `className` prop).
- CI gating:
  - Frontend UX gates workflow now runs the Stage 5 audited-route smoke gate before the broad all-pages smoke job.
- Claims extraction/verification internals:
  - Refactored claims engine, ingestion claims, adjudicator, and claims service to use shared runtime config and shared LLM response coercion helpers.
  - Standardized provider `response_format` selection (`json_schema` when supported, `json_object` fallback) with graceful downgrade when unsupported.
  - Extended claims monitoring docs and operations guidance (metrics catalog, parse/fallback alerts, runbook triage and tuning guidance).
  - Tuned parse/fallback alert rules to use ratio + minimum-volume gates for mixed traffic profiles.
  - Updated code documentation and published mirrors for claims parse mode, alignment mode, adaptive throttling, and monitoring behavior.

### Removed

- UI
  - Removed `apps/tldw-frontend/pages/chat/settings.tsx` after replacing with server-side route redirect strategy.
  - Removed duplicate chat disconnected guidance surfaces that previously showed overlapping connection messages.
  - Removed user-visible unresolved template placeholder rendering across audited chat/audio/documentation surfaces.

### Fixed

- Auth/API: deprecated `PUT /api/v1/users/me` now returns `404 User not found` when the backing `users` row is missing, instead of returning a false-success profile payload.
- Test isolation: hardened MediaDB2 `torch` stubs with minimal `Tensor`/`nn.Module` attributes so cross-suite pytest runs no longer fail during SciPy/NLTK import checks.
- fix(audio-ws): make transcribe startup resilient to Nemo probe failures
- Treat Nemo availability probe import errors as non-fatal in `websocket_transcribe`
- Default to Whisper when Nemo probing cannot be resolved at runtime
- Prevent WS startup aborts that caused downstream quota/metrics/concurrency test failures
- Document the fail-safe fallback behavior in the audio streaming protocol docs
- Hardening: Prompts/Sync/Workflows/Services
  - Improved production safety and reliability across prompt collections, sync, workflow placeholders, and ephemeral processing.
  - Documented /api/v1/prompts/collections/* as production-backed, user-scoped endpoints.
  - Enforced and documented strict /api/v1/sync/send entity validation (Media, Keywords, MediaKeywords).
  - Hardened sync error handling: internal failures stay 500; /api/v1/sync/get now fails closed on invalid sync rows instead of silently skipping.
  - Confirmed workflow process_media placeholder kinds (ebook, xml, podcast) return explicit not_implemented.
  - Added and documented ephemeral store controls (EPHEMERAL_STORE_TTL_SECONDS, EPHEMERAL_STORE_MAX_ENTRIES, EPHEMERAL_STORE_MAX_BYTES).
  - Fixed XML placeholder temp-file lifecycle (always cleaned up) and preserved intended HTTP status behavior.
  - Added regression tests for XML cleanup/error paths and rate-limiter type-hint/datetime integrity.
- Fixed several backend reliability and API-behavior issues across Prompt DB lifecycle, sync processing, and service cleanup.
  - Removed import-time async worker startup in prompts DB dependencies; worker lifecycle now starts safely under app runtime.
  - Hardened /sync/send and /sync/get error handling:
    - preserved HTTP exceptions instead of wrapping them into generic 500s,
    - restored correct 400 classification for client validation errors (including disallowed sync entities),
    - prevented silent partial /sync/get responses when invalid sync rows are encountered.
  - Expanded sync conflict timestamp parsing to support ISO-8601 Z, fractional Z, +00 (line 0), and non-UTC offsets (normalized to UTC before LWW comparison).
  - Fixed XML processing temp-file leak by ensuring cleanup on both success and failure paths.
  - Removed deprecated FastAPI 422 fallback usage to eliminate import-time deprecation warnings in exception handling.
  - Added regression coverage for all fixes, with targeted suite passing (69 passed).
- Auth/API: deprecated `PUT /api/v1/users/me` no longer returns a false-success payload when the backing `users` row is missing; it now returns `404 User not found`.
- Regression coverage: added legacy `/users/me` update tests for both missing-row (`404`) and successful update (`200`) paths.
- Test stability: hardened MediaDB2 test `torch` stubs with minimal `Tensor`/`nn.Module` attributes to prevent cross-suite import failures during SciPy/NLTK initialization.
- Security
  - Hardened API key exposure paths:
    - /api/v1/config/docs-info now always returns a safe placeholder key and api_key_configured status.
    - Startup API key logging is masked by default (full key only when explicitly enabled).
  - Hardened scheduler payload handling:
    - External payload storage now uses safe JSON serialization by default.
    - Added strict payload reference validation and payload header bounds checks.
    - Disabled legacy pickle payload deserialization by default.
  - Hardened web scraping dedupe persistence:
    - Dedupe hashes now persist as JSON instead of pickle by default.
    - Legacy pickle hash loading is disabled by default.
  - Isolated placeholder processing services:
    - Placeholder document/ebook/podcast/xml services are now blocked in production-like environments even if enabled.
- Hardened deprecated `GET /api/v1/users/me` so missing-user fallback is only allowed in single-user mode; multi-user now correctly rejects missing backing users.
- Prevented ephemeral-store “unusable key” behavior by rejecting payloads that exceed configured max-bytes before insertion.
- Improved `/api/v1/sync/get` resilience: malformed sync rows are now logged and skipped while valid rows are still returned (partial success instead of full failure).
- Added regression coverage for user fallback rules, sync client route/auth behavior, sync malformed-row handling, and ephemeral-store size-limit enforcement.
- CORS now fails closed when enabled and `ALLOWED_ORIGINS` resolves to an explicit empty list (`[]`); a non-empty explicit origin list is required unless CORS is disabled.
- Startup now rejects invalid CORS configuration when `ALLOWED_ORIGINS` includes `"*"` while credentials are enabled.
- OpenAPI CORS handling no longer reflects disallowed origins.
- `ALLOWED_ORIGINS=""` (empty string) now logs a warning and falls back to local default origins for backward compatibility.
- Security hardening (Scheduler): fixed symlink-path validation bypass so base_path now rejects symlink ancestors (including symlink/child) instead of only direct symlink paths.
- Regression coverage (Scheduler): updated test_security_fixes.py to match current backend detection (pool-based Postgres detection) and async DB mocks (AsyncMock + awaited call assertions).
- Deprecation headers: replaced hardcoded fallback sunset dates with a shared UTC helper (build_deprecation_headers) used by auth, users, and legacy character-chat endpoints, with DEPRECATION_SUNSET_DAYS support and dynamic fallback computation.
- WebSearch Module: improved reliability and diagnostics across aggregation, provider dispatch, and parsing. Aggregate mode now validates LLM configuration (returns `422` when missing and reuses `final_answer_llm` for relevance analysis when provided), Google now correctly handles `google_domain`/`googlehost`, result-language parsing (`lr` with `hl` fallback), and multi-domain blacklist behavior, Kagi endpoint construction and Searx safesearch mapping were corrected, subquery generation/parsing now sanitizes and deduplicates model output, and provider `processing_error` values are surfaced in response `error`/`warnings` instead of being silently dropped. Added regression coverage across unit and integration WebSearch paths.
- Hardened chunked image handling to reject unsupported MIME types and invalid image payloads in large `data:` image paths.
- Changed chunked image processing failure behavior from fail-open to fail-closed, preventing invalid bytes from being treated as valid images.
- Fixed `run_cpu_bound_thread()` so keyword arguments are supported correctly via `functools.partial`.
- Corrected `load_prompt()` semantics for markdown prompts: missing keys now return `None` instead of falling back to the first fenced block.
- Removed `shell=True` usage in CUDA detection (`nvidia-smi` now called with direct subprocess args).
- Added backward-compatible chat dictionary endpoint re-exports for legacy imports/tests.
- Updated telemetry dummy span compatibility (`set_attributes`, broader `record_exception` signature).
- Skills
  - Fixed built-in seed behavior so supporting and nested files are preserved during seeding.
  - Fixed overwrite seeding edge cases where soft-deleted registry rows could prevent seeded skills from being visible after overwrite.
- Admin Backup Bundles
  - Fixed restore failure error reporting so rollback diagnostics are no longer collapsed to generic `import_error` behavior.
  - `restore_failed` import errors now preserve structured rollback context for API consumers.
  - Endpoint import response now properly propagates `rollback_failures` from service results.
  - Retention cleanup now safely skips unreadable/corrupt manifests without crashing bundle creation and keeps ZIP/sidecar cleanup consistent.
- Circuit Breaker
  - AuthNZ integration assertion for circuit breaker `source` is now environment-safe by accepting both `memory` and `mixed` (persistent mode).
- Resolved web-mode extension API runtime failures by guarding `chrome.storage` access and preventing `chrome is not defined`/`chrome.storage` uncaught exceptions.
- Removed unresolved `{{...}}` template leaks from shared UI labels/tooltips on audited routes.
- Hardened shared timeout/retry flows for admin stats and STT/TTS catalog discovery to prevent permanent-loading states.
- Removed temporary Stage 5 warning allowlist (`m5-react-defaultprops-warning`) after root-cause remediation.
- Revalidated audited smoke quality gates after remediation:
  - Stage 5 gate: `11 passed`
  - Full all-pages smoke: `165 expected, 0 unexpected, 0 flaky`
- UI
  - Fixed catastrophic extension-shim/runtime overlay impact on audited routes by driving closeout to `withErrorOverlay: 0` and `withChromeRuntimeErrors: 0` in final smoke artifact.
  - Fixed wrong-content navigation defects by closing route-contract expectations across audited settings/admin/connectors/profile/config destinations.
  - Fixed audited navigation 404 regressions (Section 2 list) to zero for in-scope UX routes.
  - Fixed unresolved template-variable leaks (`{{...}}`) across previously affected chat, audio, and documentation surfaces (`templateLeakRoutes: 0` at closeout).
  - Fixed max-update-depth/infinite-rerender class regressions on critical routes (`maxDepthRoutes: 0`, `maxDepthEvents: 0` in closeout artifact).
  - Fixed permanently-loading skeleton experiences on key admin/audio surfaces via timeout + error + retry state transitions.
  - Fixed mobile interaction issues across prioritized flows (touch-target sizing, toolbar discoverability, responsive parity gaps).
  - Fixed core route identity duplication so home/setup/onboarding-test have distinct purpose and route contracts.
  - Fixed release-readiness reliability with passing gate suite reruns at closeout: Stage 5 (`12 passed`), Stage 6 (`6 passed`), Stage 7 (`4 passed`).
  - Fixed program-level UX closure criteria by completing all nine UX implementation plans and finalizing the overarching oversight plan status to complete.
- Restored `Docs/Design/rich_text_chat_rendering_v1_2026_02_15.md` after an unintended deletion in a prior docs commit.

## [0.1.20] 2026-02-07

### Added

- webui/extension fixes+improvements
  - New Guardian settings page
- **FVA Pipeline (Falsification-Verification Alignment)**: New claim verification enhancement that actively searches for contradicting evidence to provide more robust verification results. Implements the FVA-RAG paper (arXiv:2512.07015).
  - New `CONTESTED` verification status for claims with significant evidence both supporting and contradicting
  - Anti-context retrieval generates counter-queries (negation, contrary, alternative) to find contradicting evidence
  - Adjudication weighs supporting vs contradicting evidence using NLI and LLM-based stance assessment
  - API endpoints: `POST /api/v1/claims/verify/fva` and `GET /api/v1/claims/verify/fva/settings`
  - Configurable via config.txt: FVA_ENABLED, FVA_CONFIDENCE_THRESHOLD, FVA_CONTESTED_THRESHOLD, etc.
  - 9 new Prometheus metrics for observability (falsification triggers, status changes, adjudication scores, timeouts)
  - Budget management to prevent runaway costs on large claim sets
- Speech Playground history now shows metadata (duration, model/voice/provider, params summary) with a detail tooltip.
- History entries now persist STT/TTS params (task, temp, response format, segmentation, speed, split, streaming, mode) so metadata reflects actual runs.
- Global TS typecheck fixed across UI (EPUB viewer/search typings, document chat/store shapes, KnowledgeQA response normalization, MCP path typing, chunking options, safe config typing, xterm ambient types, SpeechPlayground ordering/import issues).
- Implemented TTS history end-to-end:
	- Write path (non-streaming, streaming, jobs) with status, metadata, artifacts, and error handling.
	- Read path: list/detail/favorite/delete endpoints with filters, cursor pagination, total count.
	- Retention: scheduled purge by days and max rows; artifact reference cleanup.
	- Observability: read/write counters and latency histograms (no text logging).
	- Added schema + migrations for tts_history and supporting indexes.
- Added tests:
	- Unit: schema and API list/favorite/delete, q/text_exact behavior.
	- Integration: streaming failure writes history, artifact purge updates history, cursor pagination sanity.
	- Fixed test infrastructure shims for audio/tokenizer/transcription helpers.
- Server/API
  - RAG upgrades: Doc-Researcher features (granularity router, evidence accumulator, evidence chains), batch + resume endpoints, web fallback with query rewrite, retrieval metrics, FlashRank model support.
  - Character chat: per‑chat settings + prompt preview endpoint, greeting picker/inclusion toggle, generation presets, multi‑character turn‑taking, message steering controls, pinned messages pathing.
  - Watchlists: dedup/seen inspect & reset tools; scheduler controls; performance/limits docs.
  - Guardian/Self‑monitoring: new Guardian_DB and API surfaces.
  - ACP sandbox: SSH-enabled runner and bridge; new Dockerfile + entrypoint.
  - New “Skills” framework and Kanban endpoints/modules.

### Changed

- DBs modified
	- Media DB v2: per-user SQLite DB at Media_DB_v2.db (and Postgres equivalent when configured).
- MCP Unified hardening:
  - `/api/v1/mcp/auth/refresh` now requires body-based `refresh_token` payloads (query-token transport rejected).
  - Write-tool idempotency now binds each `idempotencyKey` to the initial argument fingerprint and returns `INVALID_PARAMS` on mismatched replays.
  - Module registry lookups now use concurrency-safe snapshots to prevent `dictionary changed size during iteration` under concurrent module registration churn.
  - Sandbox stream Redis fanout now requires explicit `SANDBOX_WS_REDIS_FANOUT` opt-in (no implicit enablement from global Redis settings).
- Evaluations module hardening:
  - Parallel batch mode now enforces strict fail-fast when `continue_on_error=false` by canceling remaining tasks and stopping new scheduling.
  - Evaluation runner now safely normalizes `metrics` values, including `metrics=None`, across execution and aggregate/stat calculations.
  - RAG pipeline eval runs now use per-run user context for ephemeral vector index creation/cleanup and pipeline calls.
  - OCR-PDF usage accounting now records against the correct limiter namespace (`evals:ocr_pdf`).
  - Access policy aligned: `/api/v1/evaluations/health` remains public, `/api/v1/evaluations/metrics` is authenticated (`EVALS_READ`), and dataset endpoints enforce `EVALS_READ`/`EVALS_MANAGE` consistently.
- Server/API
  - Admin endpoints modularized: tldw_Server_API/app/api/v1/endpoints/admin/* (replaces single admin.py).
  - Audio split: new audio submodule endpoints (tts, tokenizer, streaming, history).
- Web UI/Extension
  - Document Workspace: PDF/EPUB viewers overhauled (virtualized PDF single-page mode, thumbnails, TOC keyboard nav), DocumentPicker modal, richer metadata, TTS panel improvements, retry logic.
  - Knowledge QA: web-search fallback settings, local/server threads, richer history filtering; new “All options” settings section.
  - Playground/Chat UX: composer extracted into components, Notes Dock, greeting picker, pin/steering controls, MCP tool catalog/filter UX.
  - Workflow Editor: dynamic step registry + schema-driven fields, icons, dynamic options; new tests.
  - TTS Playground: multi-voice roles, background job progress UI, presets.
  - ACP Playground: Workspace terminal tab (xterm.js).
  - Extension e2e hardening; pdf.worker exposed via web_accessible_resources.
- Tooling/ops/docs
  - Lint only-changed target and script; large doc set added/updated (PRDs, guides).
  - FlashRank reranker model vendored under models/flashrank with unignore rules.
  - New env vars for ACP sandbox, TTS history, and RAG FlashRank cache/model selection.
- Web Scraping Module
  - Hardened legacy fallback contracts in web_scraping_service.py (line 46) and web_scraping_service.py (line 276).
  - Added explicit fallback metadata (engine, fallback_context) in web_scraping_service.py (line 392) and web_scraping_service.py (line 497).
  - Added predictable fallback max_pages cap for legacy URL Level/Sitemap in web_scraping_service.py (line 340).
  - Preserved request max_pages pass-through in ingest orchestration in web_scraping_service.py (line 721) and web_scraping_service.py (line 772).
  - Added fallback-focused tests in test_legacy_fallback_behavior.py (line 44).
  - Fixed auth header fixture for friendly ingest crawl-flag tests in test_friendly_ingest_crawl_flags.py (line 19) by merging default headers and adding X-API-KEY.

### Removed

### Fixed

- Workspace selector remove handler now uses the imported MouseEvent type instead of the React namespace.
- Audit read paths no longer return empty results on DB failures; `/api/v1/audit/export` and `/api/v1/audit/count` now surface server errors when reads fail.
- Audit fallback replay now quarantines malformed JSONL lines into `audit_fallback_queue.bad.jsonl` instead of silently dropping them.
- Audit export now rejects non-positive `max_rows`, `audit_operation` ignores reserved kwarg collisions safely, and shared-audit migration stats counters now track read/inserted/skipped events correctly.
- CodeQL Bugfixes
- ruff and mypy fixes

## [0.1.19] 2026-01-31

### Added

- Soft delete support for notes/character cards
- Qwen3-STT
- JSON validation utilities with detailed error positioning (line/column information)
- [WebUI] Character generation prompt templates for full and single-field generation
- [WebUI] Flashcard undo functionality with Ctrl/Cmd+Z shortcut
- [WebUI] Media review selection and focus settings
- [WebUI] TldwApiClient methods for character export, restore, and bulk world book operations
- Comprehensive test coverage for Qwen3-ASR, Gemini tools, and character generation
- [WebUI] Documentation updates including PRD for Characters Playground UX improvements, healthcare-focused UX review prompts, and Qwen3-ASR setup guide
- README restructuring with improved formatting and version 0.1.18 release notes
- 70+ new adapters for workflows module
- [WebUI] Added Document Workspace
- [WebUI] Added Writing Playground

### Changed

- Implemented httpx/aiohttp transport adapters with centralized policy enforcement in http_client.
- Added httpx client caching and shutdown cleanup to align with aiohttp lifecycle handling.
- Formalized streaming behavior with first‑byte/idle timeouts and mid‑stream retry support.
- Expanded http_client tests for adapters, cache reuse, and streaming timeout coverage.
- [WebUI] e2e test work

### Removed

### Fixed

- Workspace selector remove handler now uses the imported MouseEvent type instead of the React namespace.

## [0.1.18] 2026-01-29

### Added

- Slides Module
- TTS:
  - Added NeuTTS, PocketTTS (ONNX), EchoTTS, Qwen3-TTS, LuxTTS, VibeVoice-ASR docs; updated streaming/format rules, default voice behavior, and setup guides.
- Moved the tldw_Browser_Assistant project and the tldw-frontent folder into the '/apps/' folder, as moving forward they will share the same base.
  - As a result, new frontend!
- New monorepo development guide, shared UI package scaffold, ambient typings, and testing guide for extension/web UI.
- Image creation API via files

### Changed

- Workflows Module
  - New "llm" step type (distinct from "prompt")
  - MCP tool allowlist and scope validation
  - Stricter approve/reject permission checks
  - Configurable LLM retry cap
- Admin-UI Updates
- New frontend, tldw-frontend
- New Storage API guide, Voice Assistant API (REST & WebSocket), Watchlists API docs, Anthropic Messages API docs, and expanded /llm/models metadata (image backends & filters).
- Wide-ranging documentation additions and edits (OCR backends, image generation, storage, benchmarks, guides, link dumps, examples).
- Kanban: vector search integration, activity logging with filtering, rate limiting on endpoints
- Reading Collections: async import jobs workflow with job monitoring endpoints
- Workflows: LLM step type with MCP tool allowlist and scope validation

### Removed

- Legacy webui

### Fixed

## [0.1.17] 2026-01-19

### Added

- File Artifacts System: Comprehensive implementation of file artifact management with support for multiple export formats (iCalendar, Markdown tables, HTML tables, XLSX, data tables) including export lifecycle management, garbage collection, and validation
- Data Tables Module: Complete backend implementation with LLM-based table generation, async job workers, database schema (tables, columns, rows, sources), REST API endpoints, and export functionality
- Media Ingestion Cancellation: Added cancellation support across audio and video processing pipelines with subprocess monitoring and graceful error handling
- Content Import Preservation: Enhanced database layer to preserve existing metadata during reimport operations with preserve_existing_on_null parameter and improved full-text search with fallback candidates
- File Adapter Registry: Dynamic adapter management system supporting multiple file types with validation, normalization, and export capabilities
- Presentation Templates: New Reveal.js slide templates and CSS styling for presentation generation with bundle export support
- API Enhancements: New endpoints for file artifacts management, data table generation/export/management, and media ingest job listing with async job tracking
- Comprehensive Test Coverage: Integration and unit tests for file artifacts, data tables, media ingestion cancellation, and database operations
- NeuTTS Support
- TTS voice registry

### Changed

### Removed

### Fixed

## [0.1.16] - 2026-01-17 / Broken Bugs

### Added

- Jobs Postgres RLS policy setup now supports `JOBS_PG_RLS_DEBUG` for policy output and `JOBS_PG_RLS_ROLE` role overrides.
- Jobs prune scheduler for retention-based cleanup (env-gated, internal scheduler).
- `CHAT_COMMANDS_ASYNC_ONLY` flag to force async chat orchestration (`achat`) and block sync `chat(...)`.
- Chat command concurrency integration test and PERF-gated p50 latency smoke test.

### Changed

- Jobs Postgres tests now default to the shared per-test Postgres fixture by wiring `JOBS_DB_URL` and ensuring Jobs tables/counters.
- Jobs RLS policy setup uses negotiated Postgres DSNs for compatibility across server versions.
- Sync `chat(...)` now offloads to a worker when invoked from a running event loop (non-streaming).

### Removed

- Legacy `command_router.dispatch_command` path (now raises with migration hint).

### Fixed

- Jobs RLS debug output now reports distinct settings fields without clobbering values.
- Fixing of 200+ bugs

## [0.1.15] - 2026-01-10

### Added

### Changed

- Jobs adapters now ignore legacy read-backend flags; Chatbooks/Prompt Studio are core Jobs only, embeddings read fallback is disabled.
- Documentation updated to reflect core Jobs defaults and the current embeddings execution modes.

### Removed

### Fixed

- Legacy AuthNZ rate limiting now bypasses only when an RG policy is attached, and cancellations propagate correctly in rate limit fallbacks.
- ChaChaNotes shutdown now drains default-character tasks and waits for the executor to prevent SQLite close races (fixes Jobs web UI TTL test segfaults).

## [0.1.14] - 2026-01-06

### Added

- Added tldw-admin react frontend for admin Mgmt of the server. Very much WIP.
- Extended feedback system/schema - Added a unified feedback system (explicit/implicit) across chat and search, integrates message IDs into chat history and streaming,
- introduced API key KDF/key_id,
- added admin effective-config endpoint/UI,

### Changed

- Centralized per-user path utilities for storage safety and consistency
- Migrated ingress rate limiting to Resource Governance (RG), removed SlowAPI decorators
- Enhanced feedback system with explicit endpoint, schemas, and idempotent merge rules
- Expanded chat streaming metadata to include system and assistant message IDs
- Integrated UI feedback across chat and search (rating, source feedback, implicit events)
- Updated documentation and configuration for feedback system and config management
- Comprehensive test coverage for feedback, chat metadata, and UI integration

### Removed

- slowapi usage

### Fixed

- Lots of Bugs

## [0.1.13] - 2025-12-29

### Added

- Next.js WebUI (apps/tldw-frontend)
- admin-ui - Full Admin UI: dashboard, users/orgs/teams, roles & permissions, API keys, jobs, usage analytics, budgets, BYOK, flags, incidents, logs, monitoring panels.
- Content Review: draft editor, sidebar, reattach-source flow, commit/review workflows.
- BYOK improvements: scoped resolution, validation, dashboard and key management.
- Option for review of media prior to ingestion.

### Changed

- Claims module expanded

### Removed

- N/A

### Fixed

- Improved frontend/backend error handling and type safety; more robust API interactions.

## [0.1.12] - 2025-12-20

### Added

- Full Kanban API (boards, lists, cards, labels, checklists, comments, import/export, bulk ops, card linking) with hybrid search (FTS vector).
- Self‑service Organizations & Teams (invites preview/redeem) and org admin flows.
- Billing & subscriptions (plans, checkout/portal, invoices, usage) and Stripe webhook handling.
- BYOK (per‑user and shared provider keys) with admin management and testing.
- TTS providers onboarding and user TTS guide.

### Changed

- Adds a full billing/subscription system (plans, limits, Stripe integration, webhooks), BYOK (per‑user and shared provider keys) with admin tooling, invitations/org/team RBAC, a Kanban module with per‑user DB FTS/vector search, many new endpoints/schemas/repos, DB migrations, audit/auth dependency changes, media visibility, and extensive docs/config updates.

### Removed

- A sense of failure.

### Fixed

- 500bugs

## [0.1.11] - 2025-11-27

### Added

- Guidance on stress testing chat server

### Changed

- RAG Documentation

### Removed

### Fixed

## [0.1.10] - 2025-11-27

### Added

- ChaChaNotes health snapshot surfaced in `/api/v1/health` to monitor init attempts/failures and cache state.
- MLX local provider scaffolding (Apple Silicon): adapters admin lifecycle endpoints, metrics parity, and config keys/tests with non-Apple skips.
- `LLM_MLX` extra in `pyproject.toml` to install `mlx-lm`/`mlx` for Apple Silicon users.
- Config-driven llama.cpp handler: `LLMInferenceManager` now constructs `LlamaCppHandler` when `[LlamaCpp]` is enabled in `config.txt` or via env, and `/llamacpp` endpoints are wired to the managed handler.
- ChaChaNotes schema v10 adds conversation metadata (state with `in-progress` default/backfill, topic labels, clusters) plus backlinks on notes (`conversation_id`, `message_id`) with covering indexes and SQLite/Postgres migrations.
- Templated hierarchical chunking for incoming documents/emails across `/api/v1/media/process_*` endpoints, including TemplateClassifier-based auto-selection of chunking templates and optional section trees.
- Streaming chunker/runtime helpers now honor shared chunking options via `prepare_chunking_options_dict`/`apply_chunking_template_if_any` for consistent behavior across document, PDF, video, audio, ebook, and email processing.
- User-facing documentation for templated chunking (`Templated_Chunking_Incoming_Documents_HowTo.md`) and the Project 2025 RAG workflow guide for policy/document ingestion.
- Chat diagnostics endpoints: `GET /api/v1/chat/queue/status` and `GET /api/v1/chat/queue/activity` exposing queue metrics and recent job activity, RBAC-gated to `system.logs` in multi-user mode.

### Changed

- Conversation title search now applies global BM25 normalization so pagination returns stable, deterministic ordering across the entire result set.
- Chunking engine improvements for streaming text and Markdown: better whitespace handling for word/semantic/token chunking and promotion of bold-only headings into hierarchical subsections under their parent section.
- ChaChaNotes dependency now ensures per-user DB directories are created, optional `message_metadata` is initialized, and default-character warmup tasks are tracked so the health snapshot accurately reflects warm starts.
- Llama.cpp integration: `LLMInferenceManager` logs model-directory creation failures instead of silently swallowing them, and `/llamacpp` endpoints resolve the manager from `app.state.llm_manager` (falling back to the module-level instance) with a clear 503 when not configured.
- Workflows and scheduler workflow routers are always mounted (without an `/api/v1` prefix) inside minimal/test apps so tests and tooling can call them consistently.

### Fixed

- ChaChaNotes warmup no longer leaves orphaned default-character tasks; background tasks are tracked and cleaned up when complete, improving shutdown and health reporting.
- Visual document ingestion from audio/video analysis now persists slide/visual artifacts via a thread executor, avoiding event-loop blocking during heavy analysis.
- MLX local provider concurrency: `MLXSessionRegistry.session_scope` snapshots the semaphore per context so dynamic concurrency updates cannot corrupt in-flight sessions.
- Media re-chunking for documents and emails remains best-effort but now logs failures at debug level instead of silently swallowing errors, making template issues easier to diagnose.
- Async chunker and template processor now handle multi-operation stages safely and preserve whitespace between overlapping chunks for all space-delimited methods.
- `/api/v1/health` now logs ChaChaNotes snapshot failures and resource-governance policy file read errors while still reporting a degraded health state instead of failing the endpoint.
- Chat queue status/activity endpoints avoid shadowing FastAPI's `status` module and enforce RBAC correctly, so authentication/authorization failures return the intended HTTP codes instead of spurious 500s.

## [0.1.9]

### Added

- ChaChaNotes health snapshot surfaced in `/api/v1/health` to monitor init attempts/failures and cache state.

### Changed

- ChaChaNotes dependency now initializes in a dedicated executor with WAL/busy-timeout tuning and background default-character creation; request path reduced to cache lookup health probe.
- Startup warms the single-user ChaChaNotes instance to avoid first-request blocking; shutdown now closes cached instances and stops the ChaChaNotes executor to prevent lingering threads.

### Removed

### Fixed

## [0.1.8] - 2025-11-22

### Added

- Auto-streaming for large audit exports exceeding configured threshold
- CSV streaming support for audit exports
- Model discovery for local LLM endpoints
- Audit event replay mechanism for failed exports
- Enhanced HTTP error handling for DNS resolution failures
- SuperSonicTTS support setup script
- STT:
  - `get_stt_config()` helper in `config.py` to centralize resolution of `[STT-Settings]` for all STT modules.
  - Documentation for `speech_to_text(...)` (segment-based) and `transcribe_audio(...)` (waveform-based) as the two canonical STT entrypoints, including guidance on error sentinel handling.

### Changed

- Audio:
  - Replace Parakeet-specific transcriber/config usage with unified UnifiedStreamingTranscriber/UnifiedStreamingConfig; add _LegacyWebSocketAdapter to adapt legacy WS to unified handler; defer imports and update tests to use unified stubs.
  - Move desktop/live audio helpers (LiveAudioStreamer, system-audio utilities) into `Audio/ARCHIVE/Desktop_Live_Audio_Samples.py` so core STT modules no longer depend on optional PyAudio/sounddevice at import time.
- Audit:
  - Add config-driven auto-stream threshold, support streaming for json/jsonl/csv, force streaming when max_rows exceeds threshold; CSV streaming generator; non-stream export caps; API-key hashing; fallback JSONL queue with background replay task; tests for streaming and replay.
- LLM:
  - Add local model discovery (short timeouts, TTL cache, candidate endpoints), get_configured_providers_async and integrate async provider loading into startup and web UI config; provider payloads include is_configured and endpoint_only.
- TTS
  - WAV output now buffered and deferred until finalize with in-memory threshold and disk spill; StreamingAudioWriter.__init__ adds max_in_memory_bytes; tests validate spill and finalize behavior.
- Web Scraping
  - Use defusedxml, broaden sitemap parse error handling, add test-mode egress bypass, add conditional process_web_scraping_task import/export, and preserve HTTPException semantics in ingestion endpoint.
- Tests
  - Extensive test updates (unified WS stub, fake HTTP client for RSS, env snapshot/restore, admin override fixtures, watchlists full-app fixture, connectors pre-mounting); CI embedding cache key changed to a static key.
  - STT unit tests for Parakeet/Qwen2Audio `return_language` branches now exercise the provider branches directly and avoid unintended Whisper fallbacks by normalizing file-path arguments.

### Removed

- Hopes, Dreams.

### Deprecated

- Efficiency.

### Fixed

- Improved WebSocket disconnect handling
- Consistent error handling in session cleanup and web ingestion
- Better network error resilience with graceful fallbacks
- Add _is_dns_resolution_error detection and mark DNS resolution errors non-retriable (DNSResolutionError signal); tests verify DNS errors are not retried while other network errors follow retry policy.
- My life.

## [0.1.6] - 2025-11-14

### Fixed

- HTTP-redirect loop
- test bugs

### Added

- Option for HTTP redirect adherence in media ingestion endpoints added in config.txt

## [0.1.5] - 2025-11-13

### Fixed

- Ollama API system_prompt
- Other stuff

### Added

- Updated WebUI
- Added PRD/initial work for cli installer/setup wizard
  - Auto-title notes
- Notes Graph CRUD
- Documentation/PRDs
- (From Gemini) New Chatbook Tools: Implemented a suite of new tools for Chatbooks, including sandboxed template variables for dynamic content in chat dictionary replacements, user-invoked slash commands (e.g., /time, /weather) for pre-LLM context enrichment, and a comprehensive dictionary validation tool (CLI and API) to lint schemas, regexes, and template syntax.

## [0.1.4] - 2025-11-9

### Fixed

- Numpy requirement in base install
- Default API now respected via config/not just ENV var.
- Too many issues to count.

### Added

- Unified requests module
- Added Resource governance module
- Moved all streaming requests to a unified pipeline (will need to revisit)
- WebUI CSP-related stuff
- Available models loaded/checked from `model_pricing.json`
- Rewrote TTS install/setup scripts (all TTS modules are likely currently broken)

## [0.1.3.0] - 2025-X

### Fixed

- Bugfixes
-

## [0.1.2.0] - 2025-X

### Fixed

- Bugfixes

## [0.1.1.0] - 2025-X

### Breaking

- Prometheus scraping for MCP metrics now requires authentication with the `system.logs` permission; `MCP_PROMETHEUS_PUBLIC` no longer enables public access, is deprecated, and will be removed. Migration: update Prometheus scrape configs to send a Bearer token (API key or JWT) that includes `system.logs` (see `Docs/Deployment/Monitoring/Metrics_Cheatsheet.md` migration note and scrape_config example).

### Features

- Version 0.1

### Fixed

- Use of gradio
