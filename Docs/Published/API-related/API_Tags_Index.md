# API Module Guide

This guide explains what each API module makes possible. It is organized by user goal rather than by source file, and each module name maps to an OpenAPI tag in `/docs` and `/redoc`.

Use this page when you are deciding which API area to explore first. Use the linked module docs and the live OpenAPI pages for request and response details.

## How To Read This Guide

| Column | Meaning |
|--------|---------|
| Module | The OpenAPI tag shown in Swagger/ReDoc. |
| What it lets you do | The capability in user-facing terms. |
| Common uses | Typical workflows or products you can build with it. |
| Docs | The most relevant existing guide, or an inline note when a dedicated guide does not exist yet. |

## Start, Auth, And Configuration

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `health` | Check whether the API server is reachable and ready. | Startup probes, load balancer checks, local setup validation. | Live OpenAPI |
| `setup` | Complete first-run setup and inspect onboarding readiness. | Guided initial configuration, setup UI checks, local single-user bootstrap. | [Authentication setup](../User_Guides/Server/Authentication_Setup.md) |
| `authentication` | Sign in, refresh tokens, inspect auth state, and use API key or JWT auth. | Single-user API clients, multi-user login flows, token refresh handling. | [AuthNZ API guide](AuthNZ-API-Guide.md) |
| `users` | Manage users, profiles, roles, and user-scoped API keys. | Account administration, user profile setup, virtual key management. | [User registration API](User_Registration_API_Documentation.md) |
| `organizations` | Manage organizations, teams, shared keys, and membership. | Multi-user deployments, team workspaces, shared provider access. | [Organizations and teams](Admin_Orgs_Teams.md) |
| `invites` | Preview, redeem, and audit organization invite codes. | Team onboarding, invite-based registration, membership workflows. | [Organizations and teams](Admin_Orgs_Teams.md) |
| `config` | Inspect server capabilities, effective config, quickstart info, and admin config diagnostics. | Client feature detection, setup screens, deployment troubleshooting. | [API notes](API_Notes.md) |
| `consent` | Record and inspect consent decisions for optional features. | Privacy-aware onboarding, user preference gates, safety workflows. | Live OpenAPI |
| `sync` | Synchronize client data and server state. | Offline-capable clients, incremental sync, cross-device state handoff. | [Sync API notes](API_Notes.md) |
| `authnz-debug` | Inspect AuthNZ debug state in development or test contexts. | Debugging auth setup and test fixtures. | Debug/internal surface |

## Media, Documents, And Ingestion

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `media` | Ingest, inspect, search, reprocess, version, and manage videos, audio, documents, web pages, and other source material. | Build media libraries, document analysis pipelines, local knowledge bases. | [API design](API_Design.md) |
| `media-embeddings` | Generate embeddings for ingested media and related content. | Semantic media search, RAG indexing, retrieval experiments. | [Embeddings API](Embeddings_API_Documentation.md) |
| `chunking` | Chunk raw text or content into retrieval-ready units. | Prepare long documents for RAG, tune chunk boundaries, test chunk strategies. | [Chunking templates API](Chunking_Templates_API_Documentation.md) |
| `chunking-templates` | Create and manage reusable chunking templates. | Per-document-type ingestion defaults, repeatable RAG preprocessing. | [Chunking templates API](Chunking_Templates_API_Documentation.md) |
| `ocr` | List OCR backends and run OCR extraction. | Image/PDF text extraction, scanned document ingestion. | [OCR API](OCR_API_Documentation.md) |
| `ingestion-sources` | Manage user-defined sources and sync operations. | Recurring imports, source catalogs, external library ingestion. | [Ingestion sources API](Ingestion_Sources_API.md) |
| `connectors` | Manage connector definitions and external-source adapters. | Integrating third-party repositories or source systems. | Experimental surface |
| `web-scraping` | Create and manage web scraping jobs and captured web content. | Save web pages, scrape research sources, monitor web content. | [Web scraping guide](../User_Guides/Server/Web_Scraping_Ingestion_Guide.md) |
| `web-clipper` | Save clipped browser content and request enrichment. | Browser extension saves, quick web capture, clipped-source cleanup. | Live OpenAPI |
| `collections-feeds` | Manage feed collections and imported feed items. | RSS/Atom readers, collection-based ingestion, reading queues. | [Collections feeds API](Collections_Feeds_API.md) |
| `collections-websub` | Receive and manage WebSub push callbacks for feed updates. | Push-driven feed ingestion and content refresh. | [Collections feeds API](Collections_Feeds_API.md) |
| `files` | Work with generated or uploaded files exposed through the API. | Download artifacts, attach files to workflows, inspect file metadata. | Live OpenAPI |
| `storage` | Manage user files, folders, downloads, quotas, trash, and storage usage. | File browsers, quota dashboards, artifact retention workflows. | [Storage API](Storage_API_Documentation.md) |
| `reading` | Manage reading-list entries and reading state. | Reading queues, saved articles, document review workflows. | [Reading list API](Reading_List_API.md) |
| `reading-highlights` | Create and manage highlights for reading items. | Annotation tools, review notes, source-linked highlights. | [Reading list API](Reading_List_API.md) |

## Audio, Voice, And Speech

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `audio` | Transcribe files, synthesize speech, list voices, manage presets, inspect audio health, and run audio utilities. | OpenAI-compatible STT/TTS clients, dictation, voice generation. | [Audio transcription API](Audio_Transcription_API.md), [TTS API](TTS_API.md) |
| `audio-websocket` | Stream audio for real-time transcription over WebSocket. | Live captions, meeting transcription, streaming dictation. | [Audio transcription API](Audio_Transcription_API.md) |
| `audio-jobs` | Run background audio processing through the Jobs pipeline. | Long recordings, fan-out transcription, async processing dashboards. | [Audio jobs API](Audio_Jobs_API.md) |
| `audiobooks` | Create and manage audiobook projects, chapters, narration, alignment, subtitles, and packaging. | Long-form narration, chapterized TTS, audiobook production. | Live OpenAPI |
| `voice-assistant` | Send voice-assistant commands and manage assistant interactions. | Push-to-talk assistants, voice command handlers, voice-enabled UI. | Live OpenAPI |
| `voice-assistant-ws` | Use the real-time voice assistant WebSocket surface. | Low-latency voice loops, interactive assistant sessions. | Live OpenAPI |

## Chat, Characters, And Persona

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `chat` | Run OpenAI-compatible chat completions and manage conversations. | Chat clients, provider-backed assistants, conversation history. | [Chat API](Chat_API_Documentation.md) |
| `messages` | Use Anthropic-style message endpoints and conversion helpers. | Anthropic-compatible clients, cross-provider message conversion. | [Anthropic messages API](Anthropic_Messages_API.md) |
| `chat-dictionaries` | Manage dictionaries for chat preprocessing and postprocessing. | Character voice consistency, replacement rules, domain vocabulary. | [Chatbook features API](Chatbook_Features_API_Documentation.md#chat-dictionary-api) |
| `chat-documents` | Generate documents from conversations and templates. | Conversation summaries, reports, structured exports. | [Chatbook features API](Chatbook_Features_API_Documentation.md#document-generator-api) |
| `chat-workflows` | Manage chat workflow templates and workflow runs. | Guided dialogues, repeatable assistant procedures, review flows. | Live OpenAPI |
| `characters` | Manage character cards, personas, imports, exports, and related assets. | Roleplay agents, persona libraries, SillyTavern-compatible cards. | [Character chat API](CHARACTER_CHAT_API_DOCUMENTATION.md) |
| `character-chat-sessions` | Create, list, update, and archive character chat sessions. | Character-specific conversations, session history, chat continuation. | [Character sessions API](Character_Chat_Sessions_API.md) |
| `character-memory` | Manage cross-session memory for characters. | Long-term persona continuity, memory inspection, memory cleanup. | [Character chat API](CHARACTER_CHAT_API_DOCUMENTATION.md) |
| `character-messages` | Create, retrieve, edit, and search character messages. | Message timelines, chat search, session exports. | [Character messages API](Character_Messages_API.md) |
| `persona` | Use the persona agent surface for voice, tools, and MCP-backed interactions. | Personal assistant flows, tool-using agents, persona runtime control. | [Personas user guide](../User_Guides/Server/Personas_User_Guide.md) |
| `persona-archetypes` | Manage persona archetype definitions. | Reusable persona templates and guided profile creation. | Experimental surface |
| `personalization` | Manage opt-in profiles, memories, and personalization signals. | User-specific RAG biasing, assistant memory, profile-aware experiences. | [Personas user guide](../User_Guides/Server/Personas_User_Guide.md) |
| `personal-context` | Manage the canonical user profile shared with Chatbook. | Profile preferences, workspace context, agent proposal review, export, and global deletion. | [Personal Context API](Personal_Context_API.md) |
| `companion` | Use companion-oriented agent endpoints. | Experimental companion experiences and profile-aware assistants. | Experimental surface |

## Search, RAG, Embeddings, And Evaluation

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `rag-unified` | Search indexed knowledge with keyword, vector, reranking, and context-building controls. | Knowledge Q&A, research assistants, retrieval tuning. | [RAG API guide](RAG-API-Guide.md) |
| `rag-health` | Inspect RAG health, cache state, and retrieval metrics. | Deployment checks, indexing diagnostics, RAG monitoring. | [RAG API guide](RAG-API-Guide.md) |
| `research` | Run research-provider and web data collection endpoints. | Web research, source discovery, aggregated research workflows. | Live OpenAPI |
| `research-discovery` | Use standalone research discovery routes. | Finding sources before deeper research runs. | Live OpenAPI |
| `research-runs` | Manage deep research session lifecycle and run state. | Long-running research jobs, progress UIs, resumable research. | Live OpenAPI |
| `research-workspace` | Inspect readiness and capability state for the Research Workspace UI. | Workspace setup checks, capability gating, troubleshooting. | Live OpenAPI |
| `paper-search` | Search scholarly providers such as arXiv, PubMed, Semantic Scholar, BioRxiv, and MedRxiv. | Literature search, citation discovery, research intake. | Live OpenAPI |
| `embeddings` | Generate OpenAI-compatible embeddings. | Semantic search, vector indexing, provider compatibility. | [Embeddings API](Embeddings_API_Documentation.md) |
| `vector-stores` | Manage and query OpenAI-compatible vector stores. | Vector search, retrieval backends, indexed knowledge stores. | [Vector stores API](Vector_Stores_Admin_and_Query.md) |
| `claims` | Extract, index, search, and maintain claims from media. | Claim review, fact-tracking, evidence workflows. | Live OpenAPI |
| `feedback` | Capture explicit feedback for RAG and chat quality. | Relevance signals, quality dashboards, tuning datasets. | Live OpenAPI |
| `evaluations` | Create datasets, run evaluations, manage recipes, and inspect metrics. | RAG evals, prompt scoring, batch model quality checks. | [Evaluations unified API](Evaluations_API_Unified_Reference.md) |
| `benchmarks` | Run benchmarking endpoints and benchmark utilities. | RAG benchmarks, regression tracking, performance comparisons. | [Benchmark guide](../User_Guides/Server/Benchmark_Creation_API_WebUI_Extension_Guide.md) |
| `vlm` | Use visual-language model processing endpoints. | Image-aware analysis and multimodal experiments. | Experimental surface |
| `text2sql` | Convert natural language questions into SQL-oriented workflows. | Database exploration, structured-data question answering. | Live OpenAPI |

## Notes, Prompts, Study, And Generated Work

<!-- Legacy discovery mapping retained for docs gate coverage: | `chatbooks` | API-related/Chatbook_API_Documentation.md | -->

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `notes` | Create, search, organize, graph, and manage notebook-style notes. | Personal knowledge bases, source-linked notes, task-backed notes. | Live OpenAPI |
| `notes-graph` | Explore graph relationships between notes, tags, and sources. | Knowledge graph views, relationship discovery, backlink analysis. | Live OpenAPI |
| `notes-semantic-index` | Manage the consented Notes semantic index and inspect similar-content graph edges. | Semantic index setup, status, rebuilds, cleanup, and grounded related-Note review. | [Semantic index API](/docs-static/API/Notes_Semantic_Index.md) / [User guide](../User_Guides/WebUI_Extension/Notes_Semantic_Graph.md) |
| `prompts` | Manage prompt library entries, imports, exports, and metadata. | Reusable prompt catalogs, prompt sharing, assistant presets. | Live OpenAPI |
| `prompt-studio` | Build, test, compare, and optimize prompts as reusable projects and runs. | Prompt engineering workflows, test cases, prompt optimization. | [Prompt Studio API](Prompt_Studio_API.md) |
| `chatbooks` | Import and export chatbooks, OpenWebUI data, related attachments, and OpenWebUI attachment hydration preview/job endpoints. | Backup/restore, migration, portable conversation bundles. | [Chatbook API](Chatbook_API_Documentation.md) |
| `flashcards` | Create and manage flashcards and decks. | Study decks, spaced-review clients, generated learning material. | Live OpenAPI |
| `quizzes` | Create and manage quizzes. | Knowledge checks, study workflows, course review. | Live OpenAPI |
| `study-suggestions` | Read and refresh study suggestion snapshots. | Learning recommendations, review queues, study dashboards. | Live OpenAPI |
| `writing` | Manage Writing Playground sessions, templates, themes, and token utilities. | Drafting tools, writing sessions, template-based writing aids. | Live OpenAPI |
| `manuscripts` | Manage writing projects, parts, chapters, and scenes. | Long-form writing, book planning, manuscript organization. | Live OpenAPI |
| `slides` | Generate and manage slide/presentation artifacts. | Presentation generation, deck workflows, teaching material. | Live OpenAPI |
| `outputs` | Create, inspect, and retrieve generated outputs and artifacts. | Artifact storage, export flows, generated report retrieval. | Live OpenAPI |
| `outputs-templates` | Create and preview reusable output templates. | Report templates, structured output formatting, reusable exports. | Live OpenAPI |
| `data-tables` | Create and manage data table generation jobs and CRUD. | Structured exports, tabular analysis, generated datasets. | Live OpenAPI |
| `kanban` | Manage boards, lists, cards, labels, links, checklists, comments, and workflow controls. | Project boards, source-linked tasks, lightweight planning. | Live OpenAPI |
| `skills` | Manage skill metadata and skill-facing API surfaces. | Assistant skills, tool catalogs, guided capabilities. | Experimental surface |
| `translation` | Translate text through configured translation providers. | Multilingual workflows, translation utilities, localization helpers. | Live OpenAPI |

## Automation, Jobs, And Integrations

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `jobs` | Manage background jobs, queues, admin sweeps, retries, and job status. | Async processing dashboards, worker monitoring, long-running tasks. | Live OpenAPI |
| `workflows` | Define and execute workflow scaffolding. | Multi-step internal orchestration, reusable processing flows. | Experimental surface |
| `scheduler` | Manage scheduler-backed workflow execution. | Dependency-aware internal tasks, registered task handlers. | Live OpenAPI |
| `scheduled-tasks` | Control user-visible scheduled task automation. | Recurring imports, repeated jobs, schedule management. | Experimental surface |
| `items` | Manage generic item records. | Lightweight item collections and cross-feature references. | Live OpenAPI |
| `tasks` | Create and list reminder-style tasks. | Reminders, task inboxes, notification flows. | [Reminder and notifications API](Reminder_Notifications_API.md) |
| `notifications` | List notification inbox entries and stream notification events. | Realtime notification UIs, task alerts, inbox badges. | [Reminder and notifications API](Reminder_Notifications_API.md) |
| `email` | Process and search email content. | Email ingestion, operator search, inbox research workflows. | [Email processing API](Email_Processing_API.md) |
| `meetings` | Manage meeting-intelligence workflows. | Meeting notes, transcript workflows, action item extraction. | [Meeting intelligence guide](../User_Guides/Server/Meeting_Intelligence_User_Guide.md) |
| `slack` | Integrate with Slack ingestion or interaction surfaces. | Team-source ingestion, chatops-style workflows. | Experimental integration |
| `discord` | Integrate with Discord ingestion or interaction surfaces. | Community-source ingestion, bot workflows. | Experimental integration |
| `telegram` | Integrate with Telegram ingestion or interaction surfaces. | Personal capture, bot workflows, message-source ingestion. | Experimental integration |
| `integrations` | Manage integration control-plane records. | Connector configuration, integration status, provider setup. | Experimental surface |
| `watchlists` | Track recurring sources, runs, alert rules, and monitored topics. | Monitoring sources, alerting, recurring research. | [Watchlists API](Watchlists_API.md) |

## Admin, Governance, And Operations

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `admin` | Run administrative operations and diagnostics. | Admin dashboards, system maintenance, operator tooling. | [Admin organizations and teams](Admin_Orgs_Teams.md) |
| `audit` | Export, count, and inspect audit records. | Compliance review, admin traceability, security investigations. | [Audit export API](Audit_Export.md) |
| `monitoring` | Read OpenTelemetry and JSON monitoring data. | Monitoring dashboards, service health checks, observability clients. | Live OpenAPI |
| `metrics` | Expose metrics and Prometheus-compatible telemetry. | Grafana dashboards, operational alerts, metrics scraping. | Live OpenAPI |
| `billing` | Manage billing and subscription surfaces. | Plan administration, subscription records, billing dashboards. | Admin-only surface |
| `privileges` | Manage privilege and permission definitions. | RBAC administration, capability audits, permission repair. | Admin-only surface |
| `resource-governor` | Inspect and manage resource governance diagnostics. | Quota debugging, rate-limit visibility, resource policy checks. | Admin-only surface |
| `llm` | Discover configured LLM providers and model capabilities. | Provider pickers, routing UIs, model availability checks. | [Providers API](Providers_API_Documentation.md) |
| `llamacpp` | Manage llama.cpp helper and model-serving endpoints. | Local model operations, llama.cpp health and control. | [llama.cpp integration modes](llamacpp_integration_modes.md) |
| `mcp-unified` | Use the unified MCP API surface with JWT/RBAC. | Tool execution, MCP status, MCP metrics, WebSocket clients. | Live OpenAPI |
| `mcp-hub` | Manage MCP hub profiles, external servers, tools, and user-facing MCP connections. | MCP client setup, external server management, tool availability. | Live OpenAPI |
| `mcp-catalogs` | Manage MCP tool catalogs for org or team leads. | Curated tool sets, team capability catalogs, MCP governance. | Live OpenAPI |
| `tools` | Access utility tooling endpoints. | Tool discovery, small API utilities, development helpers. | [Tools API](Tools_API_Documentation.md) |
| `moderation` | Check content against moderation policies, rules, review queues, and enforcement helpers. | Safety checks, policy testing, moderation review workflows. | Live OpenAPI |

## Experimental And Advanced Surfaces

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `acp` | Use experimental Agent Client Protocol session endpoints. | Agent sessions, protocol experiments, external agent clients. | Experimental surface |
| `acp-schedules` | Manage ACP schedules. | Scheduled agent sessions and recurring protocol activity. | Experimental surface |
| `acp-triggers` | Manage ACP triggers and webhook-style activation. | Event-driven agents, trigger testing, automation hooks. | Experimental surface |
| `acp-permissions` | Manage ACP permission policy endpoints. | Agent authorization, protocol permission checks. | Experimental surface |
| `acp-multiplex` | Use ACP multiplexing endpoints. | Multi-agent protocol routing and session fan-out. | Experimental surface |
| `agent-orchestration` | Coordinate higher-level agent orchestration APIs. | Multi-agent flows, orchestration experiments, agent work routing. | Experimental surface |
| `sandbox` | Run sandbox jobs and inspect artifacts, diagnostics, and execution controls. | Isolated execution, artifact review, workspace diagnostics. | [Sandbox API](Sandbox_API.md) |
| `prototype-workspaces` | Use prototype workspace collaboration endpoints. | Experimental workspace sharing and collaboration flows. | Live OpenAPI |
| `sharing` | Clone, share, revoke, and manage shared workspace resources. | Collaborative research, workspace transfer, controlled sharing. | Experimental surface |
| `workspaces` | Create and manage research workspaces, memberships, migrations, and active context. | Research Workspace clients, team workspaces, active-context selection. | [Feature map](../User_Guides/Feature_Map.md) |
| `guardian` | Configure family guardrails and guardian controls. | Household safety setup, guarded profiles, restricted modes. | Experimental surface |
| `self-monitoring` | Use self-monitoring endpoints for guarded experiences. | Personal limits, safety status, monitoring dashboards. | Experimental surface |
| `vn-capabilities` | Inspect visual-novel platform capability state. | VN client capability detection and setup checks. | Live OpenAPI |
| `vn-assets` | Manage visual-novel asset pack metadata. | VN asset libraries, asset pack browsing, runtime asset selection. | Live OpenAPI |
| `vn-scripts` | Author and manage visual-novel scripts. | VN script editing, branching stories, script validation. | Live OpenAPI |
| `vn-policy` | Manage VN policy profiles and preflight checks. | Safety checks for VN content and runtime policy gating. | Live OpenAPI |
| `vn-play` | Use visual-novel runtime play endpoints. | VN gameplay clients, runtime state, play-session control. | Live OpenAPI |

## Compatibility And Internal Tags

These tags may appear in Swagger/ReDoc because the server supports multiple runtime profiles, test routers, aliases, or future module splits.

| Module | What it lets you do | Common uses | Docs |
|--------|---------------------|-------------|------|
| `audio-ws` | Compatibility alias for streaming audio routes in minimal router profiles. | Test and compatibility routing. | Internal/test surface |

## Coverage Notes

- The guide covers the router tags found in the v1 router groups as of this documentation pass.
- Some modules share one documentation page because they are part of the same user workflow.
- `Live OpenAPI` means the module has live request/response details in `/docs` and `/redoc`, but no dedicated markdown guide in this index yet.
- `Experimental surface`, `Admin-only surface`, `Debug/internal surface`, and `Internal/test surface` are labels to help users avoid mistaking support APIs for primary workflows.
