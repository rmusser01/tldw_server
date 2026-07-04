# Page And Feature Index

This index maps WebUI and extension pages to user-facing capabilities. It is grouped by workflow rather than by source file.

## Start, Account, And Settings

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/` Home | WebUI, extension options, sidepanel entry | Resolve into setup, home, or the compact extension home state. | First launch, returning-user entry, sidepanel home. | [Start, account, and settings](Start_Account_Settings.md) |
| `/setup` | WebUI, extension options | Connect to a server and complete first-run readiness checks. | Local setup, API URL/API key entry, setup recovery. | [Getting started](../../Getting_Started/README.md) |
| `/login`, `/signup` | Hosted-only or multi-user | Authenticate or create an account when login flows are enabled. | Multi-user deployments, hosted account entry. | [Authentication setup](../Server/Authentication_Setup.md) |
| `/account`, `/profile` | Hosted-only or authenticated user | Inspect account identity, usage, roles, permissions, and profile state. | Account review, permission troubleshooting. | [Start, account, and settings](Start_Account_Settings.md) |
| `/privileges` | Advanced self-hosted | Inspect authorization privileges in deployments that expose RBAC. | Permission debugging, admin-assisted troubleshooting. | [AuthNZ guide](../../API-related/AuthNZ-API-Guide.md) |
| `/config` | Advanced self-hosted | Inspect deployment configuration outside the main settings area. | Server capability review, diagnostics. | [API notes](../../API-related/API_Notes.md) |
| `/settings` and settings subpages | WebUI, extension options, sidepanel settings | Configure server URL, auth, models, chat behavior, knowledge, RAG, speech, UI, quick ingest, image generation, sharing, and health checks. | Day-to-day configuration, provider setup, troubleshooting. | [Start, account, and settings](Start_Account_Settings.md) |
| `/billing` | Hosted-only | Manage hosted billing or subscription flows. | Hosted account management. | Hosted-only surface |
| `/404` | WebUI recovery | Recover from unknown routes. | Bad links, stale bookmarks. | Live route only |

## Chat, Characters, And Assistants

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/chat` | WebUI, extension options, sidepanel | Run the primary chat workspace with model, context, tool, web search, and history controls. | General assistants, document-aware chat, model testing. | [Chat pages](../WebUI_Extension/Chat_Pages.md) |
| `/chat/agent` | WebUI | Open the agent-oriented chat route. | Agent chat experiments and tool-backed conversations. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/quick-chat-popout` | Advanced self-hosted | Use focused quick chat outside the full workspace. | Lightweight chat window, fast prompts. | [Chat, characters, and assistants](Chat_Characters_Assistants.md) |
| `/persona` | WebUI, extension options, sidepanel | Use persona-backed assistant flows. | Personal assistant behavior, voice/persona continuity. | [Personas](../Server/Personas_User_Guide.md) |
| `/companion` and `/companion/conversation` | WebUI, extension options, sidepanel | Use companion-oriented conversations and compact sidepanel companion flows. | Ongoing assistant state, browser-adjacent companion sessions. | [Chat, characters, and assistants](Chat_Characters_Assistants.md) |
| `/characters` | WebUI, extension options | Manage character cards, roleplay setup, and character libraries. | Character chat, persona setup, roleplay assets. | [Character quickstart](../WebUI_Extension/Character_Roleplay_Quickstart.md) |
| `/agents`, `/agent-tasks` | Advanced self-hosted | Inspect agent registry and task state. | Agent orchestration experiments, task status. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/chat-workflows` | Advanced self-hosted | Manage chat workflow definitions and entry points. | Guided conversations, repeatable assistant flows. | [Workflow examples](../WebUI_Extension/Workflows_Examples.md) |
| `/chat-workspace` | Advanced self-hosted | Use workspace-focused chat with runtime, approvals, and staged context. | Project chat, tool approval review, workspace-bound conversations. | [Chat, characters, and assistants](Chat_Characters_Assistants.md) |
| `/dictionaries`, `/world-books` | WebUI, extension options | Manage chat dictionaries and lore/world-book context. | Consistent terminology, roleplay context, reusable background. | [Chat dictionaries](../WebUI_Extension/Chat_Dictionaries_Guide.md) |

## Knowledge, Media, And Sources

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/knowledge` | WebUI, extension options | Ask cited questions over selected knowledge sources. | Library Q&A, evidence review, answer export. | [Knowledge QA guide](../WebUI_Extension/Knowledge_QA_Guide.md) |
| `/search` | Legacy alias | Compatibility entry for Knowledge QA. | Old bookmarks and route compatibility. | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| `/research` | Advanced self-hosted | Work with research runs and provider-backed discovery. | Long-running research, source discovery. | [Research API](../../API-related/API_Tags_Index.md) |
| `/workspaces`, `/research-workspace`, `/document-workspace` | Advanced or labs | Manage research/project workspaces and focused document workspaces. | Source organization, project context, document-centered research. | [Prototype workspaces](../Prototype_Workspaces.md) |
| `/media`, `/media-multi`, `/media/[id]/view` | WebUI, extension options | Browse, inspect, and bulk-review ingested media. | Media library review, transcript/document inspection. | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| `/review`, `/media-trash` | Advanced self-hosted | Review queued media and recover or clean deleted media. | Cleanup, curation, recovery. | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| `/items`, `/collections`, `/reading`, `/notes` | WebUI, mixed availability | Organize library objects, saved reading, highlights, and notes. | Reading queues, notebook workflows, source-linked notes. | [Reading API](../../API-related/Reading_List_API.md) |
| `/sources`, `/sources/new`, `/sources/[sourceId]` | WebUI, extension options | Manage ingestion sources and source details. | Recurring imports, source status, folder or feed setup. | [Ingestion sources API](../../API-related/Ingestion_Sources_API.md) |
| `/connectors`, `/connectors/browse`, `/connectors/jobs`, `/connectors/sources` | Advanced self-hosted | Browse connector placeholders, jobs, and source adapters. | External repositories, third-party source workflows. | [Knowledge, media, and sources](Knowledge_Media_Sources.md) |
| `/shared`, `/share/[token]` | Advanced self-hosted | View shared resources and public share links. | Collaboration, shared research, read-only links. | [Organizations and sharing](../Server/Organizations_and_Sharing.md) |

## Audio, Speech, And Audiobooks

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/speech` | WebUI, extension options | Open the speech overview and readiness surface. | Choose between STT and TTS, check voice setup. | [Audio, speech, and audiobooks](Audio_Speech_Audiobooks.md) |
| `/stt` | WebUI, extension options | Transcribe files or use configured speech-to-text workflows. | Meeting/audio transcription, dictation workflows. | [STT/TTS quickstart](../WebUI_Extension/Getting-Started-STT_and_TTS.md) |
| `/tts` | WebUI, extension options | Generate speech from text using configured providers. | Narration, voice previews, generated audio. | [TTS getting started](../WebUI_Extension/TTS_Getting_Started.md) |
| `/audio` | Legacy alias | Compatibility route for the speech page family. | Old bookmarks and audio-route compatibility. | [Audio, speech, and audiobooks](Audio_Speech_Audiobooks.md) |
| `/audiobook-studio` | Advanced self-hosted | Build long-form audiobook projects with chapters, voices, generation, and output review. | Audiobook creation, chapterized narration. | [Audio, speech, and audiobooks](Audio_Speech_Audiobooks.md) |

## Study, Writing, And Artifacts

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/evaluations` | Advanced self-hosted | Set up evaluation workflows and inspect results. | RAG evals, model quality checks, benchmark review. | [Evaluations user guide](../Server/Evaluations_User_Guide.md) |
| `/flashcards` | WebUI, extension options, sidepanel | Create, import, review, and study flashcards. | Study decks, spaced review, quick sidepanel review. | [Flashcards guide](../WebUI_Extension/Flashcards_Study_Guide.md) |
| `/quiz` | WebUI, extension options | Create, edit, take, and review quizzes. | Knowledge checks, generated study workflows. | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| `/prompts` | WebUI, extension options | Manage prompt library and prompt studio tabs. | Prompt cataloging, testing, optimization. | [Prompt Studio API](../../API-related/Prompt_Studio_API.md) |
| `/prompt-studio` | Legacy alias | Redirects to `/prompts?tab=studio`. | Old Prompt Studio links. | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| `/chatbooks`, `/chatbooks-playground` | Advanced self-hosted | Import/export chatbooks and test chatbook content. | Backups, OpenWebUI imports, portable conversation bundles. | [Chatbook guide](../WebUI_Extension/Chatbook_User_Guide.md) |
| `/writing-playground` | Advanced self-hosted | Draft and transform writing with templates and themes. | Writing sessions, manuscript support. | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| `/presentation-studio` and child routes | Advanced self-hosted | Create, edit, and export slide/presentation artifacts. | Deck generation, project editing. | [Slides API](../../API/Slides.md) |
| `/data-tables`, `/kanban`, `/repo2txt` | Advanced self-hosted | Generate/edit tables, manage boards, and export repository text. | Structured outputs, project boards, repository context capture. | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |
| `/content-review` | Advanced self-hosted | Review generated or queued content drafts. | Editorial workflows, moderation-assisted review. | [Study, writing, and artifacts](Study_Writing_Artifacts.md) |

## Automation, Admin, And Operations

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/integrations` | Advanced self-hosted | Discover and configure integration surfaces. | Connector setup, integration status. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/scheduled-tasks`, `/scheduled-tasks/results` | Advanced self-hosted | Manage schedules and inspect scheduled run results. | Recurring jobs, automation review. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/watchlists` | Experimental/labs | Monitor sources, runs, alert rules, and recurring topics. | Watchlists, reporting, alert-driven research. | [Watchlists API](../../API-related/Watchlists_API.md) |
| `/workflow-editor` | Advanced self-hosted | Edit workflow definitions. | Multi-step processing, reusable automations. | [Workflow examples](../WebUI_Extension/Workflows_Examples.md) |
| `/mcp-hub` | Advanced self-hosted | Configure MCP hub profiles, external servers, and tool access. | Tool setup, MCP operations. | [MCP guide](../../MCP/Unified/Developer_Guide.md) |
| `/acp-playground` | Advanced self-hosted | Test Agent Client Protocol sessions, tools, permissions, and workspaces. | ACP development, protocol experiments. | [Getting started with ACP](../Integrations_Experiments/Getting_Started_with_ACP.md) |
| `/model-playground`, `/skills` | Advanced self-hosted | Test model behavior and inspect skills. | Model comparison, skill discovery. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/notifications` | WebUI | View notification inbox and alerts. | Task alerts, automation status, background job notices. | [Notifications API](../../API-related/Reminder_Notifications_API.md) |
| `/moderation`, `/moderation/rules`, `/moderation-playground`, `/claims-review` | Advanced self-hosted | Review moderation cases, content rules, safety tests, and claims. | Safety review, claim verification, policy testing. | [Automation, admin, and operations](Automation_Admin_Operations.md) |
| `/admin/*` | Admin/operator | Manage server, API keys, billing, data ops, integrations, local model runtimes, maintenance, monitoring, orgs, RBAC, rate limits, sources, usage, and watchlists. | Shared-server administration, local runtime operations. | [Organization administration](../Server/Organization_Administration.md) |

## Extension Sidepanel

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| Extension options | Extension options | Configure server URL, auth, timeouts, and shared option routes. | First-run extension setup, host permission recovery. | [Extension docs](https://github.com/rmusser01/tldw_server/tree/main/apps/extension/docs) |
| Sidepanel `/` | Extension sidepanel | Resolve to the compact sidepanel home. | Browser-adjacent starting point. | [Extension sidepanel](Extension_Sidepanel.md) |
| Sidepanel `/chat` | Extension sidepanel | Chat from the browser sidepanel, including page-aware workflows. | Ask about current page, quick chat, RAG snippets. | [Sidebar docs](https://github.com/rmusser01/tldw_server/blob/main/apps/extension/docs/sidebar/index.md) |
| Sidepanel `/clipper` | Extension sidepanel | Clip or send page content to the server. | Save pages, links, and browser context. | [Extension sidepanel](Extension_Sidepanel.md) |
| Sidepanel `/persona`, `/companion`, `/agent` | Extension sidepanel | Use compact assistant, persona, companion, and agent flows. | Browser-side assistant sessions. | [Extension sidepanel](Extension_Sidepanel.md) |
| Sidepanel `/flashcards` | Extension sidepanel | Review flashcards in a compact browser panel. | Quick study while browsing. | [Flashcards guide](../WebUI_Extension/Flashcards_Study_Guide.md) |
| Context menu actions | Extension background/content scripts | Open sidebar, open WebUI, send to server, process, transcribe, or summarize page media. | Browser capture, media ingestion, transcription handoff. | [Extension sidepanel](Extension_Sidepanel.md) |

## Experimental, Specialized, Hosted, Alias, And Debug

| Page or feature | Surface/status | What it lets you do | Common uses | More docs |
| --- | --- | --- | --- | --- |
| `/vn-assets`, `/vn-scripts`, `/vn-play` | Experimental/labs | Manage visual novel assets, scripts, and play sessions. | VN authoring and runtime experiments. | [VN API](../../API/VN.md) |
| `/prototype-workspaces` | Experimental/labs | Use prototype collaboration/workspace routes. | Workspace experiments. | [Prototype workspaces](../Prototype_Workspaces.md) |
| `/for/journalists`, `/for/osint`, `/for/researchers` | Public/hosted-oriented | Show persona-specific public landing pages. | Audience-specific entry points. | [Experimental and specialized](Experimental_And_Specialized.md) |
| `/billing/*`, `/auth/*` | Hosted-only | Handle hosted billing and auth callbacks. | Hosted account flows. | Hosted-only surface |
| `/audio`, `/search`, `/prompt-studio`, `/review`, `/moderation-playground` | Legacy alias or compatibility | Preserve old links while canonical pages evolve. | Bookmark compatibility, route migration. | [Experimental and specialized](Experimental_And_Specialized.md) |
| `/composer-variants-preview`, `/onboarding-test`, `/__debug__/*`, sidepanel `/error-boundary-test` | Internal QA/debug | Preview or test UI states. | QA and developer diagnostics only. | Internal QA/debug surface |
