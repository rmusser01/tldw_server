# User Guides

This is the main map for user-facing tldw_server documentation. Start here when you need to choose a setup path, understand what the WebUI and browser extension can do, find the right API docs, or operate a shared server.

The short version:

1. Choose a setup profile.
2. Open the WebUI.
3. Complete a first successful chat.
4. Add your first source.
5. Use the [Feature Map](Feature_Map.md) to find the next workflow.

## Start Here

Most first-time users should use the Docker single-user + WebUI path:

- [Self-hosting profiles](../Getting_Started/README.md)
- [Docker single-user + WebUI](../Getting_Started/Profile_Docker_Single_User.md)
- [Local single-user](../Getting_Started/Profile_Local_Single_User.md)
- [Docker multi-user + Postgres](../Getting_Started/Profile_Docker_Multi_User_Postgres.md)

After the base server is healthy, complete these first-value steps:

1. Open the WebUI.
2. Finish the first-time setup flow.
3. Send one successful chat message.
4. Add a first source, such as a document, URL, video, or audio file.

Optional setup paths:

- [CPU audio setup](../Getting_Started/First_Time_Audio_Setup_CPU.md)
- [GPU or accelerated audio setup](../Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md)
- [Authentication setup](Server/Authentication_Setup.md)

## Choose Your Surface

| Surface | Use it for | Start here |
| --- | --- | --- |
| WebUI | Chat, media, sources, RAG, audio, study, writing, admin, and day-to-day workflows | [WebUI user guide](WebUI_Extension/User_Guide.md) |
| Browser extension | Page capture, sidepanel chat, and browser-adjacent workflows connected to your server | [Extension docs](https://github.com/rmusser01/tldw_server/tree/main/apps/extension/docs) |
| Server API | OpenAI-compatible APIs, ingestion, RAG, evaluations, automation, and integrations | [API documentation index](../API-related/API_README.md) |
| Admin/operator docs | Multi-user setup, hardening, organizations, usage, backups, monitoring, and deployment | [Production hardening checklist](Server/Production_Hardening_Checklist.md) |

## What Can I Do?

Use the [Feature Map](Feature_Map.md) for the full task-oriented matrix.

Common workflows:

- **Chat with models and characters**: use [Chat pages](WebUI_Extension/Chat_Pages.md), [Chat API documentation](../API-related/Chat_API_Documentation.md), [Character roleplay quickstart](WebUI_Extension/Character_Roleplay_Quickstart.md), [Character cards and character chat](Server/Character_Cards_User_Guide.md), and [Personas](Server/Personas_User_Guide.md).
- **Add sources and media**: use [Media to RAG evals workflow](Server/Media_to_RAG_Evals_Workflow.md), [Web scraping and ingestion](Server/Web_Scraping_Ingestion_Guide.md), [Media ingest jobs API](../API-related/Media_Ingest_Jobs_API.md), and [Bulk conference playlist ingest](Bulk_Conference_Playlist_Ingest.md).
- **Search and ask questions over knowledge**: use [RAG API guide](../API-related/RAG-API-Guide.md), [RAG production configuration](Server/RAG_Production_Configuration_Guide.md), and [Quick Chat Docs Assistant](WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md).
- **Transcribe and generate speech**: use [Getting started with STT and TTS](WebUI_Extension/Getting-Started-STT_and_TTS.md), [Audio transcription API](../API-related/Audio_Transcription_API.md), and [TTS getting started](WebUI_Extension/TTS_Getting_Started.md).
- **Study, evaluate, and review outputs**: use [Evaluations user guide](Server/Evaluations_User_Guide.md), [Benchmark creation and runs](Server/Benchmark_Creation_API_WebUI_Extension_Guide.md), [Evaluations API unified reference](../API-related/Evaluations_API_Unified_Reference.md), and [Flashcards Study Guide](WebUI_Extension/Flashcards_Study_Guide.md).
- **Create and manage knowledge artifacts**: use [Chatbook user guide](WebUI_Extension/Chatbook_User_Guide.md) for chatbook export/import, OpenWebUI chat JSON and database import, and post-import attachment hydration; use [Prompt Studio API](../API-related/Prompt_Studio_API.md) and [Reading list API](../API-related/Reading_List_API.md) for adjacent workflows.
- **Automate and integrate**: use [Workflows examples](WebUI_Extension/Workflows_Examples.md), [Collections feeds API](../API-related/Collections_Feeds_API.md), and [Getting started with ACP](Integrations_Experiments/Getting_Started_with_ACP.md).
- **Prototype workspace flows**: use [Prototype Workspaces User Guide](Prototype_Workspaces.md) to run isolated workspace experiments.
- **Administer a shared server**: use [Organizations and sharing](Server/Organizations_and_Sharing.md), [BYOK user guide](Server/BYOK_User_Guide.md), [Usage module](Server/Usage_Module.md), and [Metrics cheatsheet](https://rmusser01.github.io/tldw_server/Monitoring/Metrics_Cheatsheet/).

## Troubleshooting

Start with the guide that matches the failing surface:

- Setup and profile issues: [self-hosting profiles](../Getting_Started/README.md) and [troubleshooting](../Getting_Started/TROUBLESHOOTING.md).
- Authentication and access issues: [Authentication setup](Server/Authentication_Setup.md), [Multi-user Postgres setup](Server/Multi-User_Postgres_Setup.md), and [Multi-user SQLite setup](Server/Multi-User_SQLite_Setup.md).
- Provider and model issues: [BYOK user guide](Server/BYOK_User_Guide.md), [OpenAI OAuth first-time setup](Server/OpenAI_OAuth_First_Time_Setup.md), [local LLM setup](Integrations_Experiments/Setting_up_a_local_LLM.md), and [Providers API documentation](../API-related/Providers_API_Documentation.md).
- Media and ingestion issues: [Web scraping and ingestion](Server/Web_Scraping_Ingestion_Guide.md), [Media ingest jobs API](../API-related/Media_Ingest_Jobs_API.md), and [Chunking templates user guide](Server/Chunking_Templates_User_Guide.md).
- Audio issues: [CPU audio setup](../Getting_Started/First_Time_Audio_Setup_CPU.md), [GPU or accelerated audio setup](../Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md), and [TTS setup guide](WebUI_Extension/TTS-SETUP-GUIDE.md).
- Production and operations issues: [Production hardening checklist](Server/Production_Hardening_Checklist.md), [Backups using Litestream](Server/Backups_Using_Litestream.md), and [Long-term admin guide](../Deployment/Long_Term_Admin_Guide.md).

## For Builders

- Live API docs on a running server: `http://127.0.0.1:8000/docs`
- [API documentation index](../API-related/API_README.md)
- [OpenAPI tag index](../API-related/API_Tags_Index.md)
- [Code documentation index](../Code_Documentation/index.md)
- [Documentation site guide](../Code_Documentation/Docs_Site_Guide.md)
- [Python SDK README](https://github.com/rmusser01/tldw_server/tree/main/sdks/python)
- [TypeScript SDK README](https://github.com/rmusser01/tldw_server/tree/main/sdks/typescript)
