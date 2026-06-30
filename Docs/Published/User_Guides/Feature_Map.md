# Feature Map

Use this page when you know what you want to do, but not which tldw_server surface or guide covers it.

Primary surfaces:

- **WebUI**: the main self-hosted browser app for chat, media, knowledge work, audio, study, and admin workflows.
- **Browser extension**: capture and sidepanel workflows that connect to your tldw_server instance.
- **Server API**: FastAPI endpoints, including OpenAI-compatible Chat, Audio, Embeddings, and vector-store APIs.
- **Admin/operator tools**: setup, authentication, hardening, usage, monitoring, backups, and multi-user operations.

## Setup And Connection

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Choose an install path | Docs | [Self-hosting profiles](../Getting_Started/README.md) |
| Run the shortest local self-hosted path | Docker + WebUI | [Docker single-user + WebUI](../Getting_Started/Profile_Docker_Single_User.md) |
| Run a contributor/debug setup | Local API + WebUI | [Local single-user](../Getting_Started/Profile_Local_Single_User.md) |
| Run a shared server | Docker + Postgres | [Docker multi-user + Postgres](../Getting_Started/Profile_Docker_Multi_User_Postgres.md) |
| Configure auth | Server/admin | [Authentication setup](Server/Authentication_Setup.md) |
| Connect the browser extension | Extension | [Extension docs](https://github.com/rmusser01/tldw_server/tree/main/apps/extension/docs) |

## Chat With Models

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Chat in the WebUI | WebUI | [WebUI user guide](WebUI_Extension/User_Guide.md) |
| Understand chat pages | WebUI | [Chat pages](WebUI_Extension/Chat_Pages.md) |
| Use the OpenAI-compatible chat API | API | [Chat API documentation](../API-related/Chat_API_Documentation.md) |
| List configured providers | API/WebUI settings | [Providers API documentation](../API-related/Providers_API_Documentation.md) |
| Use characters and roleplay | WebUI | [Character roleplay quickstart](WebUI_Extension/Character_Roleplay_Quickstart.md) |
| Tune prompt behavior | WebUI/API | [Prompt engineering notes](WebUI_Extension/Prompt_Engineering_Notes.md) |

## Add Sources And Media

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Add videos, audio, documents, or web pages | WebUI Media / API | [Media to RAG evals workflow](Server/Media_to_RAG_Evals_Workflow.md) |
| Use async media ingest jobs | API | [Media ingest jobs API](../API-related/Media_Ingest_Jobs_API.md) |
| Scrape and ingest websites | WebUI/API | [Web scraping and ingestion](Server/Web_Scraping_Ingestion_Guide.md) |
| Configure reusable ingestion sources | API/admin | [Ingestion sources API](../API-related/Ingestion_Sources_API.md) |
| Tune chunking | WebUI/API | [Chunking templates user guide](Server/Chunking_Templates_User_Guide.md) |
| Read EPUBs | WebUI | [EPUB reader guide](WebUI_Extension/EPUB_Reader_Guide.md) |

## Search And Ask Questions Over Knowledge

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Search ingested content | WebUI/API | [RAG API guide](../API-related/RAG-API-Guide.md) |
| Configure production RAG | Server/admin | [RAG production configuration](Server/RAG_Production_Configuration_Guide.md) |
| Compare or tune retrieval quality | WebUI/API | [RAG evals playbook](Server/RAG_Evals_Playbook.md) |
| Create embeddings | API | [Embeddings API documentation](../API-related/Embeddings_API_Documentation.md) |
| Query vector stores | API | [Vector stores admin and query](../API-related/Vector_Stores_Admin_and_Query.md) |
| Use docs Q&A help in the WebUI | WebUI | [Quick Chat Docs Assistant](WebUI_Extension/Quick_Chat_Docs_Assistant_Guide.md) |

## Audio, Speech, And Voice

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Set up STT and TTS for the first time | WebUI/server | [Getting started with STT and TTS](WebUI_Extension/Getting-Started-STT_and_TTS.md) |
| Configure CPU audio dependencies | Server | [CPU audio setup](../Getting_Started/First_Time_Audio_Setup_CPU.md) |
| Configure GPU or accelerated audio | Server | [GPU audio setup](../Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md) |
| Transcribe files or streams | API | [Audio transcription API](../API-related/Audio_Transcription_API.md) |
| Generate speech | WebUI/API | [TTS getting started](WebUI_Extension/TTS_Getting_Started.md) |
| Run audio jobs | API/admin | [Audio jobs API](../API-related/Audio_Jobs_API.md) |

## Study, Evaluate, And Review

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Run evaluations | WebUI/API | [Evaluations user guide](Server/Evaluations_User_Guide.md) |
| Use unified evaluation APIs | API | [Evaluations API unified reference](../API-related/Evaluations_API_Unified_Reference.md) |
| Create benchmark runs | WebUI/API/extension | [Benchmark creation and runs](Server/Benchmark_Creation_API_WebUI_Extension_Guide.md) |
| Study with flashcards | WebUI/extension | [Flashcards study guide](WebUI_Extension/Flashcards_Study_Guide.md) |
| Use kanban boards | WebUI | [Kanban board guide](WebUI_Extension/Kanban_Board_Guide.md) |

## Create And Manage Knowledge Artifacts

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Export, import, or migrate chats | WebUI/API | [Chatbook user guide](WebUI_Extension/Chatbook_User_Guide.md) |
| Use Chatbook tools | WebUI/API | [Chatbook tools getting started](WebUI_Extension/Chatbook_Tools_Getting_Started.md) |
| Import Google Keep notes | WebUI | [Google Keep import/export](WebUI_Extension/Google_Keep_Notes_Import_Export_Guide.md) |
| Manage reading items | API/WebUI | [Reading list API](../API-related/Reading_List_API.md) |
| Use prompt projects and tests | API/WebUI | [Prompt Studio API](../API-related/Prompt_Studio_API.md) |

## Automate And Integrate

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Build workflow examples | WebUI/API | [Workflows examples](WebUI_Extension/Workflows_Examples.md) |
| Manage RSS or Atom feed ingestion | API | [Collections feeds API](../API-related/Collections_Feeds_API.md) |
| Use watchlists | API/WebUI | [Watchlists API](../API-related/Watchlists_API.md) |
| Connect external agent clients | Server/API | [Getting started with ACP](Integrations_Experiments/Getting_Started_with_ACP.md) |
| Use MCP tooling | API/admin | [MCP status and tools](../API-related/Tools_API_Documentation.md) |

## Administer A Shared Server

| Goal | Start in | Best next guide |
| --- | --- | --- |
| Harden a production deployment | Server/admin | [Production hardening checklist](Server/Production_Hardening_Checklist.md) |
| Manage organizations and sharing | Admin/WebUI/API | [Organizations and sharing](Server/Organizations_and_Sharing.md) |
| Administer organizations | Admin | [Organization administration](Server/Organization_Administration.md) |
| Manage bring-your-own-key provider access | Admin/user settings | [BYOK user guide](Server/BYOK_User_Guide.md) |
| Understand usage reporting | Admin/API | [Usage module](Server/Usage_Module.md) |
| Back up SQLite deployments | Server/admin | [Backups using Litestream](Server/Backups_Using_Litestream.md) |
| Monitor the server | Admin/operator | [Metrics cheatsheet](https://rmusser01.github.io/tldw_server/Monitoring/Metrics_Cheatsheet/) |

## API Reference Entry Points

- [API documentation index](../API-related/API_README.md)
- [OpenAPI tag index](../API-related/API_Tags_Index.md)
- Live API docs on a running server: `http://127.0.0.1:8000/docs`
