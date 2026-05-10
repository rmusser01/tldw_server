# Feature Status Matrix

Legend
- Working: Stable and actively supported
- WIP: In active development; APIs or behavior may evolve
- Experimental: Available behind flags or with caveats; subject to change

## Admin Reporting
- HTTP usage (daily): `GET /api/v1/admin/usage/daily`
- HTTP top users: `GET /api/v1/admin/usage/top`
- LLM usage log: `GET /api/v1/admin/llm-usage`
- LLM usage summary: `GET /api/v1/admin/llm-usage/summary` (group_by=`user|provider|model|operation|day`)
- LLM top spenders: `GET /api/v1/admin/llm-usage/top-spenders`
- LLM CSV export: `GET /api/v1/admin/llm-usage/export.csv`
- Grafana provisioning + dashboard import guide: `https://github.com/rmusser01/tldw_server/blob/main/Helper_Scripts/Samples/Grafana/README.md`
- Prometheus alert rules (daily spend thresholds): `https://github.com/rmusser01/tldw_server/blob/main/Helper_Scripts/Samples/Prometheus/alerts.yml`

## Media Ingestion

| Capability | Status | Notes | Links |
|---|---|---|---|
| URLs/files: video, audio, PDFs, EPUB, DOCX, HTML, Markdown, XML, MediaWiki | Working | Unified ingestion + metadata | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Ingestion_Media_Processing.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/media.py) |
| Ingest jobs + cancellation | Working | Background jobs with cancel support | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Media_Ingest_Jobs_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/media/ingest_jobs.py) |
| yt-dlp downloads + ffmpeg | Working | 1000+ sites via yt-dlp | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/Ingestion_Media_Processing/Video/Video_DL_Ingestion_Lib.py) |
| Adaptive/multi-level chunking | Working | Configurable size/overlap | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Chunking_Templates_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/chunking.py) |
| OCR on PDFs/images | Working | Tesseract baseline; optional dots.ocr/POINTS | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/OCR_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/ocr.py) |
| MediaWiki import | Working | Config via YAML | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Ingestion_Pipeline_MediaWiki.md) · [config](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/Config_Files/mediawiki_import_config.yaml) |
| Browser extension capture | WIP | Web capture extension | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Completed/Content_Collections_PRD.md) |

## Audio (STT/TTS)

| Capability | Status | Notes | Links |
|---|---|---|---|
| File-based transcription | Working | faster_whisper, NeMo, Qwen2Audio | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Audio_Transcription_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio.py) |
| Real-time WS transcription | Working | `WS /api/v1/audio/stream/transcribe` | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Audio_Transcription_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio.py) |
| Diarization + VAD | Working | Optional diarization, timestamps | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Ingestion_Pipeline_Audio.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio.py) |
| TTS (OpenAI-compatible) | Working | Streaming + non-streaming; providers: OpenAI, ElevenLabs, Kokoro, PocketTTS, LuxTTS, Higgs, Chatterbox, Dia, VibeVoice, VibeVoice Realtime, NeuTTS, IndexTTS2, Supertonic, Supertonic2, Qwen3-TTS, EchoTTS | [docs](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/TTS/TTS-README.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio.py) |
| Voice catalog + management | Working | `GET /api/v1/audio/voices/catalog` | [docs](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/TTS/README.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio.py) |
| Audio jobs queue | Working | Background audio processing | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Audio_Jobs_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audio_jobs.py) |
| Audiobooks | Working | Parse, jobs, subtitles, packaging | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Completed/Audiobook_Creation_PRD.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/audiobooks.py) |

## Voice Assistant

| Capability | Status | Notes | Links |
|---|---|---|---|
| Voice assistant (REST + WS) | WIP | `POST /api/v1/voice/command`, `WS /api/v1/voice/assistant` | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API/Voice_Assistant.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/voice_assistant.py) |

## Meeting Intelligence

| Capability | Status | Notes | Links |
|---|---|---|---|
| Meetings sessions/templates/artifacts API | Working | Dedicated ` /api/v1/meetings/* ` domain for v1 flows | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Meetings_Developer_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/meetings.py) |
| Live meeting events (SSE + WS) | Working | SSE event feed + WS stream transport | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Meetings_Developer_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/Meetings/stream_adapter.py) |
| Offline finalize artifact generation | Working | Commit endpoint emits summary, actions, decisions, speaker stats | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Meetings_Developer_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/Meetings/artifact_service.py) |
| Sharing (Slack + generic webhook) | Working | Queue + retry pipeline with egress checks and DLQ worker | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Meetings_Developer_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/services/meetings_webhook_dlq_service.py) |
| Calendar/CRM integrations | WIP | Not in v1; later phase scope | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Meeting-Transcripts-PRD.md) |

## RAG & Search

| Capability | Status | Notes | Links |
|---|---|---|---|
| Full-text search (FTS5) | Working | Fast local search | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/RAG-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/rag_unified.py) |
| Embeddings + ChromaDB | Working | OpenAI-compatible embeddings | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Embeddings_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py) |
| Hybrid BM25 + vector + rerank | Working | Contextual retrieval | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/RAG-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/rag_unified.py) |
| Feedback (explicit + implicit) | Working | Explicit feedback + implicit signals; implicit can be disabled | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/feedback.py) |
| Vector Stores (OpenAI-compatible) | Working | Chroma/PG adapters | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Vector_Stores_Admin_and_Query.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/vector_stores_openai.py) |
| Media embeddings ingestion | Working | Create vectors from media | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/media_embeddings.py) |
| pgvector backend | Experimental | Optional backend | [code](https://github.com/rmusser01/tldw_server/tree/main/tldw_Server_API/app/core/RAG/rag_service/vector_stores/) |

## Chat & LLMs

| Capability | Status | Notes | Links |
|---|---|---|---|
| Chat Completions (OpenAI) | Working | Streaming supported | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Chat_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/chat.py) |
| Function calling / tools | Working | Tool schema validation | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Chat_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/chat.py) |
| Provider integrations (16+) | Working | Commercial + local | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Providers_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/llm_providers.py) |
| Local providers | Working | vLLM, llama.cpp, Ollama, etc. | [docs](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/LLM_Calls/README.md) · [code](https://github.com/rmusser01/tldw_server/tree/main/tldw_Server_API/app/core/LLM_Calls/) |
| Strict OpenAI compat filter | Working | Filter non-standard keys | [docs](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/LLM_Calls/README.md) |
| Providers listing | Working | `GET /api/v1/llm/providers` | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Providers_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/llm_providers.py) |
| Moderation endpoint | Working | Basic wrappers | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/moderation.py) |
| Chat Workflows (structured QA) | WIP | Saved templates, generated drafts, linear runs, moderated dialogue rounds, transcript view, built-in Socratic Dialogue starter, and explicit continue-to-chat handoff are available; UI remains beta and broader context resolution/resumable runs are still evolving. | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Chat_Workflows_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/chat_workflows.py) |

## Knowledge, Notes, Prompt Studio

| Capability | Status | Notes | Links |
|---|---|---|---|
| Notes + tagging | Working | Notebook-style notes | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/notes.py) |
| Notes graph API | Experimental | Graph queries are stubbed; manual links supported | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/notes_graph.py) |
| Writing Playground | WIP | Sessions, templates, themes, tokenization, wordclouds | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/writing.py) |
| Prompt library | Working | Import/export, versions, templates, bulk ops, sorting | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/prompts.py) |
| Prompt Studio: projects/prompts/tests | Working | Test cases + runs | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Prompt_Studio_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/prompt_studio_projects.py) |
| Prompt Studio: optimization + WS | Working | Live updates | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Prompt_Studio_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/prompt_studio_optimization.py) |
| Character cards & sessions | Working | SillyTavern-compatible | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/CHARACTER_CHAT_API_DOCUMENTATION.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/characters_endpoint.py) |
| Chatbooks import/export | Working | Backup/export plus OpenWebUI JSON chat import | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Chatbook_API_Documentation.md) · [guide](https://github.com/rmusser01/tldw_server/blob/main/Docs/User_Guides/WebUI_Extension/Chatbook_User_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/chatbooks.py) |
| Flashcards | Working | Decks/cards, APKG export | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/flashcards.py) |
| Quizzes | WIP | Quizzes, questions, attempts, generation | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/quizzes.py) |
| Reading & highlights | Working | Reading items mgmt | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Reading_List_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/reading.py) |
| Unified Items API | Working | Collections + Media DB view | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/items.py) |
| Kanban boards | Working | Boards, lists, cards, labels, checklists | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/User_Guides/WebUI_Extension/Kanban_Board_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/kanban_boards.py) |
| Collections feeds (RSS/Atom) | Working | Ingest feeds into collections | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Collections_Feeds_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/collections_feeds.py) |

## Evaluations

| Capability | Status | Notes | Links |
|---|---|---|---|
| G-Eval | Working | Unified eval API | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Evaluations_API_Unified_Reference.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py) |
| RAG evaluation | Working | Pipeline presets + metrics | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/RAG-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/evaluations_rag_pipeline.py) |
| OCR evaluation (JSON/PDF) | Working | Text + PDF flows | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/OCR_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py) |
| Embeddings A/B tests | Working | Provider/model compare | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Evaluations_API_Unified_Reference.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/evaluations_embeddings_abtest.py) |
| Response quality & datasets | Working | Datasets CRUD + runs | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Evaluations_API_Unified_Reference.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/evaluations_unified.py) |
| Claims extraction | Working | Answer-time claims engine | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Code_Documentation/Claims_API_and_Schema.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/claims.py) |

## Research & Web Scraping

| Capability | Status | Notes | Links |
|---|---|---|---|
| Web search (multi-provider) | Working | Google, DDG, Brave, Kagi, Tavily, Searx | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/research.py) |
| Aggregation/final answer | Working | Structured answer + evidence | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/research.py) |
| Academic paper search | Working | arXiv, BioRxiv/MedRxiv, PubMed/PMC, Semantic Scholar, OSF | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/paper_search.py) |
| Web scraping service | Working | Status, jobs, progress, cookies | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Completed/Content_Collections_PRD.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/web_scraping.py) |

## Connectors (External Sources)

| Capability | Status | Notes | Links |
|---|---|---|---|
| Google Drive connector | Working | OAuth2, browse/import | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/connectors.py) |
| Notion connector | Working | OAuth2, nested blocks→Markdown | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/connectors.py) |
| Connector policy + quotas | Working | Org policy, job quotas | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Completed/Content_Collections_PRD.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/connectors.py) |

## MCP Unified

| Capability | Status | Notes | Links |
|---|---|---|---|
| Tool execution APIs + WS | Working | Production MCP with JWT/RBAC | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/MCP/Unified/Developer_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py) |
| Catalog management | Working | Admin tool/permission catalogs | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/MCP/Unified/Modules.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/mcp_catalogs_manage.py) |
| Status/metrics endpoints | Working | Health + metrics | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/MCP/Unified/System_Admin_Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/mcp_unified_endpoint.py) |

## AuthNZ, Security, Admin/Ops

| Capability | Status | Notes | Links |
|---|---|---|---|
| Single-user (X-API-KEY) | Working | Simple local deployments | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/AuthNZ-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/auth.py) |
| Multi-user JWT + RBAC | Working | Users/roles/permissions | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/AuthNZ-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/auth.py) |
| API keys manager | Working | Create/rotate/audit | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/AuthNZ-API-Guide.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/admin/__init__.py) |
| Egress + SSRF guards | Working | Centralized guards | [docs](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/Security/README.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/core/Security/egress.py) |
| Audit logging & alerts | Working | Unified audit + alerts | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Audit_Configuration.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/admin/__init__.py) |
| Admin & Ops | Working | Users/orgs/teams, roles/perms, quotas, usage | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Admin_Orgs_Teams.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/admin/__init__.py) |
| Monitoring & metrics | Working | Prometheus text + JSON | [docs](../Monitoring/Metrics_Cheatsheet.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/metrics.py) |

## Sandbox & Tooling

| Capability | Status | Notes | Links |
|---|---|---|---|
| Sandbox (code interpreter) | Working | Spec 1.0/1.1, network policy | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Sandbox_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/sandbox.py) |

## Storage, Outputs, Watchlists, Workflows, UI

| Capability | Status | Notes | Links |
|---|---|---|---|
| SQLite defaults | Working | Local dev/small deployments | [code](https://github.com/rmusser01/tldw_server/tree/main/tldw_Server_API/app/core/DB_Management/) |
| PostgreSQL (AuthNZ, content) | Working | Postgres content mode | [docs](../Deployment/Long_Term_Admin_Guide.md) |
| Outputs: templates | Working | Markdown/HTML/MP3 via TTS | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/outputs_templates.py) |
| Outputs: artifacts | Working | Persist/list/soft-delete/purge | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/outputs.py) |
| File artifacts | WIP | Structured files + export lifecycle | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Storage_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/files.py) |
| Data tables | WIP | LLM-generated tables + async exports | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/data_tables.py) |
| Slides / presentations | WIP | Generate, version, export decks, store Presentation Studio `studio_data`, and run versioned render jobs for `mp4`/`webm` outputs | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API/Slides.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/slides.py) |
| Presentation Studio | WIP | Structured WebUI editor routes (`/presentation-studio`, `/presentation-studio/new`, `/presentation-studio/:projectId`) plus extension quick-start handoff (`/presentation-studio/start`); requires `hasPresentationStudio`, with video publish gated by `hasPresentationRender` | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API/Slides.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/apps/packages/ui/src/components/Option/PresentationStudio/) |
| Storage API | WIP | Folders, trash, quotas, downloads | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Storage_API_Documentation.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/storage.py) |
| Sync API | WIP | Change-log sync for client databases | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/API_Notes.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/sync.py) |
| Watchlists: sources/groups/tags | Working | CRUD + bulk import | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/watchlists.py) |
| Watchlists: jobs & runs | Working | Schedule, run, run details | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/watchlists.py) |
| Watchlists: templates & OPML | Working | Template store; OPML import/export | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/API-related/Watchlists_API.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/watchlists.py) |
| Watchlists: notifications | Experimental | Email/chatbook delivery | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Watchlists/Watchlist_PRD.md) |
| Workflows engine & scheduler | WIP | Defs CRUD, runs, scheduler | [docs](https://github.com/rmusser01/tldw_server/blob/main/Docs/Product/Completed/Workflows_PRD.md) · [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/workflows.py) |
| VLM backends listing | Experimental | `/api/v1/vlm/backends` | [code](https://github.com/rmusser01/tldw_server/blob/main/tldw_Server_API/app/api/v1/endpoints/vlm.py) |
| Next.js WebUI | WIP | Primary web client (`apps/tldw-frontend`) | [code](https://github.com/rmusser01/tldw_server/tree/main/apps/tldw-frontend/) |
| Admin UI | WIP | Unified admin dashboard (`admin-ui`) | [code](https://github.com/rmusser01/tldw_server/tree/main/admin-ui/) |
