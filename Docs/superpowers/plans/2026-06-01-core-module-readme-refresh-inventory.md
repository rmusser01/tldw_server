# Core Module README Refresh Inventory

Backlog: TASK-588
Design: Docs/superpowers/specs/2026-06-01-core-module-readme-refresh-design.md

## Legend

- README status: `existing`, `missing`, `created`, `refreshed`, `kept`
- Evidence status: `not inspected`, `inspected`
- Phase 2 priority: `high`, `medium`, `low`, `sufficient`

## Inventory

| Module | README status | Evidence status | Evidence inspected | Related endpoints/schemas/tests | Phase 2 priority | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Agent_Client_Protocol | created | inspected | Source: `app/core/Agent_Client_Protocol/{agent_registry.py,runner_client.py,sandbox_runner_client.py,config.py,events.py,event_bus.py,governance_filter.py,adapters/,consumers/,multiplex/}`; API/schema/docs inspected. | Endpoint: `app/api/v1/endpoints/agent_client_protocol.py` (`/acp`); schema: `app/api/v1/schemas/agent_client_protocol.py`; tests: `tests/Agent_Client_Protocol/`. | high | Protocol surface; prioritize for source-backed orientation. |
| Agent_Orchestration | created | inspected | Source: `app/core/Agent_Orchestration/{models.py,orchestration_service.py,completion_signals.py,artifact_promotion.py}`; DB and docs inspected. | Endpoint: `app/api/v1/endpoints/agent_orchestration.py` (`/agent-orchestration`); DB: `app/core/DB_Management/Orchestration_DB.py`; tests: `tests/Agent_Orchestration/`. | high | Orchestration module; prioritize for source-backed orientation. |
| Audio | created | inspected | Source: `app/core/Audio/{transcription_service.py,tts_service.py,streaming_service.py,quota_helpers.py,tokenizer_service.py,dictation_error_taxonomy.py}`; audio endpoint package inspected. | Endpoints: `app/api/v1/endpoints/audio/`; schemas: `audio_schemas.py`, `audio_health.py`, `audio_presets.py`; tests: `tests/Audio/`, `tests/AudioJobs/`. | medium | Feature module; inspect with related audio API and tests. |
| Audiobooks | created | inspected | Source: `app/core/Audiobooks/{tag_parser.py,subtitle_generator.py,subtitle_parser.py,alignment_utils.py}`; audiobook endpoint and schemas inspected. | Endpoint: `app/api/v1/endpoints/audio/audiobooks.py` (`/audiobooks`); schema: `app/api/v1/schemas/audiobook_schemas.py`; tests: `tests/Audiobooks/`. | medium | Feature module; inspect with ingestion and audio flows. |
| Audit | existing | not inspected | Pending. | Pending. | high | Operational and security-sensitive module. |
| AuthNZ | existing | not inspected | Pending. | Pending. | high | Security-sensitive authentication and authorization module. |
| Billing | existing | not inspected | Pending. | Pending. | high | Operational account and usage module. |
| Character_Chat | existing | not inspected | Pending. | Pending. | medium | Feature module with chat-facing behavior. |
| Chat | existing | not inspected | Pending. | Pending. | high | Broad user-facing LLM module. |
| Chat_Workflows | created | inspected | Source: `app/core/Chat_Workflows/{service.py,dialogue_orchestrator.py,question_renderer.py}`; dependency and DB paths inspected. | Endpoint: `app/api/v1/endpoints/chat_workflows.py` (`/api/v1/chat-workflows`); schema: `app/api/v1/schemas/chat_workflows.py`; deps: `app/api/v1/API_Deps/chat_workflows_deps.py`; tests: `tests/Chat_Workflows/`. | high | Chat orchestration module. |
| Chatbooks | existing | not inspected | Pending. | Pending. | high | Import/export and background data workflow module. |
| Chunking | existing | not inspected | Pending. | Pending. | medium | Shared processing helper used by ingestion and retrieval. |
| Claims_Extraction | existing | not inspected | Pending. | Pending. | medium | Feature module for extraction workflows. |
| CodeGraph | created | inspected | Source: `app/core/CodeGraph/{workspace.py,config.py,indexer.py,language_registry.py,context.py,jobs.py,jobs_worker.py,extractors/}`; MCP and DB surfaces inspected. | MCP: `app/core/MCP_unified/modules/implementations/codegraph_module.py`; DB: `app/core/DB_Management/codegraph/repository.py`; docs: `Docs/MCP/Unified/CodeGraph.md`; tests: `tests/CodeGraph/`. | high | Data graph and code-analysis module. |
| Collections | existing | not inspected | Pending. | Pending. | medium | Feature module for grouped user content. |
| DB_Management | existing | not inspected | Pending. | Pending. | high | Core database and persistence module. |
| Data_Tables | created | inspected | Source: `app/core/Data_Tables/jobs_worker.py`; endpoint, schemas, sidecar docs, LLM/RAG/DB connections inspected. | Endpoint: `app/api/v1/endpoints/data_tables.py` (`/data-tables`); schema: `app/api/v1/schemas/data_tables_schemas.py`; tests: `tests/DataTables/`. | high | Data-management feature module. |
| Embeddings | existing | not inspected | Pending. | Pending. | high | Shared vector and provider integration module. |
| Evaluations | existing | not inspected | Pending. | Pending. | high | Broad evaluation workflow module. |
| External_Sources | existing | not inspected | Pending. | Pending. | high | External provider and ingestion boundary. |
| File_Artifacts | created | inspected | Source: `app/core/File_Artifacts/{file_artifacts_service.py,adapter_registry.py,jobs_worker.py,metrics.py,adapters/}`; endpoint, schemas, storage docs inspected. | Endpoint: `app/api/v1/endpoints/files.py` (`/files`); schema: `app/api/v1/schemas/file_artifacts_schemas.py`; tests: `tests/FileArtifacts/`, `tests/Files/`, `tests/Storage/test_file_artifacts_storage_integration.py`. | high | Storage and artifact-management module. |
| Flashcards | existing | not inspected | Pending. | Pending. | medium | Feature module for study content. |
| Governance | missing | not inspected | Pending. | Pending. | high | Policy and governance module. |
| Image_Generation | missing | not inspected | Pending. | Pending. | medium | Feature module with provider integration. |
| Infrastructure | existing | not inspected | Pending. | Pending. | high | Broad operational support module. |
| Ingestion_Media_Processing | existing | not inspected | Pending. | Pending. | high | Broad ingestion and media-processing module. |
| Ingestion_Sources | missing | not inspected | Pending. | Pending. | high | Ingestion boundary and source-management module. |
| Integrations | missing | not inspected | Pending. | Pending. | high | External integration boundary. |
| Jobs | existing | not inspected | Pending. | Pending. | high | User-visible asynchronous work queue module. |
| LLM_Calls | existing | not inspected | Pending. | Pending. | high | Broad provider integration module. |
| Local_LLM | existing | not inspected | Pending. | Pending. | medium | Local model feature module. |
| Logging | existing | not inspected | Pending. | Pending. | low | Small operational helper unless evidence shows broader scope. |
| MCP_unified | existing | not inspected | Pending. | Pending. | high | Broad protocol, auth, and tool-execution module. |
| Meetings | missing | not inspected | Pending. | Pending. | medium | Feature module for meeting workflows. |
| Metrics | existing | not inspected | Pending. | Pending. | high | Operational observability module. |
| Moderation | existing | not inspected | Pending. | Pending. | high | Safety-sensitive policy module. |
| Monitoring | existing | not inspected | Pending. | Pending. | high | Operational monitoring module. |
| Notes | existing | not inspected | Pending. | Pending. | high | User data and persistence feature module. |
| Notes_Graph | missing | not inspected | Pending. | Pending. | medium | Feature module for note graph relationships. |
| Notifications | existing | not inspected | Pending. | Pending. | medium | Feature module for user notifications. |
| Persona | existing | not inspected | Pending. | Pending. | medium | Feature module for persona configuration. |
| Personalization | missing | not inspected | Pending. | Pending. | high | User preference and personalization data module. |
| PrivilegeMaps | existing | not inspected | Pending. | Pending. | high | Authorization and privilege mapping module. |
| Prompt_Management | existing | not inspected | Pending. | Pending. | medium | Feature module for prompt data and workflows. |
| Prototype_Workspaces | missing | not inspected | Pending. | Pending. | medium | Feature module for workspace experiments. |
| RAG | existing | not inspected | Pending. | Pending. | high | Broad retrieval pipeline module. |
| RateLimiting | existing | not inspected | Pending. | Pending. | high | Operational and abuse-control module. |
| Reminders | missing | not inspected | Pending. | Pending. | medium | Feature module for scheduled reminders. |
| Research | missing | not inspected | Pending. | Pending. | medium | Feature module for research workflows. |
| Research_Workspace | missing | not inspected | Pending. | Pending. | high | Data-heavy research workspace module. |
| Resource_Governance | existing | not inspected | Pending. | Pending. | high | Resource control and governance module. |
| Sandbox | existing | not inspected | Pending. | Pending. | high | Security-sensitive execution boundary. |
| Scheduler | existing | not inspected | Pending. | Pending. | high | Core orchestration and dependency module. |
| Search_and_Research | existing | not inspected | Pending. | Pending. | high | Broad search and research workflow module. |
| Security | existing | not inspected | Pending. | Pending. | high | Security-sensitive module. |
| Setup | existing | not inspected | Pending. | Pending. | high | Deployment and initialization module. |
| Sharing | missing | not inspected | Pending. | Pending. | high | User data sharing and access-control feature. |
| Skills | missing | not inspected | Pending. | Pending. | medium | Feature module for skill workflows. |
| Slides | missing | not inspected | Pending. | Pending. | medium | Feature module for slide generation or management. |
| Storage | missing | not inspected | Pending. | Pending. | high | Storage abstraction and persistence module. |
| Streaming | missing | not inspected | Pending. | Pending. | high | Streaming transport and runtime behavior module. |
| StudyPacks | missing | not inspected | Pending. | Pending. | medium | Feature module for study bundles. |
| StudySuggestions | missing | not inspected | Pending. | Pending. | medium | Feature module for study recommendations. |
| Sync | existing | not inspected | Pending. | Pending. | high | Cross-device or cross-store synchronization module. |
| TTS | existing | not inspected | Pending. | Pending. | medium | Feature module with provider integration. |
| Telegram | missing | not inspected | Pending. | Pending. | medium | External messaging integration feature. |
| Templating | missing | not inspected | Pending. | Pending. | low | Small helper package unless evidence shows broader scope. |
| Text2SQL | missing | not inspected | Pending. | Pending. | high | Query-generation module with database risk. |
| Third_Party | existing | not inspected | Pending. | Pending. | low | Support package for third-party helpers. |
| Tools | existing | not inspected | Pending. | Pending. | high | Tool execution and integration module. |
| Usage | existing | not inspected | Pending. | Pending. | high | Operational usage and quota-related module. |
| UserProfiles | missing | not inspected | Pending. | Pending. | high | User data module. |
| Utils | existing | not inspected | Pending. | Pending. | low | Shared helper package. |
| VN_Assets | missing | not inspected | Pending. | Pending. | medium | Feature module for visual novel assets. |
| VN_Platform | missing | not inspected | Pending. | Pending. | medium | Feature module for visual novel platform behavior. |
| VN_Play | missing | not inspected | Pending. | Pending. | medium | Feature module for visual novel play/runtime behavior. |
| VN_Policy | missing | not inspected | Pending. | Pending. | high | Policy module for visual novel behavior. |
| VN_Scripts | missing | not inspected | Pending. | Pending. | medium | Feature module for visual novel scripts. |
| VoiceAssistant | missing | not inspected | Pending. | Pending. | medium | Feature module for voice assistant behavior. |
| Watchlists | existing | not inspected | Pending. | Pending. | high | Scheduled and user-visible monitoring workflow module. |
| WebClipper | missing | not inspected | Pending. | Pending. | medium | Feature module for captured web content. |
| WebSearch | existing | not inspected | Pending. | Pending. | high | External network search boundary. |
| Web_Scraping | existing | not inspected | Pending. | Pending. | high | External network ingestion boundary. |
| Workflows | existing | not inspected | Pending. | Pending. | high | Broad orchestration module. |
| Workspaces | missing | not inspected | Pending. | Pending. | high | User workspace data and workflow module. |
| Writing | existing | not inspected | Pending. | Pending. | medium | Feature module; placeholder scan is expected to flag this README. |
| config_sections | missing | not inspected | Pending. | Pending. | low | Small configuration helper package. |
| deprecations | missing | not inspected | Pending. | Pending. | low | Small deprecation helper package. |

## Initial Red Checks

Recorded on 2026-06-01 from `/Users/appledev/Documents/GitHub/tldw_server/.worktrees/core-module-readmes`.

- README coverage check: red as expected. `test -z "$missing"` exited `1` and printed the 40 modules marked `missing` above.
- Placeholder scan: red as expected. `rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'` exited `0` and reported `tldw_Server_API/app/core/Writing/README.md` lines 3, 10, and 32.
