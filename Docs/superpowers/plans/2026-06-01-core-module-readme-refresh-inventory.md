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
| Governance | created | inspected | Source: `app/core/Governance/{types.py,resolver.py,service.py,store.py,metrics.py}`; MCP, ACP, and governance docs inspected. | MCP: `app/core/MCP_unified/modules/implementations/governance_module.py`; ACP: `app/core/Agent_Client_Protocol/runner_client.py`; tests: `tests/Governance/`, `app/core/MCP_unified/tests/test_governance_module.py`, `tests/Agent_Client_Protocol/test_acp_governance_coordinator.py`. | high | Policy and governance module. |
| Image_Generation | created | inspected | Source: `app/core/Image_Generation/{config.py,adapter_registry.py,listing.py,capabilities.py,reference_images.py,prompt_refinement.py,adapters/}`; file artifact, model catalog, persona, VN, and workflow connections inspected. | Endpoints: `app/api/v1/endpoints/files.py`, `app/api/v1/endpoints/llm_providers.py`; tests: `tests/Image_Generation/`, `tests/Files/`, `tests/FileArtifacts/test_image_adapter_allowlist.py`. | medium | Feature module with provider integration. |
| Infrastructure | existing | not inspected | Pending. | Pending. | high | Broad operational support module. |
| Ingestion_Media_Processing | existing | not inspected | Pending. | Pending. | high | Broad ingestion and media-processing module. |
| Ingestion_Sources | created | inspected | Source: `app/core/Ingestion_Sources/{models.py,service.py,local_directory.py,archive_snapshot.py,git_repository.py,diffing.py,jobs.py,access_policy.py,sinks/}`; worker, scheduler, and API docs inspected. | Endpoint: `app/api/v1/endpoints/ingestion_sources.py`; schema: `app/api/v1/schemas/ingestion_sources.py`; services: `app/services/ingestion_sources_worker.py`, `app/services/ingestion_sources_scheduler.py`; tests: `tests/Ingestion_Sources/`; docs: `Docs/API-related/Ingestion_Sources_API.md`. | high | Ingestion boundary and source-management module. |
| Integrations | created | inspected | Source: `app/core/Integrations/weather_providers.py`; chat command router and separate integration control plane inspected. | Chat: `app/core/Chat/command_router.py`; endpoint: `app/api/v1/endpoints/integrations_control_plane.py`; service: `app/services/integrations_control_plane_service.py`; tests: `tests/Chat_NEW/unit/test_weather_providers.py`, `tests/Chat_NEW/unit/test_command_router.py`, `tests/Integrations/`. | high | External integration boundary. |
| Jobs | existing | not inspected | Pending. | Pending. | high | User-visible asynchronous work queue module. |
| LLM_Calls | existing | not inspected | Pending. | Pending. | high | Broad provider integration module. |
| Local_LLM | existing | not inspected | Pending. | Pending. | medium | Local model feature module. |
| Logging | existing | not inspected | Pending. | Pending. | low | Small operational helper unless evidence shows broader scope. |
| MCP_unified | existing | not inspected | Pending. | Pending. | high | Broad protocol, auth, and tool-execution module. |
| Meetings | created | inspected | Source: `app/core/Meetings/{session_service.py,template_service.py,artifact_service.py,events_service.py,integration_service.py,stream_adapter.py}`; DB deps, Meetings DB, and developer guide inspected. | Endpoint: `app/api/v1/endpoints/meetings.py`; schema: `app/api/v1/schemas/meetings_schemas.py`; DB: `app/core/DB_Management/Meetings_DB.py`; tests: `tests/Meetings/`, `tests/API_Deps/test_meetings_db_deps_error_mapping.py`; docs: `Docs/Code_Documentation/Meetings_Developer_Guide.md`. | medium | Feature module for meeting workflows. |
| Metrics | existing | not inspected | Pending. | Pending. | high | Operational observability module. |
| Moderation | existing | not inspected | Pending. | Pending. | high | Safety-sensitive policy module. |
| Monitoring | existing | not inspected | Pending. | Pending. | high | Operational monitoring module. |
| Notes | existing | not inspected | Pending. | Pending. | high | User data and persistence feature module. |
| Notes_Graph | created | inspected | Source: `app/core/Notes_Graph/{graph_service.py,graph_cache.py,wikilink_parser.py,formatters.py}`; endpoint and schemas inspected. | Endpoint: `app/api/v1/endpoints/notes_graph.py`; schema: `app/api/v1/schemas/notes_graph.py`; tests: `tests/Notes_Graph/unit/`, `tests/Notes_Graph/integration/test_graph_endpoint.py`. | medium | Feature module for note graph relationships. |
| Notifications | existing | not inspected | Pending. | Pending. | medium | Feature module for user notifications. |
| Persona | existing | not inspected | Pending. | Pending. | medium | Feature module for persona configuration. |
| Personalization | created | inspected | Source: `app/core/Personalization/{companion_activity.py,companion_context.py,companion_derivations.py,companion_followups.py,companion_lifecycle.py,companion_proactive.py,companion_reflection_jobs.py,companion_reflection_jobs_worker.py,companion_relevance.py,companion_user_ids.py}`; companion, admin, scheduler, and adjacent activity bridges inspected. | Endpoints: `app/api/v1/endpoints/personalization.py`, `app/api/v1/endpoints/companion.py`, `app/api/v1/endpoints/admin/admin_personalization.py`; schema: `app/api/v1/schemas/personalization.py`; service: `app/services/companion_reflection_scheduler.py`; tests: `tests/Personalization/`, `tests/API_Deps/test_personalization_deps_sanitization.py`. | high | User preference and personalization data module. |
| PrivilegeMaps | existing | not inspected | Pending. | Pending. | high | Authorization and privilege mapping module. |
| Prompt_Management | existing | not inspected | Pending. | Pending. | medium | Feature module for prompt data and workflows. |
| Prototype_Workspaces | created | inspected | Source: `app/core/Prototype_Workspaces/{models.py,service.py,access.py,preview_broker.py,jobs.py,jobs_worker.py}`; AuthNZ repo, endpoint, schemas, and docs inspected. | Endpoint: `app/api/v1/endpoints/prototype_workspaces.py`; schema: `app/api/v1/schemas/prototype_workspace_schemas.py`; repo: `app/core/AuthNZ/repos/prototype_workspaces_repo.py`; tests: `tests/PrototypeWorkspaces/`; docs: `Docs/API-related/Prototype_Workspaces_API.md`, `Docs/API-related/Prototype_Workspaces_Contract_Matrix.md`, `Docs/Operations/Prototype_Workspaces_Runbook.md`, `Docs/Security/Prototype_Workspaces_Threat_Model.md`. | medium | Feature module for workspace experiments. |
| RAG | existing | not inspected | Pending. | Pending. | high | Broad retrieval pipeline module. |
| RateLimiting | existing | not inspected | Pending. | Pending. | high | Operational and abuse-control module. |
| Reminders | created | inspected | Source: `app/core/Reminders/{reminders_service.py,reminder_jobs.py}`; notifications endpoint, scheduler, worker, schemas, and API docs inspected. | Endpoints: `app/api/v1/endpoints/reminders.py`, `app/api/v1/endpoints/notifications.py`; schema: `app/api/v1/schemas/reminders_schemas.py`; services: `app/services/reminders_scheduler.py`, `app/services/reminder_jobs_worker.py`; tests: `tests/Notifications/test_reminders_*.py`, `tests/Notifications/test_reminder_jobs_worker.py`, `tests/Collections/test_reminders_notifications_db.py`; docs: `Docs/API-related/Reminder_Notifications_API.md`. | medium | Feature module for scheduled reminders. |
| Research | created | inspected | Source: `app/core/Research/{service.py,jobs.py,jobs_worker.py,artifact_store.py,broker.py,models.py,synthesizer.py,exporter.py,streaming.py,checkpoint_service.py,chat_handoff.py,providers/}`; legacy research endpoints and deep research docs/tests inspected. | Endpoints: `app/api/v1/endpoints/research_runs.py`, `app/api/v1/endpoints/research.py`; schemas: `app/api/v1/schemas/research_runs_schemas.py`, `app/api/v1/schemas/research_schemas.py`; tests: `tests/Research/`, `tests/e2e/test_deep_research_runs.py`, `tests/Workflows/adapters/test_research_adapters.py`. | medium | Feature module for research workflows. |
| Research_Workspace | created | inspected | Source: `app/core/Research_Workspace/capabilities.py`; endpoint, schema, and broader workspace docs inspected. | Endpoint: `app/api/v1/endpoints/research_workspace.py`; schema: `app/api/v1/schemas/research_workspace_capabilities.py`; tests: `tests/Research_Workspace/test_capability_derivation.py`, `tests/Research_Workspace/test_capability_endpoint.py`; docs: `Docs/Design/Research_Workspace_Shared_Workspace_Model_Contract_2026_05.md`. | high | Capability-readiness helper for Research Workspace; persistence and migration live elsewhere. |
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

- README coverage check: red as expected. `test -z "$missing"` exited `1` and printed the 40 modules marked `missing` at initial inventory time.
- Placeholder scan: red as expected. `rg -n "Replace placeholders|scaffolded from the core template|Link API routes and files|Planned improvements|T[B]D|F[I]XME" tldw_Server_API/app/core --glob 'README.md'` exited `0` and reported `tldw_Server_API/app/core/Writing/README.md` lines 3, 10, and 32.
