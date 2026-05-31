# API Pagination Route-Family Catalogue

Date: 2026-04-25

Scope: `tldw_Server_API/app/api/v1/endpoints`

This catalogue expands the initial Phase 3.2 pagination inventory by classifying endpoint modules with a corrected multi-line function-signature scan. Counts are static scan counts, not OpenAPI-derived route guarantees. A route family can appear as `raw` because it manually returns `Response`/`StreamingResponse`/`FileResponse`-like objects or because a raw response marker appears in the route body.

## Summary

- `213` endpoint modules scanned.
- `2186` route blocks with function signatures scanned.
- `93` endpoint modules include pagination-like request parameters.
- `120` endpoint modules are currently classified as `not_paginated`.
- `20` endpoint modules have no local route decorators and appear to be support/re-export modules.

Route-family styles:

| Style | Families |
| --- | ---: |
| `offset` | 56 |
| `page` | 9 |
| `cursor` | 2 |
| `offset+cursor` | 5 |
| `offset+page` | 14 |
| `offset+page+cursor` | 6 |
| `page+cursor` | 1 |
| `not_paginated` | 120 |

## Paginated Route Families

The number in parentheses is the count of route blocks in that module that reference pagination-like parameters.

### Offset

`jobs_admin` (2); `sharing` (2); `chat_dictionaries` (2); `acp_schedules` (1); `quizzes` (3); `companion` (1); `skills` (1); `writing` (3); `character_messages` (2); `chatbooks` (2); `monitoring` (2); `meetings` (1); `data_tables` (2); `orgs` (3); `flashcards` (5); `claims` (12); `chat_documents` (1); `persona` (5); `embeddings_v5_production_enhanced` (4); `notes` (14); `sandbox` (3); `scheduler_workflows` (1); `guardian_controls` (1); `media_embeddings` (1); `vector_stores_openai` (2); `writing_manuscripts` (2); `storage` (3); `character_memory` (1); `slides` (4); `agent_client_protocol` (4); `self_monitoring` (1); `voice_assistant` (1); `chat_grammars` (1); `benchmark_api` (1); `outputs_templates` (1); `research_runs` (1); `admin/admin_tools` (2); `admin/admin_monitoring` (1); `admin/admin_webhooks` (2); `admin/admin_rate_limits` (1); `admin/admin_api_keys` (2); `admin/admin_byok` (1); `admin/admin_ops` (4); `admin/admin_data_ops` (3); `admin/admin_orgs` (3); `admin/admin_storage_quotas` (1); `admin/admin_bundle_ops` (1); `admin/admin_acp_agents` (1); `admin/admin_system` (3); `audio/audiobooks` (3); `audio/audio_jobs` (2); `kanban/kanban_workflow` (2); `kanban/kanban_comments` (1); `prompt_studio/prompt_studio_evaluations` (1); `media/document_insights` (1); `media/navigation` (2)

### Page

`collections_feeds` (1); `privileges` (5); `items` (1); `outputs` (2); `evaluations/evaluations_unified` (1); `prompt_studio/prompt_studio_projects` (2); `prompt_studio/prompt_studio_test_cases` (1); `prompt_studio/prompt_studio_prompts` (2); `prompt_studio/prompt_studio_optimization` (2)

### Cursor

`llm_providers` (1); `notes_graph` (1)

### Hybrid

`email` (1); `notifications` (2); `audio/audio_history` (1); `audio/audio_transcriptions` (2); `media/ingest_jobs` (2); `characters_endpoint` (6); `watchlists` (12); `reading` (6); `prompts` (4); `personalization` (2); `admin/admin_user` (2); `admin/admin_budgets` (1); `admin/admin_profiles` (1); `admin/admin_usage` (7); `kanban/kanban_boards` (2); `kanban/kanban_search` (1); `kanban/kanban_lists` (1); `media/listing` (5); `media/versions` (1); `character_chat_sessions` (6); `paper_search` (16); `chat` (8); `workflows` (3); `kanban/kanban_cards` (3); `media/document_references` (1); `connectors` (1)

## Not-Paginated Buckets

The route families below do not expose pagination-like request parameters in the corrected static scan. They still need route-by-route review before response-envelope rollout because many contain raw responses, downloads, streams, `204` responses, or no declared `response_model`.

### Raw Response, Streaming, File, Or Manual Response Signals

`acp_permissions`; `acp_triggers`; `admin/admin_circuit_breakers`; `admin/admin_events_stream`; `admin/admin_identity_providers`; `admin/admin_impersonation`; `admin/admin_llm_providers`; `admin/admin_rbac`; `admin/admin_sessions_mfa`; `admin/admin_settings`; `admin/admin_tenant_provisioning`; `agent_orchestration`; `archetype_endpoints`; `audio/audio_streaming`; `audio/audio_tokenizer`; `audio/audio_tts`; `audio/audio_voices`; `audit`; `auth`; `chat_loop`; `chat_workflows`; `chunking_templates`; `collections_websub`; `config_admin`; `config_info`; `consent`; `discord`; `family_wizard`; `feedback`; `files`; `health`; `integrations_control_plane`; `kanban/kanban_checklists`; `kanban/kanban_labels`; `kanban/kanban_links`; `llamacpp`; `mcp_catalogs_manage`; `mcp_hub_management`; `mcp_unified_endpoint`; `media/debug`; `media/document_annotations`; `media/document_figures`; `media/document_outline`; `media/file`; `media/item`; `media/process_audios`; `media/process_code`; `media/process_documents`; `media/process_ebooks`; `media/process_emails`; `media/process_pdfs`; `media/process_videos`; `media/reading_progress`; `media/reprocess`; `moderation`; `org_invites`; `prompt_studio/prompt_studio_status`; `prompt_studio/prompt_studio_websocket`; `rag_unified`; `reminders`; `research`; `resource_governor`; `setup`; `shared_keys_scoped`; `slack`; `sync`; `test_support/admin_e2e`; `text2sql`; `tools`; `translate`; `user_keys`; `users`; `watchlist_alert_rules`; `workspaces`

### No Declared Response Model

`acp_multiplex`; `admin/admin_network`; `admin/admin_registration`; `audio/audio_health`; `authnz_debug`; `billing`; `media/add`; `media/ingest_web_content`; `media/process_mediawiki`; `media/process_web_scraping`; `messages`; `metrics`; `mlx`; `ocr`; `rag_health`; `reading_highlights`; `telegram`; `vlm`; `web_scraping`

### Normal Typed, Not Paginated

These are candidates for later response-envelope work but are not Phase 3.2 pagination targets:

`admin/admin_personalization`; `admin/admin_router_analytics`; `ingestion_sources`; `media/transcription_models`; `scheduled_tasks_control_plane`; `study_suggestions`; `web_clipper`

### Support Or Re-Export Modules With No Local Route Blocks

`audio/audio`; `chunking`; `discord_oauth_admin`; `discord_support`; `evaluations/evaluations_auth`; `evaluations/evaluations_benchmarks`; `evaluations/evaluations_crud`; `evaluations/evaluations_datasets`; `evaluations/evaluations_embeddings_abtest`; `evaluations/evaluations_rag_pipeline`; `evaluations/evaluations_recipes`; `evaluations/evaluations_synthetic`; `evaluations/evaluations_webhooks`; `media/deprecation_signals`; `media/input_contracts`; `media_navigation_policy`; `slack_oauth_admin`; `slack_support`; `telegram_support`; `workspaces_rate_limit_policy`

## Candidate Frontend Caller Notes

### `skills`

Primary caller surface is `workspace-api.ts`, `types/skill.ts`, and `components/Option/Skills/*`. The list route reads top-level `skills`, `count`, `total`, `limit`, and `offset`.

### `slides`

Primary client surface:

- `apps/packages/ui/src/services/tldw/domains/presentations.ts`
- `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- `apps/packages/ui/src/components/Option/PresentationStudio/*`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/*`

Pagination-sensitive method:

- `listVisualStyles()` loops over `/api/v1/slides/styles?limit=200&offset=...` and stops using `payload.styles` plus `payload.total_count`.

Deferrals:

- `exportPresentation()` returns binary data and should remain response-envelope exempt.
- Render job and artifact endpoints should be migrated after list/detail endpoints.

### `data_tables`

Primary client surface:

- `apps/packages/ui/src/services/tldw/domains/media.ts`
- `apps/packages/ui/src/services/tldw/data-tables.ts`
- `apps/packages/ui/src/types/data-tables.ts`
- `apps/packages/ui/src/components/Option/DataTables/*`

Pagination-sensitive methods:

- `listDataTables()` sends `limit` and `offset`, with `page`/`page_size` converted client-side.
- `getDataTable()` sends `rows_limit` and `rows_offset` for row-window retrieval.
- `mapApiListToUi()` accepts arrays or `tables`/`items`/`results`, and reads `total` or `count`.

Deferrals:

- `exportDataTable()` returns files and should remain response-envelope exempt.
- Generation and job endpoints should be migrated after list/detail row-window behavior is proven.

## Next Action

Keep `skills` as the first Phase 3.2 pilot. Use `slides` as the second offset pilot only after `listVisualStyles()` can parse both legacy `total_count` and canonical `pagination.total`. Use `data_tables` after that because it exercises both list pagination and `rows_limit`/`rows_offset` row windows.
