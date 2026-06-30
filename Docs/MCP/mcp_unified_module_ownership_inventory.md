# MCP Unified Module Ownership Inventory

Date: 2026-05-27
Backlog: TASK-480

This inventory classifies current MCP Unified modules before standalone package extraction. It is grounded in direct `tldw_Server_API` import scans of `tldw_Server_API/app/core/MCP_unified/modules/implementations/` and should be refreshed before any file move.

Classification:

- `runtime-neutral`: safe to move into a standalone package with no `tldw_Server_API` dependency.
- `adapter-backed`: reusable only after explicit host adapters replace current `tldw_Server_API` dependencies.
- `tldw-owned`: remains in `tldw_server` and is exposed as a host-provided module.

| Module file | Tool families | Classification | Data stores/dependencies | Capability/risk notes | Migration recommendation | Protecting tests |
| --- | --- | --- | --- | --- | --- | --- |
| `external_federation_module.py` | External MCP server/tool federation | `adapter-backed` | `ExternalServerManager`, MCP Hub external registry service, credential broker | High-value standalone gateway feature, but current registry and credential material are host services. | Move only after external registry store, credential broker, and upstream transport manager interfaces exist. | `test_external_federation_integration.py`, `test_external_server_manager.py`, `test_external_stdio_adapter.py`, `test_external_websocket_adapter.py` |
| `filesystem_module.py` | `fs.list`, `fs.read_text`, `fs.write_text` | `adapter-backed` | MCP Hub workspace root resolver | Generally portable, but path trust boundary depends on host workspace policy. | Keep module shape; replace workspace root resolver with a runtime workspace/path-scope adapter. | `test_filesystem_module.py` |
| `mcp_discovery_module.py` | MCP server/tool discovery | `adapter-backed` | AuthNZ DB pool, org/team membership service, `MCPProtocol`, `RequestContext` | Discovery behavior is central to profiles/modes, but current visibility rules are host AuthNZ-specific. | Extract after discovery store, principal/org membership, and protocol introspection interfaces exist. | `test_mcp_discovery_module.py` |
| `run_command_module.py` | `run` virtual CLI | `adapter-backed` | MCP command runtime package, workspace root resolver, nested governed MCP calls | Most command parsing/execution code is portable; workspace and nested-tool authorization are not. | Split neutral command runtime from host workspace/process policy adapters before moving. | `test_run_command_module.py`, `test_idempotency_and_category.py`, `test_protocol_nested_tool_preparation.py` |
| `template_module.py` | `echo` example utility | `runtime-neutral` | Base module helpers only | Safe reference implementation; no direct host import found. | Move early with the base-module package as an example module. | No dedicated test; add a small template smoke test before migration. |
| `knowledge_module.py` | Cross-domain knowledge search/retrieval | `adapter-backed` | MCP registry/server/protocol lookup, domain modules behind current registry | Aggregates host-owned modules; extraction depends on a stable module discovery/query interface. | Keep in host until notes/media/chats/prompts adapters are modeled; then consider neutral aggregator shell. | `test_knowledge_get.py`, `test_knowledge_search_defaults.py`, `test_scope_and_fallbacks.py` |
| `media_module.py` | Media search/retrieval/ingestion | `tldw-owned` | Media DB, ingestion pipeline, RAG retrievers, AuthNZ mode checks | Deeply tied to tldw media storage, ingestion jobs, and RAG conventions. | Expose as a host-provided module through a domain adapter; do not move into the standalone core. | `test_media_retrieval.py`, `test_media_retrieval_db_error_fallback.py`, `test_scope_and_fallbacks.py` |
| `notes_module.py` | Notes CRUD/search/tags | `tldw-owned` | User ChaChaNotes DB paths and project root fallback | Data model is specific to tldw notes/chats storage. | Keep host-owned; define a generic note-store interface only if another host needs notes. | `test_notes_crud_tags.py`, `test_persona_scope_stage3.py` |
| `chats_module.py` | Chat retrieval/history | `tldw-owned` | ChaChaNotes/chat DB paths and project root fallback | Host data model and conversation schema should remain under tldw ownership. | Expose through host adapter; do not move current implementation. | `test_chats_retrieval.py`, `test_persona_scope_stage3.py` |
| `characters_module.py` | Character/persona retrieval | `tldw-owned` | Character tables in ChaChaNotes DB | Persona schema and SillyTavern compatibility are tldw domain concerns. | Keep host-owned; standalone package should consume a character provider interface if needed. | `test_persona_scope_stage3.py` |
| `prompts_module.py` | Prompt retrieval/search | `tldw-owned` | Prompt tables in ChaChaNotes DB | Prompt schema is host-owned and likely evolves with Prompt Studio. | Keep host-owned; later expose a prompt catalog adapter. | `test_persona_scope_stage3.py` |
| `flashcards_module.py` | Flashcard deck/card CRUD and generation | `tldw-owned` | ChaChaNotes/project DB paths, host utility fallback | Data model and generation workflow are host-specific. | Keep host-owned; adapter only if profiles need generic study-card capabilities. | `test_flashcards_module.py` |
| `quizzes_module.py` | Quiz CRUD, quiz item handling, generation | `tldw-owned` | ChaChaNotes/project DB paths, chat API call service | Uses host storage and LLM provider plumbing. | Keep host-owned; split generation provider only if another host needs it. | `test_quizzes_module.py` |
| `slides_module.py` | Slide deck planning/export/RAG-assisted generation | `tldw-owned` | Host RAG pipeline, slide export service, project root fallback | Large domain workflow with export side effects and RAG dependencies. | Keep host-owned; define file/export/RAG adapters before any reuse attempt. | `test_slides_module.py` |
| `kanban_module.py` | Boards/lists/cards/checklists | `tldw-owned` | Project/ChaChaNotes DB paths and host utility fallback | CRUD surface is host data-model-specific. | Keep host-owned unless a generic task-board store is introduced. | `test_kanban_module.py` |
| `persona_visuals_module.py` | Persona visual pack/library/job tools | `tldw-owned` | `CharactersRAGDB`, Jobs manager, Persona visual services | Strong coupling to tldw persona media pipeline and background jobs. | Keep host-owned; expose only through a host module registration. | `test_persona_visuals_module.py` |
| `codegraph_module.py` | Repository code graph indexing/query | `tldw-owned` | CodeGraph settings/context/indexer/jobs/repository | Depends on tldw CodeGraph service layer and DB repository. | Keep host-owned; future standalone gateway can consume a code-search provider interface. | `test_codegraph_module.py` |
| `governance_module.py` | MCP governance/profile management | `adapter-backed` | Governance service and store | Governance concepts belong in the standalone roadmap, but current persistence/service are host implementations. | Split neutral profile/policy schemas from tldw Governance store before migration. | `test_governance_module.py` |
| `sandbox_module.py` | `sandbox.run` execution management | `adapter-backed` | Sandbox models and service | Capability is useful for standalone profiles, but runtime execution is security-sensitive and host-specific. | Require explicit sandbox runner, artifact store, and network policy adapters before moving. | `test_scope_and_fallbacks.py` |

## Extraction Debt

- `interfaces/`: must remain free of `tldw_Server_API` imports. Current boundary tests enforce this recursively.
- `adapters/`: intentionally imports `tldw_Server_API`; this is the host adapter layer and should not move into the standalone runtime package.
- `protocol.py`: still has direct host imports for Redis fallback, telemetry, test-mode helper, app config, Governance service/store, and AuthNZ DB access. These need follow-up seams before package extraction.
- `server.py`: still has direct host imports for AuthNZ DB/JWT/API keys, DB path utilities, WebSocket stream transport, feature flags, lifecycle guards, shutdown registry, and endpoint lifespan integration.
- `modules/base.py`: circuit breaker creation now has an injection seam, but `execute_with_circuit_breaker` still imports the host `CircuitBreakerOpenError`; standalone extraction needs a neutral breaker exception or adapter-owned exception mapping.
- `modules/implementations/`: all `adapter-backed` rows need one future migration plan per module family before moving code.
- `tldw-owned` modules should be registered as host-provided capabilities from `tldw_server`; copying them into the standalone package would drag storage, jobs, and domain service ownership across the boundary.
