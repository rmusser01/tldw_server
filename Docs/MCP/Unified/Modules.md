# MCP Unified Modules Guide

> Part of the MCP Unified documentation set. See `Docs/MCP/Unified/README.md` for the full guide index.

## Overview

- Unified MCP exposes tools, resources, and prompts through pluggable modules.
- Each module subclasses `BaseModule` and is registered through YAML configuration or environment variables.
- `GET /api/v1/mcp/status` includes a `surface` summary that groups enabled modules by user-facing risk tier and lists high-risk modules available but disabled.

## Capability Risk Tiers

Use these tiers to explain what an enabled module can do before connecting an MCP client:

| Tier | Meaning | Common modules |
|---|---|---|
| `read_only` | Reads or searches existing TLDW data without writing by default. | `media`, `knowledge`, `chats`, `prompts`, `prompts_catalog`, `mcp_discovery` |
| `write` | Creates, updates, exports, or manages TLDW data or generated artifacts. | `notes`, `template`, `quizzes`, `flashcards`, `kanban`, `slides`, `characters`, `persona_visuals`, `governance` |
| `local_files` | Reads, writes, indexes, or scopes local files/workspaces. | `filesystem`, `codegraph` |
| `external_network` | Connects to external MCP servers or networked tool providers. | `external_federation` |
| `local_process` | Runs configured commands, code, or sandbox workloads on the host. | `run_command`, `sandbox` |
| `unknown` | A module is enabled but has no registered tier yet. | Custom modules until classified |

The tier is explanatory, not a permission grant. Execution still depends on RBAC, module settings, tool schemas, and runtime policy.

High-risk modules in `local_files`, `local_process`, and `external_network`
are explicit opt-ins. The default config keeps local filesystem and command
execution modules disabled; `/api/v1/mcp/status` reports them under
`surface.disabled_available` with `requires_explicit_opt_in: true` and a
`next_action`. After changing `mcp_modules.yaml`, restart TLDW Server and
recheck `/api/v1/mcp/status`.

## Quick Start

1. Implement the module under `tldw_Server_API/app/core/MCP_unified/modules/implementations/`.
2. Add a module entry to `tldw_Server_API/Config_Files/mcp_modules.yaml` (or define `MCP_MODULES`).
3. Restart the server and verify availability with `GET /api/v1/mcp/modules` and `/api/v1/mcp/tools`.
4. Check `GET /api/v1/mcp/status` and review `surface.tiers` plus `surface.disabled_available` to confirm the effective capability surface before connecting an agent.

## Module Interface

### Required methods

- `on_initialize(self)` - set up resources using `self.config.settings`.
- `on_shutdown(self)` - release or persist resources.
- `check_health(self) -> Dict[str, bool]` - resilient health probes.
- `get_tools(self) -> List[Dict[str, Any]]` - JSON schema describing the module tools.
- `execute_tool(self, tool_name, arguments)` - dispatch execution logic.

### Optional helpers

- `get_resources`, `read_resource`
- `get_prompts`, `get_prompt`

## Template Module

- Review `modules/implementations/template_module.py` for a minimal implementation pattern.

## Configuration (YAML)

- Default file: `tldw_Server_API/Config_Files/mcp_modules.yaml`

See also: Using mcp_modules.yaml for a deeper walkthrough and common pitfalls.
`Docs/MCP/Unified/Using_Modules_YAML.md`

```yaml
modules:
  - id: media
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule
    enabled: true
    name: Media
    version: "1.0.0"
    department: media
    timeout_seconds: 30
    max_retries: 3
    circuit_breaker_threshold: 5
    circuit_breaker_timeout: 60
    settings:
      # Per-user example path; replace <content-db>.db with your configured content DB filename
      db_path: Databases/user_databases/1/<content-db>.db
      cache_ttl: 300
```

## Environment Variables

- `MCP_MODULES_CONFIG` - override path to the YAML configuration (defaults to `tldw_Server_API/Config_Files/mcp_modules.yaml`).
- `MCP_MODULES` - comma-separated definitions (`id=module.path:Class`), e.g. `MCP_MODULES="example=tldw_Server_API.app.core.MCP_unified.modules.implementations.template_module:TemplateModule"`.
- Optional accelerator: `MCP_ENABLE_MEDIA_MODULE=true` registers `MediaModule` when no YAML or explicit environment configuration is provided.
- Optional explicit opt-ins: `MCP_ENABLE_FILESYSTEM_MODULE=true`, `MCP_ENABLE_GIT_MODULE=true`, `MCP_ENABLE_SANDBOX_MODULE=true`, and `MCP_ENABLE_BROWSER_CDP_MODULE=true` register local high-risk modules only when no YAML entry already declares the module.
- `MCP_EXTERNAL_SERVERS_CONFIG` - optional override path for external federation server registry (used by `external_federation` module).

## Migration Note: Local File And Process Modules

Default installs no longer expose local filesystem or local command execution
tools. If an existing deployment intentionally used those defaults, copy the
relevant entries from
`tldw_Server_API/Config_Files/mcp_modules.local_opt_in.example.yaml` into your
selected `mcp_modules.yaml`, review the risk comments, set `enabled: true`,
restart TLDW Server, then verify the module moved from
`surface.disabled_available` to `surface.tiers`.

## External Federation Module

- Module id: `external_federation`
- Purpose: expose approved upstream MCP tools through namespaced virtual tools (`ext.<server_id>.<tool_name>`).
- Default posture: safe-by-default (`allow_writes: false`, write confirmation required when enabled).
- Full activation and security guidance: `Docs/MCP/Unified/External_Federation.md`.

## Tool Execution Result

- Tool responses include module metadata, e.g. `{ "content": [...], "module": "Media", "tool": "search_media" }`.
- The HTTP endpoint `/api/v1/mcp/tools/execute` returns the module name in the response model.

## Guidelines

- Keep health checks non-blocking and degrade gracefully.
- Store module-level settings in `ModuleConfig.settings`; avoid global config coupling.
- Sanitize inputs with `sanitize_input()` provided on `BaseModule`.
- Prefer fast failures with descriptive error reporting.

## Testing

- Register a test module via `ModuleRegistry.register_module()`.
- Exercise flows with `MCPRequest(method="tools/call", ...)` routed through `server.handle_http_request()`.

## Troubleshooting

- Inspect logs when module registration fails (class import or configuration issues).
- Ensure `PyYAML` is installed when using YAML configurations.
- Confirm tool names and input schemas match between `get_tools` and `execute_tool`.
- "Blocked module autoload" in logs: The server only autoloads modules under
  `tldw_Server_API.app.core.MCP_unified.modules.implementations`. Move your module into this namespace.
- Permission denied on tools: Check RBAC and ensure your token/role has `tools.execute:<name>`.
- Write tools blocked: If `MCP_DISABLE_WRITE_TOOLS=1`, ingestion/management tools are disabled.
- Idempotent writes not executing: Requests with the same `idempotencyKey` within TTL
  will return the cached result. Change the key to force execution or wait for TTL expiry.
- Rate limit errors: Tool or category limits may apply. Review the Security Knobs table in the MCP README
  and your category mapping.
