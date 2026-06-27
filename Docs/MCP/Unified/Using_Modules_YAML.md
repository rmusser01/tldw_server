# Using mcp_modules.yaml

This guide shows how to configure and load MCP Unified modules using a YAML file.

Location
- Default path: `tldw_Server_API/Config_Files/mcp_modules.yaml`
- Override via env: `MCP_MODULES_CONFIG=/path/to/your.yaml`

Schema
```yaml
modules:
  - id: media
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module:MediaModule
    enabled: true
    name: Media
    version: "1.0.0"
    department: media
    # Optional runtime controls
    max_concurrent: 16
    circuit_breaker_threshold: 3
    circuit_breaker_timeout: 30
    circuit_breaker_backoff_factor: 2.0
    circuit_breaker_max_timeout: 180
    # Module-specific settings
    settings:
      # Per-user example path; replace <content-db>.db with your configured content DB filename
      db_path: Databases/user_databases/1/<content-db>.db
      cache_ttl: 300
```

Rules
- Autoload is restricted to classes under:
  - `tldw_Server_API.app.core.MCP_unified.modules.implementations`
  - The server logs and ignores entries outside this namespace.
- If `modules:` is empty and `MCP_ENABLE_MEDIA_MODULE=1`, MediaModule is auto-enabled with defaults.
- Enabled modules appear in the `/api/v1/mcp/status` `surface` summary, grouped by risk tier.

Capability Risk Tiers
- `read_only`: Reads/searches existing TLDW data, such as `media`, `knowledge`, `chats`, `prompts`, and `mcp_discovery`.
- `write`: Creates or changes TLDW data/artifacts, such as `notes`, `template`, `quizzes`, `flashcards`, `kanban`, `slides`, and `governance`.
- `local_files`: Reads, writes, indexes, or scopes local files/workspaces, such as `filesystem` and `codegraph`.
- `external_network`: Connects to external servers, such as `external_federation`.
- `local_process`: Runs configured local commands or sandbox workloads, such as `run_command` and `sandbox`.
- `unknown`: Custom modules that have not been classified yet.

Before enabling a high-risk tier, verify the relevant module settings, RBAC permissions, and runtime policy. The tier explains capability shape; it does not grant execution permission.

Runtime Controls
- `max_concurrent`: Limits concurrent calls per module (0 disables guard).
- Circuit breaker knobs:
  - `circuit_breaker_threshold`: Failures before opening (default 5)
  - `circuit_breaker_timeout`: Initial open window (s, default 60)
  - `circuit_breaker_backoff_factor`: Multiplier on half-open failure (default 2.0)
  - `circuit_breaker_max_timeout`: Cap for backoff window (default 300)

Tips
- Start small: enable a single module and verify health at `GET /api/v1/mcp/modules/health`.
- For multi-node deployments, set Redis limiter and adjust security knobs (see README).
- Prefer YAML for ops; `MCP_MODULES` env is a quick, single-line alternative for development.
