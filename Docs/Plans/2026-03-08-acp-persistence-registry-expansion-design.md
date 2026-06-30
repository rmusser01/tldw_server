# ACP Persistence & Agent Registry Expansion Design

**Date:** 2026-03-08
**Status:** Approved

**ADR coverage:** Backfilled by `Docs/ADR/016-acp-session-and-orchestration-persistence.md` for implemented shared ACP session/registry persistence and per-user orchestration persistence. Owner sign-off was recorded in TASK-520. The ADR intentionally excludes unverified setup-guide consolidation claims.

## Problem

The ACP session store and orchestration service are entirely in-memory. On server restart, all session metadata, orchestration projects/tasks/runs, and review history are lost. The agent registry supports only 4 hardcoded agents with no runtime registration or health monitoring.

## Goals

1. Persist ACP sessions to a shared SQLite database
2. Persist orchestration data to per-user SQLite databases
3. Unify the dual agent configuration systems (YAML registry vs hardcoded setup guides)
4. Support more agent types out of the box
5. Allow dynamic agent registration via API
6. Add periodic health monitoring with history tracking

## Non-Goals

- PostgreSQL support (SQLite first, PG can come later)
- Migration of existing in-memory data (clean break)
- Changes to the runner client or stdio protocol
- Frontend changes (backend-only)

---

## Part 1: Session Store Persistence

### Database: `Databases/acp_sessions.db` (shared, all users)

Sessions are a global resource — admins need cross-user visibility, and the cleanup task operates globally.

### Schema

```sql
-- Schema version tracked via PRAGMA user_version

CREATE TABLE sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       INTEGER NOT NULL,
    agent_type    TEXT NOT NULL DEFAULT 'custom',
    name          TEXT NOT NULL DEFAULT '',
    status        TEXT NOT NULL DEFAULT 'active',  -- active | closed | error
    cwd           TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    last_activity_at TEXT,
    message_count INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens  INTEGER NOT NULL DEFAULT 0,
    bootstrap_ready  INTEGER NOT NULL DEFAULT 1,
    needs_bootstrap  INTEGER NOT NULL DEFAULT 0,
    forked_from   TEXT,
    tags          TEXT NOT NULL DEFAULT '[]',       -- JSON array
    mcp_servers   TEXT NOT NULL DEFAULT '[]',       -- JSON array
    persona_id    TEXT,
    workspace_id  TEXT,
    workspace_group_id TEXT,
    scope_snapshot_id  TEXT
);

CREATE INDEX idx_sessions_user_status ON sessions(user_id, status);
CREATE INDEX idx_sessions_created ON sessions(created_at DESC);
CREATE INDEX idx_sessions_forked ON sessions(forked_from);

CREATE TABLE session_messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    message_index INTEGER NOT NULL,
    role          TEXT NOT NULL,       -- user | assistant
    content       TEXT NOT NULL DEFAULT '',
    timestamp     TEXT NOT NULL,
    raw_data      TEXT,                -- JSON of raw_prompt or raw_result
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX idx_messages_session_idx ON session_messages(session_id, message_index);
```

### Implementation: `DB_Management/ACP_Sessions_DB.py`

New class `ACPSessionsDB` following the `ACP_Audit_DB` pattern:
- Thread-local connections, WAL mode, FK enforcement
- Schema versioning via `PRAGMA user_version`
- All CRUD methods match existing `ACPSessionStore` public API signatures
- Messages stored in separate table, loaded on-demand (not eagerly with session)

### Migration Path

- `ACPSessionStore` retains its public API (27 methods)
- Internal storage switches from `dict[str, SessionRecord]` to SQLite queries
- `SessionRecord` dataclass remains as the in-memory representation returned by methods
- On `get_session()`: query DB, construct `SessionRecord`
- On `list_sessions()`: query with filters, construct list
- Messages loaded lazily: `to_detail_dict()` triggers message query
- Token usage stored as denormalized columns on `sessions` table (no separate table)

### Agent Config & Permission Policy Tables

The existing `AgentConfig` and `PermissionPolicy` CRUD on `ACPSessionStore` also moves to SQLite:

```sql
CREATE TABLE agent_configs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id     INTEGER,
    team_id    INTEGER,
    agent_type TEXT NOT NULL,
    config     TEXT NOT NULL DEFAULT '{}',  -- JSON
    enabled    INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);

CREATE TABLE permission_policies (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id     INTEGER,
    team_id    INTEGER,
    tool_pattern TEXT NOT NULL,
    tier       TEXT NOT NULL,  -- auto | batch | individual
    created_at TEXT NOT NULL
);
```

---

## Part 2: Orchestration Persistence

### Database: `Databases/user_databases/<user_id>/orchestration.db` (per-user)

Orchestration data is user-owned work. Per-user databases match the existing pattern used by `Media_DB_v2` and `ChaChaNotes`.

### Schema

```sql
CREATE TABLE projects (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    user_id     INTEGER NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',  -- JSON
    created_at  TEXT NOT NULL,
    updated_at  TEXT
);

CREATE TABLE tasks (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id           INTEGER NOT NULL,
    title                TEXT NOT NULL,
    description          TEXT NOT NULL DEFAULT '',
    status               TEXT NOT NULL DEFAULT 'todo',
    agent_type           TEXT,
    dependency_id        INTEGER,
    reviewer_agent_type  TEXT,
    max_review_attempts  INTEGER NOT NULL DEFAULT 3,
    review_count         INTEGER NOT NULL DEFAULT 0,
    success_criteria     TEXT NOT NULL DEFAULT '',
    user_id              INTEGER NOT NULL,
    metadata             TEXT NOT NULL DEFAULT '{}',  -- JSON
    created_at           TEXT NOT NULL,
    updated_at           TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (dependency_id) REFERENCES tasks(id) ON DELETE SET NULL
);

CREATE INDEX idx_tasks_project ON tasks(project_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_dependency ON tasks(dependency_id);

CREATE TABLE runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id         INTEGER NOT NULL,
    session_id      TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    agent_type      TEXT,
    started_at      TEXT,
    completed_at    TEXT,
    result_summary  TEXT NOT NULL DEFAULT '',
    error           TEXT,
    token_usage     TEXT NOT NULL DEFAULT '{}',  -- JSON
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX idx_runs_task ON runs(task_id);
CREATE INDEX idx_runs_session ON runs(session_id);

CREATE TABLE reviews (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER NOT NULL,
    approved    INTEGER NOT NULL,  -- 0 or 1
    feedback    TEXT NOT NULL DEFAULT '',
    reviewer    TEXT,              -- agent type or user identifier
    created_at  TEXT NOT NULL,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE
);

CREATE INDEX idx_reviews_task ON reviews(task_id);
```

### Implementation: `DB_Management/Orchestration_DB.py`

New class `OrchestrationDB` following the `Personalization_DB` per-user pattern:
- Constructor takes `user_id`, resolves DB path via `get_user_base_directory()`
- WAL mode, FK enforcement, schema versioning
- CRUD for projects, tasks, runs, reviews

### Service Refactoring

`OrchestrationService` changes from singleton to per-user factory:

```python
# Before (global singleton):
svc = await get_orchestration_service()
svc.create_project(name, user_id=user.id)

# After (per-user DB):
db = get_orchestration_db(user.id)
db.create_project(name)
```

- `get_orchestration_db(user_id: int) -> OrchestrationDB` — factory with LRU cache
- State machine validation stays in service/DB layer (not endpoint)
- Cycle detection queries DB instead of walking in-memory dicts
- Cascade deletes handled by SQLite FK constraints

### Endpoint Changes

All endpoints in `agent_orchestration.py` change from:
```python
svc = await get_orchestration_service()
project = await svc.get_project(project_id)
if not project or project.user_id != user.id:
    raise HTTPException(404)
```
To:
```python
db = get_orchestration_db(user.id)
project = db.get_project(project_id)
if not project:
    raise HTTPException(404)
```

User-scoping is implicit in the per-user DB — no need for `user_id` checks at the endpoint layer.

---

## Part 3: Agent Registry Expansion

### Phase A: Unify Config & Add Agent Types

**Problem:** `_AGENT_SETUP_GUIDES` (hardcoded, 3 agents) and `agents.yaml` (4 agents) are not synchronized. Health and setup endpoints use the hardcoded guides, not the registry.

**Solution:** Extend `agents.yaml` schema with optional setup fields:

```yaml
agents:
  - type: claude_code
    name: Claude Code
    description: "Anthropic's Claude Code agent"
    command: claude
    requires_api_key: ANTHROPIC_API_KEY
    default: true
    install_instructions:
      - "npm install -g @anthropic-ai/claude-code"
    docs_url: "https://docs.anthropic.com/claude-code"

  - type: codex
    name: OpenAI Codex CLI
    command: codex
    requires_api_key: OPENAI_API_KEY
    install_instructions:
      - "npm install -g @openai/codex"
    docs_url: "https://github.com/openai/codex"

  - type: aider
    name: Aider
    description: "AI pair programming in your terminal"
    command: aider
    requires_api_key: null  # Supports multiple providers
    install_instructions:
      - "pip install aider-chat"
    docs_url: "https://aider.chat"

  - type: goose
    name: Goose
    description: "Block's autonomous coding agent"
    command: goose
    requires_api_key: null
    install_instructions:
      - "brew install block/goose/goose"
      - "# or: cargo install goose-cli"
    docs_url: "https://github.com/block/goose"

  - type: continue_dev
    name: Continue
    description: "Open-source AI code assistant"
    command: continue
    requires_api_key: null
    install_instructions:
      - "npm install -g @continuedev/cli"
    docs_url: "https://continue.dev"

  - type: opencode
    name: OpenCode
    command: opencode
    requires_api_key: null
    install_instructions:
      - "go install github.com/sst/opencode@latest"
    docs_url: "https://github.com/sst/opencode"

  - type: custom
    name: Custom Agent
    description: "Configure a custom agent"
    command: ""
    requires_api_key: null
```

**Registry changes:**
- `AgentRegistryEntry` gains optional fields: `install_instructions: list[str]`, `docs_url: str | None`, `required_env_vars: list[str]`
- `required_env_vars` defaults to `[requires_api_key]` if not specified

**Endpoint changes:**
- Delete `_AGENT_SETUP_GUIDES` hardcoded dict
- `_check_agent_availability()` delegates to `registry.get_entry(type).check_availability()`
- `/health` iterates `registry.entries` instead of hardcoded guides
- `/setup-guide` reads `install_instructions` and `docs_url` from registry entries

### Phase B: Dynamic Registration API

New endpoints:
- `POST /api/v1/acp/agents/register` — register a new agent at runtime
- `DELETE /api/v1/acp/agents/{agent_type}` — deregister an agent
- `PUT /api/v1/acp/agents/{agent_type}` — update agent config

Persistence: `agent_registry` table in `acp_sessions.db`:

```sql
CREATE TABLE agent_registry (
    type              TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    command           TEXT NOT NULL DEFAULT '',
    args              TEXT NOT NULL DEFAULT '[]',   -- JSON array
    env               TEXT NOT NULL DEFAULT '{}',   -- JSON dict
    requires_api_key  TEXT,
    is_default        INTEGER NOT NULL DEFAULT 0,
    install_instructions TEXT NOT NULL DEFAULT '[]', -- JSON array
    docs_url          TEXT,
    source            TEXT NOT NULL DEFAULT 'api',  -- 'yaml' | 'api'
    created_at        TEXT NOT NULL,
    updated_at        TEXT
);
```

**Merge logic:**
- YAML entries loaded first (source='yaml'), API entries override/extend
- On hot-reload, YAML entries are refreshed; API entries are preserved
- `get_available_agents()` returns merged list

**Thread safety:**
- Add `threading.RLock` to `AgentRegistry` for write operations
- Read operations (entries property, get_entry) acquire read lock
- Write operations (register, deregister) acquire write lock

### Phase C: Health Monitoring

Background task checks agent liveness periodically:

```python
class AgentHealthMonitor:
    def __init__(self, registry: AgentRegistry, check_interval: int = 60):
        self._registry = registry
        self._check_interval = check_interval
        self._health_cache: dict[str, AgentHealthStatus] = {}
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.ensure_future(self._check_loop())

    async def _check_loop(self) -> None:
        while True:
            for entry in self._registry.entries:
                status = entry.check_availability()
                self._update_health(entry.type, status)
            await asyncio.sleep(self._check_interval)

    def _update_health(self, agent_type: str, status: dict) -> None:
        # Track consecutive failures, auto-disable/re-enable
        ...
```

Health status enum: `healthy`, `degraded`, `unavailable`, `unknown`

Auto-disable logic:
- After 3 consecutive `unavailable` checks: mark as `disabled`
- On next `available` check: auto-re-enable, reset failure count

Health history table in `acp_sessions.db`:

```sql
CREATE TABLE agent_health_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_type  TEXT NOT NULL,
    status      TEXT NOT NULL,  -- healthy | degraded | unavailable
    checked_at  TEXT NOT NULL,
    details     TEXT NOT NULL DEFAULT '{}',  -- JSON
    FOREIGN KEY (agent_type) REFERENCES agent_registry(type) ON DELETE CASCADE
);

CREATE INDEX idx_health_agent_time ON agent_health_history(agent_type, checked_at DESC);
```

Retention: keep last 7 days of history, prune on cleanup cycle.

New endpoint:
- `GET /api/v1/acp/agents/health` — returns current health status for all agents plus recent history

---

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Session DB location | Shared `Databases/acp_sessions.db` | Admin cross-user visibility, global cleanup |
| Orchestration DB location | Per-user `Databases/user_databases/<id>/orchestration.db` by default, with the user DB base directory overrideable by configuration | User-owned data, matches existing pattern |
| Message storage | Separate `session_messages` table | Fork slicing by index, unbounded growth |
| Token usage | Denormalized on `sessions` table | Avoids join for quota checks |
| Agent registry persistence | Table in `acp_sessions.db` | Shared resource, merged with YAML |
| Health monitoring | Background asyncio task + in-memory cache | Fast reads, periodic DB writes |
| Review feedback | Separate `reviews` table | Preserves full audit trail |
| SQLite mode | WAL + FK ON + `PRAGMA synchronous=NORMAL` | Standard project pattern |

## Risks

1. **SQLite contention on sessions DB** — Multiple concurrent requests writing to shared DB. Mitigated by WAL mode and short transactions.
2. **Message table growth** — Long sessions accumulate many messages. Mitigated by lazy loading and potential archival.
3. **Hot-reload race** — YAML reload + API registration concurrent. Mitigated by RLock.
4. **Per-user DB proliferation** — Many users = many DB files. Same tradeoff as ChaChaNotes/Media_DB, acceptable.
