# ACP Entrypoint Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved ACP downstream entrypoint strategy model for registry metadata, classification, certification manifests, and setup/status/API visibility.

**Architecture:** Keep the compatibility matrix as the support-claim source of truth, and add a separate backend-owned entrypoint classification layer in the ACP registry. Registry rows carry explicit ACP launch metadata; a deterministic classifier turns those rows plus runtime discovery into probe state and blocker metadata; existing ACP status/setup/helper surfaces consume the same classifier output.

**Tech Stack:** FastAPI, Pydantic, SQLite, YAML registry loading, pytest, Bandit, existing ACP certification smoke helper.

---

## Source Spec And Tracking

- Spec: `Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md`
- Backlog task: `TASK-287`
- Related GitHub issues: `#1563`, `#1564`
- Relevant skills: `@superpowers:test-driven-development`, `@superpowers:verification-before-completion`

## File Structure

- Modify `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
  - Owns registry entry fields, YAML/API entry loading, dynamic registration, availability checks, and the new deterministic classifier.
- Modify `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
  - Owns persisted dynamic agent registry fields and forward migrations for old databases.
- Modify `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
  - Owns response/request models and enum validation for entrypoint strategy metadata.
- Modify `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
  - Owns health, setup-guide, agent-list, register, and update response wiring.
- Modify `tldw_Server_API/Config_Files/agents.yaml`
  - Seeds explicit strategy metadata for built-in registry rows.
- Modify `Helper_Scripts/Testing-related/acp_certification_smoke.py`
  - Emits and optionally runs bounded profile-specific certification manifests.
- Modify `Docs/Development/ACP_Compatibility_Matrix.md`
  - Adds the entrypoint-specific caveat labels introduced by the spec.
- Test files:
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py`
  - `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py`
  - `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

## Constraints

- Do not infer `acp_command` from `command`.
- Do not mark Codex CLI or Claude Code as `adapter_acp` until a concrete adapter command is selected.
- Do not install downstream agents, implement Codex/Claude adapters, or close `#1563` / `#1564`.
- Do not treat existing MCP `protocol`, `mcp_transport`, or orchestration fields as ACP certification metadata.
- Keep support-state claims controlled by `support_state` and `verification_level`.

---

### Task 1: Registry, API, And DB Strategy Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Modify: `tldw_Server_API/Config_Files/agents.yaml`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py`

- [ ] **Step 1: Write failing registry metadata tests**

Create `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py` with focused tests:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
    AgentRegistry,
    AgentRegistryEntry,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def acp_db(tmp_path):
    db = ACPSessionsDB(db_path=str(tmp_path / "acp_sessions.db"))
    try:
        yield db
    finally:
        db.close()


def test_entrypoint_strategy_defaults_to_documented_candidate() -> None:
    entry = AgentRegistryEntry(type="legacy", name="Legacy")

    assert entry.entrypoint_strategy == "documented_candidate"
    assert entry.acp_command == ""
    assert entry.acp_args == []
    assert entry.adapter_source is None
    assert entry.adapter_docs_url is None
    assert entry.certification_blocker is None


def test_registry_loads_entrypoint_strategy_fields_from_yaml(tmp_path) -> None:
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        """
agents:
  - type: opencode
    name: OpenCode
    command: opencode
    entrypoint_strategy: native_acp
    acp_command: opencode
    acp_args: ["acp"]
"""
    )

    registry = AgentRegistry(yaml_path=str(yaml_file))
    registry.load()

    entry = registry.get_entry("opencode")
    assert entry is not None
    assert entry.entrypoint_strategy == "native_acp"
    assert entry.acp_command == "opencode"
    assert entry.acp_args == ["acp"]


def test_dynamic_registration_preserves_entrypoint_strategy_fields(acp_db) -> None:
    registry = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)

    entry = registry.register_agent(
        type="adapter_agent",
        name="Adapter Agent",
        command="agent-cli",
        entrypoint_strategy="adapter_acp",
        acp_command="agent-acp",
        acp_args=["--stdio"],
        adapter_source="example/agent-acp",
        adapter_docs_url="https://example.test/agent-acp",
        certification_blocker="adapter_missing",
    )

    assert entry.entrypoint_strategy == "adapter_acp"
    assert entry.acp_command == "agent-acp"
    assert entry.acp_args == ["--stdio"]

    reloaded = AgentRegistry(yaml_path="/missing.yaml", db=acp_db)
    reloaded._load_api_entries()
    persisted = reloaded.get_entry("adapter_agent")
    assert persisted is not None
    assert persisted.entrypoint_strategy == "adapter_acp"
    assert persisted.acp_command == "agent-acp"
    assert persisted.acp_args == ["--stdio"]
    assert persisted.adapter_source == "example/agent-acp"
```

- [ ] **Step 2: Write failing DB and schema tests**

Extend `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`:

```python
import sqlite3

from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistry


def test_agent_entrypoint_strategy_fields_round_trip(self, db):
    saved = db.save_agent_entry({
        "agent_type": "adapter",
        "name": "Adapter",
        "entrypoint_strategy": "adapter_acp",
        "acp_command": "adapter-acp",
        "acp_args": '["--stdio"]',
        "adapter_source": "example/adapter",
        "adapter_docs_url": "https://example.test/adapter",
        "certification_blocker": "adapter_missing",
        "source": "api",
    })

    assert saved["entrypoint_strategy"] == "adapter_acp"
    assert saved["acp_command"] == "adapter-acp"
    assert saved["acp_args"] == '["--stdio"]'
    assert saved["adapter_source"] == "example/adapter"
    assert saved["certification_blocker"] == "adapter_missing"


def test_legacy_agent_registry_rows_get_entrypoint_defaults(tmp_path):
    db_path = tmp_path / "legacy_acp_sessions.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE agent_registry (
            agent_type TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            command TEXT NOT NULL DEFAULT '',
            args TEXT NOT NULL DEFAULT '[]',
            env TEXT NOT NULL DEFAULT '{}',
            requires_api_key TEXT,
            is_default INTEGER NOT NULL DEFAULT 0,
            install_instructions TEXT NOT NULL DEFAULT '[]',
            docs_url TEXT,
            mcp_orchestration TEXT NOT NULL DEFAULT 'agent_driven',
            mcp_entry_tool TEXT NOT NULL DEFAULT 'execute',
            mcp_structured_response INTEGER NOT NULL DEFAULT 0,
            mcp_llm_provider TEXT,
            mcp_llm_model TEXT,
            mcp_max_iterations INTEGER NOT NULL DEFAULT 20,
            mcp_refresh_tools INTEGER NOT NULL DEFAULT 0,
            source TEXT NOT NULL DEFAULT 'api',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        INSERT INTO agent_registry (
            agent_type, name, command, source, created_at, updated_at
        ) VALUES ('legacy_api', 'Legacy API', 'legacy-cli', 'api', '2026-01-01', '2026-01-01');
        PRAGMA user_version=13;
        """
    )
    conn.commit()
    conn.close()

    db = ACPSessionsDB(db_path=str(db_path))
    try:
        row = db.get_agent_entry("legacy_api")
        assert row["entrypoint_strategy"] == "documented_candidate"
        assert row["acp_command"] == ""
        assert row["acp_args"] == "[]"

        registry = AgentRegistry(yaml_path="/missing.yaml", db=db)
        registry._load_api_entries()
        entry = registry.get_entry("legacy_api")
        assert entry is not None
        assert entry.entrypoint_strategy == "documented_candidate"
        assert entry.acp_command == ""
        assert entry.acp_args == []
    finally:
        db.close()
```

Extend `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py`:

```python
from pydantic import ValidationError


def test_agent_register_request_exposes_entrypoint_strategy_fields():
    request = ACPAgentRegisterRequest(
        agent_type="native",
        name="Native",
        entrypoint_strategy="native_acp",
        acp_command="native-agent",
        acp_args=["acp"],
    )

    assert request.entrypoint_strategy == "native_acp"
    assert request.acp_command == "native-agent"
    assert request.acp_args == ["acp"]


def test_agent_register_request_rejects_invalid_entrypoint_strategy():
    with pytest.raises(ValidationError):
        ACPAgentRegisterRequest(
            agent_type="bad",
            name="Bad",
            entrypoint_strategy="maybe_acp",
        )
```

- [ ] **Step 3: Run tests and verify they fail for missing fields**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py \
  -q
```

Expected: FAIL because strategy fields do not exist yet.

- [ ] **Step 4: Add schema and persistence fields**

In `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`, add:

```python
ACPEntryPointStrategy = Literal[
    "native_acp",
    "adapter_acp",
    "documented_candidate",
    "custom_template",
]
ACPProbeState = Literal["ready_to_probe", "blocked", "custom_template", "documented_only"]
```

Add request fields to `ACPAgentRegisterRequest` and optional update fields to `ACPAgentUpdateRequest`:

```python
entrypoint_strategy: ACPEntryPointStrategy = Field(default="documented_candidate")
acp_command: str = Field(default="")
acp_args: list[str] = Field(default_factory=list)
adapter_source: str | None = Field(default=None)
adapter_docs_url: str | None = Field(default=None)
certification_blocker: str | None = Field(default=None)
```

In `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`, add matching fields to `AgentRegistryEntry`, thread them through YAML loading, `_load_api_entries()`, `register_agent()`, `_UPDATABLE_FIELDS`, and `update_agent()`.

Add small runtime coercion helpers so invalid YAML or legacy DB text cannot leak invalid enum values into API responses:

```python
def _coerce_entrypoint_strategy(value: Any) -> AgentEntrypointStrategy:
    if value in {"native_acp", "adapter_acp", "documented_candidate", "custom_template"}:
        return value
    return "documented_candidate"
```

Use the coercion helper in YAML loading and DB loading. Do not silently coerce Pydantic request payloads; request models should reject invalid literals.

In `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`, pass the new request fields from `acp_register_agent()` and the agent update endpoint into the registry methods so API-backed rows preserve the same metadata as YAML rows.

In `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`:

- Bump `_SCHEMA_VERSION` from `13` to `14`.
- Add columns to the `agent_registry` `CREATE TABLE` block:

```sql
entrypoint_strategy TEXT NOT NULL DEFAULT 'documented_candidate',
acp_command TEXT NOT NULL DEFAULT '',
acp_args TEXT NOT NULL DEFAULT '[]',
adapter_source TEXT,
adapter_docs_url TEXT,
certification_blocker TEXT,
```

- Add the same columns to `_ALLOWED_MIGRATION_COLUMNS["agent_registry"]`.
- Add an `if current_version < 14:` migration using `_ensure_column(...)`.
- Read and write those fields in `save_agent_entry()`.

- [ ] **Step 5: Seed built-in registry rows**

In `tldw_Server_API/Config_Files/agents.yaml`, add:

```yaml
entrypoint_strategy: native_acp
acp_command: opencode
acp_args:
  - acp
```

for OpenCode, and the equivalent `goose` / `["acp"]` for Goose.

Add:

```yaml
entrypoint_strategy: documented_candidate
acp_command: ""
acp_args: []
certification_blocker: adapter_required
```

for Codex CLI and Claude Code. Do not set either row to `adapter_acp`.

Add:

```yaml
entrypoint_strategy: documented_candidate
acp_command: ""
acp_args: []
certification_blocker: entrypoint_strategy_missing
```

for Aider and Continue.

Add:

```yaml
entrypoint_strategy: custom_template
acp_command: ""
acp_args: []
```

for Custom Agent.

- [ ] **Step 6: Run task tests and commit**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py \
  -q
```

Expected: PASS.

Commit:

```bash
git add \
  tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/Config_Files/agents.yaml \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py
git commit -m "feat: add ACP entrypoint strategy registry metadata"
```

---

### Task 2: Deterministic Entrypoint Classifier

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Modify: `Docs/Development/ACP_Compatibility_Matrix.md`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`

- [ ] **Step 1: Write failing classifier tests**

Add tests to `test_registry_entrypoint_strategy.py`:

```python
from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
    classify_agent_entrypoint,
)


def test_classifier_ready_to_probe_native_entrypoint(monkeypatch):
    entry = AgentRegistryEntry(
        type="opencode",
        name="OpenCode",
        entrypoint_strategy="native_acp",
        acp_command="opencode",
        acp_args=["acp"],
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
        env_getter=lambda _name: "present",
    )

    assert result.probe_state == "ready_to_probe"
    assert result.acp_command == "opencode"
    assert result.acp_args == ["acp"]
    assert result.primary_blocker is None
    assert result.blockers == []


def test_classifier_blocks_native_entrypoint_missing_command():
    entry = AgentRegistryEntry(
        type="goose",
        name="Goose",
        entrypoint_strategy="native_acp",
        acp_command="goose",
        acp_args=["acp"],
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: None)

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "binary_missing"
    assert "binary_missing" in result.blockers


def test_classifier_does_not_infer_native_acp_command_from_command():
    entry = AgentRegistryEntry(
        type="opencode",
        name="OpenCode",
        command="opencode",
        entrypoint_strategy="native_acp",
        acp_command="",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "entrypoint_strategy_missing"
    assert result.acp_command == ""


def test_classifier_does_not_infer_adapter_acp_command_from_command():
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="adapter_acp",
        acp_command="",
        adapter_source="example/codex-acp",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "entrypoint_strategy_missing"
    assert result.acp_command == ""


def test_classifier_documented_candidate_keeps_command_separate_from_acp_command():
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="documented_candidate",
        acp_command="",
        certification_blocker="adapter_required",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "documented_only"
    assert result.primary_blocker == "adapter_required"
    assert result.acp_command == ""


def test_classifier_documented_candidate_is_documented_only():
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        entrypoint_strategy="documented_candidate",
        certification_blocker="adapter_required",
    )

    result = classify_agent_entrypoint(entry)

    assert result.probe_state == "documented_only"
    assert result.primary_blocker == "adapter_required"
    assert result.acp_command == ""


def test_classifier_adapter_requires_adapter_command():
    entry = AgentRegistryEntry(
        type="adapter",
        name="Adapter",
        entrypoint_strategy="adapter_acp",
        acp_command="adapter-acp",
        acp_args=[],
        adapter_source="example/adapter",
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: None)

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "adapter_missing"


def test_classifier_custom_template_is_never_probe_ready():
    entry = AgentRegistryEntry(
        type="custom",
        name="Custom Agent",
        entrypoint_strategy="custom_template",
    )

    result = classify_agent_entrypoint(entry)

    assert result.probe_state == "custom_template"
    assert result.acp_command == ""


def test_classifier_rejects_shell_builtin_entrypoint():
    entry = AgentRegistryEntry(
        type="bad",
        name="Bad",
        entrypoint_strategy="native_acp",
        acp_command="cd",
    )

    result = classify_agent_entrypoint(entry, command_resolver=lambda _command: "/bin/cd")

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "shell_builtin_collision"
```

- [ ] **Step 2: Run classifier tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  -q
```

Expected: FAIL because `classify_agent_entrypoint` does not exist.

- [ ] **Step 3: Implement classifier**

In `agent_registry.py`, add a frozen dataclass:

```python
@dataclass(frozen=True)
class AgentEntrypointClassification:
    profile_key: str
    entrypoint_strategy: AgentEntrypointStrategy
    probe_state: AgentProbeState
    acp_command: str
    acp_args: list[str]
    primary_blocker: str | None
    blockers: list[str]
    status_message: str
    docs_url: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_key": self.profile_key,
            "entrypoint_strategy": self.entrypoint_strategy,
            "probe_state": self.probe_state,
            "acp_command": self.acp_command,
            "acp_args": list(self.acp_args),
            "primary_blocker": self.primary_blocker,
            "blockers": list(self.blockers),
            "status_message": self.status_message,
            "docs_url": self.docs_url,
        }
```

Add:

```python
_SHELL_BUILTIN_COMMANDS = frozenset({"alias", "cd", "export", "set", "source", "unset"})
```

Add `classify_agent_entrypoint(entry, *, command_resolver=shutil.which, env_getter=os.getenv)`. Rules:

- `custom_template` returns `probe_state="custom_template"` and no command.
- `documented_candidate` returns `probe_state="documented_only"` and uses `certification_blocker` when present.
- `native_acp` or `adapter_acp` with no `acp_command` returns blocked with `entrypoint_strategy_missing`.
- Shell builtins return blocked with `shell_builtin_collision`.
- Missing adapter command on `adapter_acp` returns `adapter_missing`.
- Missing native command returns `binary_missing`.
- Missing required API key returns `credentials_missing`.
- Otherwise return `ready_to_probe`.

Do not call subprocesses or run ACP initialize here.

- [ ] **Step 4: Update caveat taxonomy docs**

In `Docs/Development/ACP_Compatibility_Matrix.md`, add rows under "Caveat Taxonomy":

```markdown
| `entrypoint_strategy_missing` | Registry row has no verified ACP stdio entrypoint strategy or command. |
| `adapter_required` | Agent needs a separate ACP adapter before live ACP certification can run. |
| `adapter_missing` | Adapter-backed strategy is configured but the adapter command is unavailable. |
| `acp_initialize_failed` | ACP stdio command started but failed the bounded initialize probe. |
| `shell_builtin_collision` | Configured ACP command resolves to a shell builtin or alias-like value instead of an executable. |
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  -q
```

Expected: PASS.

Commit:

```bash
git add \
  tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  Docs/Development/ACP_Compatibility_Matrix.md
git commit -m "feat: classify ACP entrypoint probe readiness"
```

---

### Task 3: Profile-Specific Certification Manifests

**Files:**
- Modify: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Test: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [ ] **Step 1: Write failing manifest tests**

Extend `test_acp_certification_smoke.py`:

```python
def test_profile_manifest_renders_native_acp_entrypoint(monkeypatch) -> None:
    module = _load_module()

    manifest = module.build_agent_profile_manifest(
        {
            "type": "opencode",
            "name": "OpenCode",
            "entrypoint_strategy": "native_acp",
            "acp_command": "opencode",
            "acp_args": ["acp"],
            "probe_state": "ready_to_probe",
            "primary_blocker": None,
            "blockers": [],
            "status_message": "Ready to probe native ACP entrypoint.",
            "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
        }
    )

    assert manifest["profile"] == "opencode"
    assert manifest["entrypoint"]["entrypoint_strategy"] == "native_acp"
    initialize = next(command for command in manifest["commands"] if command["id"] == "acp_initialize_probe")
    assert initialize["argv"] == ["opencode", "acp"]
    assert initialize["safe_to_run_by_default"] is False
    assert [frame["method"] for frame in initialize["stdin_jsonl"]] == [
        "initialize",
        "session/new",
        "session/prompt",
    ]


def test_profile_manifest_refuses_documented_candidate() -> None:
    module = _load_module()

    manifest = module.build_agent_profile_manifest(
        {
            "type": "codex",
            "name": "Codex",
            "entrypoint_strategy": "documented_candidate",
            "acp_command": "",
            "acp_args": [],
            "probe_state": "documented_only",
            "primary_blocker": "adapter_required",
            "blockers": ["adapter_required"],
            "status_message": "Codex requires an ACP adapter.",
            "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
        }
    )

    assert manifest["requires_live_agent"] is True
    assert manifest["commands"] == []
    assert "adapter_required" in manifest["blockers"]


def test_run_profile_manifest_uses_stdio_sequence_runner(monkeypatch) -> None:
    module = _load_module()
    sequences = []

    monkeypatch.setenv("TLDW_E2E_SERVER_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("TLDW_E2E_API_KEY", "test")
    monkeypatch.setenv("ACP_AGENT_PROFILE", "opencode")
    monkeypatch.setattr(module, "_missing_executable_reason", lambda *_args: None)
    monkeypatch.setattr(
        module,
        "_run_stdio_jsonrpc_sequence",
        lambda command, cwd: sequences.append((command, cwd)) or 0,
    )

    rc = module.run_manifest_dict({
        "profile": "opencode",
        "requires_live_agent": True,
        "required_environment": ["TLDW_E2E_SERVER_URL", "TLDW_E2E_API_KEY", "ACP_AGENT_PROFILE"],
        "commands": [{
            "id": "acp_initialize_probe",
            "cwd": ".",
            "argv": ["opencode", "acp"],
            "safe_to_run_by_default": False,
            "stdin_jsonl": [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}},
                {"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {"prompt": "Reply ok."}},
            ],
            "timeout_seconds": 10,
        }],
    })

    assert rc == 0
    assert sequences
    command, _cwd = sequences[0]
    assert [frame["method"] for frame in command["stdin_jsonl"]] == [
        "initialize",
        "session/new",
        "session/prompt",
    ]
    assert command["timeout_seconds"] == 10


def test_stdio_sequence_runner_stops_after_failed_initialize(monkeypatch) -> None:
    module = _load_module()
    written = []

    class _Stdin:
        def write(self, text):
            written.append(text)

        def flush(self):
            return None

    class _Stdout:
        def readline(self):
            return '{"jsonrpc":"2.0","id":1,"error":{"message":"init failed"}}\n'

    class _Process:
        stdin = _Stdin()
        stdout = _Stdout()

        def wait(self, timeout=None):
            return 1

        def kill(self):
            return None

    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: _Process())

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}},
            {"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc != 0
    assert len(written) == 1
    assert '"method": "initialize"' in written[0]
```

- [ ] **Step 2: Run helper tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py \
  -q
```

Expected: FAIL because profile manifest helpers do not exist.

- [ ] **Step 3: Implement profile manifest builder**

In `acp_certification_smoke.py`:

- Add `build_agent_profile_manifest(entrypoint: dict[str, Any]) -> dict[str, Any]`.
- Add `run_manifest_dict(manifest: dict[str, Any]) -> int`.
- Keep existing `build_manifest("stub-smoke")` and `build_manifest("live-e2e")` behavior stable.
- Add profile command shape only when `probe_state == "ready_to_probe"`.
- Use one subprocess and an ordered JSON-RPC stdio sequence so `session/new`
  and `session/prompt` are sent only after `initialize` returns a non-error
  response:

```python
{
    "id": "acp_initialize_probe",
    "description": "Bounded ACP initialize probe for the selected downstream entrypoint.",
    "cwd": ".",
    "argv": [entrypoint["acp_command"], *entrypoint.get("acp_args", [])],
    "stdin_jsonl": [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "1",
                "clientInfo": {"name": "tldw-server-certification-smoke", "version": "0"},
            },
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": ".", "mcpServers": []},
        },
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/prompt",
            "params": {"prompt": "Reply with a short ACP certification acknowledgement."},
        },
    ],
    "timeout_seconds": 10,
    "capabilities": ["init", "session_new", "prompt"],
    "safe_to_run_by_default": False,
}
```

Update `render_manifest()` to print `stdin_jsonl` and blocker details when present.

Update command execution as follows:

- Commands without `stdin_jsonl` can keep the existing `subprocess.run(...)` path.
- Commands with `stdin_jsonl` should use a new `_run_stdio_jsonrpc_sequence(command, cwd)` helper based on `subprocess.Popen(..., stdin=PIPE, stdout=PIPE, text=True)`.
- `_run_stdio_jsonrpc_sequence()` writes the `initialize` frame first, reads one response line, and stops immediately when the response contains `error`.
- Only after `initialize` succeeds should it write the `session/new` frame; only after `session/new` succeeds should it write the `session/prompt` frame.
- Timeout handling should kill the process and return a nonzero code without hanging the test run.

- [ ] **Step 4: Add CLI profile argument**

Add an optional argument:

```python
parser.add_argument(
    "--agent-profile",
    help="Render or run a registry-backed agent profile manifest.",
)
```

When supplied:

- Load `get_agent_registry()`.
- Fetch the named entry.
- Classify with `classify_agent_entrypoint(entry)`.
- Convert classification to `build_agent_profile_manifest(classification.as_dict() | {"type": entry.type, "name": entry.name})`.
- Render or run that manifest.

- [ ] **Step 5: Run helper tests and commit**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py \
  -q
```

Expected: PASS.

Commit:

```bash
git add \
  Helper_Scripts/Testing-related/acp_certification_smoke.py \
  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
git commit -m "feat: add ACP profile certification manifests"
```

---

### Task 4: Setup, Health, And Agent API Surfaces

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py`

- [ ] **Step 1: Write failing response-surface tests**

Extend `test_acp_health.py`:

```python
def test_acp_agents_include_entrypoint_strategy_metadata(client_user_only, stub_runner_client):
    resp = client_user_only.get("/api/v1/acp/agents")
    assert resp.status_code == 200

    agent = next(item for item in resp.json()["agents"] if item["type"] == "opencode")

    assert agent["entrypoint"]["entrypoint_strategy"] == "native_acp"
    assert agent["entrypoint"]["acp_command"] == "opencode"
    assert agent["entrypoint"]["acp_args"] == ["acp"]
    assert agent["entrypoint"]["probe_state"] in {"ready_to_probe", "blocked"}


def test_acp_setup_guide_includes_entrypoint_blocker_steps(client_user_only, stub_runner_client):
    resp = client_user_only.get("/api/v1/acp/setup-guide?agent_type=codex")
    assert resp.status_code == 200

    guide = resp.json()["guides"][0]

    assert guide["entrypoint"]["entrypoint_strategy"] == "documented_candidate"
    assert guide["entrypoint"]["primary_blocker"] == "adapter_required"
    assert any("adapter" in step.lower() for step in guide["steps"])


def test_acp_health_includes_entrypoint_metadata(client_user_only, stub_runner_client):
    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200

    agent = next(item for item in resp.json()["agents"] if item["agent_type"] == "custom")

    assert "entrypoint" in agent
    assert agent["entrypoint"]["entrypoint_strategy"] == "custom_template"
```

Extend `test_acp_endpoints.py` to prove API-backed dynamic rows receive and expose the same metadata:

```python
def test_register_agent_preserves_entrypoint_strategy_kwargs(
    client_user_only,
    monkeypatch,
):
    import types
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints
    import tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry as registry_mod

    captured = {}

    class _Registry:
        def register_agent(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(type=kwargs["type"], name=kwargs["name"])

    async def _admin_user():
        return types.SimpleNamespace(id=1, is_admin=True)

    monkeypatch.setattr(registry_mod, "get_agent_registry", lambda: _Registry())
    client_user_only.app.dependency_overrides[acp_endpoints.get_request_user] = _admin_user
    try:
        response = client_user_only.post(
            "/api/v1/acp/agents/register",
            json={
                "agent_type": "dynamic_adapter",
                "name": "Dynamic Adapter",
                "command": "agent-cli",
                "entrypoint_strategy": "adapter_acp",
                "acp_command": "agent-acp",
                "acp_args": ["--stdio"],
                "adapter_source": "example/agent-acp",
                "adapter_docs_url": "https://example.test/agent-acp",
                "certification_blocker": "adapter_missing",
            },
        )
    finally:
        client_user_only.app.dependency_overrides.pop(acp_endpoints.get_request_user, None)

    assert response.status_code == 200
    assert captured["entrypoint_strategy"] == "adapter_acp"
    assert captured["acp_command"] == "agent-acp"
    assert captured["acp_args"] == ["--stdio"]
    assert captured["adapter_source"] == "example/agent-acp"
    assert captured["certification_blocker"] == "adapter_missing"


def test_dynamic_agent_list_exposes_entrypoint_strategy_metadata(
    client_user_only,
    monkeypatch,
):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    monkeypatch.setattr(
        acp_endpoints,
        "_get_registry_agents",
        lambda: (
            [
                acp_endpoints.ACPAgentInfo(
                    type="dynamic_adapter",
                    name="Dynamic Adapter",
                    is_configured=False,
                    entrypoint={
                        "profile_key": "dynamic_adapter",
                        "entrypoint_strategy": "adapter_acp",
                        "probe_state": "blocked",
                        "acp_command": "agent-acp",
                        "acp_args": ["--stdio"],
                        "primary_blocker": "adapter_missing",
                        "blockers": ["adapter_missing"],
                        "status_message": "Adapter command is missing.",
                        "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
                    },
                )
            ],
            "dynamic_adapter",
        ),
    )

    response = client_user_only.get("/api/v1/acp/agents")

    assert response.status_code == 200
    agent = response.json()["agents"][0]
    assert agent["entrypoint"]["entrypoint_strategy"] == "adapter_acp"
    assert agent["entrypoint"]["acp_command"] == "agent-acp"
    assert agent["entrypoint"]["primary_blocker"] == "adapter_missing"
```

- [ ] **Step 2: Run response tests and verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py \
  -q
```

Expected: FAIL because response models do not expose `entrypoint`.

- [ ] **Step 3: Add response model**

In `agent_client_protocol.py`, add:

```python
class ACPAgentEntrypointStatus(BaseModel):
    """ACP stdio entrypoint readiness for one downstream agent."""

    profile_key: str = Field(..., description="Registry profile key")
    entrypoint_strategy: ACPEntryPointStrategy = Field(default="documented_candidate")
    probe_state: ACPProbeState = Field(default="documented_only")
    acp_command: str = Field(default="")
    acp_args: list[str] = Field(default_factory=list)
    primary_blocker: str | None = Field(default=None)
    blockers: list[str] = Field(default_factory=list)
    status_message: str = Field(default="")
    docs_url: str | None = Field(default=ACP_COMPATIBILITY_DOCS_URL)
```

Add `entrypoint: ACPAgentEntrypointStatus` to `ACPAgentInfo` and `ACPSetupGuideAgent`.

- [ ] **Step 4: Wire endpoint helpers**

In `agent_client_protocol.py` endpoint module:

- Import `classify_agent_entrypoint`.
- Add `_entrypoint_status_from_entry(reg_entry) -> dict[str, Any]`.
- Add `_entrypoint_status_from_dict(item) -> dict[str, Any]` for runner/static fallback payloads.
- Add `_entrypoint_setup_steps(entrypoint: dict[str, Any]) -> list[str]`.

Setup steps should map:

- `adapter_required` or `adapter_missing`: "Select and install a concrete ACP adapter command before live certification."
- `binary_missing`: "Install the ACP entrypoint command and ensure it is on PATH."
- `credentials_missing`: "Set the required provider credential before live certification."
- `entrypoint_strategy_missing`: "Identify and configure a concrete ACP stdio entrypoint before live certification."
- `shell_builtin_collision`: "Use an executable ACP command, not a shell builtin or alias."
- `custom_template`: "Create a named custom ACP profile with command, args, env, workspace policy, and evidence bundle."

Wire entrypoint metadata into:

- `_check_agent_availability()`
- `acp_health()`
- `acp_setup_guide()`
- `_get_static_agents()`
- `_get_registry_agents()`
- runner-provided `/agents` normalization path
- `acp_register_agent()`
- agent update endpoint, if present in the file

- [ ] **Step 5: Run endpoint tests and commit**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py \
  -q
```

Expected: PASS.

Commit:

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py
git commit -m "feat: expose ACP entrypoint readiness in setup surfaces"
```

---

### Task 5: Verification, Backlog Finalization, And PR Prep

**Files:**
- Modify: `backlog/tasks/task-287 - Implement-ACP-downstream-entrypoint-strategy-stages-1-3.md`

- [ ] **Step 1: Run focused backend/helper tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_mcp_fields.py \
  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_endpoints.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run docs and diff checks**

Run:

```bash
git diff --check
```

Expected: no output.

Run:

```bash
rg -n "entrypoint_strategy|adapter_required|adapter_missing|shell_builtin_collision" \
  tldw_Server_API/Config_Files/agents.yaml \
  Docs/Development/ACP_Compatibility_Matrix.md \
  Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md
```

Expected: strategy fields appear in YAML and caveat labels appear in docs/spec.

- [ ] **Step 3: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  Helper_Scripts/Testing-related/acp_certification_smoke.py \
  -f json -o /tmp/bandit_acp_entrypoint_strategy.json
```

Expected: no new findings in touched code. Existing nosec comments around static manifest subprocess execution should remain justified.

- [ ] **Step 4: Update TASK-287**

Use the Backlog CLI from the worktree:

```bash
backlog task edit TASK-287 --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 --plain
backlog task edit TASK-287 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --plain
backlog task edit TASK-287 --status Done --final-summary "Implemented ACP downstream entrypoint strategy metadata, deterministic probe classification, profile-specific certification manifests, and setup/status/API exposure. Verification: focused ACP registry/API/helper tests, git diff --check, docs grep, and Bandit on touched Python scope." --plain
```

- [ ] **Step 5: Final commit**

Commit Backlog finalization:

```bash
git add "backlog/tasks/task-287 - Implement-ACP-downstream-entrypoint-strategy-stages-1-3.md"
git commit -m "chore: finalize ACP entrypoint strategy task"
```

- [ ] **Step 6: Prepare PR**

Check status:

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
```

If the branch is behind `origin/dev`, rebase before opening/updating the PR:

```bash
git fetch origin
git rebase origin/dev
```

PR summary should explicitly say:

- Adds entrypoint strategy metadata, not live support claims.
- Keeps Codex/Claude as documented candidates until concrete adapter commands are chosen.
- Adds dry-run/live-gated profile manifests but does not install or certify downstream agents.
- Leaves `#1563` and `#1564` open for live certification unless replaced by narrower per-agent evidence issues.
