# Codex ACP Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first Codex ACP integration slice by making `external_acp_adapter` a first-class strategy, seeding a pinned `codex-acp` profile, wiring runner launch/readiness semantics, and exposing actionable setup state in the WebUI.

**Architecture:** Keep native ACP and external ACP adapters on the existing ACP downstream runner path, but stop using the vague `adapter_acp` public label. The backend owns canonical registry/readiness metadata, the Go runner launches the ACP entrypoint rather than the display CLI, and the frontend consumes structured readiness instead of guessing from `is_configured`.

**Tech Stack:** FastAPI, Pydantic, SQLite, YAML registry loading, Go `tldw-agent`, React/TypeScript, Vitest, pytest, Bandit.

---

## Source Spec And Tracking

- Spec: `Docs/superpowers/specs/2026-06-01-acp-codex-orchestration-design.md`
- Planning task: `TASK-592`
- Design task: `TASK-591`
- Related existing design: `Docs/superpowers/specs/2026-05-12-acp-downstream-entrypoint-strategy-design.md`
- Relevant skills for implementation: `@superpowers:test-driven-development`, `@superpowers:verification-before-completion`

## Scope

This plan implements the first slice only:

- Add canonical `external_acp_adapter`.
- Read legacy `adapter_acp` values as an internal compatibility alias.
- Seed Codex as an external ACP adapter candidate using pinned `zed-industries/codex-acp` `0.15.0`.
- Add adapter/readiness fields needed by backend, runner, API, and WebUI.
- Keep Codex app-server, generic runner adapters, persistent normalized event tables, and live Codex certification evidence out of this implementation slice.

Do not claim Codex is `supported_with_caveats` or `live_e2e_tested` until a later live certification task passes.

## File Structure

- Modify `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
  - Owns strategy normalization, registry entry fields, deterministic entrypoint classification, and availability/readiness payloads.
- Modify `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
  - Owns public API strategy literals, registry request validation, and agent/readiness response models.
- Modify `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
  - Owns `/api/v1/acp/health`, `/api/v1/acp/setup-guide`, `/api/v1/acp/agents`, static fallback normalization, register, and update wiring.
- Modify `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
  - Owns persisted dynamic agent registry fields and forward migrations for adapter metadata.
- Modify `tldw_Server_API/Config_Files/agents.yaml`
  - Seeds the Codex external adapter candidate profile and updates field comments.
- Modify `tools/tldw-agent/internal/config/config.go`
  - Carries strategy, ACP entrypoint, adapter, and credential metadata into the Go runner.
- Modify `tools/tldw-agent/internal/acp/runner.go`
  - Resolves the launch command from strategy-specific rules and reports strategy-aware agent readiness.
- Modify `apps/packages/ui/src/services/acp/types.ts`
  - Mirrors backend entrypoint/readiness types.
- Modify `apps/packages/ui/src/services/acp/readiness.ts`
  - Normalizes ACP agent readiness and builds user-facing setup issues from stable blocker codes.
- Modify `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx`
  - Uses structured readiness for agent cards, disabled state, tooltips, and setup copy.
- Modify `Docs/Development/ACP_Compatibility_Matrix.md`
  - Adds external adapter caveat language and updates the Codex row without claiming live certification.
- Modify `Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
  - Documents the pinned Codex adapter setup path.
- Modify `Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
  - Keeps the published copy consistent with the active guide.
- Modify `Helper_Scripts/Testing-related/acp_certification_smoke.py` if the existing manifest logic rejects non-native ACP strategies.
  - Keeps certification command generation strategy-aware without requiring live evidence in this slice.

Test files:

- `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py`
- `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py`
- `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py` if certification helper changes.
- `tools/tldw-agent/internal/config/config_test.go`
- `tools/tldw-agent/internal/acp/runner_test.go`
- `apps/packages/ui/src/services/acp/__tests__/readiness.test.ts`
- `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts`

---

### Task 1: Backend Strategy Normalization And Public Schema

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py`

- [ ] **Step 1: Write failing tests for canonical and legacy strategies**

Extend `test_registry_entrypoint_strategy.py`:

```python
def test_external_acp_adapter_is_canonical_strategy() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        adapter_source="zed-industries/codex-acp",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
        env_getter=lambda _name: None,
    )

    assert result.entrypoint_strategy == "external_acp_adapter"
    assert result.probe_state == "ready_to_probe"
    assert result.acp_command == "codex-acp"
    assert result.primary_blocker is None


def test_legacy_adapter_acp_input_is_imported_as_external_acp_adapter(tmp_path) -> None:
    yaml_file = tmp_path / "agents.yaml"
    yaml_file.write_text(
        """
agents:
  - type: legacy_codex
    name: Legacy Codex
    command: codex
    entrypoint_strategy: adapter_acp
    acp_command: codex-acp
"""
    )

    registry = AgentRegistry(yaml_path=str(yaml_file))
    registry.load()

    entry = registry.get_entry("legacy_codex")
    assert entry is not None
    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert classify_agent_entrypoint(entry).entrypoint_strategy == "external_acp_adapter"
```

Add schema/API normalization tests:

```python
from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPAgentEntrypointStatus,
    ACPAgentRegisterRequest,
)


def test_agent_entrypoint_status_accepts_external_adapter() -> None:
    status = ACPAgentEntrypointStatus(
        profile_key="codex",
        entrypoint_strategy="external_acp_adapter",
        probe_state="blocked",
    )
    assert status.entrypoint_strategy == "external_acp_adapter"


def test_register_request_imports_legacy_adapter_acp_alias() -> None:
    request = ACPAgentRegisterRequest(
        agent_type="legacy_codex",
        name="Legacy Codex",
        entrypoint_strategy="adapter_acp",
    )
    assert request.entrypoint_strategy == "external_acp_adapter"


def test_static_codex_fallback_uses_external_adapter_and_delegated_credentials(monkeypatch) -> None:
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import _get_static_agents

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    agents, _default_agent = _get_static_agents()
    codex = next(agent for agent in agents if agent.type == "codex")

    assert codex.requires_api_key is None
    assert codex.entrypoint.entrypoint_strategy == "external_acp_adapter"
    assert codex.entrypoint.acp_command == "codex-acp"
    assert codex.entrypoint.credential_state == "delegated"
    assert codex.entrypoint.primary_blocker in {"adapter_missing", "agent_binary_missing", "live_certification_required"}
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py -q`

Expected: FAIL because `external_acp_adapter` is not accepted and legacy aliases are still emitted.

- [ ] **Step 2: Implement canonical strategy normalization**

In `agent_registry.py`:

```python
AgentEntrypointStrategy = Literal[
    "native_acp",
    "external_acp_adapter",
    "documented_candidate",
    "custom_template",
]
_LEGACY_ENTRYPOINT_STRATEGY_ALIASES = {
    "adapter_acp": "external_acp_adapter",
}


def _coerce_entrypoint_strategy(value: Any) -> AgentEntrypointStrategy:
    normalized = _LEGACY_ENTRYPOINT_STRATEGY_ALIASES.get(str(value), value)
    if normalized in {"native_acp", "external_acp_adapter", "documented_candidate", "custom_template"}:
        return normalized
    return "documented_candidate"
```

Add an `AgentRegistryEntry.__post_init__` if direct dataclass construction should canonicalize values:

```python
def __post_init__(self) -> None:
    self.entrypoint_strategy = _coerce_entrypoint_strategy(self.entrypoint_strategy)
```

Update classifier checks from `adapter_acp` to `external_acp_adapter`.

In `agent_client_protocol.py`, replace the public `ACPEntryPointStrategy` literal with:

```python
ACPEntryPointStrategy = Literal[
    "native_acp",
    "external_acp_adapter",
    "documented_candidate",
    "custom_template",
]
```

Use Pydantic `field_validator(..., mode="before")` on `ACPAgentRegisterRequest`, `ACPAgentUpdateRequest`, and `ACPAgentEntrypointStatus` to import `adapter_acp` as `external_acp_adapter`.

In `agent_client_protocol.py` endpoint helpers, update `_ACP_ENTRYPOINT_STRATEGIES` and `_entrypoint_status_from_dict()` so old runner/static payloads normalize to the canonical value before API output.

Update `_get_static_agents()` so the Codex fallback uses the same external-adapter/delegated-auth model as the seeded registry instead of the old `OPENAI_API_KEY`-only documented-candidate shape:

```python
ACPAgentInfo(
    type="codex",
    name="OpenAI Codex",
    description="OpenAI's Codex agent through the Codex ACP adapter",
    is_configured=False,
    requires_api_key=None,
    support_state="experimental",
    verification_level="documented_only",
    compatibility_notes="Static fallback only; Codex uses an external ACP adapter and remains experimental until live certification passes.",
    entrypoint=_entrypoint_status_from_dict({
        "type": "codex",
        "entrypoint_strategy": "external_acp_adapter",
        "acp_command": "codex-acp",
        "credential_state": "delegated",
        "primary_blocker": "live_certification_required",
        "blockers": ["live_certification_required"],
        "adapter_source": "zed-industries/codex-acp",
        "adapter_version": "0.15.0",
        "runtime_backend": "acp_downstream",
    }),
)
```

- [ ] **Step 3: Run focused backend tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py -q`

Expected: PASS.

- [ ] **Step 4: Commit Task 1**

```bash
git add tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py
git commit -m "feat: normalize external ACP adapter strategy"
```

---

### Task 2: Backend Adapter Metadata, Readiness, And Setup Guidance

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py`

- [ ] **Step 1: Write failing readiness tests**

Add tests that separate display-agent availability from adapter availability:

```python
def test_external_adapter_reports_adapter_missing_without_falling_back_to_agent_command() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        adapter_source="zed-industries/codex-acp",
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: "/usr/bin/codex" if command == "codex" else None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "adapter_missing"
    assert "adapter_missing" in result.blockers
    assert "binary_missing" not in result.blockers


def test_external_adapter_reports_display_agent_binary_missing_separately() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: "/usr/bin/codex-acp" if command == "codex-acp" else None,
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "agent_binary_missing"
    assert "agent_binary_missing" in result.blockers


def test_external_adapter_blocks_mutable_npx_latest_invocation() -> None:
    entry = AgentRegistryEntry(
        type="codex",
        name="Codex",
        command="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="npx",
        acp_args=["@zed-industries/codex-acp@latest"],
        credential_policy="delegated_to_adapter",
    )

    result = classify_agent_entrypoint(
        entry,
        command_resolver=lambda command: f"/usr/bin/{command}",
    )

    assert result.probe_state == "blocked"
    assert result.primary_blocker == "mutable_adapter_invocation"
    assert "mutable_adapter_invocation" in result.blockers
```

Add endpoint/setup-guide tests that assert setup steps use stable blocker labels:

```python
def test_entrypoint_setup_steps_include_external_adapter_specific_actions() -> None:
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import _entrypoint_setup_steps

    steps = _entrypoint_setup_steps({
        "entrypoint_strategy": "external_acp_adapter",
        "primary_blocker": "adapter_missing",
        "blockers": ["adapter_missing", "agent_binary_missing"],
    })

    joined = " ".join(steps)
    assert "ACP adapter" in joined
    assert "agent binary" in joined or "Codex" in joined
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py -q`

Expected: FAIL until new metadata/readiness fields and step mapping exist.

- [ ] **Step 2: Add adapter metadata and credential policy fields**

In `AgentRegistryEntry`, add:

```python
adapter_package: str | None = None
adapter_version: str | None = None
adapter_version_policy: Literal["exact_pin_required", "operator_managed", "unknown"] = "unknown"
adapter_install_source: Literal["github_release_preferred", "npm_pinned_allowed", "operator_managed", "unknown"] = "unknown"
credential_policy: Literal["env_var", "delegated_to_adapter", "none", "unknown"] = "unknown"
runtime_backend: Literal["acp_downstream", "codex_app_server", "runner_adapter", "unknown"] = "acp_downstream"
```

Add matching fields to `ACPAgentEntrypointStatus` where the UI needs to display setup state:

```python
display_command: str = ""
display_binary_found: bool | None = None
adapter_found: bool | None = None
credential_state: Literal["ready", "missing", "delegated", "unknown"] = "unknown"
adapter_source: str | None = None
adapter_package: str | None = None
adapter_version: str | None = None
runtime_backend: str = "acp_downstream"
```

Keep `is_configured` as "ready to start an ACP session", not "fully certified" and not "auth-proven":

```python
is_configured = entrypoint.probe_state == "ready_to_probe"
```

For `credential_policy="delegated_to_adapter"`, passive readiness should use `credential_state="delegated"` and should not block on `OPENAI_API_KEY`.

- [ ] **Step 3: Expand stable blocker vocabulary**

Add blocker codes and setup text:

```python
_ACP_ENTRYPOINT_STEP_MAP = {
    "adapter_required": "Select and install a concrete ACP adapter command before live certification.",
    "adapter_missing": "Install the configured ACP adapter command and ensure it is on PATH.",
    "agent_binary_missing": "Install the downstream agent CLI that the adapter controls.",
    "binary_missing": "Install the ACP entrypoint command and ensure it is on PATH.",
    "credentials_missing": "Set the required provider credential before live certification.",
    "entrypoint_strategy_missing": "Identify and configure a concrete ACP stdio entrypoint before live certification.",
    "shell_builtin_collision": "Use an executable ACP command, not a shell builtin or alias.",
    "custom_template": "Create a named custom ACP profile with command, args, env, workspace policy, and evidence bundle.",
    "live_certification_required": "Run live ACP certification before claiming this agent is supported.",
    "mutable_adapter_invocation": "Install a pinned ACP adapter binary instead of using a mutable package invocation.",
    "adapter_auth_missing": "Authenticate the ACP adapter or configure its accepted credential source.",
    "adapter_auth_failed": "Re-authenticate the ACP adapter; its provider login failed during active probing.",
    "agent_auth_failed": "Re-authenticate the downstream agent; the adapter started but the agent rejected auth.",
}
```

Do not add active auth probing in this task.

- [ ] **Step 4: Run focused backend tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py
git commit -m "feat: expose ACP adapter readiness metadata"
```

---

### Task 3: Dynamic Registry Persistence For Adapter Metadata

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py`
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`

- [ ] **Step 1: Write failing persistence tests**

Extend `test_acp_sessions_db.py`:

```python
def test_agent_registry_adapter_metadata_round_trips(db) -> None:
    saved = db.save_agent_entry({
        "agent_type": "codex",
        "name": "Codex",
        "command": "codex",
        "entrypoint_strategy": "external_acp_adapter",
        "acp_command": "codex-acp",
        "adapter_source": "zed-industries/codex-acp",
        "adapter_package": "@zed-industries/codex-acp",
        "adapter_version": "0.15.0",
        "adapter_version_policy": "exact_pin_required",
        "adapter_install_source": "github_release_preferred",
        "credential_policy": "delegated_to_adapter",
        "runtime_backend": "acp_downstream",
        "source": "api",
    })

    assert saved["entrypoint_strategy"] == "external_acp_adapter"
    assert saved["adapter_version"] == "0.15.0"
    assert saved["credential_policy"] == "delegated_to_adapter"
    assert saved["runtime_backend"] == "acp_downstream"
```

Add a legacy DB migration test by creating an `agent_registry` table without the new columns and then initializing `ACPSessionsDB`.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py -q`

Expected: FAIL because the new columns are missing.

- [ ] **Step 2: Add DB columns and save/load plumbing**

In the `agent_registry` schema and forward-migration column map, add:

```sql
adapter_package TEXT,
adapter_version TEXT,
adapter_version_policy TEXT NOT NULL DEFAULT 'unknown',
adapter_install_source TEXT NOT NULL DEFAULT 'unknown',
credential_policy TEXT NOT NULL DEFAULT 'unknown',
runtime_backend TEXT NOT NULL DEFAULT 'acp_downstream'
```

Update `save_agent_entry()`, `list_agent_entries()`, and registry `_load_api_entries()` to preserve these fields.

Update register/update request models and endpoint calls to pass these fields.

- [ ] **Step 3: Run focused DB and registry tests**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py -q`

Expected: PASS.

- [ ] **Step 4: Commit Task 3**

```bash
git add tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
  tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
  tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py
git commit -m "feat: persist ACP adapter registry metadata"
```

---

### Task 4: Seed Codex External Adapter Profile And Docs

**Files:**
- Modify: `tldw_Server_API/Config_Files/agents.yaml`
- Modify: `Docs/Development/ACP_Compatibility_Matrix.md`
- Modify: `Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
- Modify: `Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md`
- Test: `tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py`

- [ ] **Step 1: Write failing seeded-registry test**

Add:

```python
def test_seeded_codex_profile_uses_pinned_external_acp_adapter() -> None:
    registry = AgentRegistry()
    registry.load()

    entry = registry.get_entry("codex")

    assert entry is not None
    assert entry.entrypoint_strategy == "external_acp_adapter"
    assert entry.command == "codex"
    assert entry.acp_command == "codex-acp"
    assert entry.adapter_source == "zed-industries/codex-acp"
    assert entry.adapter_version == "0.15.0"
    assert entry.adapter_version_policy == "exact_pin_required"
    assert entry.adapter_install_source == "github_release_preferred"
    assert entry.credential_policy == "delegated_to_adapter"
    assert entry.support_state == "experimental"
    assert entry.verification_level == "documented_only"
```

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py::test_seeded_codex_profile_uses_pinned_external_acp_adapter -q`

Expected: FAIL because Codex is still a documented candidate.

- [ ] **Step 2: Update `agents.yaml`**

Change the Codex row to:

```yaml
  - type: codex
    name: OpenAI Codex CLI
    description: "OpenAI's coding agent via the Codex ACP adapter"
    command: codex
    args: []
    env: {}
    requires_api_key: null
    install_instructions:
      - "Install OpenAI Codex CLI using the current OpenAI instructions."
      - "Install zed-industries/codex-acp 0.15.0 from the GitHub release artifact and ensure codex-acp is on PATH."
    docs_url: "https://github.com/openai/codex"
    support_state: experimental
    verification_level: documented_only
    compatibility_notes: "Codex can be reached through zed-industries/codex-acp, but this tldw profile remains experimental until live ACP certification passes."
    compatibility_docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md"
    entrypoint_strategy: external_acp_adapter
    acp_command: codex-acp
    acp_args: []
    adapter_source: zed-industries/codex-acp
    adapter_docs_url: "https://github.com/zed-industries/codex-acp"
    adapter_package: "@zed-industries/codex-acp"
    adapter_version: "0.15.0"
    adapter_version_policy: exact_pin_required
    adapter_install_source: github_release_preferred
    credential_policy: delegated_to_adapter
    runtime_backend: acp_downstream
    certification_blocker: live_certification_required
```

Do not use `npx @latest` in seeded runtime config.

- [ ] **Step 3: Update docs without overclaiming support**

In `ACP_Compatibility_Matrix.md`:

- Add `external_acp_adapter` to terminology where entrypoint strategies are described.
- Add caveats `live_certification_required`, `agent_binary_missing`, `adapter_auth_missing`, `adapter_auth_failed`, and `agent_auth_failed`.
- Update the Codex row mode/evidence to describe `codex-acp` `0.15.0` as a documented external adapter path.
- Keep `support_state=experimental` and `verification_level=documented_only`.
- Keep all live capability checks as `skip` until certification passes.

In both ACP getting-started docs:

- Tell users to install Codex CLI separately.
- Prefer the `zed-industries/codex-acp` `0.15.0` GitHub release artifact.
- Allow pinned npm only as an operator setup/certification alternative.
- State passive readiness checks never install packages or run `npx @latest`.

- [ ] **Step 4: Run focused registry test**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py::test_seeded_codex_profile_uses_pinned_external_acp_adapter -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add tldw_Server_API/Config_Files/agents.yaml \
  Docs/Development/ACP_Compatibility_Matrix.md \
  Docs/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md \
  Docs/Published/User_Guides/Integrations_Experiments/Getting_Started_with_ACP.md \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py
git commit -m "docs: seed Codex external ACP adapter profile"
```

---

### Task 5: Go Runner Config And Strategy-Aware Launch Rules

**Files:**
- Modify: `tools/tldw-agent/internal/config/config.go`
- Modify: `tools/tldw-agent/internal/acp/runner.go`
- Test: `tools/tldw-agent/internal/config/config_test.go`
- Test: `tools/tldw-agent/internal/acp/runner_test.go`

- [ ] **Step 1: Write failing config parsing tests**

Create or extend `tools/tldw-agent/internal/config/config_test.go`:

```go
func TestRegisteredAgentParsesACPEntrypointFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	err := os.WriteFile(path, []byte(`
agents:
  default: codex
  agents:
    - type: codex
      name: Codex
      command: codex
      args: ["--display"]
      entrypoint_strategy: external_acp_adapter
      acp_command: codex-acp
      acp_args: ["--stdio"]
      adapter_source: zed-industries/codex-acp
      adapter_version: 0.15.0
      credential_policy: delegated_to_adapter
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadFrom(path)
	if err != nil {
		t.Fatal(err)
	}
	agent := cfg.Agents.Agents[0]

	if agent.EntrypointStrategy != "external_acp_adapter" {
		t.Fatalf("strategy = %q", agent.EntrypointStrategy)
	}
	if agent.ACPCommand != "codex-acp" {
		t.Fatalf("acp command = %q", agent.ACPCommand)
	}
	if !reflect.DeepEqual(agent.ACPArgs, []string{"--stdio"}) {
		t.Fatalf("acp args = %#v", agent.ACPArgs)
	}
}
```

Run: `cd tools/tldw-agent && go test ./internal/config`

Expected: FAIL because fields do not exist.

- [ ] **Step 2: Write failing runner launch-rule tests**

Extend `runner_test.go` with small tests around a new pure helper, for example `resolveLaunchAgentConfig(entry config.RegisteredAgent)`.

```go
func TestRunnerLaunchesACPCommandForExternalAdapter(t *testing.T) {
	entry := config.RegisteredAgent{
		Type: "codex",
		Command: "codex",
		Args: []string{"--display"},
		EntrypointStrategy: "external_acp_adapter",
		ACPCommand: "codex-acp",
		ACPArgs: []string{"--stdio"},
	}

	agentCfg, err := resolveLaunchAgentConfig(entry)
	if err != nil {
		t.Fatalf("resolve failed: %v", err)
	}

	if agentCfg.Command != "codex-acp" {
		t.Fatalf("command = %q, want codex-acp", agentCfg.Command)
	}
	if !reflect.DeepEqual(agentCfg.Args, []string{"--stdio"}) {
		t.Fatalf("args = %#v", agentCfg.Args)
	}
}

func TestRunnerDoesNotFallbackExternalAdapterToDisplayCommand(t *testing.T) {
	entry := config.RegisteredAgent{
		Type: "codex",
		Command: "codex",
		EntrypointStrategy: "external_acp_adapter",
		ACPCommand: "",
	}

	_, err := resolveLaunchAgentConfig(entry)
	if err == nil || !strings.Contains(err.Error(), "acp_command is required") {
		t.Fatalf("expected missing acp command error, got %v", err)
	}
}

func TestRunnerLegacyNativeACPFallsBackToCommand(t *testing.T) {
	entry := config.RegisteredAgent{
		Type: "goose",
		Command: "goose",
		Args: []string{"acp"},
		EntrypointStrategy: "native_acp",
		ACPCommand: "",
	}

	agentCfg, err := resolveLaunchAgentConfig(entry)
	if err != nil {
		t.Fatalf("resolve failed: %v", err)
	}
	if agentCfg.Command != "goose" {
		t.Fatalf("command = %q", agentCfg.Command)
	}
}
```

Add passive inventory tests that prove `agent/list` does not launch downstream agents:

```go
func TestRunnerInitializeDoesNotSpawnDownstreamForPassiveCapabilities(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type: "codex",
			Name: "Codex",
			Command: "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand: "codex-acp",
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("initialize must not spawn downstream agents for passive capabilities")
		return nil, nil, nil
	})

	resp := callRunnerInitialize(t, runner)
	if resp.AgentCapabilities == nil {
		t.Fatalf("initialize should still return the default capability envelope")
	}
}

func TestRunnerAgentListUsesPassiveReadinessWithoutSpawning(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type: "codex",
			Name: "Codex",
			Command: "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand: "codex-acp",
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("agent/list must not spawn or initialize downstream agents")
		return nil, nil, nil
	})
	runner.SetLookPathFunc(func(command string) (string, error) {
		switch command {
		case "codex", "codex-acp":
			return "/usr/bin/" + command, nil
		default:
			return "", exec.ErrNotFound
		}
	})

	resp := callRunnerAgentList(t, runner)
	agent := findAgentListItem(t, resp, "codex")

	if !agent.IsConfigured {
		t.Fatalf("codex should be passively configured when display and adapter commands resolve")
	}
	if agent.ProbeState != "ready_to_probe" {
		t.Fatalf("probe state = %q", agent.ProbeState)
	}
}

func TestRunnerAgentListBlocksMutableNpxLatestWithoutSpawning(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type: "codex",
			Name: "Codex",
			Command: "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand: "npx",
			ACPArgs: []string{"@zed-industries/codex-acp@latest"},
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("agent/list must not execute npx")
		return nil, nil, nil
	})
	runner.SetLookPathFunc(func(command string) (string, error) {
		switch command {
		case "codex", "npx":
			return "/usr/bin/" + command, nil
		default:
			return "", exec.ErrNotFound
		}
	})

	resp := callRunnerAgentList(t, runner)
	agent := findAgentListItem(t, resp, "codex")

	if agent.IsConfigured {
		t.Fatalf("mutable npx @latest adapter invocation must not be passively configured")
	}
	if agent.PrimaryBlocker != "mutable_adapter_invocation" {
		t.Fatalf("primary blocker = %q", agent.PrimaryBlocker)
	}
}
```

Implement small test helpers such as `callRunnerInitialize()`, `callRunnerAgentList()`, and `findAgentListItem()` using the same `net.Pipe` pattern already present in `runner_test.go`.

Run: `cd tools/tldw-agent && go test ./internal/acp`

Expected: FAIL until strategy-aware launch resolution and passive readiness exist.

- [ ] **Step 3: Add Go config fields**

Update `RegisteredAgent`:

```go
type RegisteredAgent struct {
	Type               string   `yaml:"type"`
	Name               string   `yaml:"name"`
	Description        string   `yaml:"description"`
	Command            string   `yaml:"command"`
	Args               []string `yaml:"args"`
	Env                []string `yaml:"env"`
	RequiresAPIKey     string   `yaml:"requires_api_key"`
	EntrypointStrategy string   `yaml:"entrypoint_strategy"`
	ACPCommand         string   `yaml:"acp_command"`
	ACPArgs            []string `yaml:"acp_args"`
	AdapterSource      string   `yaml:"adapter_source"`
	AdapterDocsURL     string   `yaml:"adapter_docs_url"`
	AdapterPackage     string   `yaml:"adapter_package"`
	AdapterVersion     string   `yaml:"adapter_version"`
	CredentialPolicy   string   `yaml:"credential_policy"`
	RuntimeBackend     string   `yaml:"runtime_backend"`
}
```

Normalize blank `EntrypointStrategy` to legacy native behavior only inside launch resolution; do not mutate config globally unless there is an existing pattern for config normalization.

- [ ] **Step 4: Implement strategy-aware launch resolution and passive readiness**

In `runner.go`, add a pure launch helper and use it in `handleSessionNew()` and any explicit active probe/certification path only. Do not call it from `agent/list` or from `initialize` just to build passive capabilities:

```go
func resolveLaunchAgentConfig(entry config.RegisteredAgent) (config.AgentConfig, error) {
	strategy := strings.TrimSpace(entry.EntrypointStrategy)
	if strategy == "" {
		strategy = "native_acp"
	}

	switch strategy {
	case "native_acp":
		if strings.TrimSpace(entry.ACPCommand) != "" {
			return config.AgentConfig{
				Command: entry.ACPCommand,
				Args: entry.ACPArgs,
				Env: expandAgentEnv(entry.Env),
			}, nil
		}
		if strings.TrimSpace(entry.Command) == "" {
			return config.AgentConfig{}, fmt.Errorf("agent.command is required")
		}
		return config.AgentConfig{Command: entry.Command, Args: entry.Args, Env: expandAgentEnv(entry.Env)}, nil
	case "external_acp_adapter":
		if strings.TrimSpace(entry.ACPCommand) == "" {
			return config.AgentConfig{}, fmt.Errorf("agent.acp_command is required for external_acp_adapter")
		}
		return config.AgentConfig{
			Command: entry.ACPCommand,
			Args: entry.ACPArgs,
			Env: expandAgentEnv(entry.Env),
		}, nil
	default:
		return config.AgentConfig{}, fmt.Errorf("agent strategy %q is not launchable by ACP downstream runner", strategy)
	}
}
```

Keep the session error message stable and actionable.

Change `buildAgentCapabilities()` so it returns default capabilities plus cached capabilities from a prior real downstream session, but does not call `refreshCapabilities()` when no cache exists. Either remove `refreshCapabilities()` or leave it available only for an explicit active health/certification action. This prevents a status/handshake path from launching `codex-acp` or any package-manager command.

Add a passive readiness helper for `agent/list` that never calls `spawnFunc`, never starts downstream `initialize`, and never runs package managers:

```go
type agentReadiness struct {
	IsConfigured bool
	ProbeState string
	PrimaryBlocker string
	Blockers []string
	StatusMessage string
}

func (r *Runner) passiveAgentReadiness(entry config.RegisteredAgent) agentReadiness {
	// Use r.lookPathFunc, defaulting to exec.LookPath, for display command and ACP entrypoint checks.
	// For external_acp_adapter, require acp_command and do not fall back to command.
	// If acp_command is "npx" and acp_args contain "@latest", return mutable_adapter_invocation.
	// Return ready_to_probe only when the deterministic launch entrypoint is present.
}
```

Add `lookPathFunc func(string) (string, error)` to `Runner`, initialize it to `exec.LookPath` in `NewRunner()`, and add `SetLookPathFunc()` for tests. Existing active capability probing can remain in `probeAgentCapabilities()` for explicit session/certification paths, but `handleAgentList()` must call only `passiveAgentReadiness()`.

- [ ] **Step 5: Include entrypoint metadata in `agent/list`**

Extend the runner `agentListItem` response with fields the backend can normalize when it falls back to runner inventory:

```go
EntrypointStrategy string   `json:"entrypoint_strategy,omitempty"`
ACPCommand         string   `json:"acp_command,omitempty"`
ACPArgs            []string `json:"acp_args,omitempty"`
AdapterSource      string   `json:"adapter_source,omitempty"`
AdapterVersion     string   `json:"adapter_version,omitempty"`
CredentialPolicy   string   `json:"credential_policy,omitempty"`
RuntimeBackend     string   `json:"runtime_backend,omitempty"`
DisplayCommand     string   `json:"display_command,omitempty"`
DisplayBinaryFound bool     `json:"display_binary_found"`
AdapterFound       bool     `json:"adapter_found"`
CredentialState    string   `json:"credential_state,omitempty"`
ProbeState         string   `json:"probe_state,omitempty"`
PrimaryBlocker     string   `json:"primary_blocker,omitempty"`
Blockers           []string `json:"blockers,omitempty"`
StatusMessage      string   `json:"status_message,omitempty"`
```

Populate `DisplayCommand`, `DisplayBinaryFound`, `AdapterFound`, `CredentialState`, and `IsConfigured` from `passiveAgentReadiness()`, not from a spawned capability probe. For delegated Codex adapter credentials, emit `CredentialState: "delegated"` until an explicit active probe/session proves auth success or failure.

- [ ] **Step 6: Run Go tests**

Run: `cd tools/tldw-agent && go test ./internal/config ./internal/acp`

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add tools/tldw-agent/internal/config/config.go \
  tools/tldw-agent/internal/config/config_test.go \
  tools/tldw-agent/internal/acp/runner.go \
  tools/tldw-agent/internal/acp/runner_test.go
git commit -m "feat: launch ACP adapters through explicit entrypoints"
```

---

### Task 6: Frontend Readiness Types And ACP Create Modal

**Files:**
- Modify: `apps/packages/ui/src/services/acp/types.ts`
- Modify: `apps/packages/ui/src/services/acp/readiness.ts`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx`
- Test: `apps/packages/ui/src/services/acp/__tests__/readiness.test.ts`
- Test: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts`

- [x] **Step 1: Write failing readiness normalization tests**

Extend `readiness.test.ts`:

```ts
import {
  buildACPAgentSetupSummary,
  isACPAgentReadyToStart
} from "@/services/acp/readiness"

it("treats external ACP adapter readiness as launchable when ready to probe", () => {
  const agent = {
    type: "codex",
    name: "Codex",
    description: "Codex via adapter",
    is_configured: true,
    entrypoint: {
      profile_key: "codex",
      entrypoint_strategy: "external_acp_adapter",
      probe_state: "ready_to_probe",
      acp_command: "codex-acp",
      acp_args: [],
      blockers: [],
      credential_state: "delegated"
    }
  }

  expect(isACPAgentReadyToStart(agent)).toBe(true)
})

it("explains missing codex-acp adapter distinctly from missing provider keys", () => {
  const summary = buildACPAgentSetupSummary({
    type: "codex",
    name: "Codex",
    description: "Codex via adapter",
    is_configured: false,
    entrypoint: {
      profile_key: "codex",
      entrypoint_strategy: "external_acp_adapter",
      probe_state: "blocked",
      acp_command: "codex-acp",
      acp_args: [],
      primary_blocker: "adapter_missing",
      blockers: ["adapter_missing"],
      status_message: "Adapter missing",
      adapter_version: "0.15.0"
    }
  })

  expect(summary.disabledReason).toContain("codex-acp")
  expect(summary.disabledReason).not.toContain("API key")
})

it("does not let stale is_configured override a blocked structured entrypoint", () => {
  const agent = {
    type: "codex",
    name: "Codex",
    description: "Codex via adapter",
    is_configured: true,
    entrypoint: {
      profile_key: "codex",
      entrypoint_strategy: "external_acp_adapter",
      probe_state: "blocked",
      acp_command: "codex-acp",
      acp_args: [],
      primary_blocker: "mutable_adapter_invocation",
      blockers: ["mutable_adapter_invocation"],
      status_message: "Mutable adapter invocation is blocked",
      credential_state: "delegated"
    }
  }

  const summary = buildACPAgentSetupSummary(agent)

  expect(isACPAgentReadyToStart(agent)).toBe(false)
  expect(summary.disabled).toBe(true)
  expect(summary.disabledReason).toContain("pinned")
})
```

Run: `bunx vitest run apps/packages/ui/src/services/acp/__tests__/readiness.test.ts`

Expected: FAIL because these helper functions/types do not exist.

- [x] **Step 2: Add TypeScript entrypoint/readiness types**

In `types.ts`, add:

```ts
export type ACPEntryPointStrategy =
  | "native_acp"
  | "external_acp_adapter"
  | "documented_candidate"
  | "custom_template"

export type ACPProbeState =
  | "ready_to_probe"
  | "blocked"
  | "custom_template"
  | "documented_only"
  | "unsupported_backend"

export type ACPCredentialState = "ready" | "missing" | "delegated" | "unknown"

export interface ACPAgentEntrypointStatus {
  profile_key: string
  entrypoint_strategy: ACPEntryPointStrategy
  probe_state: ACPProbeState
  acp_command: string
  acp_args: string[]
  primary_blocker?: string | null
  blockers: string[]
  status_message: string
  docs_url?: string | null
  display_command?: string
  display_binary_found?: boolean | null
  adapter_found?: boolean | null
  credential_state?: ACPCredentialState
  adapter_source?: string | null
  adapter_package?: string | null
  adapter_version?: string | null
  runtime_backend?: string
}
```

Add `entrypoint: ACPAgentEntrypointStatus` to `ACPAgentInfo`.

- [x] **Step 3: Add frontend readiness helpers**

In `readiness.ts`, implement:

```ts
export const isACPAgentReadyToStart = (agent: Pick<ACPAgentInfo, "is_configured" | "entrypoint">): boolean => {
  if (agent.entrypoint) {
    return agent.entrypoint.probe_state === "ready_to_probe"
  }
  return agent.is_configured === true
}

export const buildACPAgentSetupSummary = (agent: ACPAgentInfo) => {
  const entrypoint = agent.entrypoint
  const blockers = entrypoint?.blockers ?? []
  const adapterName = entrypoint?.acp_command || "the configured ACP adapter"
  const readyToStart = isACPAgentReadyToStart(agent)
  if (blockers.includes("adapter_missing")) {
    return {
      disabled: true,
      disabledReason: `Install ${adapterName}${entrypoint?.adapter_version ? ` ${entrypoint.adapter_version}` : ""} and ensure it is on PATH.`
    }
  }
  if (blockers.includes("agent_binary_missing")) {
    return {
      disabled: true,
      disabledReason: `Install ${agent.name}'s CLI before starting this adapter-backed session.`
    }
  }
  if (blockers.includes("mutable_adapter_invocation")) {
    return {
      disabled: true,
      disabledReason: "Install a pinned ACP adapter binary instead of using a mutable package invocation."
    }
  }
  if (entrypoint?.credential_state === "delegated" && readyToStart) {
    return {
      disabled: false,
      disabledReason: "Credentials are handled by the adapter and will be verified when the session starts."
    }
  }
  return {
    disabled: !readyToStart,
    disabledReason: entrypoint?.status_message || "This ACP agent is not ready to start."
  }
}
```

Refine copy as needed, but keep stable blocker mapping rather than generic string matching. Structured `entrypoint.probe_state` is authoritative when present; `is_configured` is only a legacy fallback for payloads without entrypoint metadata.

- [x] **Step 4: Update `ACPSessionCreateModal`**

Use the helper in `AgentCard`:

```tsx
const setup = buildACPAgentSetupSummary(agent)
const disabled = setup.disabled
```

Replace the current `requiresApiKey`-only tooltip with strategy-aware copy:

```tsx
{disabled && (
  <Tooltip title={setup.disabledReason}>
    <AlertCircle className="h-4 w-4 text-warning" />
  </Tooltip>
)}
{agent.entrypoint?.entrypoint_strategy === "external_acp_adapter" && (
  <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-text-muted">
    Adapter
  </span>
)}
```

Do not add a new banner bar. Keep the information inside the existing agent card density.

- [x] **Step 5: Run focused frontend tests**

Run: `bunx vitest run apps/packages/ui/src/services/acp/__tests__/readiness.test.ts apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts`

Expected: PASS.

Actual verification:
- Red run: `./node_modules/.bin/vitest run src/services/acp/__tests__/readiness.test.ts` failed on missing `buildACPAgentSetupSummary` and `isACPAgentReadyToStart`.
- Review follow-up red run: `./node_modules/.bin/vitest run src/services/acp/__tests__/readiness.test.ts` failed on blocker precedence when `primary_blocker="mutable_adapter_invocation"` and secondary blockers included `adapter_missing`.
- Green run: `./node_modules/.bin/vitest run src/services/acp/__tests__/readiness.test.ts src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts` passed with 2 files and 11 tests after blocker precedence, binary blocker alias, and cached-agent default-selection fixes.
- `git diff --check` passed.
- UI package typecheck with `NODE_OPTIONS=--max-old-space-size=8192 ./node_modules/.bin/tsc --noEmit --project tsconfig.json` failed on existing non-ACP errors in QuickIngest, Layout, Playground, Sidepanel, onboarding, option-index, and quick-ingest-open files; no errors were reported for the ACP files changed in this task.

- [x] **Step 6: Commit Task 6**

```bash
git add apps/packages/ui/src/services/acp/types.ts \
  apps/packages/ui/src/services/acp/readiness.ts \
  apps/packages/ui/src/services/acp/__tests__/readiness.test.ts \
  apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionCreateModal.tsx \
  apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
git commit -m "feat: show ACP adapter readiness in session creation"
```

---

### Task 7: Certification Manifest Compatibility Without Live Evidence

**Files:**
- Inspect: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Modify if needed: `Helper_Scripts/Testing-related/acp_certification_smoke.py`
- Test if modified: `tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

- [x] **Step 1: Inspect certification helper strategy assumptions**

Run: `rg -n "adapter_acp|native_acp|entrypoint_strategy|codex" Helper_Scripts/Testing-related/acp_certification_smoke.py tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py`

Expected: identify whether the helper rejects `external_acp_adapter`.

Actual: `build_agent_profile_manifest` does not filter by strategy name; it emits the supplied entrypoint metadata and creates the ACP initialize probe whenever `probe_state == "ready_to_probe"`. Inspection found no helper rejection path for `external_acp_adapter`.

- [x] **Step 2: If needed, write failing helper test**

Only if the helper has strategy filtering, add:

```python
def test_certification_manifest_accepts_external_acp_adapter_strategy(tmp_path, monkeypatch) -> None:
    # Build the same manifest path used for native ACP, but with strategy external_acp_adapter.
    manifest = build_agent_profile_manifest(
        agent_profile="codex",
        entrypoint_strategy="external_acp_adapter",
        acp_command="codex-acp",
        acp_args=[],
    )

    assert manifest["entrypoint_strategy"] == "external_acp_adapter"
    assert manifest["acp_command"] == "codex-acp"
```

Use the actual helper function names after inspection.

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q`

Expected: FAIL only if code changes are required.

Actual: not needed because the helper has no strategy filter to fail; existing helper coverage remains compatible with the external adapter path.

- [x] **Step 3: Normalize certification manifest handling**

Update helper logic so `external_acp_adapter` is treated as ACP downstream for command-manifest purposes, but do not auto-run live Codex certification and do not update compatibility evidence.

Actual: no code change needed; current command-manifest handling is strategy-agnostic and gated by `probe_state`, so external adapters are already treated as ACP downstream when ready to probe.

- [x] **Step 4: Run helper tests if modified**

Run: `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py -q`

Expected: PASS.

Actual: ran despite no code changes. `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py` passed with 37 tests.

- [x] **Step 5: Commit Task 7 if files changed**

```bash
git add Helper_Scripts/Testing-related/acp_certification_smoke.py \
  tldw_Server_API/tests/Helper_Scripts/test_acp_certification_smoke.py
git commit -m "test: allow external ACP adapter certification manifests"
```

If no helper changes are needed, record the inspection result in `TASK-593` implementation notes instead of committing an empty task.

Actual: no helper/test files changed; inspection result recorded in the implementation tracker instead of creating an empty helper commit.

---

### Task 8: Verification, Backlog Closeout, And PR Readiness

**Files:**
- Modify: `backlog/tasks/task-592 - Plan-Codex-ACP-adapter-implementation.md` only if this plan is being finalized in this branch.
- Modify later implementation task records as they are created/executed.

- [x] **Step 1: Run backend verification**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py \
  -q
```

Expected: PASS.

Actual: first run failed because `test_acp_setup_guide_codex_includes_entrypoint_blocker_steps` still expected Codex `entrypoint_strategy == "documented_candidate"`. Root cause was a stale assertion from the pre-`external_acp_adapter` Codex model; the test was updated to expect `external_acp_adapter` and concrete adapter blockers. Rerun passed with 114 tests and 6 warnings:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Agent_Client_Protocol/test_registry_entrypoint_strategy.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_sessions_db.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_health.py tldw_Server_API/tests/Agent_Client_Protocol/test_acp_config_validation.py -q
```

- [x] **Step 2: Run Go verification**

Run: `cd tools/tldw-agent && go test ./internal/config ./internal/acp`

Expected: PASS.

Actual: passed with `GOCACHE=/private/tmp/tldw-go-cache go test ./internal/config ./internal/acp` from `tools/tldw-agent`.

- [x] **Step 3: Run frontend verification**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/services/acp/__tests__/readiness.test.ts \
  apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts
```

Expected: PASS.

Actual: passed from `apps/packages/ui` with `./node_modules/.bin/vitest run src/services/acp/__tests__/readiness.test.ts src/components/Option/ACPPlayground/__tests__/ACPSessionCreateModal.modal-prop-guard.test.ts` with 2 files and 11 tests. The package-local command is required in this worktree because root `bunx vitest` does not resolve the UI package alias/dependency layout.

- [x] **Step 4: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate && python -m bandit \
  -r tldw_Server_API/app/core/Agent_Client_Protocol/agent_registry.py \
     tldw_Server_API/app/api/v1/schemas/agent_client_protocol.py \
     tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py \
     tldw_Server_API/app/core/DB_Management/ACP_Sessions_DB.py \
     Helper_Scripts/Testing-related/acp_certification_smoke.py \
  -f json -o /tmp/bandit_task_592_codex_acp_adapter.json
```

Expected: no new findings in touched code. If `Helper_Scripts/Testing-related/acp_certification_smoke.py` was not touched, remove it from the command.

Actual: ran without the helper file because Task 7 did not touch it. Bandit exited 1 for the known `ACP_Sessions_DB.py` baseline only: B105 at line 205 and B608 at lines 1058, 1063, 1099, 1117, 1947, 2162, and 2247. `agent_registry.py`, `agent_client_protocol.py` schemas, and endpoint files had zero findings. JSON output: `/tmp/bandit_task_593_codex_acp_adapter.json`.

- [x] **Step 5: Run diff hygiene checks**

Run: `git diff --check`

Expected: no whitespace errors.

Run: `rg -n "adapter_acp" tldw_Server_API apps tools Docs | cat`

Expected: only legacy-import compatibility tests/comments mention `adapter_acp`; current UI, emitted API examples, active docs, and seeded config use `external_acp_adapter`.

Actual: `git diff --check` passed. `rg -n "adapter_acp" tldw_Server_API apps tools Docs` returned legacy-import tests, compatibility alias code, historical plan/spec docs, and the compatibility matrix note that legacy input may be imported as `external_acp_adapter`; no current UI, emitted API example, seeded config, or active Codex docs use `adapter_acp`.

- [x] **Step 6: Update Backlog final summary and commit any remaining docs/task changes**

Use Backlog MCP for final notes. Record:

- Verification commands and results.
- Explicit live-certification skip: Codex live ACP certification remains a separate follow-up.
- Any helper inspection result from Task 7.

Commit:

```bash
git add backlog/tasks/task-592\ -\ Plan-Codex-ACP-adapter-implementation.md
git commit -m "docs: finalize Codex ACP adapter plan"
```

Expected: branch is clean except for intentional follow-up implementation commits.

---

## Follow-Up Tasks Not In This Plan

- Live Codex ACP certification using the local backend, WebUI, runner, and installed `codex-acp` `0.15.0`.
- Codex app-server backend design and implementation.
- Generic runner-adapter fallback implementation for agents that do not speak ACP and do not expose app-server-like APIs.
- Persistent normalized event table for app-server and runner-adapter backends, unless a later readiness/session-status task proves it is needed earlier.
