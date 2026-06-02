"""Tests for ACP health check and setup-guide endpoints (Phase 0)."""
import importlib.machinery
import os
import sys
import types

import pytest

pytestmark = pytest.mark.unit


# Stub heavyweight audio deps before app import in shared fixtures.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


class StubRunnerClient:
    def __init__(self) -> None:
        self.agent_capabilities = {"promptCapabilities": {"image": False}}
        self.is_running = True

    async def create_session(self, cwd, mcp_servers=None, agent_type=None, user_id=None, **kwargs):
        return "session-123"

    async def prompt(self, session_id, prompt):
        return {"stopReason": "end"}

    async def cancel(self, session_id):
        pass

    async def close_session(self, session_id):
        pass

    def pop_updates(self, session_id, limit=100):
        return []


@pytest.fixture()
def stub_runner_client(monkeypatch):
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_endpoints

    stub = StubRunnerClient()

    async def _get_runner_client():
        return stub

    monkeypatch.setattr(acp_endpoints, "get_runner_client", _get_runner_client)
    return stub


# ---- Config tests ----

def test_config_binary_path_loaded(monkeypatch):
    """runner_binary_path config option is loaded from env."""
    monkeypatch.setenv("ACP_RUNNER_BINARY_PATH", "/usr/local/bin/tldw-agent-acp")
    monkeypatch.setenv("ACP_RUNNER_COMMAND", "")  # Clear command

    import tldw_Server_API.app.core.Agent_Client_Protocol.config as acp_config
    # Mock get_config_section to return empty dict so config.txt doesn't interfere
    monkeypatch.setattr(acp_config, "get_config_section", lambda _: {})

    cfg = acp_config.load_acp_runner_config()
    assert cfg.binary_path == "/usr/local/bin/tldw-agent-acp"
    # When binary_path is set and command is empty, command becomes the binary_path
    assert cfg.command == "/usr/local/bin/tldw-agent-acp"


def test_config_binary_path_does_not_override_command(monkeypatch):
    """When both binary_path and command are set, command takes precedence."""
    monkeypatch.setenv("ACP_RUNNER_BINARY_PATH", "/usr/local/bin/tldw-agent-acp")
    monkeypatch.setenv("ACP_RUNNER_COMMAND", "go")
    monkeypatch.setenv("ACP_RUNNER_ARGS", '["run", "./cmd"]')

    import tldw_Server_API.app.core.Agent_Client_Protocol.config as acp_config
    monkeypatch.setattr(acp_config, "get_config_section", lambda _: {})

    cfg = acp_config.load_acp_runner_config()
    assert cfg.binary_path == "/usr/local/bin/tldw-agent-acp"
    assert cfg.command == "go"  # command takes precedence


def test_sandbox_config_new_fields(monkeypatch):
    """New sandbox config fields have correct defaults."""
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config
    cfg = load_acp_sandbox_config()
    assert cfg.session_ttl_seconds == 86400
    assert cfg.max_concurrent_sessions_per_user == 5
    assert cfg.max_tokens_per_session == 1_000_000
    assert cfg.max_session_duration_seconds == 14400
    assert cfg.session_retention_days == 30
    assert cfg.audit_retention_days == 30
    assert cfg.allowed_egress_hosts == []


def test_sandbox_config_custom_values(monkeypatch):
    """New sandbox config fields can be customized via env vars."""
    monkeypatch.setenv("ACP_SESSION_TTL_SECONDS", "3600")
    monkeypatch.setenv("ACP_MAX_CONCURRENT_SESSIONS_PER_USER", "10")
    monkeypatch.setenv("ACP_SESSION_RETENTION_DAYS", "7")
    monkeypatch.setenv("ACP_AUDIT_RETENTION_DAYS", "14")
    monkeypatch.setenv("ACP_SANDBOX_ALLOWED_EGRESS_HOSTS", "api.anthropic.com,api.openai.com")

    from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config
    cfg = load_acp_sandbox_config()
    assert cfg.session_ttl_seconds == 3600
    assert cfg.max_concurrent_sessions_per_user == 10
    assert cfg.session_retention_days == 7
    assert cfg.audit_retention_days == 14
    assert cfg.allowed_egress_hosts == ["api.anthropic.com", "api.openai.com"]


def test_sandbox_config_allowed_egress_json(monkeypatch):
    """allowed_egress_hosts accepts JSON array format."""
    monkeypatch.setenv("ACP_SANDBOX_ALLOWED_EGRESS_HOSTS", '["api.anthropic.com", "api.openai.com"]')

    from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config
    cfg = load_acp_sandbox_config()
    assert cfg.allowed_egress_hosts == ["api.anthropic.com", "api.openai.com"]


# ---- Health endpoint tests ----

def test_acp_health_returns_structured_response(client_user_only, stub_runner_client, monkeypatch):
    """Health endpoint returns structured diagnostics."""
    # Ensure no real binaries interfere
    monkeypatch.setenv("ACP_RUNNER_COMMAND", "nonexistent-binary-12345")
    monkeypatch.delenv("ACP_RUNNER_BINARY_PATH", raising=False)

    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "timestamp" in data
    assert "runner" in data
    assert "agents" in data
    assert "runner_probe" in data
    assert "overall" in data
    assert data["runner"]["status"] in ("ok", "missing", "error")


def test_acp_health_runner_missing(client_user_only, stub_runner_client, monkeypatch):
    """Health endpoint reports missing runner."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod
    from tldw_Server_API.app.core.Agent_Client_Protocol.config import ACPRunnerConfig

    def _check_missing():
        return {"status": "missing", "detail": "No runner configured"}

    monkeypatch.setattr(acp_mod, "_check_runner_binary", _check_missing)

    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["runner"]["status"] == "missing"


def test_acp_health_agents_detection(client_user_only, stub_runner_client):
    """Health endpoint lists agent availability."""
    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    agents = data["agents"]
    assert len(agents) >= 3  # claude_code, codex, opencode
    agent_types = [a["agent_type"] for a in agents]
    assert "claude_code" in agent_types
    assert "codex" in agent_types
    assert "opencode" in agent_types
    for agent in agents:
        assert "status" in agent
        assert agent["status"] in ("available", "unavailable", "requires_setup", "unknown")
        assert "support_state" in agent
        assert "verification_level" in agent


def test_acp_health_includes_custom_entrypoint_metadata(client_user_only, stub_runner_client):
    """Health endpoint exposes classifier entrypoint metadata for custom profiles."""
    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    custom = next(agent for agent in data["agents"] if agent["agent_type"] == "custom")

    entrypoint = custom["entrypoint"]
    assert entrypoint["entrypoint_strategy"] == "custom_template"
    assert entrypoint["probe_state"] == "custom_template"
    assert entrypoint["primary_blocker"] == "custom_template"
    assert "custom_template" in entrypoint["blockers"]
    assert "custom ACP profile" in entrypoint["status_message"]


def test_acp_health_runner_probe_with_running_client(client_user_only, stub_runner_client):
    """Health endpoint probes running runner client."""
    # stub_runner_client has is_running = True
    resp = client_user_only.get("/api/v1/acp/health")
    assert resp.status_code == 200
    data = resp.json()
    # Runner probe should try to check the client
    assert "runner_probe" in data


# ---- Setup guide endpoint tests ----

def test_acp_setup_guide_returns_guides(client_user_only, stub_runner_client):
    """Setup guide endpoint returns actionable guides."""
    resp = client_user_only.get("/api/v1/acp/setup-guide")
    assert resp.status_code == 200
    data = resp.json()
    assert "runner" in data
    assert "guides" in data
    assert len(data["guides"]) >= 3
    for guide in data["guides"]:
        assert "agent_type" in guide
        assert "name" in guide
        assert "status" in guide
        assert "steps" in guide
        assert "compatibility" in guide
        assert guide["compatibility"]["support_state"] in (
            "supported",
            "supported_with_caveats",
            "experimental",
            "documented_unverified",
            "unsupported",
        )
        assert guide["compatibility"]["verification_level"] in (
            "documented_only",
            "stub_smoke_tested",
            "live_e2e_tested",
            "sandbox_tested",
            "production_supported",
        )
        assert len(guide["steps"]) > 0


def test_acp_setup_guide_marks_configured_but_unverified_agents(client_user_only, stub_runner_client):
    """Setup guide keeps runtime availability separate from compatibility support state."""
    resp = client_user_only.get("/api/v1/acp/setup-guide?agent_type=custom")
    assert resp.status_code == 200
    data = resp.json()
    guide = data["guides"][0]
    compatibility = guide["compatibility"]
    assert compatibility["support_state"] == "documented_unverified"
    assert compatibility["verification_level"] == "documented_only"
    assert compatibility["docs_url"] == "/docs-static/Development/ACP_Compatibility_Matrix.md"
    assert any("certification checklist" in step.lower() for step in guide["steps"])


def test_acp_setup_guide_codex_includes_entrypoint_blocker_steps(client_user_only, stub_runner_client):
    """Setup guide surfaces documented-candidate ACP adapter blockers."""
    resp = client_user_only.get("/api/v1/acp/setup-guide?agent_type=codex")
    assert resp.status_code == 200
    data = resp.json()
    guide = data["guides"][0]

    entrypoint = guide["entrypoint"]
    assert entrypoint["entrypoint_strategy"] == "documented_candidate"
    assert entrypoint["primary_blocker"] == "adapter_required"
    assert any("adapter" in step.lower() for step in guide["steps"])


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


def test_acp_agents_includes_opencode_entrypoint_metadata(client_user_only, stub_runner_client):
    """Agent list exposes native ACP command metadata separately from compatibility support."""
    resp = client_user_only.get("/api/v1/acp/agents")
    assert resp.status_code == 200
    data = resp.json()
    opencode = next(agent for agent in data["agents"] if agent["type"] == "opencode")

    entrypoint = opencode["entrypoint"]
    assert entrypoint["entrypoint_strategy"] == "native_acp"
    assert entrypoint["acp_command"] == "opencode"
    assert entrypoint["acp_args"] == ["acp"]
    assert entrypoint["probe_state"] in {"ready_to_probe", "blocked"}


def test_entrypoint_status_from_entry_normalizes_docs_url() -> None:
    """Registry-backed entrypoint status uses served docs-static URLs."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod
    from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import AgentRegistryEntry

    status = acp_mod._entrypoint_status_from_entry(
        AgentRegistryEntry(
            type="docs_agent",
            name="Docs Agent",
            entrypoint_strategy="documented_candidate",
            certification_blocker="adapter_required",
            compatibility_docs_url="Docs/Development/ACP_Compatibility_Matrix.md",
        )
    )

    assert status["docs_url"] == "/docs-static/Development/ACP_Compatibility_Matrix.md"


def test_acp_agents_normalizes_invalid_runner_compatibility_values(
    client_user_only,
    stub_runner_client,
    monkeypatch,
):
    """Runner-provided compatibility values are normalized before Pydantic validation."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod

    async def _list_agents():
        return {
            "agents": [
                {
                    "type": "runner_agent",
                    "name": "Runner Agent",
                    "description": "Provided by runner",
                    "isConfigured": True,
                    "support_state": "not-a-real-state",
                    "verification_level": "not-a-real-level",
                    "compatibility_docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
                    "entrypoint": {
                        "profile_key": "runner_agent",
                        "entrypoint_strategy": "adapter_acp",
                        "probe_state": "blocked",
                        "acp_command": "runner-acp",
                        "acp_args": ["--stdio"],
                        "primary_blocker": "adapter_missing",
                        "blockers": ["adapter_missing"],
                        "status_message": "Adapter command is not installed.",
                        "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
                    },
                }
            ],
            "defaultAgentType": "runner_agent",
        }

    monkeypatch.setattr(acp_mod, "_get_registry_agents", lambda: None)
    stub_runner_client.list_agents = _list_agents

    resp = client_user_only.get("/api/v1/acp/agents")
    assert resp.status_code == 200
    data = resp.json()
    agent = data["agents"][0]
    assert agent["support_state"] == "documented_unverified"
    assert agent["verification_level"] == "documented_only"
    assert agent["compatibility_docs_url"] == "/docs-static/Development/ACP_Compatibility_Matrix.md"
    assert agent["entrypoint"]["entrypoint_strategy"] == "external_acp_adapter"
    assert agent["entrypoint"]["primary_blocker"] == "adapter_missing"
    assert agent["entrypoint"]["blockers"] == ["adapter_missing"]


def test_acp_agents_static_fallback_includes_conservative_entrypoint_metadata(
    client_user_only,
    monkeypatch,
):
    """Static fallback rows include conservative entrypoint readiness metadata."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod

    async def _runner_unavailable():
        raise RuntimeError("runner unavailable")

    monkeypatch.setattr(acp_mod, "_get_registry_agents", lambda: None)
    monkeypatch.setattr(acp_mod, "get_runner_client", _runner_unavailable)

    resp = client_user_only.get("/api/v1/acp/agents")
    assert resp.status_code == 200
    data = resp.json()
    custom = next(agent for agent in data["agents"] if agent["type"] == "custom")

    entrypoint = custom["entrypoint"]
    assert entrypoint["profile_key"] == "custom"
    assert entrypoint["entrypoint_strategy"] == "custom_template"
    assert entrypoint["primary_blocker"] == "custom_template"


def test_acp_setup_guide_filter_agent(client_user_only, stub_runner_client):
    """Setup guide endpoint can filter to a specific agent type."""
    resp = client_user_only.get("/api/v1/acp/setup-guide?agent_type=claude_code")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["guides"]) == 1
    assert data["guides"][0]["agent_type"] == "claude_code"


def test_acp_setup_guide_unknown_agent(client_user_only, stub_runner_client):
    """Setup guide endpoint returns all agents for unknown agent_type."""
    resp = client_user_only.get("/api/v1/acp/setup-guide?agent_type=unknown_agent_xyz")
    assert resp.status_code == 200
    data = resp.json()
    # Falls back to listing all agents
    assert len(data["guides"]) >= 3


def test_acp_setup_guide_runner_steps(client_user_only, stub_runner_client, monkeypatch):
    """Setup guide includes runner setup steps when runner is missing."""
    import tldw_Server_API.app.api.v1.endpoints.agent_client_protocol as acp_mod

    def _check_missing():
        return {"status": "missing", "detail": "No runner configured"}

    monkeypatch.setattr(acp_mod, "_check_runner_binary", _check_missing)

    resp = client_user_only.get("/api/v1/acp/setup-guide")
    assert resp.status_code == 200
    data = resp.json()
    runner = data["runner"]
    assert runner["status"] == "missing"
    assert len(runner["steps"]) > 1  # Should have setup instructions
    assert any("setup_acp.sh" in step for step in runner["steps"])
