from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "Helper_Scripts"
    / "Testing-related"
    / "acp_certification_smoke.py"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("acp_certification_smoke", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stub_smoke_manifest_reuses_existing_acp_gates() -> None:
    module = _load_module()

    manifest = module.build_manifest("stub-smoke")

    command_ids = {command["id"] for command in manifest["commands"]}
    capability_ids = {
        capability
        for command in manifest["commands"]
        for capability in command["capabilities"]
    }

    assert manifest["verification_level"] == "stub_smoke_tested"
    assert manifest["requires_live_agent"] is False
    assert "backend_acp_smoke" in command_ids
    assert "go_runner_verify" in command_ids
    assert "browser_mocked_setup_run_diagnose" in command_ids
    assert {
        "init",
        "session_new",
        "prompt",
        "structured_completion",
        "diagnostics",
        "cancel_close",
        "redacted_support_view",
    }.issubset(capability_ids)


def test_live_e2e_manifest_requires_operator_supplied_runtime_state() -> None:
    module = _load_module()

    manifest = module.build_manifest("live-e2e")

    assert manifest["verification_level"] == "live_e2e_tested"
    assert manifest["requires_live_agent"] is True
    assert manifest["required_environment"] == [
        "TLDW_E2E_SERVER_URL",
        "TLDW_E2E_API_KEY",
        "ACP_AGENT_PROFILE",
    ]
    assert any(command["id"] == "live_backend_acp_e2e" for command in manifest["commands"])
    assert all(command["safe_to_run_by_default"] is False for command in manifest["commands"])


def test_manifest_json_output_is_stable_and_machine_readable() -> None:
    module = _load_module()

    rendered = module.render_manifest("stub-smoke", output_format="json")
    payload = json.loads(rendered)

    assert payload["profile"] == "stub-smoke"
    assert payload["commands"][0]["argv"][0] in {"python", "python3"}
    assert payload["commands"][0]["cwd"] == "."
    assert payload["commands"][0]["safe_to_run_by_default"] is True


def test_manifest_markdown_quotes_shell_tokens_with_spaces() -> None:
    module = _load_module()

    rendered = module.render_manifest("stub-smoke")

    assert "'guide ACP setup'" in rendered
    assert "TLDW_WEB_CMD='bun run dev -- -p 18080'" in rendered


def test_manifest_markdown_keeps_live_env_placeholders_expandable() -> None:
    module = _load_module()

    rendered = module.render_manifest("live-e2e")

    assert "TLDW_E2E_SERVER_URL=${TLDW_E2E_SERVER_URL}" in rendered
    assert "TLDW_E2E_API_KEY=${TLDW_E2E_API_KEY}" in rendered
    assert "ACP_AGENT_PROFILE=${ACP_AGENT_PROFILE}" in rendered


def test_run_manifest_live_e2e_refuses_without_required_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.delenv("TLDW_E2E_SERVER_URL", raising=False)
    monkeypatch.delenv("TLDW_E2E_API_KEY", raising=False)
    monkeypatch.delenv("ACP_AGENT_PROFILE", raising=False)

    rc = module.run_manifest("live-e2e")
    captured = capsys.readouterr()

    assert rc == 2
    assert "Refusing to run live ACP certification" in captured.err


def test_run_manifest_returns_first_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()
    commands = module.build_manifest("stub-smoke")["commands"][:1]
    monkeypatch.setattr(
        module,
        "build_manifest",
        lambda _profile: {
            "profile": "stub-smoke",
            "requires_live_agent": False,
            "required_environment": [],
            "commands": commands,
        },
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7),
    )

    rc = module.run_manifest("stub-smoke")

    assert rc == 7


def test_run_manifest_skips_optional_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module,
        "build_manifest",
        lambda _profile: {
            "profile": "stub-smoke",
            "requires_live_agent": False,
            "required_environment": [],
            "commands": [
                {
                    "id": "optional_missing_browser",
                    "cwd": ".",
                    "argv": ["./definitely-missing-playwright"],
                    "safe_to_run_by_default": True,
                    "optional": True,
                }
            ],
        },
    )
    monkeypatch.setattr(module.subprocess, "run", lambda *_args, **_kwargs: None)

    rc = module.run_manifest("stub-smoke")
    captured = capsys.readouterr()

    assert rc == 0
    assert "SKIP optional_missing_browser" in captured.out


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
