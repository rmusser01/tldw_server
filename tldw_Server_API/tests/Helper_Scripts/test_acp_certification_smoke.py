from __future__ import annotations

import importlib.util
import json
import time
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


def test_profile_manifest_refuses_blocked_entrypoint() -> None:
    module = _load_module()

    manifest = module.build_agent_profile_manifest(
        {
            "type": "claude-code",
            "name": "Claude Code",
            "entrypoint_strategy": "adapter_acp",
            "acp_command": "tldw-acp-claude",
            "acp_args": ["--stdio"],
            "probe_state": "blocked",
            "primary_blocker": "adapter_missing",
            "blockers": [],
            "status_message": "Configured ACP adapter command is not available on PATH.",
            "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
        }
    )

    assert manifest["profile"] == "claude-code"
    assert manifest["requires_live_agent"] is True
    assert manifest["commands"] == []
    assert manifest["blockers"] == ["adapter_missing"]
    assert manifest["entrypoint"]["entrypoint_strategy"] == "adapter_acp"
    assert manifest["entrypoint"]["probe_state"] == "blocked"
    assert manifest["entrypoint"]["acp_command"] == "tldw-acp-claude"


def test_profile_manifest_refuses_custom_template() -> None:
    module = _load_module()

    manifest = module.build_agent_profile_manifest(
        {
            "type": "custom",
            "name": "Custom",
            "entrypoint_strategy": "custom_template",
            "acp_command": "",
            "acp_args": [],
            "probe_state": "custom_template",
            "primary_blocker": None,
            "blockers": [],
            "status_message": "Custom agent templates require operator-supplied ACP entrypoint metadata.",
            "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
        }
    )

    assert manifest["profile"] == "custom"
    assert manifest["commands"] == []
    assert manifest["blockers"] == []
    assert manifest["entrypoint"]["entrypoint_strategy"] == "custom_template"
    assert manifest["entrypoint"]["probe_state"] == "custom_template"
    assert manifest["entrypoint"]["acp_command"] == ""
    assert "operator-supplied ACP entrypoint metadata" in manifest["notes"][0]


def test_render_manifest_dict_prints_stdin_jsonl_and_blockers() -> None:
    module = _load_module()
    manifest = module.build_agent_profile_manifest(
        {
            "type": "opencode",
            "name": "OpenCode",
            "entrypoint_strategy": "native_acp",
            "acp_command": "opencode",
            "acp_args": ["acp"],
            "probe_state": "ready_to_probe",
            "primary_blocker": "manual_review_required",
            "blockers": [],
            "status_message": "Ready to probe native ACP entrypoint.",
            "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
        }
    )

    rendered = module.render_manifest_dict(manifest)

    assert "- blockers: `manual_review_required`" in rendered
    assert "## Blockers" in rendered
    assert "- manual_review_required" in rendered
    assert "stdin_jsonl:" in rendered
    assert "```jsonl" in rendered
    assert '"method": "initialize"' in rendered
    assert '"method": "session/new"' in rendered
    assert '"method": "session/prompt"' in rendered


def test_agent_profile_cli_uses_registry_classification_path(monkeypatch, capsys) -> None:
    module = _load_module()
    from tldw_Server_API.app.core.Agent_Client_Protocol import agent_registry

    calls = []
    entry = SimpleNamespace(type="opencode", name="OpenCode")

    class _Registry:
        def get_entry(self, profile):
            calls.append(("get_entry", profile))
            return entry

    class _Classification:
        def as_dict(self):
            calls.append(("as_dict",))
            return {
                "entrypoint_strategy": "native_acp",
                "acp_command": "opencode",
                "acp_args": ["acp"],
                "probe_state": "ready_to_probe",
                "primary_blocker": None,
                "blockers": [],
                "status_message": "Configured ACP entrypoint is ready for a bounded initialize probe.",
                "docs_url": "/docs-static/Development/ACP_Compatibility_Matrix.md",
            }

    def _classify_agent_entrypoint(received_entry):
        calls.append(("classify", received_entry.type))
        return _Classification()

    monkeypatch.setattr(agent_registry, "get_agent_registry", lambda: _Registry())
    monkeypatch.setattr(agent_registry, "classify_agent_entrypoint", _classify_agent_entrypoint)

    rc = module.main(["--agent-profile", "opencode", "--format", "json"])
    captured = capsys.readouterr()
    manifest = json.loads(captured.out)

    assert rc == 0
    assert calls == [("get_entry", "opencode"), ("classify", "opencode"), ("as_dict",)]
    assert manifest["profile"] == "opencode"
    assert manifest["name"] == "OpenCode"
    assert manifest["entrypoint"]["entrypoint_strategy"] == "native_acp"
    assert manifest["commands"][0]["argv"] == ["opencode", "acp"]


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


def test_stdio_sequence_runner_ignores_notification_before_initialize_error(monkeypatch) -> None:
    module = _load_module()
    written = []
    responses = iter(
        [
            '{"jsonrpc":"2.0","method":"log/message","params":{"message":"starting"}}\n',
            '{"jsonrpc":"2.0","id":99,"result":{"ignored":true}}\n',
            '{"jsonrpc":"2.0","id":1,"error":{"message":"init failed"}}\n',
        ]
    )

    class _Stdin:
        def write(self, text):
            written.append(text)

        def flush(self):
            return None

    class _Stdout:
        def readline(self):
            return next(responses, "")

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
    assert '"method": "session/new"' not in "".join(written)


def test_stdio_sequence_runner_ignores_notification_before_initialize_success(monkeypatch) -> None:
    module = _load_module()
    written = []
    responses = iter(
        [
            '{"jsonrpc":"2.0","method":"log/message","params":{"message":"starting"}}\n',
            '{"jsonrpc":"2.0","id":99,"result":{"ignored":true}}\n',
            '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"1"}}\n',
            '{"jsonrpc":"2.0","id":2,"result":{"sessionId":"session-1"}}\n',
            '{"jsonrpc":"2.0","id":3,"result":{"content":[]}}\n',
        ]
    )

    class _Stdin:
        def write(self, text):
            written.append(text)

        def flush(self):
            return None

        def close(self):
            return None

    class _Stdout:
        def readline(self):
            return next(responses, "")

    class _Process:
        stdin = _Stdin()
        stdout = _Stdout()

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

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

    assert rc == 0
    assert [json.loads(frame)["method"] for frame in written] == [
        "initialize",
        "session/new",
        "session/prompt",
    ]


def test_stdio_sequence_runner_times_out_on_partial_line_and_cleans_up(monkeypatch) -> None:
    module = _load_module()
    written = []

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, text):
            written.append(text)

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def readline(self):
            time.sleep(0.2)
            return '{"jsonrpc":"2.0","id":1'

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    started = time.monotonic()
    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}},
        ],
        "timeout_seconds": 0.01,
    }, module.ROOT)
    elapsed = time.monotonic() - started

    assert rc == 124
    assert elapsed < 0.35
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert len(written) == 1


def test_stdio_sequence_runner_sanitizes_error_output_and_cleans_up(
    monkeypatch,
    capsys,
) -> None:
    module = _load_module()

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            return None

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def readline(self):
            return (
                '{"jsonrpc":"2.0","id":1,'
                '"error":{"code":-32000,"message":"bad secret-token-value",'
                '"data":{"token":"SECRET_DATA_SHOULD_NOT_PRINT"}}}\n'
            )

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)
    captured = capsys.readouterr()

    assert rc != 0
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True
    assert "code" in captured.err
    assert "bad secret-token-value" in captured.err
    assert "SECRET_DATA_SHOULD_NOT_PRINT" not in captured.err
    assert "data" not in captured.err


def test_stdio_sequence_runner_cleans_up_after_broken_pipe(monkeypatch) -> None:
    module = _load_module()

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            raise BrokenPipeError("closed")

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def readline(self):
            return ""

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc != 0
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_stdio_sequence_runner_drops_noisy_notifications_without_unbounded_queue(
    monkeypatch,
) -> None:
    module = _load_module()
    queue_sizes = []
    lines = [
        '{"jsonrpc":"2.0","method":"log/message","params":{"index":%d}}\n' % index
        for index in range(200)
    ]
    lines.append('{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"1"}}\n')
    responses = iter(lines)

    class _TrackingQueue(module.queue.Queue):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            queue_sizes.append(self.maxsize)

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            return None

        def flush(self):
            return None

        def close(self):
            self.closed = True

    class _Stdout(_Pipe):
        def readline(self, _limit=-1):
            return next(responses, "")

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 0

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.queue, "Queue", _TrackingQueue)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc == 0
    assert queue_sizes
    assert queue_sizes[0] > 0
    assert queue_sizes[0] <= 64
    assert process.killed is False
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_stdio_sequence_runner_fails_and_cleans_up_on_overlong_stdout_line(monkeypatch) -> None:
    module = _load_module()

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            return None

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def readline(self, limit=-1):
            return "{" * max(limit, 1)

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc != 0
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_stdio_sequence_runner_cleans_up_after_write_oserror(monkeypatch) -> None:
    module = _load_module()

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            raise OSError("write failed")

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def readline(self, _limit=-1):
            return ""

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc != 0
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_stdio_sequence_runner_cleans_up_after_flush_valueerror(monkeypatch) -> None:
    module = _load_module()

    class _Pipe:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _Stdin(_Pipe):
        def write(self, _text):
            return None

        def flush(self):
            raise ValueError("flush failed")

    class _Stdout(_Pipe):
        def readline(self, _limit=-1):
            return ""

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            self.wait_calls += 1
            return 1

        def kill(self):
            self.killed = True

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc != 0
    assert process.killed is True
    assert process.wait_calls >= 1
    assert process.stdin.closed is True
    assert process.stdout.closed is True


def test_stdio_sequence_runner_success_closes_stdin_before_terminating(monkeypatch) -> None:
    module = _load_module()
    events = []
    responses = iter(['{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"1"}}\n'])

    class _Pipe:
        def __init__(self, name):
            self.name = name
            self.closed = False

        def close(self):
            self.closed = True
            events.append(f"{self.name}.close")

    class _Stdin(_Pipe):
        def __init__(self):
            super().__init__("stdin")

        def write(self, _text):
            return None

        def flush(self):
            return None

    class _Stdout(_Pipe):
        def __init__(self):
            super().__init__("stdout")

        def readline(self, _limit=-1):
            return next(responses, "")

    class _Process:
        def __init__(self):
            self.stdin = _Stdin()
            self.stdout = _Stdout()
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True
            events.append("terminate")

        def wait(self, timeout=None):
            self.wait_calls += 1
            events.append("wait")
            return 0

        def kill(self):
            self.killed = True
            events.append("kill")

    process = _Process()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_args, **_kwargs: process)

    rc = module._run_stdio_jsonrpc_sequence({
        "id": "acp_initialize_probe",
        "cwd": ".",
        "argv": ["opencode", "acp"],
        "stdin_jsonl": [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ],
        "timeout_seconds": 10,
    }, module.ROOT)

    assert rc == 0
    assert events.index("stdin.close") < events.index("wait")
    assert "terminate" not in events
    assert "kill" not in events
    assert process.terminated is False
    assert process.killed is False
