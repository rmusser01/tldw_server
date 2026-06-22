import json
from pathlib import Path

import pytest
from mcp_unified.lsp import (
    LspBackendStatus,
    LspCodeAction,
    LspCodeActionsResult,
    LspDiagnostic,
    LspDiagnosticsResult,
    LspHover,
    LspLocation,
    LspLocationsResult,
    LspPosition,
    LspPreview,
    LspRange,
    LspRuntimeConfig,
    LspSignatureHelp,
    LspSymbol,
    LspSymbolsResult,
    LspTextEdit,
    LspToolError,
    redact_lsp_detail,
)


def test_lsp_position_is_zero_based_utf16_contract():
    position = LspPosition(line=0, character=4)

    assert position.line == 0
    assert position.character == 4


def test_lsp_position_rejects_negative_offsets():
    with pytest.raises(ValueError, match="line"):
        LspPosition(line=-1, character=0)


def test_lsp_position_rejects_negative_character_offsets():
    with pytest.raises(ValueError, match="character"):
        LspPosition(line=0, character=-1)


def test_lsp_error_payload_redacts_absolute_paths(tmp_path):
    error = LspToolError("backend_unhealthy", detail=f"failed in {tmp_path}")

    payload = error.to_payload(workspace_root=tmp_path)

    assert str(tmp_path) not in str(payload)
    assert payload["reason_code"] == "backend_unhealthy"


def test_lsp_error_payload_redacts_absolute_paths_from_message(tmp_path):
    error = LspToolError("backend_unhealthy", message=f"failed in {tmp_path}")

    payload = error.to_payload(workspace_root=tmp_path)

    assert str(tmp_path) not in str(payload)
    assert payload["message"] == "failed in <workspace>"


def test_lsp_error_rejects_unknown_reason_code():
    with pytest.raises(ValueError, match="unknown LSP reason_code"):
        LspToolError("not_a_reason")


def test_lsp_detail_truncation_bounds_final_output_length():
    redacted = redact_lsp_detail("abcdefghij", max_length=8)

    assert redacted is not None
    assert len(redacted) <= 8
    assert redacted.endswith("...")


def test_lsp_runtime_config_uses_conservative_defaults():
    config = LspRuntimeConfig()

    assert config.request_timeout_seconds == 5.0
    assert config.startup_timeout_seconds == 10.0
    assert config.idle_ttl_seconds == 300
    assert config.max_diagnostics == 500
    assert config.max_symbols == 500
    assert config.max_references == 500
    assert config.max_hover_bytes == 16_000
    assert config.max_preview_bytes == 200_000
    assert config.max_stderr_bytes == 8_000


def test_lsp_runtime_config_accepts_valid_mapping_values():
    config = LspRuntimeConfig.from_mapping(
        {
            "request_timeout_seconds": 2,
            "startup_timeout_seconds": 3.5,
            "idle_ttl_seconds": 60,
            "max_diagnostics": 10,
            "unknown": -1,
        }
    )

    assert config.request_timeout_seconds == 2.0
    assert config.startup_timeout_seconds == 3.5
    assert config.idle_ttl_seconds == 60
    assert config.max_diagnostics == 10
    assert config.max_symbols == 500


@pytest.mark.parametrize(
    "settings",
    [
        {"request_timeout_seconds": -1},
        {"startup_timeout_seconds": 0},
        {"idle_ttl_seconds": -1},
        {"max_diagnostics": 0},
        {"max_symbols": -1},
        {"max_references": 0},
        {"max_hover_bytes": -1},
        {"max_preview_bytes": 0},
        {"max_stderr_bytes": -1},
    ],
)
def test_lsp_runtime_config_rejects_non_positive_values(settings):
    with pytest.raises(ValueError):
        LspRuntimeConfig.from_mapping(settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"request_timeout_seconds": True},
        {"idle_ttl_seconds": False},
        {"max_diagnostics": True},
    ],
)
def test_lsp_runtime_config_rejects_bool_values(settings):
    with pytest.raises(TypeError):
        LspRuntimeConfig.from_mapping(settings)


@pytest.mark.parametrize(
    "settings",
    [
        {"request_timeout_seconds": "5"},
        {"idle_ttl_seconds": "300"},
        {"max_diagnostics": "500"},
    ],
)
def test_lsp_runtime_config_rejects_invalid_strings(settings):
    with pytest.raises(TypeError):
        LspRuntimeConfig.from_mapping(settings)


def test_lsp_code_action_data_is_json_serializable_and_deterministic():
    action = LspCodeAction(
        title="fix",
        data={
            "z": [1, "two", None],
            "a": {"nested": True},
        },
    )

    payload = action.to_dict()

    assert list(payload["data"]) == ["a", "z"]
    assert json.dumps(payload, sort_keys=True)


@pytest.mark.parametrize("bad_value", [Path("x.py"), b"bytes", {"set"}, object()])
def test_lsp_code_action_rejects_non_json_data_values(bad_value):
    action = LspCodeAction(title="fix", data={"bad": bad_value})

    with pytest.raises(TypeError):
        action.to_dict()


def test_lsp_result_shapes_are_json_serializable():
    position = LspPosition(1, 2)
    lsp_range = LspRange(start=position, end=LspPosition(1, 5))
    location = LspLocation(path="pkg/app.py", range=lsp_range)
    diagnostic = LspDiagnostic(path="pkg/app.py", range=lsp_range, message="unused import")
    edit = LspTextEdit(range=lsp_range, new_text="x")
    payloads = [
        LspDiagnosticsResult(diagnostics=[diagnostic]).to_dict(),
        LspSymbolsResult(symbols=[LspSymbol(name="func", kind="function", location=location)]).to_dict(),
        LspLocationsResult(locations=[location]).to_dict(),
        LspHover(contents="int", range=lsp_range).to_dict(),
        LspSignatureHelp(signatures=["func(x: int)"]).to_dict(),
        LspBackendStatus(name="ruff", healthy=True, capabilities=["diagnostics"]).to_dict(),
        LspPreview(path="pkg/app.py", text_edits=[edit], preview="x").to_dict(),
        LspCodeActionsResult(actions=[LspCodeAction(title="fix", edits=[edit])]).to_dict(),
    ]

    for payload in payloads:
        assert json.dumps(payload, sort_keys=True)


def test_lsp_location_rejects_non_json_direct_path_field():
    location = LspLocation(path=Path("pkg/app.py"), range=LspRange(LspPosition(1, 2), LspPosition(1, 5)))

    with pytest.raises(TypeError):
        location.to_dict()


def test_lsp_signature_help_rejects_non_json_signature_items():
    signature_help = LspSignatureHelp(signatures=[Path("pkg/app.py")])

    with pytest.raises(TypeError):
        signature_help.to_dict()


@pytest.mark.parametrize(
    "status",
    [
        LspBackendStatus(name=Path("ruff"), healthy=True),
        LspBackendStatus(name=1, healthy=True),
        LspBackendStatus(name="ruff", healthy=True, capabilities=[Path("diagnostics")]),
        LspBackendStatus(name="ruff", healthy=True, capabilities=[1]),
        LspBackendStatus(name="ruff", healthy=True, version=Path("0.13")),
        LspBackendStatus(name="ruff", healthy=True, version=1),
        LspBackendStatus(name="ruff", healthy=True, detail=Path("stderr")),
        LspBackendStatus(name="ruff", healthy=True, detail=1),
    ],
)
def test_lsp_backend_status_rejects_non_json_direct_fields(status):
    with pytest.raises(TypeError):
        status.to_dict()
