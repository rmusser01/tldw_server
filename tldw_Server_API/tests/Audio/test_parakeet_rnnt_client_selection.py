"""Run the actual first-party config path without importing audio backends."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import dataclasses
import json
import os
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
AUDIO = ROOT / "tldw_Server_API/app/core/Ingestion_Media_Processing/Audio"


def definitions(path: Path, names: set[str], namespace: dict) -> None:
    """Load only audited first-party definitions, excluding audio backend imports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in names:
            nodes.append(node)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), namespace
    )  # nosec B102


def handler_namespace() -> dict:
    """Keep real config/protocol logic; replace only lifecycle and model boundaries."""
    namespace = {
        "__name__": __name__,
        "asyncio": asyncio,
        "contextlib": contextlib,
        "dataclass": dataclasses.dataclass,
        "json": json,
        "os": os,
        "deque": deque,
        "DiarizationError": type("DiarizationError", (Exception,), {}),
        "QuotaExceeded": type("QuotaExceeded", (Exception,), {}),
        "logger": Mock(),
        "WebSocketDisconnect": type("Disconnected", (Exception,), {}),
        "is_truthy": lambda v: str(v).lower() in {"1", "true"},
        "WSControlSession": lambda _: SimpleNamespace(apply_config=lambda _: SimpleNamespace(events=[])),
        "_get_ws_control_protocol_config": lambda: None,
    }
    definitions(
        ROOT / "tldw_Server_API/app/core/exceptions.py", {"StreamingProtocolError", "AudioProtocolError"}, namespace
    )
    definitions(AUDIO / "Audio_Streaming_Parakeet.py", {"StreamingConfig"}, namespace)
    definitions(
        AUDIO / "model_utils.py",
        {"ALLOWED_PARAKEET_VARIANTS", "CANONICAL_PARAKEET_ONNX_ALIASES", "normalize_model_and_variant"},
        namespace,
    )
    definitions(
        AUDIO / "audio_stream_protocol.py",
        {
            "AUDIO_CHAT_ENDPOINT",
            "AUDIO_TRANSCRIBE_ENDPOINT",
            "_ALLOWED_MODES",
            "AudioProtocolConfig",
            "validate_audio_stream_config",
            "audio_protocol_error_payload",
        },
        namespace,
    )
    definitions(
        AUDIO / "Audio_Streaming_Unified.py",
        {
            "_AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS",
            "UnifiedStreamingConfig",
            "_clamp_float",
            "_clamp_int",
            "_audio_protocol_frame_for_config",
            "handle_unified_websocket",
        },
        namespace,
    )
    return namespace


class ModelBoundaryReached(BaseException):
    """Stop before any model code; the real handler has resolved its config."""


async def run_frame(frame: dict, server_model: str = "nvidia/parakeet-tdt-0.6b-v3") -> tuple:
    """Run the actual handler until rejection or the stubbed model boundary."""
    namespace = handler_namespace()
    config = namespace["UnifiedStreamingConfig"](parakeet_rnnt_model_name=server_model)
    websocket = SimpleNamespace(receive_text=AsyncMock(return_value=json.dumps(frame)), close=AsyncMock())
    stream = SimpleNamespace(start=AsyncMock(), stop=AsyncMock(), send_json=AsyncMock(), error=AsyncMock())
    namespace["WebSocketStream"] = lambda *_args, **_kwargs: stream
    model_entry = Mock(side_effect=ModelBoundaryReached)
    namespace["UnifiedStreamingTranscriber"] = model_entry
    try:
        await namespace["handle_unified_websocket"](websocket, config)
    except ModelBoundaryReached:
        pass
    return config, websocket, stream, model_entry


@pytest.mark.parametrize("strict_protocol", [False, True])
@pytest.mark.parametrize(
    "client_model",
    [
        "example.invalid/unapproved-model",
        "nvidia/parakeet-tdt-0.6b-v3-extra",
        "https://example.invalid/model",
        "./models/example",
        "",
        "nvidia/parakeet-tdt-0.6b-v3 ",
        None,
        True,
        42,
        [],
        {},
    ],
)
def test_unapproved_client_model_is_rejected_before_config_mutation_or_model_entry(
    client_model: object,
    strict_protocol: bool,
) -> None:
    """An arbitrary override must never reach the model initialization boundary."""
    frame = {"type": "config", "parakeet_rnnt_model_name": client_model}
    if strict_protocol:
        frame.update(protocol_version=1, mode="dictate", audio_format="pcm16", sample_rate=16000, channels=1)
    config, websocket, stream, model_entry = asyncio.run(run_frame(frame))
    assert model_entry.call_count == 0
    assert config.parakeet_rnnt_model_name == "nvidia/parakeet-tdt-0.6b-v3"
    error = stream.send_json.call_args.args[0]
    assert error["code"] == "bad_request"
    assert "parakeet_rnnt_model_name" in error["message"]
    assert websocket.close.call_args.kwargs["code"] == 4400


@pytest.mark.parametrize(
    "client_model",
    [
        "nvidia/parakeet-tdt-0.6b-v3",
        "nvidia/parakeet_realtime_eou_120m-v1",
        "trusted-server/custom-model",
    ],
)
def test_approved_client_selection_reaches_model_entry(client_model: str) -> None:
    """Existing documented choices and the trusted server selection stay usable."""
    config, _websocket, stream, model_entry = asyncio.run(
        run_frame(
            {"type": "config", "parakeet_rnnt_model_name": client_model},
            "trusted-server/custom-model",
        )
    )
    assert model_entry.call_count == 1
    assert model_entry.call_args.args[0].parakeet_rnnt_model_name == client_model
    assert config.parakeet_rnnt_model_name == client_model
    assert not stream.send_json.called


@pytest.mark.parametrize("server_model", ["nvidia/parakeet-tdt-0.6b-v3", "trusted-server/custom-model"])
def test_omitted_client_selection_preserves_server_model(server_model: str) -> None:
    """Omitting the field does not replace or restrict a trusted server value."""
    config, _websocket, stream, model_entry = asyncio.run(run_frame({"type": "config"}, server_model))
    assert model_entry.call_count == 1
    assert config.parakeet_rnnt_model_name == server_model
    assert not stream.send_json.called


@pytest.mark.parametrize(
    "settings",
    [
        {"parakeet_use_rnnt_streamer": False},
        {"model": "whisper"},
        {"model": "parakeet-mlx"},
    ],
)
def test_other_client_settings_cannot_bypass_model_source_validation(settings: dict) -> None:
    """Disabling RNNT or choosing another provider cannot admit an arbitrary source."""
    frame = {"type": "config", "parakeet_rnnt_model_name": "example.invalid/unapproved-model", **settings}
    config, _websocket, stream, model_entry = asyncio.run(run_frame(frame))
    assert model_entry.call_count == 0
    assert config.model == "parakeet"
    assert config.parakeet_use_rnnt_streamer is True
    assert stream.send_json.call_args.args[0]["code"] == "bad_request"
