#!/usr/bin/env python3
"""Manually verify realtime speech against a configured server (TASK-12089).

Run from the project venv with --audio pointing to spoken 16 kHz mono PCM16 WAV.
This explicitly invoked command can use paid providers; it is outside pytest.
Set TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN or SINGLE_USER_API_KEY for authentication.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import wave
from pathlib import Path
from typing import Any

from websockets.sync.client import ClientConnection, connect


def load_audio(path: Path) -> bytes:
    """Load a WAV matching the endpoint's documented input contract."""
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (16000, 1, 2):
            raise ValueError("Audio must be 16 kHz mono PCM16 WAV")
        if wav.getnframes() > 16000 * 30:
            raise ValueError("Audio must be at most 30 seconds")
        return wav.readframes(wav.getnframes())


def wait_for_event(ws: ClientConnection, event_type: str, *, timeout: float = 180.0) -> dict[str, Any]:
    """Wait within one deadline and fail on protocol error events."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Timed out waiting for {event_type}")
        event = json.loads(ws.recv(timeout=remaining))
        if event.get("type") == "error":
            raise RuntimeError("Realtime server returned an error event; inspect server logs")
        if event.get("type") == event_type:
            return event


def run_smoke(url: str, audio: bytes, auth_token: str) -> None:
    """Submit one manual turn through the running server's configured providers."""
    with connect(url, additional_headers={"Authorization": f"Bearer {auth_token}"}, open_timeout=10) as ws:
        wait_for_event(ws, "session.created", timeout=10)
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Answer with one short sentence."},
                }
            )
        )
        wait_for_event(ws, "session.updated", timeout=10)
        # Base64 plus JSON stays comfortably below the 256 KiB frame limit.
        for start in range(0, len(audio), 48 * 1024):
            chunk = base64.b64encode(audio[start : start + 48 * 1024]).decode("ascii")
            ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": chunk}))
        ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        wait_for_event(ws, "conversation.item.done")
        ws.send(json.dumps({"type": "response.create"}))
        done = wait_for_event(ws, "response.done")
        status = done.get("response", {}).get("status")
        if status != "completed":
            raise RuntimeError(f"Realtime response ended with status {status}")


def main(argv: list[str] | None = None) -> int:
    """Validate explicit operator inputs before connecting to any providers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime", help="Configured realtime server URL")
    parser.add_argument("--audio", type=Path, required=True, help="Spoken 16 kHz mono PCM16 WAV file")
    args = parser.parse_args(argv)
    token = os.getenv("TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN") or os.getenv("SINGLE_USER_API_KEY")
    if not token:
        parser.error("Set TLDW_REALTIME_LIVE_SMOKE_AUTH_TOKEN or SINGLE_USER_API_KEY")
    try:
        audio = load_audio(args.audio)
    except (OSError, ValueError, wave.Error) as exc:
        parser.error(str(exc))
    if not audio:
        parser.error("Audio file is empty")
    run_smoke(args.url, audio, token)
    print("Realtime speech smoke completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
