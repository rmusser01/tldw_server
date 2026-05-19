from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal

SessionState = Literal["awaiting_config", "running", "paused", "closing"]
_ACTIVE_STATES = {"running", "paused"}


@dataclass(slots=True)
class WSControlProtocolConfig:
    ws_control_v2_enabled: bool = False
    paused_audio_queue_cap_seconds: float = 2.0
    overflow_warning_interval_seconds: float = 5.0


@dataclass(slots=True)
class ProtocolDecision:
    protocol_version: int
    negotiated_v2: bool
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None


@dataclass(slots=True)
class FrameDecision:
    intent: str
    events: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, Any] | None = None
    should_emit_full_transcript: bool = False
    should_reset: bool = False
    should_close: bool = False
    dropped_buffered_seconds: float = 0.0
    queued_audio_seconds: float = 0.0


@dataclass(slots=True)
class PausedAudioDecision:
    accepted_seconds: float
    dropped_seconds: float
    buffered_seconds: float
    events: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class WSControlSession:
    config: WSControlProtocolConfig
    state: SessionState = "awaiting_config"
    protocol_version: int = 1
    negotiated_v2: bool = False
    _paused_audio_segments: deque[float] = field(default_factory=deque)
    _paused_audio_seconds: float = 0.0
    _last_overflow_warning_at: float | None = None

    def apply_config(self, frame: dict[str, Any] | None) -> ProtocolDecision:
        requested = 1
        if isinstance(frame, dict):
            requested = _normalize_protocol_version(frame.get("protocol_version"))

        negotiated_v2 = requested == 2 and self.config.ws_control_v2_enabled
        self.protocol_version = 2 if negotiated_v2 else 1
        self.negotiated_v2 = negotiated_v2
        self.state = "running"

        events: list[dict[str, Any]] = []
        if negotiated_v2:
            events.append({"type": "status", "state": "configured", "protocol_version": 2})

        return ProtocolDecision(
            protocol_version=self.protocol_version,
            negotiated_v2=self.negotiated_v2,
            events=events,
        )

    def handle_frame(self, frame: dict[str, Any] | None) -> FrameDecision:
        if not isinstance(frame, dict):
            return self._invalid_control("Control frame must be a JSON object")

        frame_type = str(frame.get("type") or "").strip().lower()
        if frame_type == "control":
            if not self.negotiated_v2:
                return self._invalid_control("Control frames require protocol_version=2")
            action = str(frame.get("action") or "").strip().lower()
            if self.state not in _ACTIVE_STATES:
                return self._invalid_control(
                    f"Control action {action or 'unknown'} not allowed in state {self.state}"
                )
            return self._handle_action(action, emit_status=True)

        if frame_type == "commit":
            if self.state not in _ACTIVE_STATES:
                return self._invalid_control(f"Frame type commit not allowed in state {self.state}")
            return self._handle_action("commit", emit_status=False)
        if frame_type == "stop":
            if self.state not in _ACTIVE_STATES:
                return self._invalid_control(f"Frame type stop not allowed in state {self.state}")
            return self._handle_action("stop", emit_status=self.negotiated_v2)
        if frame_type == "reset":
            if self.state not in _ACTIVE_STATES:
                return self._invalid_control(f"Frame type reset not allowed in state {self.state}")
            self._reset_runtime_state()
            self.state = "running"
            return FrameDecision(
                intent="reset",
                events=[{"type": "status", "state": "reset"}],
                should_reset=True,
            )

        return FrameDecision(intent="ignored")

    def buffer_paused_audio(self, duration_seconds: float, *, now: float) -> PausedAudioDecision:
        if self.state != "paused":
            return PausedAudioDecision(
                accepted_seconds=0.0,
                dropped_seconds=0.0,
                buffered_seconds=self._paused_audio_seconds,
                events=[],
            )

        duration = max(0.0, float(duration_seconds))
        self._paused_audio_segments.append(duration)
        self._paused_audio_seconds += duration

        dropped_seconds = 0.0
        cap = max(0.0, float(self.config.paused_audio_queue_cap_seconds))
        while self._paused_audio_seconds > cap and self._paused_audio_segments:
            overflow = self._paused_audio_seconds - cap
            oldest = self._paused_audio_segments[0]
            drop_now = min(oldest, overflow)
            dropped_seconds += drop_now
            if drop_now >= oldest:
                self._paused_audio_segments.popleft()
            else:
                self._paused_audio_segments[0] = oldest - drop_now
            self._paused_audio_seconds -= drop_now

        buffered_seconds = self._paused_audio_seconds
        events: list[dict[str, Any]] = []
        if dropped_seconds > 0 and self._should_emit_overflow_warning(now):
            self._last_overflow_warning_at = float(now)
            events.append(
                {
                    "type": "warning",
                    "warning_type": "audio_dropped_during_pause",
                    "message": (
                        f"Paused audio queue exceeded {cap:.1f}s; "
                        f"dropped {dropped_seconds:g}s using drop_oldest policy"
                    ),
                    "dropped_seconds": dropped_seconds,
                    "buffered_seconds": buffered_seconds,
                    "policy": "drop_oldest",
                }
            )

        return PausedAudioDecision(
            accepted_seconds=duration,
            dropped_seconds=dropped_seconds,
            buffered_seconds=buffered_seconds,
            events=events,
        )

    def release_paused_audio(self) -> float:
        """Transfer buffered paused-audio accounting to the caller and clear local state."""
        return self._clear_paused_audio_queue()

    def _handle_action(self, action: str, *, emit_status: bool) -> FrameDecision:
        if action == "pause":
            self.state = "paused"
            events = [self._status("paused")] if emit_status else []
            return FrameDecision(intent="pause", events=events, queued_audio_seconds=self._paused_audio_seconds)

        if action == "resume":
            self.state = "running"
            events = [self._status("resumed")] if emit_status else []
            return FrameDecision(intent="resume", events=events, queued_audio_seconds=self._paused_audio_seconds)

        if action == "commit":
            return FrameDecision(
                intent="commit",
                should_emit_full_transcript=True,
                queued_audio_seconds=self._paused_audio_seconds,
            )

        if action == "stop":
            dropped = self._clear_paused_audio_queue()
            self.state = "closing"
            events = [self._status("closing")] if emit_status else []
            return FrameDecision(
                intent="stop",
                events=events,
                should_emit_full_transcript=True,
                should_close=True,
                dropped_buffered_seconds=dropped,
            )

        return self._invalid_control(f"Unsupported control action: {action or 'unknown'}")

    def _invalid_control(self, message: str) -> FrameDecision:
        return FrameDecision(
            intent="invalid_control",
            error={"type": "error", "error_type": "invalid_control", "message": message},
        )

    def _status(self, state: str) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "status", "state": state}
        if self.negotiated_v2:
            payload["protocol_version"] = 2
        return payload

    def _reset_runtime_state(self) -> None:
        self._clear_paused_audio_queue()
        self._last_overflow_warning_at = None

    def _clear_paused_audio_queue(self) -> float:
        dropped = self._paused_audio_seconds
        self._paused_audio_segments.clear()
        self._paused_audio_seconds = 0.0
        return dropped

    def _should_emit_overflow_warning(self, now: float) -> bool:
        interval = max(0.0, float(self.config.overflow_warning_interval_seconds))
        if self._last_overflow_warning_at is None:
            return True
        return (float(now) - self._last_overflow_warning_at) >= interval


def _normalize_protocol_version(value: Any) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError):
        return 1
    return 2 if version == 2 else 1
