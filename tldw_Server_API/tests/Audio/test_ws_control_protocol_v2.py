from __future__ import annotations

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ws_control_protocol import (
    WSControlProtocolConfig,
    WSControlSession,
)


def _session(*, v2_enabled: bool = True) -> WSControlSession:
    return WSControlSession(
        WSControlProtocolConfig(
            ws_control_v2_enabled=v2_enabled,
            paused_audio_queue_cap_seconds=2.0,
            overflow_warning_interval_seconds=5.0,
        )
    )


def test_absent_protocol_version_defaults_to_v1() -> None:
    session = _session(v2_enabled=True)

    decision = session.apply_config({"type": "config"})

    assert decision.protocol_version == 1
    assert decision.negotiated_v2 is False
    assert decision.events == []
    assert session.state == "running"


def test_protocol_version_2_negotiates_only_when_enabled() -> None:
    disabled = _session(v2_enabled=False)

    disabled_decision = disabled.apply_config({"type": "config", "protocol_version": 2})

    assert disabled_decision.protocol_version == 1
    assert disabled_decision.negotiated_v2 is False
    assert disabled_decision.error is None
    assert disabled_decision.events == []

    enabled = _session(v2_enabled=True)

    enabled_decision = enabled.apply_config({"type": "config", "protocol_version": 2})

    assert enabled_decision.protocol_version == 2
    assert enabled_decision.negotiated_v2 is True
    assert enabled_decision.error is None
    assert enabled_decision.events == [
        {"type": "status", "state": "configured", "protocol_version": 2}
    ]
    assert enabled.state == "running"


def test_control_frames_are_rejected_when_v2_not_negotiated() -> None:
    session = _session(v2_enabled=False)
    session.apply_config({"type": "config", "protocol_version": 2})

    decision = session.handle_frame({"type": "control", "action": "pause"})

    assert decision.intent == "invalid_control"
    assert decision.error == {
        "type": "error",
        "error_type": "invalid_control",
        "message": "Control frames require protocol_version=2",
    }
    assert decision.events == []
    assert session.state == "running"


def test_valid_control_actions_pause_resume_commit_stop() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})

    pause = session.handle_frame({"type": "control", "action": "pause"})
    resume = session.handle_frame({"type": "control", "action": "resume"})
    commit = session.handle_frame({"type": "control", "action": "commit"})
    stop = session.handle_frame({"type": "control", "action": "stop"})

    assert pause.intent == "pause"
    assert pause.events == [{"type": "status", "state": "paused", "protocol_version": 2}]
    assert resume.intent == "resume"
    assert resume.events == [{"type": "status", "state": "resumed", "protocol_version": 2}]
    assert commit.intent == "commit"
    assert commit.should_emit_full_transcript is True
    assert commit.events == []
    assert stop.intent == "stop"
    assert stop.should_emit_full_transcript is True
    assert stop.should_close is True
    assert stop.events == [{"type": "status", "state": "closing", "protocol_version": 2}]
    assert session.state == "closing"


def test_pause_resume_are_idempotent() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})

    first_pause = session.handle_frame({"type": "control", "action": "pause"})
    second_pause = session.handle_frame({"type": "control", "action": "pause"})
    first_resume = session.handle_frame({"type": "control", "action": "resume"})
    second_resume = session.handle_frame({"type": "control", "action": "resume"})

    assert first_pause.events == [{"type": "status", "state": "paused", "protocol_version": 2}]
    assert second_pause.events == [{"type": "status", "state": "paused", "protocol_version": 2}]
    assert first_resume.events == [{"type": "status", "state": "resumed", "protocol_version": 2}]
    assert second_resume.events == [{"type": "status", "state": "resumed", "protocol_version": 2}]
    assert session.state == "running"


def test_legacy_top_level_reset_remains_valid_in_running_and_paused_states() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})

    running_reset = session.handle_frame({"type": "reset"})
    session.handle_frame({"type": "control", "action": "pause"})
    paused_reset = session.handle_frame({"type": "reset"})

    assert running_reset.intent == "reset"
    assert running_reset.should_reset is True
    assert running_reset.events == [{"type": "status", "state": "reset"}]
    assert paused_reset.intent == "reset"
    assert paused_reset.should_reset is True
    assert paused_reset.events == [{"type": "status", "state": "reset"}]
    assert session.state == "running"


def test_paused_queue_cap_accounting_with_drop_oldest_overflow() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})
    session.handle_frame({"type": "control", "action": "pause"})

    first = session.buffer_paused_audio(1.25, now=10.0)
    second = session.buffer_paused_audio(1.5, now=11.0)

    assert first.accepted_seconds == 1.25
    assert first.dropped_seconds == 0.0
    assert first.buffered_seconds == 1.25
    assert first.events == []

    assert second.accepted_seconds == 1.5
    assert second.dropped_seconds == 0.75
    assert second.buffered_seconds == 2.0
    assert second.events == [
        {
            "type": "warning",
            "warning_type": "audio_dropped_during_pause",
            "message": "Paused audio queue exceeded 2.0s; dropped 0.75s using drop_oldest policy",
            "dropped_seconds": 0.75,
            "buffered_seconds": 2.0,
            "policy": "drop_oldest",
        }
    ]


def test_overflow_warnings_are_rate_limited_by_configured_interval() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})
    session.handle_frame({"type": "control", "action": "pause"})
    session.buffer_paused_audio(2.0, now=10.0)

    first_overflow = session.buffer_paused_audio(1.0, now=12.0)
    second_overflow = session.buffer_paused_audio(1.0, now=14.0)
    third_overflow = session.buffer_paused_audio(1.0, now=17.1)

    assert len(first_overflow.events) == 1
    assert first_overflow.dropped_seconds == 1.0
    assert second_overflow.events == []
    assert second_overflow.dropped_seconds == 1.0
    assert len(third_overflow.events) == 1
    assert third_overflow.dropped_seconds == 1.0


def test_legacy_frames_are_rejected_before_config() -> None:
    session = _session(v2_enabled=True)

    for frame_type in ("commit", "reset", "stop"):
        decision = session.handle_frame({"type": frame_type})
        assert decision.intent == "invalid_control"
        assert decision.error == {
            "type": "error",
            "error_type": "invalid_control",
            "message": f"Frame type {frame_type} not allowed in state awaiting_config",
        }
        assert session.state == "awaiting_config"


def test_control_frames_are_rejected_once_session_is_closing() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})
    stop = session.handle_frame({"type": "control", "action": "stop"})

    assert stop.intent == "stop"
    assert session.state == "closing"

    decision = session.handle_frame({"type": "control", "action": "pause"})

    assert decision.intent == "invalid_control"
    assert decision.error == {
        "type": "error",
        "error_type": "invalid_control",
        "message": "Control action pause not allowed in state closing",
    }
    assert session.state == "closing"


def test_buffer_paused_audio_is_noop_outside_paused_state() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})

    running = session.buffer_paused_audio(1.0, now=10.0)
    assert running.accepted_seconds == 0.0
    assert running.dropped_seconds == 0.0
    assert running.buffered_seconds == 0.0
    assert running.events == []

    session.handle_frame({"type": "control", "action": "pause"})
    session.buffer_paused_audio(1.5, now=11.0)
    session.handle_frame({"type": "control", "action": "resume"})

    resumed = session.buffer_paused_audio(1.0, now=12.0)
    assert resumed.accepted_seconds == 0.0
    assert resumed.dropped_seconds == 0.0
    assert resumed.buffered_seconds == 1.5
    assert resumed.events == []


def test_release_paused_audio_clears_accounting_before_later_stop() -> None:
    session = _session(v2_enabled=True)
    session.apply_config({"type": "config", "protocol_version": 2})
    session.handle_frame({"type": "control", "action": "pause"})
    session.buffer_paused_audio(1.5, now=10.0)

    resume = session.handle_frame({"type": "control", "action": "resume"})
    assert resume.queued_audio_seconds == 1.5

    released_seconds = session.release_paused_audio()
    assert released_seconds == 1.5

    stop = session.handle_frame({"type": "control", "action": "stop"})
    assert stop.dropped_buffered_seconds == 0.0
