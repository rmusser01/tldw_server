import asyncio
import base64
import json
import numpy as np
import pytest


class DummyWebSocket:
    def __init__(self, messages):
        """
        Initialize the DummyWebSocket with a sequence of incoming messages.

        Parameters:
            messages (iterable): Iterable of inbound messages to be consumed by receive_text; a shallow copy is stored so the original iterable is not mutated.

        Initial state:
            _messages: list copy of provided messages (acts as the input queue).
            _out: empty list collecting outbound messages sent via send_json.
            accepted: False until accept() is called.
            closed: False until close() is called.
        """
        self._messages = list(messages)
        self._out = []
        self.accepted = False
        self.closed = False

    async def accept(self):
        """
        Mark the dummy WebSocket connection as accepted.

        Sets the internal `accepted` flag to True so the DummyWebSocket is treated as an accepted connection by tests.
        """
        self.accepted = True

    async def receive_text(self):
        """
        Get the next queued incoming text message.

        Returns:
            str: The next message string from the internal queue.

        Raises:
            RuntimeError: If no messages remain.
        """
        if not self._messages:
            raise RuntimeError("No more messages")
        return self._messages.pop(0)

    async def send_json(self, data):
        """
        Record a JSON-serializable message on the mock websocket's outbound queue.

        Parameters:
            data: The message object to send (will be appended to the websocket's outbound list for later inspection).
        """
        self._out.append(data)

    async def close(self, code=None, reason=None):
        """
        Mark the in-memory websocket as closed.

        This sets the `closed` flag to True. The optional `code` and `reason` parameters are accepted for API compatibility but are ignored.
        """
        self.closed = True

    # Helpers
    @property
    def outputs(self):
        """
        Get the list of outbound messages recorded by the dummy WebSocket.

        Returns:
            list: The internal list of messages previously sent via send_json (mutable, in insertion order).
        """
        return self._out


@pytest.mark.asyncio
async def test_core_ws_with_diarization_and_insights(monkeypatch):
    # Import the core WS handler
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.ws_server as wsmod

    # Patch in fake diarizer and insights classes if not available
    class _FakeSettings:
        def __init__(self, enabled=True):
            """
            Create a fake settings object indicating whether insights are enabled.

            Parameters:
                enabled (bool): Whether insights are enabled for this settings instance.
            """
            self.enabled = enabled

        @classmethod
        def from_client_payload(cls, payload):
            """
            Create a settings instance from a client-supplied payload.

            Parameters:
                payload (dict): Client configuration dictionary. If it contains the key `"enabled"`,
                    its truthiness determines the instance's enabled state; if absent, enabled defaults to True.

            Returns:
                cls: An instance of the settings class with `enabled` set according to `payload`.
            """
            enabled = bool(payload.get("enabled", True))
            return cls(enabled=enabled)

    class _FakeInsights:
        def __init__(self, websocket, settings, **kwargs):
            """
            Initialize the fake insights helper used by tests.

            Parameters:
                websocket: The websocket-like object the helper will reference for sending or receiving messages.
                settings: Settings object that controls insights behavior (e.g., whether insights are enabled).
                **kwargs: Additional ignored keyword arguments.

            The instance records observed transcript segments and commit events in the internal list `_seen`.
            """
            self.websocket = websocket
            self.settings = settings
            self._seen = []

        def describe(self):

            """
            Report whether insights are enabled.

            Returns:
                dict: A dictionary containing the key "enabled" with a boolean value indicating if insights are enabled.
            """
            return {"enabled": self.settings.enabled}

        async def on_transcript(self, segment):
            """
            Record an incoming transcript segment for later inspection.

            Parameters:
                segment (dict): Transcript segment received from the transcription pipeline; appended to the instance's internal `_seen` list.
            """
            self._seen.append(segment)

        async def on_commit(self, full_text):
            """
            Record a committed transcript fragment into the internal seen list.

            Parameters:
                full_text (str): The finalized transcript text to record.
            """
            self._seen.append({"commit": full_text})

        async def reset(self):
            """
            Reset the insights instance's recorded observations.

            Clears any stored transcript segments and commit records tracked by this instance.
            """
            self._seen.clear()

        async def close(self):
            """
            Mark the dummy WebSocket connection as closed.

            Sets the instance's closed flag to True to indicate the connection has been closed.
            """
            pass

    class _FakeDiarizer:
        def __init__(self, sample_rate, store_audio=False, storage_dir=None, num_speakers=None):
            """
            Create a test diarizer configured with the given audio parameters.

            Parameters:
                sample_rate (int): Sample rate in Hz used by the diarizer.
                store_audio (bool): If True, indicates raw audio should be stored (test stub only).
                storage_dir (str | None): Path where audio would be stored if enabled.
                num_speakers (int | None): Optional expected number of speakers.

            Notes:
                Initializes an empty internal map for storing labeled segment metadata.
            """
            self.sample_rate = sample_rate
            self.map = {}

        async def ensure_ready(self):
            """
            Ensure the diarizer is ready to process audio.

            For this test implementation, always reports readiness.

            Returns:
                True indicating the diarizer is ready.
            """
            return True

        async def label_segment(self, audio_np, meta):
            """
            Assigns a speaker label to an audio segment and stores the result in the diarizer's internal map.

            Parameters:
                audio_np (numpy.ndarray): Audio samples for the segment (not inspected by this fake implementation).
                meta (Mapping): Metadata for the segment; the value of `"segment_id"` (converted to int) is used to index the stored label. If `"segment_id"` is missing, `0` is used.

            Returns:
                dict: A mapping containing `speaker_id` and `speaker_label` for the segment (also stored at `self.map[segment_id]`).
            """
            seg_id = int(meta.get("segment_id", 0))
            info = {"speaker_id": 1, "speaker_label": "SPEAKER_TEST"}
            self.map[seg_id] = info
            return info

        async def finalize(self):
            """
            Return the diarization mapping and placeholders for additional results.

            Returns:
                tuple: A 3-tuple (map, None, None) where `map` is a dictionary mapping segment IDs to speaker info objects, and the second and third elements are placeholders that are always `None`.
            """
            return self.map, None, None

        async def reset(self):
            """
            Reset the diarizer's internal state by removing all stored segment labels.

            This clears the internal `map` used to store labeled segment information so the instance
            behaves as if no segments have been processed.
            """
            self.map.clear()

        async def close(self):
            """
            Mark the dummy WebSocket connection as closed.

            Sets the instance's closed flag to True to indicate the connection has been closed.
            """
            pass

    monkeypatch.setattr(wsmod, "LiveInsightSettings", _FakeSettings, raising=False)
    monkeypatch.setattr(wsmod, "LiveMeetingInsights", _FakeInsights, raising=False)
    monkeypatch.setattr(wsmod, "StreamingDiarizer", _FakeDiarizer, raising=False)

    # Build messages: config (enable diarization/insights), audio, commit, stop
    cfg = {
        "type": "config",
        "model": "parakeet",
        "sample_rate": 10,
        "chunk_duration": 1.0,
        "enable_partial": False,
        "diarize": True,
        "insights": {"enabled": True},
    }
    audio = (np.ones(11, dtype=np.float32)).tobytes()  # 1.1s at 10 Hz
    messages = [
        json.dumps(cfg),
        json.dumps({"type": "audio", "data": base64.b64encode(audio).decode("ascii")}),
        json.dumps({"type": "commit"}),
        json.dumps({"type": "stop"}),
    ]
    ws = DummyWebSocket(messages)

    # Provide a simple decode function
    async def _run():
        """
        Execute the Parakeet core websocket handler using a stub decode function that returns the fixed transcript "hello".

        This coroutine invokes wsmod.websocket_parakeet_core with the test DummyWebSocket and a minimal decode callback to drive the handler through config, audio, commit, and stop events.
        """
        def _decode(audio_np, sr):
            return "hello"

        await wsmod.websocket_parakeet_core(ws, decode_fn=_decode)

    await _run()

    outs = ws.outputs
    # Expect configuration status
    assert any(o.get("type") == "status" and o.get("state") == "configured" for o in outs)
    # Expect insights enabled status
    assert any(o.get("type") == "status" and o.get("state") in {"insights_enabled", "insights_disabled"} for o in outs)
    # Expect a final frame with speaker info
    finals = [o for o in outs if o.get("type") == "final"]
    assert finals, f"No final frames: {outs}"
    assert any("speaker_id" in f and "speaker_label" in f for f in finals)
    # Expect full transcript
    assert any(o.get("type") == "full_transcript" for o in outs)
    # Expect diarization summary
    assert any(o.get("type") == "diarization_summary" for o in outs)


@pytest.mark.asyncio
async def test_core_ws_finally_flushes_and_closes_transcriber(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.ws_server as wsmod

    calls = []

    class _FakeTranscriber:
        def __init__(self, config, decode_fn=None):
            self.decode_fn = decode_fn or (lambda _a, _b: "ok")

        async def process_audio_chunk(self, audio):
            return None

        async def flush(self):
            calls.append("flush")
            return None

        def close(self):
            calls.append("close")

        def reset(self):
            calls.append("reset")

        def get_full_transcript(self):
            return ""

    monkeypatch.setattr(wsmod, "ParakeetCoreTranscriber", _FakeTranscriber, raising=True)

    ws = DummyWebSocket([json.dumps({"type": "stop"})])
    await wsmod.websocket_parakeet_core(ws, decode_fn=lambda _a, _b: "ok")

    assert "flush" in calls
    assert "close" in calls
    assert calls.index("flush") < calls.index("close")


@pytest.mark.asyncio
async def test_core_ws_model_reconfigure_closes_previous_transcriber(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.ws_server as wsmod

    closes = []

    class _FakeTranscriber:
        _next_id = 0

        def __init__(self, config, decode_fn=None):
            type(self)._next_id += 1
            self.transcriber_id = type(self)._next_id
            self.decode_fn = decode_fn or (lambda _a, _b: "ok")

        async def process_audio_chunk(self, audio):
            return None

        async def flush(self):
            return None

        def close(self):
            closes.append(self.transcriber_id)

        def reset(self):
            return None

        def get_full_transcript(self):
            return ""

    monkeypatch.setattr(wsmod, "ParakeetCoreTranscriber", _FakeTranscriber, raising=True)

    ws = DummyWebSocket(
        [
            json.dumps({"type": "config", "model": "parakeet", "model_variant": "standard"}),
            json.dumps({"type": "config", "model": "parakeet", "model_variant": "onnx"}),
            json.dumps({"type": "stop"}),
        ]
    )
    await wsmod.websocket_parakeet_core(ws, decode_fn=lambda _a, _b: "ok")

    # Old transcriber should be closed during model swap, and active one on teardown.
    assert len(closes) >= 2


@pytest.mark.asyncio
async def test_core_ws_outer_error_sanitizes_internal_message(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.ws_server as wsmod

    class _FakeTranscriber:
        def __init__(self, config, decode_fn=None):
            self.decode_fn = decode_fn

        async def flush(self):
            return None

        def close(self):
            return None

    monkeypatch.setattr(wsmod, "ParakeetCoreTranscriber", _FakeTranscriber, raising=True)

    ws = DummyWebSocket([])
    await wsmod.websocket_parakeet_core(ws, decode_fn=lambda _a, _b: "ok")

    errors = [out for out in ws.outputs if out.get("type") == "error"]
    assert errors
    assert errors[-1]["message"] == "Parakeet core websocket failed"
    assert "No more messages" not in str(errors)
    assert ws.closed is True
