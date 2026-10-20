"""A cancelled request keeps a local TTS engine until its thread call returns (TASK-13856).

asyncio.to_thread cannot stop a thread, so an adapter that guards its engine with a lock must not
hand the engine to the next request while a cancelled request's call still runs. Starlette cancels
a disconnected request through an anyio cancel scope, which re-cancels on every await, so each
site is checked with a plain task cancel and with an anyio scope.
"""

from __future__ import annotations

import asyncio
import platform
import sys
import threading
import types
from collections.abc import Awaitable, Callable

import anyio
import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters import supertonic2_adapter, supertonic_adapter
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest

pytestmark = pytest.mark.unit

CANCEL_MODES = pytest.mark.parametrize("via_anyio_scope", [False, True], ids=["task-cancel", "anyio-scope"])


class _Gate:
    """Engine calls record that they started, then block until released, so a caller can be cancelled mid-call."""

    def __init__(self) -> None:
        self.started: list[str] = []
        self.release = threading.Event()

    def enter(self, name: str) -> None:
        self.started.append(name)
        if not self.release.wait(5):
            raise RuntimeError("the test never released the engine call")


async def _assert_cancelled_call_keeps_the_engine(
    call: Callable[[], Awaitable[object]],
    gate: _Gate,
    *,
    via_anyio_scope: bool,
) -> None:
    """Cancel one call mid-engine-call while a second waits; the second must not start until the first returns."""
    scopes: list[anyio.CancelScope] = []

    async def _first() -> None:
        if not via_anyio_scope:
            await call()
            return
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await call()

    first = asyncio.create_task(_first())
    try:
        for _ in range(500):
            if gate.started:
                break
            await asyncio.sleep(0.01)
        assert gate.started, "the first call never reached the engine"
        second = asyncio.create_task(call())
        await asyncio.sleep(0.05)
        if via_anyio_scope:
            scopes[0].cancel()
        else:
            first.cancel()
        await asyncio.sleep(0.1)
        # The cancelled request's engine call is still running, so the next one has not started.
        assert len(gate.started) == 1
    finally:
        gate.release.set()
    # Bounded waits: a lock-leak regression must fail here, not hang the run until pytest-timeout kills it (Review M1).
    await asyncio.wait_for(second, 5)
    assert len(gate.started) == 2
    if via_anyio_scope:
        await asyncio.wait_for(first, 5)
    else:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 5)


_WAV_HEADER = b"RIFF\x24\x00\x00\x00WAVEfmt "


def _pocket_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter

    class _Engine:
        def generate(self, text, voice, max_frames=500):
            gate.enter("generate")
            return np.zeros(240, dtype=np.float32)

        def stream(self, text, voice, **_kwargs):
            gate.enter("stream")
            yield np.zeros(240, dtype=np.float32)

    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = _Engine()
    return adapter


def _pocket_request(*, stream: bool) -> TTSRequest:
    return TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=stream,
        voice_reference=_WAV_HEADER,
        extra_params={"validate_reference": False, "convert_reference": False},
    )


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_generate_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        return await adapter.generate(_pocket_request(stream=False))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_pocket_tts_stream_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _pocket_adapter(gate)

    async def call():
        response = await adapter.generate(_pocket_request(stream=True))
        return [chunk async for chunk in response.audio_stream]

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _luxtts_adapter(gate: _Gate):
    from tldw_Server_API.app.core.TTS.adapters.luxtts_adapter import LuxTTSAdapter

    class _Engine:
        def encode_prompt(self, prompt_audio, duration=5, rms=0.001):
            return {"prompt_tokens": [1]}

        def generate_speech(self, text, encode_dict, **_kwargs):
            # The lock spans encode_prompt and generate_speech, so this is the holder's last call.
            gate.enter("generate_speech")
            return np.zeros((1, 3200), dtype=np.float32)

    adapter = LuxTTSAdapter({"validate_reference": False, "convert_reference": False})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = _Engine()
    return adapter


@pytest.mark.asyncio
@CANCEL_MODES
async def test_luxtts_keeps_the_engine_until_its_call_returns(via_anyio_scope):
    gate = _Gate()
    adapter = _luxtts_adapter(gate)

    async def call():
        return await adapter.generate(
            TTSRequest(text="hello", format=AudioFormat.PCM, voice_reference=_WAV_HEADER, stream=False)
        )

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _supertonic_adapter(monkeypatch, tmp_path, gate: _Gate, *, version: int):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "M1.json").write_text("{}")

    def _engine(*_args):
        gate.enter("synthesize")
        return np.zeros((1, 2400), dtype=np.float32), np.array([0.1])

    vendor = types.SimpleNamespace(
        load_text_to_speech=lambda _onnx_dir, _use_gpu=False: _engine,
        load_voice_style=lambda _paths, _verbose=False: {"style": "dummy"},
    )
    if version == 1:
        adapter_cls = supertonic_adapter.SupertonicOnnxAdapter
        module_name, voice = "tldw_Server_API.app.core.TTS.vendors.supertonic", "supertonic_m1"
    else:
        adapter_cls = supertonic2_adapter.Supertonic2OnnxAdapter
        module_name, voice = "tldw_Server_API.app.core.TTS.vendors.supertonic2", "supertonic2_m1"
    monkeypatch.setitem(sys.modules, module_name, vendor)
    adapter = adapter_cls(
        {
            "model_path": str(onnx_dir),
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": voice,
                "voice_files": {voice: "M1.json"},
            },
        }
    )
    return adapter, voice


@pytest.mark.asyncio
@CANCEL_MODES
@pytest.mark.parametrize("version", [1, 2], ids=["supertonic", "supertonic2"])
async def test_supertonic_keeps_the_engine_until_its_call_returns(monkeypatch, tmp_path, version, via_anyio_scope):
    gate = _Gate()
    adapter, voice = _supertonic_adapter(monkeypatch, tmp_path, gate, version=version)

    async def call():
        # WAV, not PCM: Supertonic accepts only MP3 and WAV, and MP3 needs PyAV.
        return await adapter.generate(TTSRequest(text="hello", voice=voice, format=AudioFormat.WAV, stream=False))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _omnivoice_runtime(monkeypatch, tmp_path, gate: _Gate, *, gate_load: bool):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_runtime import OmniVoiceRuntime

    class _OmniVoice:
        @classmethod
        def from_pretrained(cls, _model_path, **_kwargs):
            if gate_load:
                gate.enter("load")
            return cls()

        def generate(self, **_kwargs):
            if not gate_load:
                gate.enter("generate")
            return np.zeros(2400, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "omnivoice", types.SimpleNamespace(OmniVoice=_OmniVoice))
    (tmp_path / "model").mkdir()
    return OmniVoiceRuntime({"model_path": str(tmp_path / "model")})


@pytest.mark.asyncio
@CANCEL_MODES
async def test_omnivoice_runtime_synthesis_keeps_the_model_until_its_call_returns(monkeypatch, tmp_path, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.omnivoice_sidecar_protocol import OmniVoiceSynthesizeRequest

    gate = _Gate()
    runtime = _omnivoice_runtime(monkeypatch, tmp_path, gate, gate_load=False)

    async def call():
        return await runtime.synthesize(OmniVoiceSynthesizeRequest(text="hello", mode="auto", language_id="en"))

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _vibevoice_adapter(monkeypatch, gate: _Gate):
    import contextlib

    from tldw_Server_API.app.core.TTS.adapters import vibevoice_adapter as vibevoice_mod

    class _Tensor:
        def cpu(self):
            return self

        def numpy(self):
            return np.zeros(2400, dtype=np.float32)

    class _Model:
        def generate(self, **_kwargs):
            gate.enter("generate")
            return types.SimpleNamespace(speech_outputs=[_Tensor()])

    class _Processor:
        tokenizer = object()

        def __call__(self, **_kwargs):
            return {}

    fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext, is_tensor=lambda _value: False)
    monkeypatch.setattr(vibevoice_mod, "_get_torch", lambda **_kwargs: fake_torch)
    adapter = vibevoice_mod.VibeVoiceAdapter({"device": "cpu"})
    adapter.model = _Model()
    adapter.processor = _Processor()
    return adapter


@pytest.mark.asyncio
@CANCEL_MODES
async def test_vibevoice_keeps_the_model_until_its_call_returns(monkeypatch, via_anyio_scope):
    """A cancelled generate must keep a second generate from running on the model until the first thread returns.

    The variant reload path (`_ensure_model_variant`) is not exercised here; it is covered only
    because it shares `_model_state_lock` with generate.
    """
    gate = _Gate()
    adapter = _vibevoice_adapter(monkeypatch, gate)

    async def call():
        request = TTSRequest(text="Speaker 1: hello", format=AudioFormat.PCM, stream=True)
        return [chunk async for chunk in adapter._stream_audio_vibevoice(request, "speaker_1", 1)]

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_omnivoice_runtime_load_keeps_the_lock_until_the_load_returns(monkeypatch, tmp_path, via_anyio_scope):
    gate = _Gate()
    runtime = _omnivoice_runtime(monkeypatch, tmp_path, gate, gate_load=True)

    await _assert_cancelled_call_keeps_the_engine(runtime.load, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_mlx_model_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

    gate = _Gate()

    def load_model(_model_id):
        gate.enter("load_model")
        return object()

    utils_module = types.ModuleType("mlx_audio.tts.utils")
    utils_module.load_model = load_model
    monkeypatch.setitem(sys.modules, "mlx_audio", types.ModuleType("mlx_audio"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", types.ModuleType("mlx_audio.tts"))
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", utils_module)
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    adapter = Qwen3TTSAdapter({"runtime": "mlx", "device": "mps"})
    await adapter.ensure_initialized()
    runtime = adapter._get_runtime()

    async def call():
        return await runtime._get_model("auto")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_qwen3_pipeline_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter

    gate = _Gate()

    def build_pipeline(_model_id):
        gate.enter("build_pipeline")
        return object()

    adapter = Qwen3TTSAdapter({})
    adapter._pipeline_builders = [("fake", build_pipeline)]

    async def call():
        return await adapter._get_pipeline("Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_echo_tts_model_load_keeps_the_lock_until_the_load_returns(via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters.echo_tts_adapter import EchoTTSAdapter

    gate = _Gate()

    def load_model_from_hf(**_kwargs):
        gate.enter("load_model_from_hf")
        return object()

    adapter = EchoTTSAdapter({})
    adapter.device = "cuda"  # initialize() resolves it; this test skips initialize()
    adapter._echo_modules_loaded = True
    adapter._echo_inference = types.SimpleNamespace(
        load_model_from_hf=load_model_from_hf,
        load_fish_ae_from_hf=lambda **_kwargs: object(),
        load_pca_state_from_hf=lambda **_kwargs: object(),
    )
    adapter._import_torch = lambda: types.SimpleNamespace(bfloat16="bf16", float16="f16", float32="f32")

    await _assert_cancelled_call_keeps_the_engine(adapter._ensure_models_loaded, gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@CANCEL_MODES
async def test_kitten_tts_runtime_load_keeps_the_lock_until_the_load_returns(monkeypatch, via_anyio_scope):
    from tldw_Server_API.app.core.TTS.adapters import kitten_tts_adapter as kitten_mod

    gate = _Gate()

    def kitten_runtime(assets):
        gate.enter("KittenRuntime")
        return types.SimpleNamespace(sample_rate=24000)

    monkeypatch.setattr(kitten_mod, "resolve_model_revision", lambda _model, revision: revision)
    monkeypatch.setattr(kitten_mod, "download_model_assets", lambda model, **_kwargs: types.SimpleNamespace(repo_id=model))
    monkeypatch.setattr(kitten_mod, "KittenRuntime", kitten_runtime)
    adapter = kitten_mod.KittenTTSAdapter({})

    async def call():
        return await adapter._load_runtime_for_model("KittenML/kitten-tts-nano-0.8")

    await _assert_cancelled_call_keeps_the_engine(call, gate, via_anyio_scope=via_anyio_scope)


def _kokoro_adapter(monkeypatch, tmp_path, gate: _Gate, *, backend: str, init_timeout_sec: float | None = None):
    """A Kokoro adapter whose model load (the ONNX constructor or KModel) blocks on the gate."""
    from tldw_Server_API.app.core.TTS.adapters import kokoro_adapter as kokoro_mod

    async def _resource_manager():
        return types.SimpleNamespace(register_model=lambda **_kwargs: None)

    monkeypatch.setattr(kokoro_mod, "get_resource_manager", _resource_manager)
    # The adapter falls back to this env var; a shell that sets it would bound the "unbounded" cases.
    monkeypatch.delenv("KOKORO_INIT_TIMEOUT_SEC", raising=False)
    config: dict[str, object] = {"kokoro_device": "cpu", "kokoro_lazy_init": False}
    if init_timeout_sec is not None:
        config["kokoro_init_timeout_sec"] = init_timeout_sec
    if backend == "onnx":

        class _Kokoro:
            def __init__(self, *_args, **_kwargs):
                gate.enter("load")

        monkeypatch.setitem(
            sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=_Kokoro, EspeakConfig=lambda **_kwargs: None)
        )
        model, voices = tmp_path / "model.onnx", tmp_path / "voices.bin"
        model.touch()
        voices.touch()
        config.update(kokoro_use_onnx=True, kokoro_model_path=str(model), kokoro_voices_json=str(voices))
    else:

        class _KModel:
            def __init__(self, **_kwargs):
                gate.enter("load")

            def eval(self):
                return self

            def cpu(self):
                return self

        model_module = types.ModuleType("kokoro.model")
        model_module.KModel = _KModel
        monkeypatch.setitem(sys.modules, "kokoro", types.ModuleType("kokoro"))
        monkeypatch.setitem(sys.modules, "kokoro.model", model_module)
        monkeypatch.setattr(kokoro_mod, "safe_import_torch", lambda: types.SimpleNamespace())
        model = tmp_path / "kokoro-v1_0.pth"
        model.touch()
        (tmp_path / "config.json").write_text("{}")
        config.update(kokoro_use_onnx=False, kokoro_model_path=str(model))
    return kokoro_mod.KokoroAdapter(config)


# ensure_initialized() is the production entry (the registry and generate()); initialize() is what it runs.
_KOKORO_ENTRIES = pytest.mark.parametrize("entry", ["ensure_initialized", "initialize", "_ensure_model_loaded"])


@pytest.mark.asyncio
@CANCEL_MODES
@_KOKORO_ENTRIES
@pytest.mark.parametrize(
    ("backend", "init_timeout_sec"),
    [("onnx", None), ("pytorch", None), ("pytorch", 30.0)],
    ids=["onnx", "pytorch-unbounded", "pytorch-bounded"],
)
async def test_kokoro_model_load_keeps_the_lock_until_the_load_returns(
    monkeypatch, tmp_path, backend, init_timeout_sec, entry, via_anyio_scope
):
    """A cancelled Kokoro load keeps the next load waiting for its thread.

    "pytorch-bounded" cancels the caller while it waits inside init_timeout_sec.
    """
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend=backend, init_timeout_sec=init_timeout_sec)

    await _assert_cancelled_call_keeps_the_engine(getattr(adapter, entry), gate, via_anyio_scope=via_anyio_scope)


@pytest.mark.asyncio
@_KOKORO_ENTRIES
async def test_kokoro_load_past_init_timeout_keeps_the_lock_until_the_load_returns(monkeypatch, tmp_path, entry):
    """init_timeout_sec bounds each caller's wait; the lock stays with the load until its thread returns."""
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="pytorch", init_timeout_sec=0.1)
    call = getattr(adapter, entry)
    try:
        # The first load outlasts the timeout; its thread keeps running.
        assert await asyncio.wait_for(call(), 5) is False
        # The next caller waits behind that thread, up to its own timeout, and starts no load.
        assert await asyncio.wait_for(call(), 5) is False
        assert gate.started == ["load"]
    finally:
        gate.release.set()
    # Only the stuck load had to outlast the bound; give the reload room on a slow runner.
    adapter.init_timeout_sec = 5.0
    # The timed-out load's model was discarded, so once its thread returns the next caller loads again.
    assert await asyncio.wait_for(call(), 5) is True
    assert gate.started == ["load", "load"]


@pytest.mark.asyncio
async def test_kokoro_onnx_load_ignores_init_timeout_sec(monkeypatch, tmp_path):
    """init_timeout_sec is PyTorch-only: an ONNX load that outlasts it still completes for its caller."""
    gate = _Gate()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="onnx", init_timeout_sec=0.1)
    load = asyncio.create_task(adapter._ensure_model_loaded())
    try:
        await asyncio.sleep(0.3)  # three times the bound
        assert not load.done()
    finally:
        gate.release.set()
    assert await asyncio.wait_for(load, 5) is True


@pytest.mark.asyncio
async def test_kokoro_load_timeout_error_is_not_reported_as_the_bound(monkeypatch, tmp_path):
    """A TimeoutError raised by the load itself (a hub download, say) propagates; only the bound's expiry returns False."""
    adapter = _kokoro_adapter(monkeypatch, tmp_path, _Gate(), backend="pytorch", init_timeout_sec=30.0)

    class _TimingOutKModel:
        def __init__(self, **_kwargs):
            raise TimeoutError("hub download timed out")

    monkeypatch.setattr(sys.modules["kokoro.model"], "KModel", _TimingOutKModel)
    with pytest.raises(TimeoutError, match="hub download"):
        await asyncio.wait_for(adapter._ensure_model_loaded(), 5)


@pytest.mark.asyncio
async def test_kokoro_close_releases_the_pytorch_model(monkeypatch, tmp_path):
    """close() drops the PyTorch model and pipelines, so the next ensure_initialized() loads the model again."""
    gate = _Gate()
    gate.release.set()
    adapter = _kokoro_adapter(monkeypatch, tmp_path, gate, backend="pytorch")
    assert await adapter.ensure_initialized()
    adapter.kokoro_pt_pipelines["a"] = object()

    await adapter.close()

    assert (adapter.kokoro_pt_model, adapter.kokoro_pt_pipelines) == (None, {})
    assert await adapter.ensure_initialized()
    assert gate.started == ["load", "load"]
