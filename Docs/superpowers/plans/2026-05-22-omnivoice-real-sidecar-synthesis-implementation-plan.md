# OmniVoice Real Sidecar Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the existing OmniVoice sidecar silent-WAV stub with a real, managed OmniVoice Python API runtime while preserving the current TTS registry boundary.

**Architecture:** Keep the main API process responsible for public TTS semantics, stored voice resolution, validation, fallback, and output conversion. Add a focused sidecar runtime module that imports OmniVoice, loads one local model directory in no-download mode, runs auto/design/clone synthesis behind a lock, and returns native WAV/PCM plus structured metadata/errors over the existing authenticated loopback sidecar API.

**Tech Stack:** FastAPI, Pydantic v2, httpx, asyncio subprocess supervision, pytest, soundfile/wave, OmniVoice Python API, existing `TTSServiceV2`, `OmniVoiceAdapter`, and Backlog.md task tracking.

---

## Spec And Task Context

- Approved spec: `Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md`
- Planning Backlog task: `TASK-454`
- Design Backlog task: `TASK-453`
- Current implementation gap: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py` returns a generated silent WAV instead of real OmniVoice output.

This plan intentionally does not change the default TTS provider, does not enable OmniVoice by default, and does not add runtime downloads.

## File Structure And Responsibilities

- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_protocol.py`
  - Own the sidecar request/response/error/status schemas and internal auth header helper.
- Create: `tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py`
  - Own sidecar-local OmniVoice model loading, no-download checks, request mapping, serialization, runtime status, and structured runtime errors.
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py`
  - Own HTTP endpoints, token auth, runner injection for tests, route-level error mapping, and control/status behavior.
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py`
  - Own subprocess config propagation, scratch directory paths, local model path env, and readiness polling against the real status/health contract.
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_adapter.py`
  - Own public TTSRequest to sidecar payload normalization, voice design aliases, language alias conflicts, generation allowlist, native sample-rate handling, and structured sidecar error mapping.
- Modify: `tldw_Server_API/app/core/TTS/tts_validation.py`
  - Relax OmniVoice language allowlist and validate OmniVoice mode/design/generation parameters before adapter dispatch.
- Modify: `Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py`
  - Record local `model_path`, add verification helpers, keep config patch scoped to the OmniVoice provider block.
- Modify: `tldw_Server_API/Config_Files/tts_providers_config.yaml`
  - Add disabled-by-default `model_path`, scratch/runtime hints, and clarify native 24 kHz.
- Modify: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
  - Document explicit OmniVoice install/verify and no runtime downloads.
- Modify: `tldw_Server_API/app/core/TTS/TTS-README.md`
  - Document real sidecar behavior and supported OmniVoice modes.
- Tests:
  - Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py`
  - Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py`
  - Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py`
  - Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py`
  - Modify: `tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py`
  - Create: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py`
  - Create: `tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py`

## Implementation Notes

- Use `source .venv/bin/activate` before Python, pytest, or Bandit commands.
- Keep sidecar runtime imports lazy. `omnivoice`, `torch`, `numpy`, and `soundfile` must not be imported by the main API process during normal module import.
- Do not loosen general outbound HTTP/private-IP policies. The existing sidecar-specific `httpx.AsyncClient(trust_env=False)` path remains the only loopback sidecar HTTP client.
- Store direct clone references in a managed scratch directory, not arbitrary default `/tmp`.
- When a task changes code, commit after its tests pass.

---

### Task 1: Expand Sidecar Protocol Schemas

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_protocol.py`
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py`

- [x] **Step 1: Write failing protocol tests**

Add tests that instantiate `OmniVoiceSynthesizeRequest` directly and assert:

```python
def test_synthesize_request_requires_generation_object_shape():
    req = OmniVoiceSynthesizeRequest(text="hi", mode="auto", generation={"num_step": 8})
    assert req.generation.compact() == {"num_step": 8}


def test_synthesize_request_rejects_unknown_top_level_keys():
    with pytest.raises(ValidationError):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", temperature=0.7)


def test_synthesize_request_rejects_unknown_generation_keys():
    with pytest.raises(ValidationError, match="generation"):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", generation={"unknown": 1})


def test_synthesize_request_rejects_mode_field_conflicts():
    with pytest.raises(ValidationError, match="mode=auto"):
        OmniVoiceSynthesizeRequest(text="hi", mode="auto", instruct="warm", generation={})


def test_synthesize_request_rejects_mixed_design_and_clone_inputs():
    with pytest.raises(ValidationError, match="instruct"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="clone",
            instruct="warm",
            reference_audio_path="/managed/ref.wav",
            reference_text="reference transcript",
            generation={},
        )
```

Also add assertions for `design` and `clone`:

```python
def test_synthesize_request_accepts_design_mode_with_instruct():
    req = OmniVoiceSynthesizeRequest(text="hi", mode="design", instruct="warm narrator", generation={})
    assert req.instruct == "warm narrator"


def test_synthesize_request_clone_requires_reference_text_and_path():
    with pytest.raises(ValidationError, match="reference_text"):
        OmniVoiceSynthesizeRequest(
            text="hi",
            mode="clone",
            reference_audio_path="/tmp/ref.wav",
            generation={},
        )
```

- [x] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py -k "synthesize_request" -q
```

Expected: FAIL because the current protocol has no `generation`, `instruct`, `language_id`, `requested_sample_rate`, or conflict validation.

- [x] **Step 3: Implement protocol models**

Update `omnivoice_sidecar_protocol.py` with focused models:

```python
class OmniVoiceSidecarError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    retryable: bool = False


class OmniVoiceRuntimeStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "idle_stopped"
    ready: bool = False
    provider: str = "omnivoice"
    runtime: str = "sidecar"
    model: str | None = None
    model_path: str | None = None
    sample_rate: int = 24000
    last_error_code: str | None = None


class OmniVoiceGenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_step: int | None = Field(default=None, ge=1, le=128)
    guidance_scale: float | None = Field(default=None, ge=0.0, le=30.0)
    denoise: bool | None = None
    t_shift: float | None = None
    position_temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    class_temperature: float | None = Field(default=None, ge=0.0, le=10.0)
    layer_penalty_factor: float | None = Field(default=None, ge=0.0, le=10.0)
    duration: float | None = Field(default=None, gt=0.0)
    speed: float | None = Field(default=None, gt=0.0, le=4.0)
    postprocess_output: bool | None = None
    preprocess_prompt: bool | None = None
    audio_chunk_duration: float | None = Field(default=None, gt=0.0)
    audio_chunk_threshold: float | None = Field(default=None, gt=0.0)

    def compact(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class OmniVoiceSynthesizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(..., min_length=1)
    mode: Literal["auto", "design", "clone"] = "auto"
    voice: str | None = None
    instruct: str | None = None
    language_id: str | None = None
    reference_audio_path: str | None = None
    reference_text: str | None = None
    requested_sample_rate: int | None = Field(default=None, ge=1)
    generation: OmniVoiceGenerationParams = Field(default_factory=OmniVoiceGenerationParams)

    @model_validator(mode="after")
    def validate_mode_inputs(self) -> "OmniVoiceSynthesizeRequest":
        if self.mode == "auto" and (self.instruct or self.reference_audio_path):
            raise ValueError("mode=auto cannot include instruct or reference_audio_path")
        if self.mode == "design":
            if not (self.instruct and self.instruct.strip()):
                raise ValueError("instruct is required for mode=design")
            if self.reference_audio_path:
                raise ValueError("mode=design cannot include reference_audio_path")
        if self.mode == "clone":
            if self.instruct:
                raise ValueError("mode=clone cannot include instruct")
            if not self.reference_audio_path:
                raise ValueError("reference_audio_path is required for mode=clone")
            if not (self.reference_text and self.reference_text.strip()):
                raise ValueError("reference_text is required for mode=clone")
        return self
```

Keep `build_sidecar_auth_headers(...)` unchanged. Keep `OmniVoiceHealthResponse` or replace it with `OmniVoiceRuntimeStatus` only if endpoint tests are updated at the same time.

- [x] **Step 4: Run focused protocol tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py -k "synthesize_request" -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_protocol.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py
git commit -m "test: define omnivoice sidecar protocol contract"
```

---

### Task 2: Normalize OmniVoice Adapter Payloads And Structured Errors

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_adapter.py`
- Modify: `tldw_Server_API/app/core/TTS/tts_validation.py`
- Modify: `tldw_Server_API/app/core/TTS/tts_service_v2.py`
- Modify: `tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py`
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py`
- Modify or create: `tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_omnivoice_fallback_policy.py`

- [x] **Step 1: Write failing adapter payload tests**

Add tests that verify canonical payload shape:

```python
@pytest.mark.asyncio
async def test_omnivoice_adapter_sends_generation_object_and_design_mode(monkeypatch):
    adapter = OmniVoiceAdapter({"sample_rate": 24000, "timeout": 5})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        lambda *, timeout: _FakeClient(recorded, _ok_wav_response()),
        raising=True,
    )

    await adapter.generate(TTSRequest(
        text="hello",
        voice="auto",
        format=AudioFormat.WAV,
        stream=False,
        language="es",
        extra_params={"instruct": "calm teacher", "num_step": 8, "guidance_scale": 4.0},
    ))

    assert recorded["json"]["mode"] == "design"
    assert recorded["json"]["instruct"] == "calm teacher"
    assert recorded["json"]["language_id"] == "es"
    assert recorded["json"]["generation"] == {"num_step": 8, "guidance_scale": 4.0}
    assert "sample_rate" not in recorded["json"]
    assert recorded["json"]["requested_sample_rate"] == 24000
```

Add one conflict test:

```python
def test_omnivoice_adapter_rejects_conflicting_instruct_aliases():
    request = TTSRequest(
        text="hello",
        voice="auto",
        format=AudioFormat.WAV,
        stream=False,
        extra_params={"instruct": "warm", "voice_design": "cold"},
    )
    with pytest.raises(TTSValidationError, match="instruct"):
        OmniVoiceAdapter({})._build_sidecar_payload(request, mode="auto", reference_audio_path=None)
```

Adjust helper signatures in tests if `_build_sidecar_payload` remains private and needs `requested_sample_rate`.

Add a direct-reference materialization test:

```python
@pytest.mark.asyncio
async def test_omnivoice_reference_audio_materializes_under_configured_scratch_dir(tmp_path, monkeypatch):
    scratch_dir = tmp_path / "runtime" / "scratch"
    adapter = OmniVoiceAdapter({
        "sample_rate": 24000,
        "timeout": 5,
        "extra_params": {"scratch_dir": str(scratch_dir)},
    })
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter.set_supervisor(_FakeSupervisor())
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.omnivoice_adapter.create_sidecar_async_client",
        lambda *, timeout: _FakeClient(recorded, _ok_wav_response()),
        raising=True,
    )

    await adapter.generate(TTSRequest(
        text="clone me",
        voice="clone",
        format=AudioFormat.WAV,
        stream=False,
        voice_reference=_make_reference_wav(3.5),
        extra_params={"reference_text": "reference transcript"},
    ))

    reference_path = Path(recorded["json"]["reference_audio_path"])
    assert reference_path.parent == scratch_dir
```

- [x] **Step 2: Write failing validation tests**

Add tests in `test_tts_validation_omnivoice.py`:

```python
def test_omnivoice_allows_non_english_language_passthrough():
    validator = TTSInputValidator()
    req = TTSRequest(text="hola", voice="auto", language="es", format=AudioFormat.WAV)
    valid, error = validator.validate_request(req, provider="omnivoice")
    assert valid is True
    assert error is None


def test_omnivoice_rejects_unknown_generation_param():
    validator = TTSInputValidator()
    req = TTSRequest(
        text="hello",
        voice="auto",
        format=AudioFormat.WAV,
        extra_params={"omnivoice_unknown_knob": 1},
    )
    valid, error = validator.validate_request(req, provider="omnivoice")
    assert valid is False
    assert "generation" in str(error).lower() or "unknown" in str(error).lower()
```

- [x] **Step 3: Write failing service fallback-policy tests**

Add `tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_omnivoice_fallback_policy.py` or extend an existing service-policy test file. Use a fake factory/adapter setup that makes the OmniVoice adapter fail and a second provider succeed if fallback is allowed.

Cover at least:

```python
@pytest.mark.asyncio
async def test_explicit_omnivoice_instruct_request_does_not_fallback():
    request = OpenAISpeechRequest(
        model="omnivoice",
        input="hello",
        voice="auto",
        response_format="wav",
        stream=False,
        extra_params={"instruct": "calm narrator"},
    )
    service = _service_with_failing_omnivoice_and_successful_openai()
    with pytest.raises(TTSGenerationError):
        chunks = [chunk async for chunk in service.generate_speech(request, provider="omnivoice", fallback=True)]
    assert service.fake_openai.calls == 0


@pytest.mark.asyncio
async def test_omnivoice_direct_voice_reference_does_not_fallback():
    request = OpenAISpeechRequest(
        model="tts-1",
        input="hello",
        voice="clone",
        voice_reference=base64.b64encode(_make_reference_wav(3.5)).decode("ascii"),
        response_format="wav",
        stream=False,
        extra_params={"reference_text": "reference transcript"},
    )
    service = _service_with_failing_omnivoice_and_successful_openai()
    with pytest.raises(TTSGenerationError):
        chunks = [chunk async for chunk in service.generate_speech(request, provider="omnivoice", fallback=True)]
    assert service.fake_openai.calls == 0
```

Also cover `custom:<voice_id>` and an OmniVoice-only generation parameter such as `num_step`.

- [x] **Step 4: Run focused tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py \
  tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_omnivoice_fallback_policy.py \
  -q
```

Expected: FAIL because the current adapter sends `sample_rate` top-level, has no `design` mode, does not build `generation`, and validation is English-only.

- [x] **Step 5: Implement adapter normalization helpers**

Add focused helpers to `OmniVoiceAdapter`:

```python
GENERATION_PARAM_TYPES = {
    "num_step": int,
    "guidance_scale": float,
    "denoise": bool,
    "t_shift": float,
    "position_temperature": float,
    "class_temperature": float,
    "layer_penalty_factor": float,
    "duration": float,
    "speed": float,
    "postprocess_output": bool,
    "preprocess_prompt": bool,
    "audio_chunk_duration": float,
    "audio_chunk_threshold": float,
}
INSTRUCT_KEYS = ("instruct", "voice_design", "voice_description")
LANGUAGE_KEYS = ("language_id", "language")
REFERENCE_TEXT_KEYS = ("reference_text", "ref_text", "voice_reference_text")
```

Implementation rules:

- `_resolve_instruct(extras)` returns a single stripped string or `None`; different values across aliases raise `TTSValidationError`.
- `_resolve_language_id(request)` checks `extra_params.language_id`, `extra_params.language`, then `request.language`; conflicting explicit values raise `TTSValidationError`.
- `_resolve_generation(extras)` accepts only `GENERATION_PARAM_TYPES`; type-coerce numeric values; reject unknown OmniVoice-only keys that start with `omnivoice_` or are in a local unsupported set.
- `_resolve_mode(...)` rejects mixed design+clone inputs when both `instruct` and reference audio/custom clone indicators are present; otherwise returns `clone` if reference audio exists, `design` if instruct exists, and `auto` for plain requests. Explicit mode conflicts raise `TTSValidationError`.
- `_build_sidecar_payload(...)` emits only canonical keys: `text`, `mode`, optional `voice`, optional `instruct`, optional `language_id`, optional `reference_audio_path`, optional `reference_text`, `requested_sample_rate`, and `generation`.
- `_materialize_reference_audio_sync(...)` writes direct clone reference files under `extra_params.scratch_dir` from provider config, falling back to `temp_dir` only for test-only configurations where no sidecar containment is required.

- [x] **Step 6: Use sidecar native sample-rate headers on response**

In `generate(...)`, read sidecar native rate:

```python
native_sample_rate = int(response.headers.get("X-OmniVoice-Sample-Rate", self.DEFAULT_SAMPLE_RATE) or self.DEFAULT_SAMPLE_RATE)
target_rate = request.target_sample_rate or native_sample_rate
```

Use `native_sample_rate` for WAV/PCM normalization and response metadata unless conversion explicitly changes it. Do not pass requested target sample rate as sidecar native output rate.

- [x] **Step 7: Implement structured sidecar error mapping**

When `response.status_code != 200`, parse JSON if present:

```python
payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
error = payload.get("error") if isinstance(payload, dict) else None
code = error.get("code") if isinstance(error, dict) else None
```

Map:

- `MODEL_NOT_AVAILABLE`, `RUNTIME_IMPORT_FAILED`, `MODEL_LOAD_FAILED` -> `TTSProviderNotConfiguredError` or `TTSGenerationError` with sanitized details
- `INVALID_REFERENCE_AUDIO`, `INVALID_GENERATION_PARAMETER`, `REFERENCE_PATH_NOT_ALLOWED` -> `TTSValidationError`
- everything else -> `TTSGenerationError`

Keep `_sanitize_sidecar_error_text(...)`; never surface raw response text.

- [x] **Step 8: Relax validation language allowlist and add generation checks**

In `tts_validation.py`:

- Remove `omnivoice` from `SUPPORTED_LANGUAGES` or include an explicit sentinel that skips language rejection.
- Add OmniVoice-specific parameter validation for `instruct`, `voice_design`, `voice_description`, `language_id`, `omnivoice_mode`/`mode`, and generation allowlist/ranges.
- Keep clone `reference_text` and duration checks unchanged.

- [x] **Step 9: Implement service-level no-fallback policy**

In `tts_service_v2.py`, harden `_is_explicit_omnivoice_request(...)` or the fallback decision near `generate_speech(...)` so fallback is disabled when any of these are true:

- provider hint is `omnivoice`
- model is an OmniVoice alias
- `voice` starts with `custom:`
- `voice_reference` is present
- `extra_params.instruct`, `voice_design`, or `voice_description` is present
- `extra_params` contains any OmniVoice generation key from the approved allowlist
- `extra_params.mode` or `extra_params.omnivoice_mode` selects `design` or `clone`

Keep generic fallback behavior when OmniVoice is only an implicit priority candidate and no OmniVoice-specific semantic is present.

- [x] **Step 10: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py \
  tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_omnivoice_fallback_policy.py \
  -q
```

Expected: PASS.

- [x] **Step 11: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_adapter.py \
  tldw_Server_API/app/core/TTS/tts_validation.py \
  tldw_Server_API/app/core/TTS/tts_service_v2.py \
  tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py \
  tldw_Server_API/tests/TTS_NEW/unit/service/test_tts_omnivoice_fallback_policy.py
git commit -m "feat: normalize omnivoice sidecar requests"
```

---

### Task 3: Add Sidecar-Local OmniVoice Runtime

**Files:**
- Create: `tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py`
- Create: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py`

- [x] **Step 1: Write failing runtime unit tests**

Use fake import hooks instead of importing real OmniVoice. Cover:

```python
@pytest.mark.asyncio
async def test_runtime_rejects_missing_local_model_path(tmp_path):
    runtime = OmniVoiceRuntime({"model_path": str(tmp_path / "missing")})
    with pytest.raises(OmniVoiceRuntimeError) as exc:
        await runtime.load()
    assert exc.value.code == "MODEL_NOT_AVAILABLE"
```

Fake model behavior:

```python
class _FakeOmniVoice:
    calls = []

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        cls.model_path = model_path
        cls.kwargs = kwargs
        return cls()

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return np.zeros(2400, dtype=np.float32)
```

Assert:

- `from_pretrained` receives a local directory.
- auto mode calls `generate(text=..., language=...)`.
- design mode includes `instruct`.
- clone mode includes `ref_audio` and `ref_text`.
- result is parseable WAV at 24000 Hz.
- empty arrays raise `EMPTY_AUDIO_OUTPUT`.
- clone reference paths outside configured `scratch_dir` raise `REFERENCE_PATH_NOT_ALLOWED`.

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py -q
```

Expected: FAIL because `omnivoice_runtime.py` does not exist.

- [x] **Step 3: Implement runtime module**

Create `omnivoice_runtime.py` with:

```python
class OmniVoiceRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
```

Runtime shape:

```python
class OmniVoiceRuntime:
    NATIVE_SAMPLE_RATE = 24000

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.model = None
        self.status = "idle_stopped"
        self._load_lock = asyncio.Lock()
        self._generate_lock = asyncio.Lock()
```

Key implementation details:

- Resolve `model_path` from config first, then `model` only if it is an existing directory.
- If resolved path is missing or not a directory, raise `OmniVoiceRuntimeError("MODEL_NOT_AVAILABLE", ...)`.
- Lazy import `omnivoice.OmniVoice` inside `load()`.
- Catch import failures as `RUNTIME_IMPORT_FAILED`.
- Catch load failures as `MODEL_LOAD_FAILED`.
- Run blocking `from_pretrained` and `generate` via `asyncio.to_thread`.
- Build kwargs with `language=request.language_id` only when present.
- Resolve `scratch_dir` from config and reject clone `reference_audio_path` values that are not contained under `scratch_dir` or another explicitly configured managed reference directory.
- Convert generated output into mono float/PCM WAV bytes. Prefer `soundfile.write(BytesIO(), array, 24000, format="WAV")` when available; fallback to `wave` with clipped int16 conversion.
- Return a small dataclass `OmniVoiceSynthesizeResult(audio_bytes, audio_format, sample_rate, channels, cold_start, model)`.

- [x] **Step 4: Run runtime tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py -q
```

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py
git commit -m "feat: add omnivoice sidecar runtime"
```

---

### Task 4: Wire Runtime Into Sidecar HTTP Server

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py`
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py`

- [x] **Step 1: Write failing server tests with injected fake runner**

Extend `create_app(...)` tests:

```python
class _FakeRuntime:
    async def status(self):
        return OmniVoiceRuntimeStatus(status="ready", ready=True, model="local-model")

    async def load(self):
        return None

    async def synthesize(self, request):
        return OmniVoiceSynthesizeResult(
            audio_bytes=_make_wav_bytes(),
            audio_format="wav",
            sample_rate=24000,
            channels=1,
            cold_start=False,
            model="local-model",
        )
```

Assert:

- `GET /status` returns runtime status.
- `POST /control/warmup` calls `load()`.
- `POST /v1/synthesize` returns fake WAV headers and bytes.
- `OmniVoiceRuntimeError("MODEL_NOT_AVAILABLE", ...)` returns JSON `{"error": {"code": "MODEL_NOT_AVAILABLE", ...}}` with a non-200 status.
- clone requests with a reference path outside `scratch_dir` return structured `REFERENCE_PATH_NOT_ALLOWED`.
- Existing auth tests still pass.

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py -q
```

Expected: FAIL because current server has no runner injection or `/status`.

- [x] **Step 3: Implement sidecar app factory injection**

Change signature:

```python
def create_app(*, sidecar_token: str, runtime: OmniVoiceRuntime | None = None) -> FastAPI:
```

When no runtime is passed, build one from env:

```python
runtime = runtime or OmniVoiceRuntime(load_runtime_config_from_env())
```

Add env config loader:

```python
def load_runtime_config_from_env() -> dict[str, Any]:
    return {
        "model": os.environ.get("OMNIVOICE_MODEL", "omnivoice"),
        "model_path": os.environ.get("OMNIVOICE_MODEL_PATH"),
        "runtime_path": os.environ.get("OMNIVOICE_RUNTIME_PATH"),
        "scratch_dir": os.environ.get("OMNIVOICE_SCRATCH_DIR"),
        "device_map": os.environ.get("OMNIVOICE_DEVICE_MAP"),
        "dtype": os.environ.get("OMNIVOICE_DTYPE"),
    }
```

Add `/status`, make `/health` cheap and status-backed, and route `warmup/reload/shutdown` through runtime methods where available.

- [x] **Step 4: Implement structured error responses**

Add helper:

```python
def _runtime_error_response(exc: OmniVoiceRuntimeError) -> JSONResponse:
    status_code = 503 if exc.code in {"MODEL_NOT_AVAILABLE", "MODEL_LOAD_FAILED", "RUNTIME_IMPORT_FAILED"} else 422
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": exc.code, "message": str(exc), "retryable": exc.retryable}},
    )
```

Keep raw exception details out of responses.

- [x] **Step 5: Run sidecar server tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py
git commit -m "feat: run omnivoice runtime from sidecar API"
```

---

### Task 5: Pass Runtime Config Through The Supervisor

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py`
- Modify: `tldw_Server_API/app/core/TTS/tts_service_v2.py` only if adapter overrides need scratch-dir defaults from service config
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py`

- [x] **Step 1: Write failing supervisor env tests**

Add assertions to existing spawn env tests:

```python
@pytest.mark.asyncio
async def test_supervisor_spawn_sets_omnivoice_runtime_env(tmp_path, monkeypatch):
    spawn_envs: list[dict[str, str]] = []
    # Patch subprocess spawn and readiness client the same way existing supervisor tests do.

    supervisor = OmniVoiceSidecarSupervisor(
        provider_config={
            "model": "omnivoice",
            "extra_params": {
                "model_path": str(tmp_path / "models" / "omnivoice"),
                "runtime_path": str(tmp_path / "runtime"),
                "scratch_dir": str(tmp_path / "runtime" / "scratch"),
                "device_map": "cpu",
                "dtype": "float32",
            },
        },
        repo_root=tmp_path,
    )
    await supervisor.ensure_started()
    env = spawn_envs[0]
    assert env["OMNIVOICE_MODEL"] == "omnivoice"
    assert env["OMNIVOICE_MODEL_PATH"].endswith("models/omnivoice")
    assert env["OMNIVOICE_RUNTIME_PATH"].endswith("runtime")
    assert env["OMNIVOICE_SCRATCH_DIR"].endswith("scratch")
    assert env["OMNIVOICE_DEVICE_MAP"] == "cpu"
    assert env["OMNIVOICE_DTYPE"] == "float32"
```

The body should reuse the existing fake process and ready-client patterns in `test_omnivoice_sidecar_supervisor.py`.

- [x] **Step 2: Run supervisor tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py -q
```

Expected: FAIL on missing env vars.

- [x] **Step 3: Implement config env propagation**

In `OmniVoiceSidecarSupervisor.__init__`, resolve:

- provider top-level `model`
- extra `model_path`
- extra `runtime_path`
- extra `scratch_dir`, default `<runtime_path>/scratch` when runtime path exists
- extra `device_map`
- extra `dtype`

In `_build_subprocess_env(...)`, add only non-empty values. Create `scratch_dir` and `runtime_path` directories before spawn if configured.

- [x] **Step 4: Ensure readiness remains cheap**

Keep `_wait_for_ready()` polling `/health`. Since `/health` becomes status-backed, it must not trigger real synthesis or model download. If `/health` reports `ready=False` because model is missing, supervisor should treat startup as successful only if the sidecar process is alive. Do not spin forever waiting for a model that setup has not provisioned.

Implementation option:

```python
if response.status_code == 200:
    payload = response.json()
    status_value = str(payload.get("status", ""))
    if payload.get("ready") is True or status_value in {"idle_stopped", "model_unavailable", "runtime_missing", "degraded"}:
        return
```

Do not hide synth-time provider errors; this only means the sidecar process is reachable.

- [x] **Step 5: Run supervisor tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py
git commit -m "feat: pass omnivoice runtime config to sidecar"
```

---

### Task 6: Harden Installer And Config For Local Model Paths

**Files:**
- Modify: `Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py`
- Modify: `tldw_Server_API/Config_Files/tts_providers_config.yaml`
- Modify: `tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py`

- [x] **Step 1: Write failing installer tests**

Add tests for config patch output:

```python
def test_omnivoice_installer_records_model_path(tmp_path):
    ...
    model_path = tmp_path / "models" / "OmniVoice"
    model_path.mkdir(parents=True)
    changed = patch_tts_config(..., model_path=model_path)
    parsed = yaml.safe_load(config_path.read_text())
    assert parsed["providers"]["omnivoice"]["extra_params"]["model_path"] == "models/OmniVoice"
```

Add a helper test:

```python
def test_validate_local_model_path_rejects_missing_path(tmp_path):
    from Helper_Scripts.TTS_Installers.install_tts_omnivoice_sidecar import validate_local_model_path
    with pytest.raises(SystemExit, match="model"):
        validate_local_model_path(tmp_path / "missing")
```

- [x] **Step 2: Run installer tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py -q
```

Expected: FAIL because installer does not record `model_path`.

- [x] **Step 3: Implement model path installer options**

Add CLI args:

```python
parser.add_argument("--model-path", help="Resolved local OmniVoice model directory")
parser.add_argument("--skip-model-check", action="store_true")
```

Add helper:

```python
def validate_local_model_path(model_path: Path) -> Path:
    resolved = model_path.expanduser().resolve()
    if not resolved.is_dir():
        raise SystemExit(f"OmniVoice model path is not a directory: {resolved}")
    return resolved
```

Patch config to include:

```yaml
extra_params:
  model_path: "models/omnivoice_sidecar/models/OmniVoice"
  runtime_path: "models/omnivoice_sidecar/runtime"
  scratch_dir: "models/omnivoice_sidecar/runtime/scratch"
```

If the installer is going to patch `providers.omnivoice.enabled: true`, `--model-path` is required unless `--skip-model-check` is paired with a config mode that leaves the provider disabled. For this implementation slice, choose the simpler behavior: fail fast when `--model-path` is absent. Do not silently enable OmniVoice without a local model directory.

- [x] **Step 4: Update default config**

In `tts_providers_config.yaml`, keep `enabled: false` and add comments/keys:

```yaml
model: "omnivoice"
sample_rate: 24000  # Native OmniVoice output; public resampling happens in the main adapter.
extra_params:
  model_path: "models/omnivoice_sidecar/models/OmniVoice"
  runtime_path: "models/omnivoice_sidecar/runtime"
  scratch_dir: "models/omnivoice_sidecar/runtime/scratch"
```

Do not add OmniVoice to provider priority.

- [x] **Step 5: Run installer tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py -q
```

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py \
  tldw_Server_API/Config_Files/tts_providers_config.yaml \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py
git commit -m "feat: require local omnivoice model path"
```

---

### Task 7: Add Opt-In Real Runtime Integration Test

**Files:**
- Create: `tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py`

- [x] **Step 1: Write skipped-by-default integration tests**

Create tests guarded by `TLDW_TEST_OMNIVOICE_REAL=1`:

```python
pytestmark = pytest.mark.integration


def _require_real_omnivoice():
    if os.environ.get("TLDW_TEST_OMNIVOICE_REAL") != "1":
        pytest.skip("Set TLDW_TEST_OMNIVOICE_REAL=1 to run real OmniVoice tests")
    model_path = os.environ.get("TLDW_OMNIVOICE_MODEL_PATH")
    if not model_path or not Path(model_path).is_dir():
        pytest.skip("TLDW_OMNIVOICE_MODEL_PATH must point to a local model directory")
    return Path(model_path)
```

Tests:

- `test_real_omnivoice_runtime_auto_voice_smoke`
- `test_real_omnivoice_runtime_design_smoke`
- `test_real_omnivoice_runtime_clone_smoke`

Each should:

- instantiate `OmniVoiceRuntime({"model_path": str(model_path)})`
- call `synthesize(...)`
- assert non-empty WAV bytes
- parse with `wave.open(BytesIO(...))`
- assert sample rate is 24000
- instantiate the runtime with a `scratch_dir` under `tmp_path`
- write the clone reference WAV inside that `scratch_dir` before calling clone synth

Use a tiny generated WAV fixture for clone reference and a short `reference_text`.

- [x] **Step 2: Run integration test in default environment**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py -q
```

Expected: SKIPPED unless `TLDW_TEST_OMNIVOICE_REAL=1` and `TLDW_OMNIVOICE_MODEL_PATH` are set.

- [x] **Step 3: Commit**

```bash
git add tldw_Server_API/tests/TTS_NEW/integration/test_omnivoice_real_runtime.py
git commit -m "test: add opt-in omnivoice real runtime smoke tests"
```

---

### Task 8: Update OmniVoice Documentation

**Files:**
- Modify: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- Modify: `tldw_Server_API/app/core/TTS/TTS-README.md`

- [x] **Step 1: Update TTS setup guide**

Document:

- OmniVoice is optional and disabled by default.
- PyTorch/backend prerequisites must be installed explicitly for the target hardware before real synthesis is expected to work.
- Run installer with explicit local model path.
- Runtime requests do not download model assets.
- Supported modes: auto, voice design via `extra_params.instruct`, cloning via direct `voice_reference` or stored `custom:<voice_id>` plus `reference_text`.
- `response_format` conversion happens in tldw; sidecar native output is 24 kHz WAV/PCM.

Example API body:

```json
{
  "model": "omnivoice",
  "voice": "auto",
  "input": "A short test sentence.",
  "response_format": "wav",
  "stream": false,
  "extra_params": {
    "instruct": "A calm documentary narrator",
    "language_id": "en",
    "num_step": 8
  }
}
```

- [x] **Step 2: Update TTS README**

Add a concise OmniVoice provider section pointing to setup guide and noting:

- sidecar runtime
- one configured local model
- no incremental streaming v1
- structured sidecar errors are mapped by the adapter

- [x] **Step 3: Run docs link/check smoke**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Docs/test_stt_tts_link_hygiene.py tldw_Server_API/tests/Docs/test_speech_api_guide_map.py -q
```

Expected: PASS, or document existing unrelated failures before proceeding.

- [x] **Step 4: Commit**

```bash
git add Docs/STT-TTS/TTS-SETUP-GUIDE.md tldw_Server_API/app/core/TTS/TTS-README.md
git commit -m "docs: document omnivoice real sidecar setup"
```

---

### Task 9: Final Verification And Security Scan

**Files:**
- No planned source edits unless verification finds issues.
- Update Backlog task(s) with final verification notes.

- [x] **Step 1: Run focused OmniVoice test suite**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_runtime.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_server.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_supervisor.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_installer.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_omnivoice.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_registry.py \
  -q
```

Expected: PASS.

- [x] **Step 2: Run broader TTS regression slice**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/unit tldw_Server_API/tests/TTS/adapters/test_omnivoice_adapter_mock.py -q
```

Expected: PASS, or document unrelated baseline failures.

- [x] **Step 3: Run Bandit on touched code scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/TTS/adapters/omnivoice_adapter.py \
  tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_protocol.py \
  tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_server.py \
  tldw_Server_API/app/core/TTS/adapters/omnivoice_sidecar_supervisor.py \
  tldw_Server_API/app/core/TTS/adapters/omnivoice_runtime.py \
  tldw_Server_API/app/core/TTS/tts_validation.py \
  Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py \
  -f json -o /tmp/bandit_omnivoice_real_sidecar.json
```

Expected: no new high/medium findings in touched code. Fix new findings before finalizing.

- [x] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files changed.

- [x] **Step 5: Update Backlog implementation task**

Use Backlog.md MCP to record:

- touched files
- tests run and results
- Bandit result path
- known skipped real-runtime integration test if no local model path was available
- final summary

- [x] **Step 6: Commit final tracking update**

```bash
git add "backlog/tasks/<implementation-task>.md"
git commit -m "chore: record omnivoice sidecar verification"
```

---

## Definition Of Done

- The sidecar no longer returns hard-coded silent audio when a runtime is configured.
- Auto voice, voice design, and clone requests map to real OmniVoice API calls.
- Runtime no-download behavior is enforced by local model directory validation.
- Adapter emits canonical sidecar payloads with nested `generation`.
- Sidecar structured errors are sanitized and mapped to existing TTS exception classes.
- OmniVoice language validation no longer blocks non-English passthrough.
- Public response format conversion remains in the main app.
- Unit tests pass for protocol, adapter, runtime, server, supervisor, installer, validation, and registry behavior.
- Real runtime smoke tests exist and skip by default without explicit local model env.
- Docs describe setup and supported modes.
- Bandit is run on touched code scope.
