# Chatterbox Upstream Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update tldw's Chatterbox integration to support the current upstream TTS family (standard, multilingual, Turbo) and add dedicated voice-conversion support while keeping watermark removal enabled by default.

**Architecture:** Keep `chatterbox` as one provider for TTS, add a Chatterbox family mode resolver inside the adapter/validation layers, and expose voice conversion through a dedicated audio endpoint instead of as a text-to-speech model alias. Refactor shared audio response helpers so TTS and VC can reuse persistence/history/header behavior without duplicating endpoint logic.

**Tech Stack:** FastAPI, Pydantic, existing TTS adapter registry/service, Chatterbox upstream package, pytest, Bandit.

---

## Stage 1: Establish Chatterbox Catalog And Config Surface
**Goal**: Create one backend source of truth for Chatterbox TTS model aliases and update config/install surfaces for the current upstream family.
**Success Criteria**: Canonical model ids exist in one place, registry/schema/config reference them consistently, and the Chatterbox install extra covers the runtime imports needed for Turbo/VC.
**Tests**: Alias resolution unit tests; config/registry smoke tests.
**Status**: In Progress

**TASK-531 slice update (2026-06-07)**: Added a shared Chatterbox catalog, wired standard/emotion/multilingual/turbo aliases through `TTSAdapterFactory`, refreshed the OpenAI speech model description, updated provider config defaults, and added current Turbo runtime dependencies. Setup docs remain to be refreshed.

**TASK-539 slice update (2026-06-08)**: Added local model path handling for the Chatterbox runtime loaders. Existing local `model_path`, `multilingual_model_path`, `turbo_model_path`, and `vc_model_path` values use upstream `from_local()`; repo IDs and unset paths preserve the current `from_pretrained()` behavior.

### Task 1: Add a canonical Chatterbox model catalog

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapter_registry.py`
- Modify: `tldw_Server_API/app/core/Audio/tts_service.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`

**Step 1: Write the failing test**

Add assertions covering new aliases and the rule that VC is not part of the TTS model alias set.

```python
def test_chatterbox_model_aliases_resolve_to_provider():
    assert model_to_provider_map["chatterbox"] == TTSProvider.CHATTERBOX
    assert model_to_provider_map["chatterbox-emotion"] == TTSProvider.CHATTERBOX
    assert model_to_provider_map["chatterbox-multilingual"] == TTSProvider.CHATTERBOX
    assert model_to_provider_map["chatterbox-turbo"] == TTSProvider.CHATTERBOX
    assert "chatterbox-vc" not in model_to_provider_map
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k aliases -v
```

Expected: FAIL because the new aliases are not registered yet.

**Step 3: Write minimal implementation**

- Add a Chatterbox alias catalog constant.
- Reuse that constant in registry alias mapping and provider inference.
- Update `OpenAISpeechRequest.model` description to list canonical Chatterbox TTS ids only.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k aliases -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapter_registry.py tldw_Server_API/app/core/Audio/tts_service.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
git commit -m "feat: centralize chatterbox model aliases"
```

### Task 2: Update config and install extras for current upstream family

**Files:**
- Modify: `pyproject.toml`
- Modify: `tldw_Server_API/Config_Files/tts_providers_config.yaml`
- Modify: `Docs/STT-TTS/CHATTERBOX_SETUP.md`
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`

**Step 1: Write the failing test**

Add a config-focused test that expects the adapter to understand a `variant` setting and preserve `disable_watermark=True`.

```python
def test_chatterbox_variant_config_defaults():
    adapter = ChatterboxAdapter({"variant": "turbo", "disable_watermark": True})
    assert adapter.config.get("variant") == "turbo"
    assert adapter.disable_watermark is True
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k variant_config -v
```

Expected: FAIL or require adapter changes.

**Step 3: Write minimal implementation**

- Expand `TTS_chatterbox` extras to include current runtime imports needed by Turbo/VC.
- Replace the stale `model_path ... unused` config note with real `model_path`, `turbo_model_path`, and `variant` fields.
- Update Chatterbox setup docs accordingly.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k variant_config -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml tldw_Server_API/Config_Files/tts_providers_config.yaml Docs/STT-TTS/CHATTERBOX_SETUP.md tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
git commit -m "feat: refresh chatterbox config and install surface"
```

## Stage 2: Implement Chatterbox Family Mode Resolution
**Goal**: Teach the adapter to resolve standard, multilingual, and Turbo families cleanly and clean them up correctly.
**Success Criteria**: Request model aliases and config variant select the right upstream runtime, watermark stripping still works, and cleanup clears every loaded family.
**Tests**: Adapter unit tests for family selection, cleanup, and unsupported Turbo controls.
**Status**: In Progress

**TASK-531 slice update (2026-06-07)**: Added request/config family resolution for standard, multilingual, and Turbo; lazy-loaded Turbo via `chatterbox.tts_turbo`; cleared all TTS family model handles during cleanup; added safe generate-kwarg filtering and seed handling. Dedicated VC runtime support remains for later stages.

**TASK-532 slice update (2026-06-07)**: Added explicit Turbo ignored-control metadata and stopped passing no-op CFG/exaggeration/min-p controls to the upstream Turbo runtime.

### Task 3: Add family resolution and runtime caching to the adapter

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py`
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`

**Step 1: Write the failing test**

Add tests for mode resolution and runtime cleanup.

```python
@pytest.mark.asyncio
async def test_chatterbox_resolves_turbo_from_request_model():
    adapter = ChatterboxAdapter({"variant": "standard"})
    request = TTSRequest(text="Hi", model="chatterbox-turbo")
    assert adapter._resolve_family_mode(request, language_id="en") == "turbo"

@pytest.mark.asyncio
async def test_close_clears_all_chatterbox_runtimes():
    adapter = ChatterboxAdapter({})
    adapter.model_en = object()
    adapter.model_multi = object()
    adapter.model_turbo = object()
    adapter.model_vc = object()
    await adapter.close()
    assert adapter.model_en is None
    assert adapter.model_multi is None
    assert adapter.model_turbo is None
    assert adapter.model_vc is None
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k "turbo or close_clears" -v
```

Expected: FAIL because Turbo/VC runtime state does not exist yet.

**Step 3: Write minimal implementation**

- Add explicit family resolution helper.
- Add lazy runtime slots for Turbo and VC.
- Route standard/multilingual/turbo generation through the correct upstream loader.
- Expand cleanup/resource-manager registration to cover every family.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k "turbo or close_clears" -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
git commit -m "feat: add chatterbox family mode resolution"
```

### Task 4: Preserve transparent Turbo behavior

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py`
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`

**Step 1: Write the failing test**

Add a test that Turbo ignores unsupported controls but reports that fact.

```python
@pytest.mark.asyncio
async def test_turbo_ignores_unsupported_cfg_and_exaggeration(monkeypatch):
    adapter = ChatterboxAdapter({})
    request = TTSRequest(
        text="Hello [laugh]",
        model="chatterbox-turbo",
        extra_params={"cfg_weight": 0.5},
        emotion="happy",
    )
    metadata = adapter._build_generation_metadata(request, family_mode="turbo")
    assert metadata["family_mode"] == "turbo"
    assert metadata["ignored_controls"] == ["cfg_weight", "emotion", "emotion_intensity", "exaggeration"]
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k ignored_controls -v
```

Expected: FAIL because the adapter does not expose this metadata yet.

**Step 3: Write minimal implementation**

- Add Turbo-specific metadata reporting.
- Ensure unsupported controls are ignored intentionally rather than applied accidentally.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k ignored_controls -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
git commit -m "feat: make chatterbox turbo control handling explicit"
```

## Stage 3: Make Validation And TTS Request Plumbing Mode-Aware
**Goal**: Update validation and service plumbing so the Chatterbox family behaves correctly without breaking existing callers.
**Success Criteria**: TTS validation distinguishes standard/multilingual/turbo, existing aliases remain backward compatible, and schema/discovery surfaces reflect the new TTS ids.
**Tests**: Validation unit tests and endpoint integration tests.
**Status**: In Progress

**TASK-531 slice update (2026-06-07)**: Validation now distinguishes Chatterbox standard/Turbo English-only behavior from multilingual language support, and OpenAI request conversion maps `extra_params.seed` into `TTSRequest.seed`.

### Task 5: Add Chatterbox family-aware validation

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/tts_validation.py`
- Test: `tldw_Server_API/tests/TTS/test_tts_validation.py`

**Step 1: Write the failing test**

Add tests for multilingual languages and Turbo-specific behavior.

```python
def test_chatterbox_multilingual_accepts_fr():
    validator = TTSInputValidator({})
    request = TTSRequest(text="Bonjour", model="chatterbox-multilingual", language="fr")
    ok, error = validator.validate_request(request, provider="chatterbox")
    assert ok is True
    assert error is None

def test_chatterbox_standard_rejects_fr():
    validator = TTSInputValidator({})
    request = TTSRequest(text="Bonjour", model="chatterbox", language="fr")
    ok, error = validator.validate_request(request, provider="chatterbox")
    assert ok is False
    assert "Language" in error
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py -k chatterbox -v
```

Expected: FAIL because validation only knows provider-wide `chatterbox`.

**Step 3: Write minimal implementation**

- Add Chatterbox family resolution in the validator from `request.model`.
- Update supported languages/formats/reference handling to use family-specific rules.
- Keep `model="chatterbox"` and `model="chatterbox-emotion"` backward compatible.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py -k chatterbox -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/tts_validation.py tldw_Server_API/tests/TTS/test_tts_validation.py
git commit -m "feat: add chatterbox family-aware validation"
```

## Stage 4: Add Dedicated Voice Conversion Support
**Goal**: Add a dedicated Chatterbox VC endpoint and adapter path without polluting the text-to-speech model catalog.
**Success Criteria**: VC requests accept source audio plus target voice input, reuse stored custom voices when provided, and return converted audio successfully.
**Tests**: VC adapter unit tests and endpoint integration tests.
**Status**: In Progress

**TASK-533 slice update (2026-06-08)**: Added a dedicated Chatterbox VC runtime path in the adapter. The adapter now lazy-loads `chatterbox.vc.ChatterboxVC`, calls upstream `generate(audio=..., target_voice_path=...)`, encodes the resulting waveform through the shared streamer, and clears `model_vc` during cleanup.

**TASK-534 slice update (2026-06-08)**: Added `POST /api/v1/audio/voice-conversion` as a protected multipart endpoint accepting `source_audio`, optional `target_voice`, `response_format`, and `stream`. The endpoint materializes uploads to temporary files, delegates through `TTSServiceV2.convert_chatterbox_voice`, defers temp-file cleanup until streaming responses are consumed, and registers the `audio.voice_conversion` privilege scope.

**TASK-535 slice update (2026-06-08)**: Added `target_voice_id` form support to the voice-conversion endpoint. Stored target voices now resolve through `VoiceManager.load_voice_reference_audio()` for the authenticated user, materialize as the same temporary target reference path used by uploaded `target_voice`, and requests that provide both target reference forms fail fast with HTTP 400.

### Task 6: Add VC schema and endpoint with shared response helpers

**Files:**
- Create: `tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/audio.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/audio/__init__.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/audio_schemas.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/audio/audio_response_helpers.py`
- Test: `tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py`

**Step 1: Write the failing test**

Add an integration test for the new endpoint.

```python
def test_chatterbox_voice_conversion_requires_target_voice(test_client, auth_headers):
    payload = {
        "input_audio": BASE64_WAV,
        "input_audio_format": "wav",
        "response_format": "wav",
    }
    response = test_client.post("/api/v1/audio/voice-conversion", json=payload, headers=auth_headers)
    assert response.status_code == 422
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py -v
```

Expected: FAIL because the endpoint does not exist yet.

**Step 3: Write minimal implementation**

- Add a dedicated multipart endpoint that:
  - materializes uploaded source audio
  - materializes optional uploaded target voice audio
  - resolves stored custom voice IDs into target voice reference files
  - calls into the Chatterbox adapter VC path
  - returns streaming or non-streaming audio responses
  - defers temp-file cleanup until streaming responses finish
- Register the route in audio router exports and privilege maps.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/audio/audio_voice_conversion.py tldw_Server_API/app/api/v1/endpoints/audio/audio.py tldw_Server_API/app/api/v1/endpoints/audio/__init__.py tldw_Server_API/app/api/v1/schemas/audio_schemas.py tldw_Server_API/app/api/v1/endpoints/audio/audio_response_helpers.py tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py
git commit -m "feat: add chatterbox voice conversion endpoint"
```

### Task 7: Add VC runtime support to the Chatterbox adapter

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py`
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`

**Step 1: Write the failing test**

Add a unit test that checks VC uses separate source audio and target voice inputs.

```python
@pytest.mark.asyncio
async def test_chatterbox_voice_conversion_uses_source_and_target_paths(monkeypatch, tmp_path):
    adapter = ChatterboxAdapter({})
    source = tmp_path / "source.wav"
    target = tmp_path / "target.wav"
    source.write_bytes(b"RIFF" + b"\x00" * 100)
    target.write_bytes(b"RIFF" + b"\x00" * 100)
    call = {}

    class FakeVC:
        sr = 24000
        def generate(self, audio, target_voice_path=None):
            call["audio"] = audio
            call["target_voice_path"] = target_voice_path
            return FAKE_TENSOR

    adapter.model_vc = FakeVC()
    await adapter._convert_voice_with_chatterbox(str(source), str(target), AudioFormat.WAV)
    assert call["audio"] == str(source)
    assert call["target_voice_path"] == str(target)
```

**Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k voice_conversion -v
```

Expected: FAIL because VC helpers do not exist yet.

**Step 3: Write minimal implementation**

- Add VC lazy loader and conversion helper.
- Accept source/target temp-file paths from the endpoint/service layer.
- Reuse encoding helpers to return requested output format.

**Step 4: Run test to verify it passes**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -k voice_conversion -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py
git commit -m "feat: add chatterbox vc runtime support"
```

## Stage 5: Update Docs, UI Discovery, And Verification
**Goal**: Finish all user-facing surfaces and verify the touched scope.
**Success Criteria**: Docs/UI show the new TTS model family, VC is documented separately, tests pass, and Bandit is clean on touched paths.
**Tests**: Targeted pytest runs plus Bandit.
**Status**: In Progress

**TASK-531 slice update (2026-06-07)**: Added Chatterbox family ids to the UI fallback model catalog, corrected the frontend Chatterbox voice-reference sample-rate requirement to 24000 Hz, and recorded targeted pytest/Vitest/Bandit verification. Broader docs and VC user-facing coverage remain outstanding.

**TASK-532 slice update (2026-06-07)**: Refreshed the Chatterbox setup runbook and TTS module README for standard, multilingual, and Turbo model behavior. Dedicated VC user-facing docs remain outstanding until that endpoint/runtime slice exists.

**TASK-533/TASK-534 slice update (2026-06-08)**: Documented the dedicated Chatterbox VC endpoint in the setup runbook and TTS module README. The endpoint is intentionally separate from `/api/v1/audio/speech` and `chatterbox-vc` remains outside the TTS model alias catalog.

**TASK-540 slice update (2026-06-08)**: Added an authenticated `POST /api/v1/audio/tts/providers/{provider}/unload` route for releasing cached heavy TTS runtimes, with Chatterbox as the primary operational use case. The route closes one cached adapter and lets the next request reload it on demand.

**TASK-542 slice update (2026-06-08)**: Exposed the provider unload route through the frontend TTS/voice service layer via `unloadTtsProvider()`, kept the fallback capability spec and strict client path metadata aligned with the new backend route, and added focused Vitest coverage for the helper and route metadata.

**TASK-543 slice update (2026-06-08)**: Added typed `TldwApiClient.unloadTtsProvider()` support in the transitional base client and `models-audio` domain, with ownership-guard coverage so future domain cleanup keeps the provider unload route accounted for.

**TASK-544 slice update (2026-06-08)**: Aligned backend `VoiceManager` Chatterbox voice-reference processing requirements with the upstream 24 kHz sample rate already used by `AudioProcessor`, frontend voice requirements, and setup docs.

**TASK-545 slice update (2026-06-08)**: Hardened `VoiceManager` provider normalization so same-format voice uploads are only copied when ffprobe confirms the sample rate already matches the provider target; otherwise uploads are normalized through ffmpeg, covering Chatterbox WAV references that need 24 kHz resampling.

### Task 8: Update frontend discovery and voice requirements

**Files:**
- Modify: `apps/packages/ui/src/services/tldw/audio-models.ts`
- Modify: `apps/packages/ui/src/services/tldw/voice-cloning.ts`
- Modify: `Docs/STT-TTS/CHATTERBOX_SETUP.md`
- Modify: `tldw_Server_API/app/core/TTS/TTS-README.md`

**Step 1: Write the failing test**

Add or update a small test/snapshot around Chatterbox model fallback ids if a test exists nearby; if no test exists, add one in the most local UI service test area.

```ts
it("includes canonical chatterbox tts model ids", async () => {
  const models = await fetchTldwTtsModels()
  expect(models.some((m) => m.id === "chatterbox-turbo")).toBe(true)
  expect(models.some((m) => m.id === "chatterbox-multilingual")).toBe(true)
})
```

**Step 2: Run test to verify it fails**

Run the nearest existing frontend test command for the touched scope.

```bash
bunx vitest run apps/packages/ui/src/services/tldw
```

Expected: FAIL or missing coverage until the fallback list/docs are updated.

**Step 3: Write minimal implementation**

- Add canonical Chatterbox TTS ids to the frontend fallback list.
- Update voice requirement copy to stop presenting one hardcoded Chatterbox contract for every family.
- Document VC as a separate endpoint/feature.

**Step 4: Run test to verify it passes**

```bash
bunx vitest run apps/packages/ui/src/services/tldw
```

Expected: PASS.

**Step 5: Commit**

```bash
git add apps/packages/ui/src/services/tldw/audio-models.ts apps/packages/ui/src/services/tldw/voice-cloning.ts Docs/STT-TTS/CHATTERBOX_SETUP.md tldw_Server_API/app/core/TTS/TTS-README.md
git commit -m "docs: expose updated chatterbox family and vc support"
```

### Task 9: Verify the touched scope

**Files:**
- Test: `tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py`
- Test: `tldw_Server_API/tests/TTS/test_tts_validation.py`
- Test: `tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py`
- Test: nearest touched frontend tests

**Step 1: Run targeted backend tests**

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py \
  tldw_Server_API/tests/TTS/test_tts_validation.py \
  tldw_Server_API/tests/TTS_NEW/integration/test_chatterbox_voice_conversion_endpoint.py -v
```

Expected: PASS.

**Step 2: Run touched frontend tests**

```bash
bunx vitest run apps/packages/ui/src/services/tldw
```

Expected: PASS.

**Step 3: Run Bandit on touched backend paths**

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py \
  tldw_Server_API/app/core/TTS/tts_validation.py \
  tldw_Server_API/app/api/v1/endpoints/audio \
  -f json -o /tmp/bandit_chatterbox_upstream_parity.json
```

Expected: no new findings in touched code.

**Step 4: Review diff**

```bash
git diff --stat
git diff -- tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py
git diff -- tldw_Server_API/app/api/v1/endpoints/audio
```

Expected: only intended Chatterbox parity changes in touched files.

### 2026-06-08 slice update: Chatterbox validation formats

Backlog task: `TASK-529`

- Aligned central Chatterbox request validation with the adapter's advertised output formats by allowing `opus`, `flac`, and `pcm` alongside `wav` and `mp3`.
- Added focused validation coverage for Chatterbox FLAC/PCM requests and provider-limits metadata.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/test_tts_validation.py tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`.

### 2026-06-08 slice update: Chatterbox family capability metadata

Backlog task: `TASK-530`

- Added adapter capability metadata for discoverable Chatterbox family support: Original/Emotion, Multilingual language codes, Turbo tags, and the voice-conversion endpoint.
- Kept top-level `supported_languages` behavior unchanged so existing clients still see the configured runtime default.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`.

### 2026-06-08 slice update: Chatterbox voice conditionals cache

Backlog task: `TASK-531`

- Added an in-process adapter cache for upstream `prepare_conditionals()` output keyed by model family, reference-audio hash, and exaggeration.
- Reused cached conditionals on repeated reference-audio requests and omitted `audio_prompt_path` once conditionals are prepared, preserving fallback behavior for runtimes without `prepare_conditionals()`.
- Cleared cached conditionals on adapter close/cleanup.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`.

### 2026-06-08 slice update: Chatterbox conditionals cache bounds

Backlog task: `TASK-533`

- Replaced the plain in-process conditionals cache with a bounded LRU cache.
- Added `chatterbox_conditionals_cache_size` / `conditionals_cache_size` with a default of 16 entries; setting it to 0 disables retention while preserving per-request `prepare_conditionals()` use.
- Refreshed the Chatterbox setup runbook and provider YAML so operators can tune the cache explicitly.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`, `python -m pytest tldw_Server_API/tests/TTS/test_tts_default_policy.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task533.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox VC upload bounds

Backlog task: `TASK-532`

- Replaced unbounded multipart reads in the Chatterbox voice-conversion endpoint with chunked temp-file materialization capped at 50 MiB per upload/payload.
- Return `413` and clean partial temp files when a source or target voice-conversion upload exceeds the limit.
- Verified with `python -m pytest tldw_Server_API/tests/TTS_NEW/integration/test_tts_endpoints.py -k "chatterbox_voice_conversion" -v`.

### 2026-06-08 slice update: Chatterbox VC response-format docs

Backlog task: `TASK-534`

- Corrected the setup runbook to list only the response formats accepted by the current voice-conversion endpoint: `wav`, `mp3`, `flac`, `opus`, `aac`, and `pcm`.
- Added the 50 MiB per-upload cap to the runbook note next to the voice-conversion request examples.
- Verified with `git diff --check`.

### 2026-06-08 slice update: Chatterbox provider config aliases

Backlog task: `TASK-535`

- Expanded provider config normalization for Chatterbox YAML settings so generic keys are duplicated to adapter-prefixed `chatterbox_*` keys.
- Covered family selection, standard/multilingual/Turbo/VC model paths, auto-download, conditionals cache size, and generation default controls.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/test_tts_adapters.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapter_registry.py -f json -o /tmp/bandit_tts_adapter_registry_task535.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox target latency config

Backlog task: `TASK-536`

- Made `ChatterboxAdapter` honor `chatterbox_target_latency_ms` / `target_latency_ms` instead of always using the hardcoded 200 ms progressive streaming hint.
- Added positive-integer coercion so invalid or non-positive values fall back to 200 ms.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task536.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox generation-default config hardening

Backlog task: `TASK-537`

- Made generation default numeric settings parse defensively with prefixed-key precedence and unprefixed aliases.
- Invalid numeric values for default exaggeration, CFG weight, temperature, repetition penalty, min-p, or top-p now fall back to conservative defaults instead of raising during adapter construction.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task537.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox streaming voice-reference cleanup

Backlog task: `TASK-538`

- Deferred cleanup of temporary Chatterbox voice-reference files for streaming TTS responses until the returned audio stream is consumed or closed.
- Preserved immediate temp-file cleanup for non-streaming requests and error paths before a streaming response is returned.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task538.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox stream chunk latency

Backlog task: `TASK-539`

- Applied `target_latency_ms` / `chatterbox_target_latency_ms` to the actual `stream_encoded_waveform()` chunk duration for Chatterbox TTS streams.
- Reused the same configured chunk duration for Chatterbox voice-conversion streams so response streaming behavior matches capability metadata.
- Verified with `python -m pytest tldw_Server_API/tests/TTS/adapters/test_chatterbox_adapter_mock.py -v`, `python -m bandit -r tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py -f json -o /tmp/bandit_chatterbox_adapter_task539.json`, and `git diff --check`.

### 2026-06-08 slice update: Chatterbox streaming latency docs

Backlog task: `TASK-540`

- Refreshed the Chatterbox setup runbook so streaming chunk guidance describes `target_latency_ms` / `chatterbox_target_latency_ms` as configurable, with 200 ms as the default.
- Clarified that both TTS and voice-conversion streaming use the configured chunk duration.
- Verified with `rg -n "200ms|0\\.2s|~200" Docs/STT-TTS/CHATTERBOX_SETUP.md` and `git diff --check`.

### 2026-06-08 slice update: Chatterbox BF16 inference

Backlog task: `TASK-541`

- Added opt-in BF16 mode resolution for Chatterbox TTS via `chatterbox_use_bf16`, `use_bf16`, or `TTS_BF16=off|on|auto`, with the default remaining off.
- When BF16 is enabled, the adapter prepares the Chatterbox T3 module with `torch.bfloat16` and wraps TTS generation in `torch.autocast` when available.
- Added provider-config alias coverage for generic `use_bf16` and refreshed the provider YAML/setup runbook.

### 2026-06-08 slice update: Chatterbox split_text/chunk_size aliases

Backlog task: `TASK-542`

- Mapped upstream-style `extra_params.split_text` into the existing service-level chunking enable flag.
- Mapped upstream-style `extra_params.chunk_size` to the service chunk target/max character settings while preserving existing `chunking_service` / `chunking` precedence.
- Documented that these aliases apply to non-streaming long-text Chatterbox requests so the service can assemble generated PCM segments into one encoded response.

### 2026-06-08 slice update: Chatterbox speed_factor pass-through

Backlog task: `TASK-543`

- Added safe `speed_factor` candidate kwargs for standard, multilingual, and Turbo Chatterbox generation.
- Explicit `extra_params.speed_factor` takes precedence; otherwise non-default OpenAI-compatible `speed` is offered as `speed_factor`.
- The existing runtime signature filter still drops `speed_factor` for installed Chatterbox versions that do not support it.

### 2026-06-08 slice update: Chatterbox generation-control capability metadata

Backlog task: `TASK-544`

- Added discoverable Chatterbox capability metadata for standard/multilingual generation controls, Turbo controls, speed-factor request fields, service chunking aliases, and BF16 modes.
- Kept the top-level `supports_speech_rate` flag false because Chatterbox speed changes are runtime-conditional metadata, not a guaranteed generic speech-rate capability.
- Refreshed the Chatterbox setup runbook to point operators at the expanded provider capability metadata.

### 2026-06-08 slice update: Chatterbox predefined voice alias

Backlog task: `TASK-545`

- Added safe upstream-style `extra_params.voice_mode="predefined"` plus `extra_params.predefined_voice_id` support for Chatterbox speech requests.
- The alias resolves through the authenticated user's stored custom voice manager and leaves arbitrary `reference_audio_filename` values untouched.
- Refreshed the OpenAI speech schema and Chatterbox setup runbook with the supported mapping and security boundary.

### 2026-06-08 slice update: Chatterbox output_format alias

Backlog task: `TASK-546`

- Added optional OpenAI speech request `output_format` as a Chatterbox-family compatibility alias for upstream clients.
- The alias applies only when `response_format` was omitted; explicit `response_format` keeps precedence, and non-Chatterbox request conversion remains unchanged.
- Refreshed schema docs and the Chatterbox setup runbook with the alias behavior.

### 2026-06-08 slice update: Chatterbox language alias

Backlog task: `TASK-547`

- Added optional OpenAI speech request `language` as a Chatterbox-family compatibility alias for upstream multilingual clients.
- The alias applies only when `lang_code` was omitted; explicit `lang_code` keeps precedence, and non-Chatterbox request conversion remains unchanged.
- Refreshed schema docs and the Chatterbox setup runbook with the alias behavior.

### 2026-06-08 slice update: OpenAI-style provider voice catalog

Backlog task: `TASK-548`

- Added opt-in `format=openai` support to `GET /api/v1/audio/voices/catalog` so clients can receive flattened provider voice discovery as `{ "object": "list", "data": [...] }`.
- Kept the default provider-to-voices catalog response unchanged and left `GET /api/v1/audio/voices` as the authenticated custom voice list.
- Documented the Chatterbox voice-discovery mapping in the setup runbook.

### 2026-06-08 slice update: TTS provider model-info

Backlog task: `TASK-549`

- Added `GET /api/v1/audio/tts/providers/{provider}/model-info` for focused provider status, loaded state, supported model IDs, family metadata, voice-conversion metadata, and unload route discovery.
- Return HTTP 404 for unknown providers instead of an empty model-info payload.
- Sanitized model-info values from capabilities/status data so provider config secrets and local filesystem paths are not exposed.

**Step 5: Commit**

```bash
git add tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py tldw_Server_API/app/core/TTS/tts_validation.py tldw_Server_API/app/api/v1/endpoints/audio tldw_Server_API/app/api/v1/schemas/audio_schemas.py apps/packages/ui/src/services/tldw/audio-models.ts apps/packages/ui/src/services/tldw/voice-cloning.ts Docs/STT-TTS/CHATTERBOX_SETUP.md tldw_Server_API/app/core/TTS/TTS-README.md pyproject.toml tldw_Server_API/Config_Files/tts_providers_config.yaml
git commit -m "feat: add full chatterbox upstream parity"
```
