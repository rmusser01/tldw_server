# Fish S2 Commercial API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add hosted Fish Audio commercial S2 support behind the existing `fish_s2` TTS provider.

**Architecture:** Keep `fish_s2` as the public provider key and add a `commercial_api` backend alongside the existing `native_http` backend. The adapter delegates backend-specific HTTP contracts while `TTSServiceV2` preserves user-scoped local voice metadata and remote Fish model IDs.

**Tech Stack:** FastAPI, pytest, existing `tldw_Server_API.app.core.http_client` helpers, Loguru, existing TTS adapter/service abstractions.

---

### Task 1: Commercial Backend TTS Contract

**Files:**
- Create: `tldw_Server_API/app/core/TTS/backends/fish_s2_commercial_api.py`
- Modify: `tldw_Server_API/app/core/TTS/backends/fish_s2_base.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_commercial_api_backend.py`

- [x] **Step 1: Write failing tests**

Add tests that assert `FishS2CommercialApiBackend.synthesize()`:

```python
backend = FishS2CommercialApiBackend({
    "base_url": "https://api.fish.audio",
    "api_key": "secret",
    "model": "s2-pro",
    "timeout": 30,
})

audio = await backend.synthesize(
    text="hello",
    response_format="mp3",
    streaming=False,
    reference_id="voice-model-id",
    extra_params={
        "sample_rate": 44100,
        "mp3_bitrate": 128,
        "latency": "balanced",
        "prosody": {"speed": 1.2, "volume": 0},
    },
)
```

Expected request:

```python
method == "POST"
url == "https://api.fish.audio/v1/tts"
headers == {
    "Authorization": "Bearer secret",
    "Content-Type": "application/json",
    "model": "s2-pro",
}
json == {
    "text": "hello",
    "format": "mp3",
    "reference_id": "voice-model-id",
    "sample_rate": 44100,
    "mp3_bitrate": 128,
    "latency": "balanced",
    "prosody": {"speed": 1.2, "volume": 0},
}
```

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_commercial_api_backend.py
```

Expected: import failure for missing `fish_s2_commercial_api.py`.

- [x] **Step 3: Implement minimal backend**

Implement:

- config normalization for `base_url`, `api_key`, `model`, `timeout`
- `health_check()` requiring configured base URL and API key
- `_headers()`
- `_build_tts_payload()`
- `synthesize()` using `afetch` and `astream_bytes`
- status/error mapping using existing TTS exceptions

- [x] **Step 4: Run test to verify it passes**

Run the same test file and expect pass.

### Task 2: Commercial Voice Model Creation

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/backends/fish_s2_commercial_api.py`
- Modify: `tldw_Server_API/app/core/TTS/backends/fish_s2_base.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_commercial_api_backend.py`

- [x] **Step 1: Write failing tests**

Add tests for `add_reference()` that assert it posts multipart data to `/model`:

```python
await backend.add_reference(
    reference_id="tldw_u1_voice-1",
    audio_b64="QUJD",
    reference_text="hello there",
    title="Voice One",
    description="private clone",
)
```

Expected:

```python
url == "https://api.fish.audio/model"
data includes type="tts", title="Voice One", train_mode="fast", visibility="private", texts="hello there"
files includes voices=("reference.wav", b"ABC", "audio/wav")
result["reference_id"] == returned["_id"]
```

- [x] **Step 2: Run test to verify it fails**

Expected: method signature or behavior mismatch.

- [x] **Step 3: Implement minimal creation support**

Update protocol and both backends to accept optional metadata. Commercial backend returns `{"reference_id": remote_id, "remote_reference_id": remote_id, ...}`.

- [x] **Step 4: Run backend tests**

Expected: commercial backend tests pass and native backend tests still pass.

### Task 3: Adapter Backend Selection And Format Capabilities

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/adapters/fish_s2_adapter.py`
- Modify: `tldw_Server_API/app/core/TTS/tts_validation.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_adapter.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_fish_s2.py`

- [x] **Step 1: Write failing tests**

Add tests that:

- `backend="commercial_api"` builds `FishS2CommercialApiBackend`
- Fish capabilities include `opus`
- commercial pass-through includes `sample_rate`, `mp3_bitrate`, `opus_bitrate`, `latency`, `max_new_tokens`, `min_chunk_length`, `condition_on_previous_chunks`, `early_stop_threshold`, and `prosody`

- [x] **Step 2: Run tests to verify failure**

Expected: unknown backend or missing pass-through fields.

- [x] **Step 3: Implement adapter updates**

Update `_build_backend()`, supported formats, capabilities, and pass-through list.

- [x] **Step 4: Run adapter/validation tests**

Expected: Fish adapter/validation tests pass.

### Task 4: Service Metadata Flow

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/tts_service_v2.py`
- Test: `tldw_Server_API/tests/TTS_NEW/unit/test_tts_service.py`

- [x] **Step 1: Write failing tests**

Add a test where adapter `add_reference()` returns a hosted ID different from the deterministic local ID:

```python
return {"reference_id": "fish-hosted-model-id", "remote_reference_id": "fish-hosted-model-id"}
```

Expected stored metadata:

```python
metadata.provider_artifacts["fish_s2"]["remote_reference_id"] == "fish-hosted-model-id"
```

- [x] **Step 2: Run test to verify failure**

Expected: current code stores deterministic `tldw_u...` instead of returned hosted ID.

- [x] **Step 3: Implement service update**

Use backend return payload when present; fall back to deterministic ID for `native_http`.

- [x] **Step 4: Run service Fish tests**

Run:

```bash
python -m pytest -q tldw_Server_API/tests/TTS_NEW/unit/test_tts_service.py -k fish
```

Expected: pass.

### Task 5: Config And Docs

**Files:**
- Modify: `tldw_Server_API/app/core/TTS/tts_config.py`
- Modify: `tldw_Server_API/Config_Files/tts_providers_config.yaml`
- Modify: `Docs/STT-TTS/TTS-SETUP-GUIDE.md`
- Test: focused config test if existing pattern is present, otherwise backend config tests cover env use.

- [x] **Step 1: Write/update tests**

Add coverage that `FISH_AUDIO_API_KEY` and `FISH_API_KEY` map to `providers.fish_s2.api_key`.

- [x] **Step 2: Run test to verify failure**

Expected: env var is ignored before implementation.

- [x] **Step 3: Implement config/docs**

Document both modes:

- `backend: commercial_api` for hosted Fish Audio
- `backend: native_http` for self-hosted Fish Speech

- [x] **Step 4: Run config/docs-adjacent tests**

Run focused backend/config tests.

### Task 6: Final Verification

**Files:**
- All touched files.

- [x] **Step 1: Run focused tests**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_adapter.py \
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_native_http_backend.py \
  tldw_Server_API/tests/TTS_NEW/unit/adapters/test_fish_s2_commercial_api_backend.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_fish_s2_registry.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_validation_fish_s2.py \
  tldw_Server_API/tests/TTS_NEW/unit/test_tts_service.py -k fish \
  tldw_Server_API/tests/TTS_NEW/integration/test_fish_s2_reference_endpoints.py
```

- [x] **Step 2: Run Bandit on touched Python scope**

```bash
python -m bandit -r \
  tldw_Server_API/app/core/TTS/backends \
  tldw_Server_API/app/core/TTS/adapters/fish_s2_adapter.py \
  tldw_Server_API/app/core/TTS/tts_service_v2.py \
  tldw_Server_API/app/core/TTS/tts_config.py \
  tldw_Server_API/app/core/TTS/tts_validation.py \
  tldw_Server_API/app/api/v1/endpoints/audio/audio_voices.py \
  -f json -o /tmp/bandit_fish_s2_commercial_api.json
```

- [x] **Step 3: Update Backlog task**

Record tests, Bandit result, final summary, and known limitations.
