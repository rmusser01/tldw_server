# Hunyuan OCR GGUF Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `ocr_backend=hunyuan` so it supports a native Hunyuan family plus a Hunyuan-specific GGUF llama.cpp family with `remote`, `managed`, and `cli` runtime modes while preserving the existing OCR request and structured-output contract.

**Architecture:** Keep `HunyuanOCRBackend` as the only public Hunyuan backend and turn it into a thin orchestrator. Add a dedicated Hunyuan GGUF runtime helper for the llama.cpp-specific family, add a family-aware auto-eligibility hook to the OCR registry, tighten native-family readiness rules so `auto` does not get stuck on importable Transformers dependencies, and expose namespaced discovery metadata for the new dual-family backend without changing the PDF pipeline contract.

**Tech Stack:** Python 3, FastAPI, Pydantic, Loguru, existing OCR runtime helpers, pytest, Bandit, Markdown docs

---

## File Structure

### OCR Contract And Registry

- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/base.py`
  - Add an explicit default class hook for registry auto-selection so backends can override generic `auto` / `auto_high_quality` participation without hardcoded name checks.
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/registry.py`
  - Replace the current backend-name-specific `llamacpp` / `chatllm` auto-eligibility logic with the backend hook while preserving explicit backend selection behavior and the existing registry ordering.

### Shared OCR Runtime Parsing

- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py`
  - Add a small explicit-key loader or equivalent helper so Hunyuan GGUF can reuse the existing profile parsing, safe argv rendering, and process-lifecycle helpers without being forced into the current `<PREFIX>_OCR_*` env shape.

### Hunyuan GGUF Runtime Helper

- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_llamacpp_runtime.py`
  - Own Hunyuan GGUF family configuration parsing, native-vs-llamacpp family resolution helpers, llama.cpp `remote` / `managed` / `cli` execution, prompt transport details, structured-output parsing helpers, and sanitized discovery metadata for the GGUF family.

### Public Hunyuan Backend

- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py`
  - Keep Hunyuan as the public backend, add runtime-family orchestration, preserve prompt presets and structured OCR normalization, implement stricter native-family readiness rules, and delegate GGUF execution to the new runtime helper.

### Discovery Endpoint And Schema

- Modify: `tldw_Server_API/app/api/v1/schemas/ocr_schemas.py`
  - Expand the discovery schema for Hunyuan’s dual-family metadata while keeping compatibility with the current endpoint shape.
- Modify: `tldw_Server_API/app/api/v1/endpoints/ocr.py`
  - Publish namespaced Hunyuan discovery metadata and keep the existing top-level discovery contract stable enough for current callers.

### Tests

- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py`
  - Unit tests for Hunyuan GGUF family env parsing, remote/managed/cli mode resolution, safe argv handling, and local-only availability rules.
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py`
  - Unit tests for family selection, stricter native readiness, explicit backend behavior, prompt preset mapping, GGUF structured-output parsing, and Hunyuan-specific auto-eligibility behavior.
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py`
  - Extend registry coverage for the new backend hook and Hunyuan family-aware `auto` / `auto_high_quality` behavior.
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py`
  - Assert the new Hunyuan namespaced discovery payload and backward-compatible top-level fields.
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py`
  - Cover the new explicit-key runtime-profile loading path if `runtime_support.py` is generalized.
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py`
  - Focused PDF-pipeline coverage for `ocr_backend=hunyuan` with both native-family and GGUF-family metadata propagation.

### Docs

- Modify: `Docs/OCR/HunyuanOCR.md`
  - Document the new family split, env surface, operator guidance, and coexistence with the generic `llamacpp` backend.
- Modify: `Docs/OCR/OCR_Providers.md`
  - Update backend overview, examples, and operator guidance for Hunyuan GGUF.
- Modify: `Docs/API-related/OCR_API_Documentation.md`
  - Document the expanded Hunyuan discovery payload and example ingestion requests.
- Modify: `Docs/Operations/Env_Vars.md`
  - Add the new `HUNYUAN_RUNTIME_FAMILY` and `HUNYUAN_LLAMACPP_*` variables.

## Implementation Notes

- Preserve `ocr_backend=hunyuan` as the only public Hunyuan selector. Do not introduce a second public backend such as `hunyuan_llamacpp`.
- Keep `ocr_backend=llamacpp` intact as the generic llama.cpp OCR backend. Do not alias it to Hunyuan behavior.
- Preserve explicit backend selection semantics: `get_backend("hunyuan")` must bypass generic auto-eligibility gating exactly as current explicit backend selection does.
- Tighten native-family readiness for Hunyuan family resolution:
  - native `vllm` should count as available when `HUNYUAN_VLLM_URL` is configured
  - native `transformers` should count as family-available only with explicit operator intent, not merely importable Python dependencies
- Preserve the public OCR contract:
  - `ocr_output_format`
  - `ocr_prompt_preset`
  - `OCRResult`
  - `analysis_details.ocr.structured`
- Keep GGUF `auto` mode order `remote -> managed -> cli` unless implementation evidence proves the order must change for correctness.
- Reuse `runtime_support.py` helpers instead of re-implementing argv parsing, managed process lifecycle, or readiness checks inside the Hunyuan runtime helper.
- Preserve sanitized PDF output. Never surface raw argv values, local paths, ports, or prompt internals in `analysis_details.ocr` or `/api/v1/ocr/backends`.
- Keep top-level Hunyuan discovery fields backward-compatible where possible:
  - `available`
  - `mode` or equivalent effective mode
  - `configured`
  - `prompt_preset`
  - `backend_concurrency_cap`
  Add namespaced `native` and `llamacpp` sub-objects rather than replacing the current flat shape outright.
- Bandit is required before claiming the implementation complete because the touched scope is Python.

### Task 1: Add The Registry Hook And Family-Aware Auto-Selection Contract

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/base.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/registry.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py`
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py`

- [ ] **Step 1: Write the failing registry and explicit-selection tests**

Add tests that prove:

- `OCRBackend` exposes a default `auto_eligible(high_quality: bool) -> bool` hook returning `True`
- `registry.get_backend("auto")` consults the backend hook instead of hardcoded backend-name checks
- `registry.get_backend("auto_high_quality")` passes `high_quality=True`
- explicit selection `get_backend("hunyuan")` still bypasses the hook
- Hunyuan-specific tests can force `auto_eligible(False)` and `auto_eligible(True)` independently

```python
@pytest.mark.unit
def test_registry_auto_uses_backend_auto_eligible_hook(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.auto_eligible",
        classmethod(lambda cls, high_quality: False),
    )

    assert get_backend("auto") is None


@pytest.mark.unit
def test_registry_explicit_hunyuan_selection_bypasses_auto_eligible(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.auto_eligible",
        classmethod(lambda cls, high_quality: False),
    )

    backend = get_backend("hunyuan")
    assert backend is not None
    assert backend.name == "hunyuan"
```

- [ ] **Step 2: Run the focused registry tests to verify the current code fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py -k "auto_eligible or explicit_hunyuan"
```

Expected:

- FAIL because `OCRBackend` does not define the hook yet
- FAIL because `registry.py` still hardcodes only `llamacpp` and `chatllm`

- [ ] **Step 3: Implement the minimal registry hook**

Make the smallest change set that satisfies the new tests:

- add the default class hook to `OCRBackend`
- update `registry.py` to call the hook for generic `auto` / `auto_high_quality`
- preserve explicit backend selection behavior unchanged
- preserve current backend ordering and config-based priority rules

```python
class OCRBackend(ABC):
    @classmethod
    def auto_eligible(cls, high_quality: bool) -> bool:
        return True
```

```python
def _backend_auto_eligible(cls: type[OCRBackend], *, high_quality: bool) -> bool:
    try:
        return bool(cls.auto_eligible(high_quality))
    except AttributeError:
        return True
```

- [ ] **Step 4: Re-run the registry-focused tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py -k "auto_eligible or explicit_hunyuan"
```

Expected:

- PASS for the new hook coverage
- PASS for the existing `llamacpp` / `chatllm` auto-selection coverage

- [ ] **Step 5: Commit the registry-contract slice**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/base.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/registry.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py
git commit -m "feat(ocr): add family-aware auto eligibility hook"
```

### Task 2: Add Shared Runtime Parsing For Explicit-Key Hunyuan GGUF Config

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py`
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py`

- [ ] **Step 1: Write the failing explicit-key runtime-profile tests**

Add tests that prove:

- Hunyuan GGUF can load `remote`, `managed`, and `cli` profiles from explicit keys such as:
  - `HUNYUAN_LLAMACPP_MODE`
  - `HUNYUAN_LLAMACPP_MODEL`
  - `HUNYUAN_LLAMACPP_MODEL_PATH`
  - `HUNYUAN_LLAMACPP_SERVER_ARGV`
  - `HUNYUAN_LLAMACPP_CLI_ARGV`
- remote model uses a logical model identifier instead of a local path
- managed and CLI use separate argv surfaces

```python
@pytest.mark.unit
def test_load_ocr_runtime_profiles_from_explicit_keys_parses_hunyuan_llamacpp_env():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.runtime_support import (
        load_ocr_runtime_profiles_from_keys,
    )

    env = {
        "HUNYUAN_LLAMACPP_MODE": "managed",
        "HUNYUAN_LLAMACPP_ALLOW_MANAGED_START": "true",
        "HUNYUAN_LLAMACPP_HOST": "127.0.0.1",
        "HUNYUAN_LLAMACPP_PORT": "19092",
        "HUNYUAN_LLAMACPP_MODEL_PATH": "ggml-org/HunyuanOCR-GGUF:Q8_0",
        "HUNYUAN_LLAMACPP_SERVER_ARGV": '["llama-server", "-hf", "{model_path}", "--port", "{port}"]',
        "HUNYUAN_LLAMACPP_MAX_PAGE_CONCURRENCY": "2",
    }

    profiles = load_ocr_runtime_profiles_from_keys(
        env=env,
        mode_key="HUNYUAN_LLAMACPP_MODE",
        allow_managed_start_key="HUNYUAN_LLAMACPP_ALLOW_MANAGED_START",
        max_page_concurrency_key="HUNYUAN_LLAMACPP_MAX_PAGE_CONCURRENCY",
        host_key="HUNYUAN_LLAMACPP_HOST",
        port_key="HUNYUAN_LLAMACPP_PORT",
        model_path_key="HUNYUAN_LLAMACPP_MODEL_PATH",
        argv_key="HUNYUAN_LLAMACPP_SERVER_ARGV",
    )

    assert profiles.active.mode == "managed"
    assert profiles.active.argv[0] == "llama-server"
```

- [ ] **Step 2: Run the runtime-support tests to verify the current code fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py -k "explicit_keys or hunyuan_llamacpp"
```

Expected:

- FAIL because `runtime_support.py` only understands `<PREFIX>_OCR_*`
- FAIL because no Hunyuan GGUF runtime helper exists yet

- [ ] **Step 3: Implement the minimal shared parsing helper**

Add the smallest reusable API to `runtime_support.py` that avoids duplicating profile parsing:

- keep `load_ocr_runtime_profiles(prefix)` intact for current backends
- add a new explicit-key loader or equivalent wrapper for Hunyuan GGUF
- keep safe argv rendering and local-only availability semantics unchanged

```python
def load_ocr_runtime_profiles_from_keys(
    *,
    env: Mapping[str, Any] | None = None,
    mode_key: str,
    allow_managed_start_key: str,
    max_page_concurrency_key: str,
    host_key: str | None = None,
    port_key: str | None = None,
    model_path_key: str | None = None,
    argv_key: str | None = None,
    prompt_key: str | None = None,
) -> OCRRuntimeProfiles:
    ...
```

- [ ] **Step 4: Re-run the runtime-support tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py -k "explicit_keys or hunyuan_llamacpp"
```

Expected:

- PASS for the explicit-key loader tests
- PASS for the existing runtime-support tests

- [ ] **Step 5: Commit the shared parsing slice**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py
git commit -m "feat(ocr): add explicit-key OCR runtime profile parsing"
```

### Task 3: Build The Hunyuan GGUF Runtime Helper And Hunyuan Family Orchestration

**Files:**
- Create: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_llamacpp_runtime.py`
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py`
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py`

- [ ] **Step 1: Write the failing Hunyuan-family orchestration tests**

Add tests that prove:

- `HUNYUAN_RUNTIME_FAMILY=native` uses only native family
- `HUNYUAN_RUNTIME_FAMILY=llamacpp` uses only GGUF family
- `HUNYUAN_RUNTIME_FAMILY=auto` prefers configured native runtime but does not treat importable Transformers deps alone as enough to block GGUF
- GGUF `auto` order is `remote -> managed -> cli`
- prompt presets still map to `general|doc|table|spotting|json`
- JSON-like GGUF outputs become normalized `OCRResult` values
- explicit `ocr_backend=hunyuan` still works when Hunyuan GGUF auto-eligibility would be false

```python
@pytest.mark.unit
def test_hunyuan_auto_family_prefers_native_vllm_when_configured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "auto")
    monkeypatch.setenv("HUNYUAN_MODE", "auto")
    monkeypatch.setenv("HUNYUAN_VLLM_URL", "http://127.0.0.1:9000/v1/chat/completions")

    description = HunyuanOCRBackend().describe()

    assert description["runtime_family"] == "native"
    assert description["effective_mode"] == "vllm"


@pytest.mark.unit
def test_hunyuan_auto_family_falls_back_to_llamacpp_when_native_is_unconfigured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "auto")
    monkeypatch.setenv("HUNYUAN_MODE", "auto")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODE", "remote")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_HOST", "127.0.0.1")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_PORT", "19092")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODEL", "ggml-org/HunyuanOCR-GGUF:Q8_0")

    description = HunyuanOCRBackend().describe()

    assert description["runtime_family"] == "llamacpp"
    assert description["effective_mode"] == "remote"
```

- [ ] **Step 2: Run the Hunyuan backend tests to verify the current code fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py
```

Expected:

- FAIL because Hunyuan has no runtime family orchestration yet
- FAIL because no Hunyuan llama.cpp runtime helper exists yet

- [ ] **Step 3: Implement the minimal runtime helper and Hunyuan backend changes**

Create the helper and update `hunyuan_ocr.py` to orchestrate families:

- parse `HUNYUAN_RUNTIME_FAMILY`
- compute stricter native-family readiness
- delegate GGUF execution and discovery to the helper
- keep the current prompt preset map authoritative in `hunyuan_ocr.py`
- preserve best-effort structured parsing
- add `HunyuanOCRBackend.auto_eligible(...)` using the resolved family rules from the approved design

```python
class HunyuanOCRBackend(OCRBackend):
    @classmethod
    def auto_eligible(cls, high_quality: bool) -> bool:
        family = _resolve_runtime_family_for_auto()
        if family == "llamacpp":
            env_name = (
                "HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE"
                if high_quality
                else "HUNYUAN_LLAMACPP_AUTO_ELIGIBLE"
            )
            return _env_bool(env_name, False)
        return True
```

Guardrails:

- do not delete the current native Hunyuan code paths
- do not let `auto` silently retry the other family after a runtime failure
- keep GGUF remote payloads OpenAI-compatible and Hunyuan-specific
- validate and freeze text/image content ordering in tests

- [ ] **Step 4: Re-run the Hunyuan backend tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py -k "hunyuan or auto_eligible"
```

Expected:

- PASS for Hunyuan family orchestration
- PASS for Hunyuan GGUF runtime helper tests
- PASS for registry integration with the new hook

- [ ] **Step 5: Commit the Hunyuan backend slice**

```bash
git add \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_llamacpp_runtime.py \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py
git commit -m "feat(ocr): add Hunyuan GGUF runtime orchestration"
```

### Task 4: Publish Namespaced Hunyuan Discovery Metadata

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/ocr_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/ocr.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py`

- [ ] **Step 1: Write the failing discovery-schema and endpoint tests**

Add tests that prove:

- `OCRBackendDiscoveryEntry` accepts namespaced `native` and `llamacpp` sub-objects
- Hunyuan discovery still includes the top-level backward-compatible fields:
  - `available`
  - `mode`
  - `configured`
  - `prompt_preset`
- Hunyuan discovery also includes:
  - `runtime_family`
  - `configured_runtime_family`
  - `native`
  - `llamacpp`
- ambiguous flat-only booleans such as `url_configured` are not the only source of truth anymore

```python
def test_list_ocr_backends_enriches_hunyuan_discovery_with_namespaced_runtime_families(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    class _StubHunyuanBackend:
        def describe(self):
            return {
                "mode": "remote",
                "configured": True,
                "runtime_family": "llamacpp",
                "configured_runtime_family": "auto",
                "prompt_preset": "json",
                "native": {"mode": "vllm", "configured": False, "url_configured": False},
                "llamacpp": {
                    "mode": "remote",
                    "configured": True,
                    "url_configured": True,
                    "model": "ggml-org/HunyuanOCR-GGUF:Q8_0",
                },
            }

    monkeypatch.setattr(ocr_mod, "_list_backends", lambda: {"hunyuan": {"available": True}})
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend",
        _StubHunyuanBackend,
    )

    payload = ocr_mod.list_ocr_backends()
    assert payload["hunyuan"]["runtime_family"] == "llamacpp"
    assert payload["hunyuan"]["llamacpp"]["url_configured"] is True
```

- [ ] **Step 2: Run the discovery tests to verify the current code fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py
```

Expected:

- FAIL because the schema does not yet support namespaced Hunyuan runtime metadata
- FAIL because the endpoint does not emit family-specific Hunyuan discovery details

- [ ] **Step 3: Implement the minimal schema and endpoint changes**

Expand the schema and endpoint while preserving compatibility:

- add a nested runtime-family discovery model
- keep current top-level fields for existing callers
- map Hunyuan’s effective mode to top-level `mode`
- publish family-specific metadata under `native` and `llamacpp`

```python
class OCRRuntimeFamilyDiscoveryEntry(BaseModel):
    mode: Optional[str] = None
    configured: Optional[bool] = None
    url_configured: Optional[bool] = None
    managed_configured: Optional[bool] = None
    managed_running: Optional[bool] = None
    allow_managed_start: Optional[bool] = None
    cli_configured: Optional[bool] = None
    auto_eligible: Optional[bool] = None
    auto_high_quality_eligible: Optional[bool] = None
    model: Optional[str] = None
    model_path: Optional[str] = None


class OCRBackendDiscoveryEntry(BaseModel):
    ...
    runtime_family: Optional[str] = None
    configured_runtime_family: Optional[str] = None
    native: Optional[OCRRuntimeFamilyDiscoveryEntry] = None
    llamacpp: Optional[OCRRuntimeFamilyDiscoveryEntry] = None
```

- [ ] **Step 4: Re-run the discovery tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py
```

Expected:

- PASS for the Hunyuan discovery coverage
- PASS for the existing `llamacpp` / `chatllm` discovery assertions

- [ ] **Step 5: Commit the discovery slice**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/ocr_schemas.py \
  tldw_Server_API/app/api/v1/endpoints/ocr.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py
git commit -m "feat(ocr): add namespaced Hunyuan discovery metadata"
```

### Task 5: Prove PDF Pipeline Integration And Update Operator Docs

**Files:**
- Create: `tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py`
- Modify: `Docs/OCR/HunyuanOCR.md`
- Modify: `Docs/OCR/OCR_Providers.md`
- Modify: `Docs/API-related/OCR_API_Documentation.md`
- Modify: `Docs/Operations/Env_Vars.md`

- [ ] **Step 1: Write the failing PDF-pipeline and doc-contract tests**

Add focused Hunyuan pipeline tests that prove:

- `ocr_backend=hunyuan` with native family preserves `analysis_details["ocr"]["runtime_family"] == "native"`
- `ocr_backend=hunyuan` with GGUF family preserves `analysis_details["ocr"]["runtime_family"] == "llamacpp"`
- `analysis_details["ocr"]["structured"]` remains populated for Hunyuan GGUF
- sanitized metadata omits private argv, host, port, and raw model-path details from the PDF output

```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_task_attaches_hunyuan_llamacpp_runtime_metadata(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    class _StubHunyuanBackend:
        name = "hunyuan"

        @classmethod
        def available(cls) -> bool:
            return True

        def describe(self):
            return {
                "mode": "remote",
                "configured": True,
                "runtime_family": "llamacpp",
                "llamacpp": {"mode": "remote", "url_configured": True},
                "backend_concurrency_cap": 2,
            }

    ...
    assert result["analysis_details"]["ocr"]["runtime_family"] == "llamacpp"
    assert result["analysis_details"]["ocr"]["structured"]["pages"][0]["raw"]["page"] == 1
```

- [ ] **Step 2: Run the focused pipeline tests to verify the current code fails**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py
```

Expected:

- FAIL because the current Hunyuan backend does not emit runtime-family metadata into the PDF pipeline

- [ ] **Step 3: Update the docs and finalize the pipeline expectations**

Update the source docs only:

- `Docs/OCR/HunyuanOCR.md`
  - new `HUNYUAN_RUNTIME_FAMILY`
  - new `HUNYUAN_LLAMACPP_*` surface
  - operator guidance for `hunyuan` vs `llamacpp`
- `Docs/OCR/OCR_Providers.md`
  - Hunyuan GGUF overview and coexistence rules
- `Docs/API-related/OCR_API_Documentation.md`
  - Hunyuan discovery payload examples
- `Docs/Operations/Env_Vars.md`
  - exact env var catalog

Keep the documentation explicit that:

- native family is still supported
- GGUF family is native-first fallback or explicit family selection
- `ocr_backend=llamacpp` remains a separate generic backend

- [ ] **Step 4: Re-run the pipeline tests and the focused doc-adjacent OCR tests**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_llamacpp_chatllm_pdf_pipeline.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py -k "hunyuan or llamacpp or chatllm"
```

Expected:

- PASS for the new Hunyuan pipeline coverage
- PASS for the existing llama.cpp / ChatLLM pipeline coverage
- PASS for discovery regressions

- [ ] **Step 5: Commit the pipeline-and-docs slice**

```bash
git add \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py \
  Docs/OCR/HunyuanOCR.md \
  Docs/OCR/OCR_Providers.md \
  Docs/API-related/OCR_API_Documentation.md \
  Docs/Operations/Env_Vars.md
git commit -m "docs(ocr): document Hunyuan GGUF runtime support"
```

### Task 6: Run Final Verification, Security Checks, And Handoff

**Files:**
- Verify only; no new files required unless the verification uncovers gaps

- [ ] **Step 1: Run the full focused Hunyuan OCR test slice**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_llamacpp_runtime.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_hunyuan_ocr_pdf_pipeline.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py
```

Expected:

- PASS for all Hunyuan-focused runtime, registry, discovery, and PDF-pipeline tests

- [ ] **Step 2: Run a regression slice for adjacent OCR backends**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_llamacpp_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_chatllm_ocr_backend.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_llamacpp_chatllm_pdf_pipeline.py
```

Expected:

- PASS to show the Hunyuan work did not regress the existing generic `llamacpp` or `chatllm` OCR backends

- [ ] **Step 3: Run Bandit on the touched Python scope**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/OCR \
  tldw_Server_API/app/api/v1/endpoints/ocr.py \
  tldw_Server_API/app/api/v1/schemas/ocr_schemas.py \
  -f json -o /tmp/bandit_hunyuan_ocr_gguf.json
```

Expected:

- JSON report written to `/tmp/bandit_hunyuan_ocr_gguf.json`
- no new high-confidence findings in the touched code

- [ ] **Step 4: Summarize verification results and document any residual risk**

Record:

- exact pytest commands run
- Bandit result
- whether Hunyuan GGUF remote/managed/cli runtime behavior is covered entirely by mocks or by any opt-in real-runtime test
- any remaining follow-up work that is intentionally deferred

- [ ] **Step 5: Commit the final verification or cleanup changes if needed**

```bash
git add -A
git commit -m "test(ocr): verify Hunyuan GGUF runtime integration"
```

## Local Review Notes

- This plan intentionally keeps the implementation bounded to the OCR subsystem. It does not attempt to generalize all multimodal runtime management across the codebase.
- The biggest correctness risk is family selection. The first implementation task after the registry hook should keep failing tests around native-family readiness until the Hunyuan backend no longer treats importable Transformers deps as enough to block GGUF fallback.
- The biggest operator-clarity risk is discovery output. Do not ship the new dual-family `hunyuan` backend until `/api/v1/ocr/backends` makes it obvious whether `hunyuan` is using native or GGUF family and which subfamily configuration is actually active.
