# Hunyuan OCR GGUF Runtime Design

Date: 2026-04-19
Status: Proposed
Owner: Codex brainstorming session

## Summary

Extend the existing `ocr_backend=hunyuan` backend so it can run through two internal runtime families:

- native Hunyuan execution:
  - `vllm`
  - `transformers`
- GGUF Hunyuan execution through `llama.cpp`:
  - `remote`
  - `managed`
  - `cli`

The public API does not change. Clients continue to request OCR with the existing media and evaluation fields such as:

- `enable_ocr`
- `ocr_backend=hunyuan`
- `ocr_mode`
- `ocr_output_format`
- `ocr_prompt_preset`

Operators opt into the GGUF path through a new `HUNYUAN_LLAMACPP_*` configuration surface. The backend keeps its current prompt presets and structured OCR contract, and `auto` prefers the current native Hunyuan stack before falling back to the GGUF path.

This design deliberately keeps `hunyuan` as the single user-facing backend name. It does not add a second public backend selector for GGUF Hunyuan.

## Problem

The repository already ships a `hunyuan` OCR backend, but today it supports only two runtime families:

- OpenAI-compatible `vllm`
- local `transformers`

That leaves a gap for operators who want to run the newer GGUF version of HunyuanOCR locally through `llama.cpp`.

This gap matters because upstream support is now real and recent:

- `llama.cpp` merged HunyuanOCR support in PR `#21395` on April 5, 2026.
- `ggml-org/HunyuanOCR-GGUF` is published on Hugging Face and references that PR.
- ggml-org's April 10, 2026 article shows the intended server pattern: `llama-server -hf ggml-org/HunyuanOCR-GGUF` with OpenAI-compatible `/v1/chat/completions` requests.

Without first-class support in `tldw_server`, users who prefer GGUF deployment currently have three bad options:

1. abandon `ocr_backend=hunyuan` and use another backend that does not match the model they want
2. build an external compatibility layer and pretend it is generic OCR infrastructure
3. patch the server locally without a stable contract for prompts, discovery, or structured OCR output

The repository also already has proven OCR runtime patterns in:

- `llamacpp_ocr.py`
- `chatllm_ocr.py`
- `runtime_support.py`

The missing piece is not whether the server can manage remote, managed, and CLI OCR runtimes. The missing piece is how to expose GGUF Hunyuan in a way that preserves the current `hunyuan` backend contract instead of introducing another parallel backend name.

## Goals

- Keep `ocr_backend=hunyuan` as the only public selector for Hunyuan OCR.
- Add an internal GGUF runtime family for Hunyuan based on `llama.cpp`.
- Support `remote`, `managed`, and `cli` execution for the GGUF family in v1.
- Preserve the existing native Hunyuan family:
  - `HUNYUAN_MODE=auto|vllm|transformers`
  - `HUNYUAN_VLLM_*`
  - `HUNYUAN_MODEL_PATH`, `HUNYUAN_DEVICE`, and related vars
- Add a new `HUNYUAN_LLAMACPP_*` operator-facing config surface for the GGUF family.
- Keep `HUNYUAN_RUNTIME_FAMILY=auto` native-first:
  1. native family first
  2. GGUF family second
- Preserve existing prompt presets:
  - `general`
  - `doc`
  - `table`
  - `spotting`
  - `json`
- Preserve the current OCR output contract:
  - `OCRResult`
  - `text|markdown|json`
  - `analysis_details.ocr.structured`
- Expose enough discovery metadata through `GET /api/v1/ocr/backends` for operators to understand which family and mode are active.
- Reuse proven OCR runtime support patterns where practical instead of inventing a new multimodal runtime framework.

## Non-Goals

- Add a new public backend name such as `hunyuan_llamacpp`.
- Replace `ocr_backend=hunyuan` with `ocr_backend=llamacpp`.
- Merge the native and GGUF configuration surfaces into one overloaded env namespace.
- Introduce request-level overrides for:
  - host
  - port
  - model path
  - binary path
  - managed startup policy
- Build a generic multimodal runtime layer for all vision features.
- Refactor unrelated OCR backends just because they use similar lifecycle code.
- Change the PDF pipeline contract for structured OCR persistence.
- Guarantee exact OCR quality parity between native Hunyuan and GGUF Hunyuan.
- Make `auto` silently retry another family after a real inference failure. Family fallback is availability-based, not output-quality-based.

## User-Confirmed Decisions

The following design choices were explicitly confirmed during brainstorming:

- Public backend shape:
  - extend `ocr_backend=hunyuan`
  - do not introduce a second public backend name
- GGUF runtime support in v1:
  - `remote + managed + cli`
- `HUNYUAN_RUNTIME_FAMILY=auto` preference:
  - native family first
  - GGUF family second
- Operator-facing config split:
  - keep existing `HUNYUAN_*` vars for native family
  - introduce new `HUNYUAN_LLAMACPP_*` vars for GGUF family
- Prompt and output behavior:
  - preserve existing presets
  - preserve structured output contract

## Current State

### Existing Public Backend Contract

The OCR subsystem already uses a stable backend abstraction:

- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/base.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/registry.py`

The PDF pipeline and OCR evaluation flows resolve a backend instance and then call:

- `ocr_image(...)`
- `ocr_image_structured(...)`
- `describe()` where available

That means the new GGUF capability can fit without changing the public OCR pipeline, as long as the `hunyuan` backend still satisfies the same contract.

### Existing Hunyuan Backend

The current `HunyuanOCRBackend` already supports:

- native `vllm`
- local `transformers`

It already understands:

- prompt presets
- basic text and markdown output
- best-effort structured JSON normalization

Its current configuration is centered on:

- `HUNYUAN_MODE`
- `HUNYUAN_VLLM_*`
- `HUNYUAN_MODEL_PATH`
- `HUNYUAN_DEVICE`

### Existing OCR Runtime Patterns

The repository already has strong precedent for server-owned OCR runtimes through:

- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/llamacpp_ocr.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/chatllm_ocr.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/runtime_support.py`

Those modules already solve the hard operational pieces:

- config parsing
- local-only availability checks
- JSON argv rendering without shell interpolation
- managed process registry and reuse
- managed HTTP readiness checks
- remote OpenAI-compatible request assembly
- safe CLI invocation
- backend-local concurrency caps

The proposed Hunyuan GGUF work should reuse those ideas directly instead of re-deriving them.

## Approaches Considered

### Approach 1: Public `hunyuan` Backend, Internal Family Orchestrator

Keep `ocr_backend=hunyuan` public. Turn `HunyuanOCRBackend` into a thin orchestrator that chooses between:

- native family
- GGUF llama.cpp family

Internally, the GGUF path is delegated to a dedicated helper or helper module.

Pros:

- preserves the public API and docs
- matches the user-requested operator config split
- keeps backward compatibility for existing native Hunyuan deployments
- lets the implementation reuse existing OCR runtime patterns
- keeps Hunyuan-specific prompt and output semantics in one public backend

Cons:

- requires an extra family-selection layer
- `describe()` becomes richer and slightly more complex

### Approach 2: Expand `hunyuan_ocr.py` In Place With All GGUF Logic

Keep one file and embed native plus GGUF runtime logic directly in `hunyuan_ocr.py`.

Pros:

- fewer files
- fewer imports

Cons:

- grows a mixed-responsibility backend file quickly
- makes testing harder
- makes native and GGUF concerns harder to separate later

### Approach 3: Separate Public GGUF Backend Name

Add a new backend such as `hunyuan_llamacpp` and let users choose that explicitly.

Pros:

- simplest internal logic
- clean operational separation

Cons:

- contradicts the user-approved design
- creates duplicate Hunyuan behavior in the public OCR registry
- makes docs and backend choice more confusing

## Recommendation

Use Approach 1, implemented with the internal separation discipline of Approach 2 avoided and Approach 3 kept internal-only.

Concretely:

- `HunyuanOCRBackend` remains the public backend.
- Family selection happens inside `HunyuanOCRBackend`.
- GGUF llama.cpp behavior lives in a dedicated helper module or internal runner so native and GGUF concerns stay isolated.

## Proposed Architecture

### Public Boundary

The public OCR surface remains unchanged:

- clients request `ocr_backend=hunyuan`
- the backend returns `OCRResult`
- the PDF pipeline persists `analysis_details.ocr` and `analysis_details.ocr.structured`

No request field is added for runtime family or llama.cpp mode.

### Internal Structure

Recommended structure:

- `tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends/hunyuan_ocr.py`
  - public backend
  - family selection
  - shared prompt preset logic
  - final result normalization
- new helper module, likely near the OCR backends:
  - Hunyuan GGUF llama.cpp runtime resolution
  - remote/managed/cli execution
  - GGUF-specific `describe()` metadata
  - GGUF prompt adaptation and response parsing

The backend should be split conceptually into:

- family selection
- runtime execution
- OCR result normalization

That keeps the backend understandable even after adding three GGUF runtime modes.

## Configuration Design

### Family Selector

Add:

- `HUNYUAN_RUNTIME_FAMILY=auto|native|llamacpp`

Default:

- `auto`

Behavior:

- `native`
  - only native family is eligible
  - use current `HUNYUAN_MODE`
- `llamacpp`
  - only GGUF family is eligible
  - use `HUNYUAN_LLAMACPP_MODE`
- `auto`
  - try native family first
  - if native family is not locally available, try GGUF family

This separation is intentional. It avoids overloading `HUNYUAN_MODE` with both native and GGUF concepts.

### Native Family

The current native surface remains intact:

- `HUNYUAN_MODE=auto|vllm|transformers`
- `HUNYUAN_VLLM_URL`
- `HUNYUAN_VLLM_MODEL`
- `HUNYUAN_VLLM_TIMEOUT`
- `HUNYUAN_VLLM_USE_DATA_URL`
- `HUNYUAN_MODEL_PATH`
- `HUNYUAN_MODEL_REVISION`
- `HUNYUAN_DEVICE`
- generation and post-processing vars already in use

Existing native operators should not have to change config to keep the backend working.

### GGUF Family

Add a new GGUF configuration surface:

- `HUNYUAN_LLAMACPP_MODE=auto|remote|managed|cli`
- `HUNYUAN_LLAMACPP_AUTO_ELIGIBLE=true|false`
- `HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE=true|false`
- `HUNYUAN_LLAMACPP_MAX_PAGE_CONCURRENCY=<int>`

Remote mode:

- `HUNYUAN_LLAMACPP_HOST`
- `HUNYUAN_LLAMACPP_PORT`
- `HUNYUAN_LLAMACPP_MODEL`
- `HUNYUAN_LLAMACPP_USE_DATA_URL=true|false`
- `HUNYUAN_LLAMACPP_TIMEOUT`
- `HUNYUAN_LLAMACPP_TEMPERATURE`
- `HUNYUAN_LLAMACPP_MAX_TOKENS`

Managed mode:

- `HUNYUAN_LLAMACPP_ALLOW_MANAGED_START=true|false`
- `HUNYUAN_LLAMACPP_HOST`
- `HUNYUAN_LLAMACPP_PORT`
- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_SERVER_ARGV`
- `HUNYUAN_LLAMACPP_STARTUP_TIMEOUT_SEC`

CLI mode:

- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_CLI_ARGV`

`HUNYUAN_LLAMACPP_SERVER_ARGV` and `HUNYUAN_LLAMACPP_CLI_ARGV` must be JSON argv arrays and must continue to use safe placeholder replacement rather than shell interpolation.

## Runtime Resolution

### Family Resolution Rules

Family resolution should be deterministic:

1. Read `HUNYUAN_RUNTIME_FAMILY`.
2. If `native`, resolve only native family.
3. If `llamacpp`, resolve only GGUF family.
4. If `auto`, attempt native family first.
5. Only if native family is not locally available, attempt GGUF family.

Important rule:

- `auto` family fallback is availability-based, not a second-pass retry after a failed inference call.

This keeps behavior predictable and avoids hiding real runtime failures behind implicit cross-family fallbacks.

Native-first family resolution must be stricter than the current broad `available()` heuristic used by the native Hunyuan backend. For family selection, "native available" should mean a configured native runtime, not merely that `transformers`, `torch`, and `PIL` are importable on the host.

Recommended native-family readiness rules:

- native `vllm` is family-available when `HUNYUAN_VLLM_URL` is configured
- native `transformers` is family-available only when the operator has explicitly selected it, for example through:
  - `HUNYUAN_MODE=transformers`
  - or an explicit native-runtime opt-in variable added during implementation
- if `HUNYUAN_MODE=auto`, the presence of importable native Python dependencies alone must not prevent GGUF family fallback

This avoids a bad default where any machine with general ML dependencies installed would silently stay on native family and never use the configured GGUF runtime.

### Native-First Auto

User-confirmed behavior is:

- preserve current `hunyuan` semantics first
- add GGUF as a fallback family

That means a system with a configured native runtime should keep using native Hunyuan unless the operator explicitly selects `HUNYUAN_RUNTIME_FAMILY=llamacpp`.

### GGUF Mode Resolution

Inside the GGUF family:

- `remote`
- `managed`
- `cli`
- `auto`

should follow the same basic mode semantics as current `llamacpp` OCR:

- explicit mode forces one mode
- `auto` chooses the first configured and locally eligible mode
- managed startup remains explicit and operator-controlled

Recommended default order for GGUF family `auto`:

1. `remote` if a remote profile is configured
2. `managed` if a managed profile is configured and allowed
3. `cli` if a CLI profile is configured

This is intentionally slightly safer than the current `llamacpp` OCR backend default. For a new Hunyuan GGUF namespace, preferring an already configured remote endpoint over auto-starting a local managed process reduces surprise for operators while still keeping managed mode available when they explicitly choose it or when remote is not configured.

## Prompt And Output Contract

### Prompt Presets

The public `hunyuan` backend keeps the existing presets:

- `general`
- `doc`
- `table`
- `spotting`
- `json`

The GGUF family should not collapse these down to only `"OCR"`.

Instead, it should map the current presets to llama.cpp-compatible prompts that are conservative and close to upstream guidance:

- `general`
  - minimal extraction prompt
- `doc`
  - markdown-preserving document extraction
- `table`
  - table-focused faithful extraction
- `spotting`
  - JSON-oriented prompt for spans and bounding boxes
- `json`
  - JSON-oriented prompt for full structured output

The minimal upstream example uses `OCR`, but this backend already promises richer behavior. The design should preserve that contract rather than narrow it.

### Multimodal Payload Shape

For GGUF remote and managed server modes, requests should use the OpenAI-compatible multimodal chat payload already supported by `llama.cpp`.

The exact ordering of `text` and `image_url` content should follow the backend-specific integration choice and be validated in tests. Upstream examples currently show:

- text prompt
- image

The backend should use the ordering that actually works for HunyuanOCR in `llama.cpp` and keep it consistent across remote and managed modes.

### Structured Output

The GGUF family must preserve the existing OCR structured output contract:

- return normalized `OCRResult`
- support `text|markdown|json`
- write `analysis_details.ocr.structured` in the PDF pipeline

Parsing should remain best-effort:

- if JSON parsing succeeds, preserve blocks and text in structured form
- if JSON parsing fails, return plain text/markdown and record raw output where appropriate

This matches the broader OCR backend pattern already in the repo.

## Discovery And Metadata

`GET /api/v1/ocr/backends` should continue to expose `hunyuan` as a single backend entry, but with richer details.

Recommended top-level fields:

- `runtime_family`
- `configured_runtime_family`
- `effective_mode`
- `configured`
- `backend_concurrency_cap`
- `prompt_preset`

Recommended namespaced family fields:

- `native`
  - `mode`
  - `configured`
  - `url_configured`
  - `transformers_configured`
  - `model`
  - `model_path`
- `llamacpp`
  - `mode`
  - `configured`
  - `auto_eligible`
  - `auto_high_quality_eligible`
  - `url_configured`
  - `managed_configured`
  - `managed_running`
  - `allow_managed_start`
  - `cli_configured`
  - `model`
  - `model_path`
  - `backend_concurrency_cap`

If nested family objects are awkward for the current endpoint contract, the fallback is explicit prefixes such as:

- `native_url_configured`
- `native_mode`
- `llamacpp_url_configured`
- `llamacpp_managed_configured`

What should be avoided is a single ambiguous field such as `url_configured` once both families exist.

Sanitization rules:

- do not return secrets
- do not return unsanitized argv arrays if they include sensitive data
- do not expose internal-only values that are not useful to operators

## Registry Auto-Eligibility

The current OCR registry hardcodes auto-eligibility exceptions only for `llamacpp` and `chatllm`. That is not sufficient for this design because `hunyuan` remains a single public backend while gaining a second internal runtime family.

The design therefore needs an explicit registry change rather than only new environment variables.

### Recommended Registry Improvement

Add an optional backend class hook for registry auto-selection, for example:

- `auto_eligible(high_quality: bool) -> bool`

Default behavior for backends that do not implement the hook:

- `True`

Registry behavior:

- for explicit backend selection such as `ocr_backend=hunyuan`, bypass this hook exactly as current explicit selection bypasses backend auto-eligibility gates
- for generic OCR registry `auto` and `auto_high_quality`, consult the backend hook instead of hardcoding backend-name checks

This keeps the registry from accumulating more backend-name-specific conditionals and makes Hunyuan family-aware auto selection possible.

### Hunyuan-Specific Auto-Eligibility Rules

`HunyuanOCRBackend.auto_eligible(...)` should be family-aware:

- if the resolved family is `native`, preserve current Hunyuan participation in generic OCR `auto` and `auto_high_quality`
- if the resolved family is `llamacpp`, generic OCR `auto` participation requires `HUNYUAN_LLAMACPP_AUTO_ELIGIBLE=true`
- if the resolved family is `llamacpp`, generic OCR `auto_high_quality` participation requires `HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE=true`
- if `HUNYUAN_RUNTIME_FAMILY=auto`, the hook should evaluate the same native-first family resolution used at execution time and apply the matching rule for the family that would actually satisfy availability

This keeps three behaviors aligned:

1. explicit `ocr_backend=hunyuan` always works if the backend is available
2. native Hunyuan keeps current generic auto behavior
3. GGUF Hunyuan does not unexpectedly start participating in generic OCR auto flows without an explicit operator opt-in

## Coexistence With The Generic `llamacpp` Backend

The repository already has a public generic OCR backend:

- `ocr_backend=llamacpp`

This design does not remove or replace it.

The intended distinction after this change is:

- `ocr_backend=hunyuan`
  - model-specific backend semantics
  - Hunyuan-specific prompt presets and structured-output normalization
  - native Hunyuan family plus Hunyuan GGUF family
- `ocr_backend=llamacpp`
  - generic llama.cpp OCR backend
  - not specific to Hunyuan
  - remains appropriate for operators who want a model-agnostic llama.cpp OCR surface

Operator guidance:

- if you are running `ggml-org/HunyuanOCR-GGUF` and you want Hunyuan-specific semantics, configure and use `ocr_backend=hunyuan`
- if you want a generic llama.cpp OCR surface independent of model-specific semantics, use `ocr_backend=llamacpp`
- do not point both public backend surfaces at the same deployment unless you intentionally need both behaviors and understand that discovery output and generic OCR auto selection will treat them as separate backends

This keeps the new Hunyuan GGUF support from turning into a de facto alias for the existing generic llama.cpp backend.

## PDF Pipeline Impact

The PDF pipeline should require minimal change because it already relies on the generic OCR backend contract.

Expected behavior stays the same:

1. resolve backend
2. call structured OCR if available
3. normalize into `OCRResult`
4. persist `analysis_details.ocr`
5. optionally replace or append OCR content according to `ocr_mode`

The only meaningful pipeline-level change should be better `describe()` metadata coming from the `hunyuan` backend, including family and mode information and any backend-specific concurrency cap.

## Error Handling

### Availability Versus Runtime Failure

Availability rules should stay local-only and cheap:

- remote mode available if required remote config exists
- managed mode available if managed config exists and startup is allowed or a managed process is already running
- CLI mode available if required CLI config exists

Do not perform expensive remote reachability checks inside `available()`.

### Family Fallback Rules

If `HUNYUAN_RUNTIME_FAMILY=native`:

- never fall back to GGUF

If `HUNYUAN_RUNTIME_FAMILY=llamacpp`:

- never fall back to native

If `HUNYUAN_RUNTIME_FAMILY=auto`:

- use GGUF only if native is unavailable during selection
- do not silently retry GGUF after a native inference failure
- do not silently retry native after a GGUF inference failure

This preserves debuggability and operator intent.

### Managed Mode Constraints

Managed GGUF mode should inherit the same v1 limitation already used by current OCR-managed runtimes:

- single-process ownership only

Multi-worker or shared-host deployments should use:

- `remote`
- or `cli`

### Failure Reporting

When OCR fails, metadata should remain useful:

- record backend name as `hunyuan`
- include `runtime_family`
- include effective mode
- include sanitized error text

That keeps user-visible and operator-visible behavior consistent even when the runtime path changes underneath.

## Testing Strategy

### Unit Tests

Add or extend tests for:

- family resolution:
  - `native`
  - `llamacpp`
  - `auto` native-first
- GGUF mode resolution:
  - `remote`
  - `managed`
  - `cli`
  - `auto`
- local-only availability semantics for GGUF family
- managed lifecycle behavior for Hunyuan GGUF runtime
- remote request payload construction
- prompt preset mapping for GGUF family
- structured JSON parsing and fallback behavior
- `describe()` output shape and sanitization
- Hunyuan family-aware `auto_eligible(...)` behavior

### Registry Tests

Add coverage proving:

- `ocr_backend=hunyuan` remains a single backend name
- Hunyuan generic OCR `auto` participation remains unchanged for native family
- Hunyuan GGUF family does not unexpectedly change generic OCR `auto`
- `auto_high_quality` participation for Hunyuan GGUF is controlled by explicit Hunyuan GGUF eligibility flags

### Pipeline Tests

Add or extend PDF OCR tests proving:

- `ocr_backend=hunyuan` still writes stable `analysis_details.ocr`
- structured OCR survives both native and GGUF families
- concurrency metadata reflects backend-local caps for the GGUF family

### Docs And Contract Tests

Update docs and tests for:

- `Docs/OCR/HunyuanOCR.md`
- `Docs/OCR/OCR_Providers.md`
- `Docs/API-related/OCR_API_Documentation.md`
- `Docs/Operations/Env_Vars.md`
- any OCR backend discovery tests that assert backend metadata shape

## Rollout Plan

Recommended staged rollout:

1. Introduce internal family-selection plumbing and GGUF helper module.
2. Add GGUF remote mode first in code structure, but ship the full `remote + managed + cli` surface in the same feature wave once tests are in place.
3. Update discovery metadata and docs.
4. Keep native-first `auto` as the default.
5. Leave any future registry-level reprioritization of Hunyuan GGUF for a separate design.

## Risks

### Risk: Config Surface Drift

There is a real risk that `HUNYUAN_MODE` and `HUNYUAN_LLAMACPP_MODE` get conflated over time.

Mitigation:

- keep family and mode as distinct concepts in code and docs
- reflect both in `describe()`
- add tests that enforce native-first `auto`

### Risk: Growing `hunyuan_ocr.py` Into A Mixed Runtime File

Mitigation:

- keep GGUF runtime execution in a dedicated helper module
- keep the public backend focused on orchestration and normalization

### Risk: Implicit Cross-Family Retry Masks Real Failures

Mitigation:

- availability-based fallback only
- no hidden retry after runtime failure

### Risk: Discovery Metadata Becomes Inconsistent With Runtime

Mitigation:

- derive `describe()` fields from the same resolution helpers used by execution
- add tests for both configured mode and effective mode

## Open Questions

These are implementation details, not unresolved product questions:

- whether the GGUF helper should reuse `runtime_support.py` directly or through a tiny Hunyuan-specific wrapper
- the exact minimal prompt strings for each preset after validating `llama.cpp` HunyuanOCR behavior locally
- whether the GGUF helper should expose a dedicated timeout var for CLI separate from remote if current defaults prove too coarse

None of these questions change the approved design shape.

## Recommendation

Proceed with implementation planning for a native-first `hunyuan` backend that adds an internal GGUF `llama.cpp` family with:

- `remote`
- `managed`
- `cli`

using a new `HUNYUAN_LLAMACPP_*` operator surface and preserving the current prompt and structured output contract.
