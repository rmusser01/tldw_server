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
- `HUNYUAN_MODE=auto` preference:
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
- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_USE_DATA_URL=true|false`
- `HUNYUAN_LLAMACPP_TIMEOUT`
- `HUNYUAN_LLAMACPP_TEMPERATURE`
- `HUNYUAN_LLAMACPP_MAX_TOKENS`

Managed mode:

- `HUNYUAN_LLAMACPP_ALLOW_MANAGED_START=true|false`
- `HUNYUAN_LLAMACPP_HOST`
- `HUNYUAN_LLAMACPP_PORT`
- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_ARGV`
- `HUNYUAN_LLAMACPP_STARTUP_TIMEOUT_SEC`

CLI mode:

- `HUNYUAN_LLAMACPP_MODEL_PATH`
- `HUNYUAN_LLAMACPP_ARGV`

`HUNYUAN_LLAMACPP_ARGV` must be a JSON argv array and must continue to use safe placeholder replacement rather than shell interpolation.

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

### Native-First Auto

User-confirmed behavior is:

- preserve current `hunyuan` semantics first
- add GGUF as a fallback family

That means a system with a valid `HUNYUAN_VLLM_URL` or usable native transformers stack should keep using native Hunyuan unless the operator explicitly selects `HUNYUAN_RUNTIME_FAMILY=llamacpp`.

### GGUF Mode Resolution

Inside the GGUF family:

- `remote`
- `managed`
- `cli`
- `auto`

should work the same way current `llamacpp` OCR does:

- explicit mode forces one mode
- `auto` chooses the first configured and locally eligible mode
- managed startup remains explicit and operator-controlled

Recommended default order for GGUF family `auto`:

1. `managed` if a managed profile is configured and allowed
2. `remote` if a remote profile is configured
3. `cli` if a CLI profile is configured

This matches the existing OCR precedent that a server-owned reusable local runtime can be preferred when it is explicitly allowed.

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

Recommended fields:

- `runtime_family`
- `configured_runtime_family`
- `native_mode`
- `llamacpp_mode`
- `effective_mode`
- `configured`
- `auto_eligible`
- `auto_high_quality_eligible`
- `managed_configured`
- `managed_running`
- `allow_managed_start`
- `url_configured`
- `cli_configured`
- `backend_concurrency_cap`
- `prompt_preset`

Sanitization rules:

- do not return secrets
- do not return unsanitized argv arrays if they include sensitive data
- do not expose internal-only values that are not useful to operators

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

### Registry Tests

Add coverage proving:

- `ocr_backend=hunyuan` remains a single backend name
- GGUF family auto eligibility does not unexpectedly change generic `auto`
- `auto_high_quality` participation is controlled by explicit Hunyuan GGUF eligibility flags

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
