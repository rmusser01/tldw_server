# Embeddings Request Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved Embeddings request-orchestrator refactor behind `EMBEDDINGS_ORCHESTRATOR_ENABLED`, preserving current `/api/v1/embeddings` behavior while extracting request normalization, provider resolution, policy, cache planning, execution, and response metadata into core Embeddings modules.

**Architecture:** Keep FastAPI, AuthNZ, RBAC, ResourceGovernor, and billing at the endpoint boundary. Add pure core modules for request types, input normalization, provider/model resolution, policy decisions, and a fakeable `EmbeddingRequestOrchestrator` with explicit `prepare` and `execute` phases. Preserve endpoint helper shims while tests and callers migrate.

**Tech Stack:** Python 3.11, FastAPI, Pydantic schemas from the existing endpoint, numpy, existing Embeddings cache and provider execution helpers, pytest, pytest-asyncio, Loguru, Prometheus metric shims, Bandit.

---

## Source References

- Approved spec: `Docs/superpowers/specs/2026-06-24-embeddings-request-orchestrator-refactor-design.md`
- Backlog task for this plan: `TASK-12015`
- Primary endpoint: `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
- Existing helper tests: `tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py`
- Existing endpoint behavior tests: `tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py`, `tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py`, `tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py`, `tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py`

## File Structure

- Create `tldw_Server_API/app/core/Embeddings/request_types.py`
  - Owns internal dataclasses, stable domain exceptions, sanitized error fields, and response metadata instructions.
  - `EmbeddingRequestContext` must not contain raw request input text.
- Create `tldw_Server_API/app/core/Embeddings/input_normalizer.py`
  - Owns OpenAI-compatible `input` parsing, token-array decoding, token counting, per-item token limit errors, and batch-size validation.
- Create `tldw_Server_API/app/core/Embeddings/provider_resolution.py`
  - Owns provider-qualified model splitting, explicit provider handling, provider/model mismatch errors, default helper behavior, and current HuggingFace model-name heuristics.
- Create `tldw_Server_API/app/core/Embeddings/embedding_policy.py`
  - Owns dimensions validation, dimensions postprocessing policy, L2 policy, allowlist checks, admin bypass checks, unsupported-provider checks, fallback-chain decisions, fallback model mapping, and explicit-header fallback suppression.
- Create `tldw_Server_API/app/core/Embeddings/orchestrator.py`
  - Owns `EmbeddingRequestOrchestrator.prepare()` and `EmbeddingRequestOrchestrator.execute()`, fakeable cache/executor protocols, cache read/write planning, fallback execution, vector-count validation, and result metadata.
- Modify `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
  - Preserve old helper symbols as delegate shims.
  - Extract current create handler body into a legacy private function.
  - Add feature-flagged delegation to the orchestrator path.
  - Map core domain exceptions to existing HTTP status codes and response bodies.
- Create `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py`
  - Characterizes current endpoint and batch-helper behavior before extraction.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_request_types.py`
  - Unit tests for internal dataclasses and domain errors.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py`
  - Unit tests for input parsing and token accounting.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py`
  - Unit tests for provider/model resolution.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py`
  - Unit tests for dimensions, fallback, allowlists, unsupported providers, and L2 policy.
- Create `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`
  - Unit tests with fake cache and fake executor.
- Create `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`
  - Endpoint parity tests for old path versus feature-flagged orchestrator path.

## Data Contracts

Add these public internal contracts in `request_types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


EmbeddingErrorCode = Literal[
    "empty_input",
    "invalid_input_type",
    "too_many_inputs",
    "input_too_long",
    "invalid_token_array",
    "unknown_provider",
    "provider_model_mismatch",
    "invalid_dimensions",
    "provider_denied",
    "model_denied",
    "provider_unsupported",
    "missing_provider_credentials",
    "provider_malformed_response",
    "provider_rate_limited",
    "provider_unavailable",
    "fallback_exhausted",
]


class EmbeddingDomainError(Exception):
    def __init__(
        self,
        code: EmbeddingErrorCode,
        message: str,
        *,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
        retry_after: int | None = None,
        cause_class: str | None = None,
        details: list[dict[str, int]] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.cause_class = cause_class
        self.details = details or []


class EmbeddingInputError(EmbeddingDomainError):
    pass


class EmbeddingPolicyError(EmbeddingDomainError):
    pass


class EmbeddingProviderError(EmbeddingDomainError):
    pass


class EmbeddingRateLimitError(EmbeddingDomainError):
    pass


class EmbeddingExecutionError(EmbeddingDomainError):
    pass


@dataclass(frozen=True, slots=True)
class EmbeddingRequestContext:
    user_id: str | None
    model_field: str | None
    provider_header: str | None
    dimensions: int | None
    encoding_format: str
    request_id: str | None = None
    endpoint_path: str = "/api/v1/embeddings"
    testing: bool = False
    adapters_enabled: bool = False


@dataclass(frozen=True, slots=True)
class NormalizedEmbeddingInput:
    texts: list[str]
    token_counts: list[int]
    total_tokens: int
    provided_token_arrays: bool = False
    token_input_mode: Literal["none", "single", "batch"] = "none"


@dataclass(frozen=True, slots=True)
class ProviderModelIntent:
    provider: str
    model: str
    requested_provider: str
    requested_model: str
    provider_was_explicit: bool
    model_was_provider_qualified: bool


@dataclass(frozen=True, slots=True)
class EmbeddingPolicyDecision:
    provider: str
    model: str
    dimensions: int | None
    fallback_chain: list[str]
    fallback_allowed: bool
    enforce_policy: bool
    bypass_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingExecutionPlan:
    provider: str
    model: str
    dimensions: int | None
    backend_identity: str | None
    fallback_chain: list[str]
    cache_namespace: str | None = None
    batch_size: int | None = None
    execution_path: Literal["legacy", "adapter"] = "legacy"
    observability_tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EmbeddingExecutionResult:
    vectors: list[list[float]]
    provider: str
    model: str
    prompt_tokens: int
    total_tokens: int
    cache_hits: int
    cache_misses: int
    fallback_from: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
```

`EmbeddingExecutionPlan` and `EmbeddingRequestContext` must never contain request text, API keys, authorization headers, nonce secrets, or full provider error bodies. `NormalizedEmbeddingInput.texts` is allowed to contain the execution texts and must not be logged or serialized by plan/debug helpers.

## Implementation Tasks

### Task 1: Characterize Current Endpoint And Cache Behavior

- [ ] Add `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py`.
- [ ] Add a fixture that sets `TESTING=true`, clears `EMBEDDINGS_ORCHESTRATOR_ENABLED`, creates a FastAPI `TestClient`, sets CSRF and auth headers, and overrides `get_request_user` with an active non-admin user.
- [ ] Add `test_batch_full_cache_hit_skips_provider_and_preserves_order`:
  - Import `embeddings_v5_production_enhanced` as `mod`.
  - Patch `mod.embedding_cache.get` with `AsyncMock(side_effect=[[1.0, 0.0], [0.0, 1.0]])`.
  - Patch `mod.embedding_cache.set` with `AsyncMock()`.
  - Patch `mod.create_embeddings_with_circuit_breaker` with an async function that raises `AssertionError("provider should not be called on a full cache hit")`.
  - Call `await mod.create_embeddings_batch_async(["a", "b"], provider="huggingface", model_id="sentence-transformers/all-MiniLM-L6-v2")`.
  - Assert the result is `[[1.0, 0.0], [0.0, 1.0]]` and `embedding_cache.set.await_count == 0`.
- [ ] Add `test_batch_partial_cache_hit_executes_only_misses_and_writes_float_vectors`:
  - Patch `embedding_cache.get` so item 0 is cached and item 1 misses.
  - Patch `create_embeddings_with_circuit_breaker` to assert `texts == ["miss"]` and return `[[0.25, 0.75]]`.
  - Assert result order is preserved and `embedding_cache.set` receives a `list[float]`, not a base64 string.
- [ ] Add `test_endpoint_base64_response_does_not_write_base64_to_cache`:
  - Patch cache miss, fake provider vector, and `embedding_cache.set`.
  - POST `/api/v1/embeddings` with `x-provider: huggingface`, model `sentence-transformers/all-MiniLM-L6-v2`, input `"cache me"`, `encoding_format: "base64"`, and `dimensions: 2`.
  - Assert the response embedding decodes as two float32 values and `embedding_cache.set.await_args.args[1]` is a numeric list.
- [ ] Add `test_endpoint_dimension_adjustment_cache_write_order_is_characterized`:
  - Patch cache miss and fake a HuggingFace provider vector with four floats.
  - POST `/api/v1/embeddings` with `dimensions: 2` and numeric response format.
  - Assert response vector length is two.
  - Assert the current cache write vector length exactly as observed. If the old path stores pre-adjustment provider vectors, mark this as compatibility behavior in the test comment. The orchestrator path still follows the approved spec by writing postprocessed canonical float vectors behind the feature flag.
- [ ] Add `test_endpoint_full_cache_hit_still_reserves_and_commits_rg_tokens`:
  - Install fake `request.app.state.rg_governor` and `request.app.state.rg_policy_loader`.
  - Patch cache to full hit and provider execution to raise if called.
  - POST a HuggingFace request.
  - Assert reserve and commit were called with token units greater than zero. This preserves the current accounting order where token reservation happens before batch-helper cache knowledge.
- [ ] Add `test_endpoint_vector_count_mismatch_maps_to_502`:
  - Patch provider execution to return one fewer vector than requested.
  - Assert HTTP `502`.
- [ ] Run the characterization tests and the existing narrow helper tests:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_length_mismatch_raises \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_rate_limit_maps_to_429 \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_batch_generic_provider_error_is_sanitized \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_strips_prefix \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_rejects_mismatch
```

Expected result after this task is implemented: all selected tests pass. If any characterization fails, update the test expectation to match observed current behavior and add a short comment classifying it as contract or compatibility.

### Task 2: Add Internal Request Types And Domain Errors

- [ ] Add `request_types.py` with the contracts from this plan.
- [ ] Add `to_http_payload()` on `EmbeddingDomainError` returning sanitized dict fields only: `error_code`, `message`, `provider`, `model`, `retryable`, `retry_after`, `details`, and `cause_class`.
- [ ] Add `tldw_Server_API/tests/Embeddings_isolated/test_request_types.py`.
- [ ] Test `EmbeddingRequestContext` has no `raw_input`, `texts`, `input`, `api_key`, or `authorization` attributes when constructed with normal endpoint metadata.
- [ ] Test `EmbeddingExecutionPlan` serializes only provider, model, dimensions, backend identity, fallback chain, cache namespace, batch size, execution path, and low-cardinality tags.
- [ ] Test `EmbeddingDomainError.to_http_payload()` does not include exception `__cause__`, raw provider body strings, or arbitrary attributes.
- [ ] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_request_types.py
```

Expected result before implementation: import failure for `tldw_Server_API.app.core.Embeddings.request_types`. Expected result after implementation: all request-type tests pass.

### Task 3: Extract Input Normalization

- [ ] Add `input_normalizer.py`.
- [ ] Implement `normalize_embedding_input(raw_input, *, model, max_tokens, count_tokens, tokens_to_texts) -> NormalizedEmbeddingInput`.
- [ ] Preserve current invalid input messages through stable error codes:
  - Empty string: `EmbeddingInputError("empty_input", "Input cannot be empty")`.
  - Empty list: `EmbeddingInputError("empty_input", "Input list cannot be empty")`.
  - Mixed list: `EmbeddingInputError("invalid_input_type", "Invalid input type")`.
  - More than 2048 inputs: `EmbeddingInputError("too_many_inputs", "Maximum 2048 inputs allowed")`.
  - Token-array decode failure: `EmbeddingInputError("invalid_token_array", "Invalid token array input")`.
  - Token limit: `EmbeddingInputError("input_too_long", "One or more inputs exceed max tokens {max_tokens} for model {model}", details=[{"index": i, "tokens": tok}])`.
- [ ] Add `tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py`.
- [ ] Cover string, list of strings, single token array, batch token arrays, empty inputs, mixed inputs, list-size limit, and token-limit details.
- [ ] In endpoint code, keep existing `tokens_to_texts`, `count_tokens`, and `_get_model_max_tokens` helpers until all callers migrate. The first endpoint integration should call the new normalizer only on the orchestrator path.
- [ ] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py
```

Expected result after implementation: all normalizer tests pass and no FastAPI imports are required by `input_normalizer.py`.

### Task 4: Extract Provider And Model Resolution

- [ ] Add `provider_resolution.py`.
- [ ] Move provider-qualified model parsing into `split_provider_model(model: str) -> tuple[str | None, str]`.
- [ ] Add `resolve_provider_model(model, provider_header, *, settings_config, require_model) -> ProviderModelIntent`.
- [ ] Preserve resolution order:
  - Explicit `x-provider` wins and is lowercased.
  - Provider-qualified model IDs strip the provider prefix.
  - Prefix mismatch with explicit provider raises `EmbeddingPolicyError("provider_model_mismatch", "Model provider prefix 'openai' does not match provider 'huggingface'")`.
  - With no explicit provider and no prefix, use current HuggingFace heuristics, then `guess_provider_for_model`, then current defaults for helper compatibility.
  - Endpoint create path uses `require_model=True`; helper shim `_resolve_model_and_provider` uses `require_model=False` to preserve its current default-model behavior.
- [ ] Add `tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py`.
- [ ] Cover explicit provider, provider-qualified model, mismatch, HuggingFace heuristic patterns, OpenAI model default, and compatibility default when `model is None` and `require_model=False`.
- [ ] Modify `embeddings_v5_production_enhanced.py` shims:

```python
def _split_provider_model(model: str) -> tuple[str | None, str]:
    from tldw_Server_API.app.core.Embeddings.provider_resolution import split_provider_model
    return split_provider_model(model)


def _resolve_model_and_provider(model: str | None, provider: str | None) -> tuple[str, str]:
    from tldw_Server_API.app.core.Embeddings.provider_resolution import resolve_provider_model
    try:
        intent = resolve_provider_model(
            model,
            provider,
            settings_config=settings.get("EMBEDDING_CONFIG", {}) or {},
            require_model=False,
        )
    except EmbeddingPolicyError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=exc.message) from exc
    return intent.model, intent.provider
```

- [ ] Keep `guess_provider_for_model` import/export behavior stable for current tests.
- [ ] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_strips_prefix \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py::test_resolve_model_and_provider_rejects_mismatch
```

Expected result after implementation: new resolver tests and existing shim tests pass.

### Task 5: Extract Embedding Policy Decisions

- [x] Add `embedding_policy.py`.
- [x] Move `_supports_openai_dimensions`, `_validate_dimensions_request`, `_dimension_policy`, `adjust_dimensions`, `decide_and_apply_l2`, `resolve_fallback_chain`, `_fallback_model_map`, and `map_model_for_provider` into the policy module.
- [x] Add `enforce_embedding_policy(intent, context, *, allowed_providers, allowed_models, implemented_providers, enforce_policy, allow_fallback_with_header, settings_fallback_chain, settings_fallback_model_map) -> EmbeddingPolicyDecision`.
- [x] Preserve policy behavior:
  - OpenAI dimensions are allowed only for `text-embedding-3-small` and `text-embedding-3-large`.
  - Non-OpenAI dimensions must be positive and no greater than 4096.
  - Explicit provider header suppresses fallback unless `EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER` is truthy.
  - Allowlist failures raise `EmbeddingPolicyError` with provider/model denial codes.
  - Recognized but unimplemented providers raise `EmbeddingPolicyError("provider_unsupported", "Provider 'voyage' not implemented")`.
  - Admin bypass remains claim-first through endpoint-provided `enforce_policy` value; the policy module receives the final boolean and does not import AuthNZ.
- [x] Add `tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py`.
- [x] Cover dimensions, fallback chain defaults, configured fallback chain, fallback model mapping, explicit-header suppression, allowlist deny, wildcard model allowlist, unsupported provider, and L2 normalization/base64 behavior.
- [x] Modify endpoint shims to delegate and map `EmbeddingPolicyError` to existing `HTTPException` status codes.
- [x] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py
```

Expected result after implementation: new policy tests pass and existing dimensions/fallback tests still pass through endpoint shims.

Task 5 completed and reviewed. Additional endpoint regressions cover unknown and unsupported providers with invalid dimensions for both create and batch endpoints. Spec review approved after adding unknown-provider classification. Quality review approved after moving endpoint provider/dimension classification into the centralized policy wrapper.

Verification:
- Red endpoint check before the final fix: `test_embeddings_unsupported_provider.py` failed with 3 failures exposing the ordering bug.
- `python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py` -> 5 passed, 195 warnings.
- `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py tldw_Server_API/tests/Embeddings/test_l2_normalization_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_batch_dimensions.py tldw_Server_API/tests/Embeddings/test_embeddings_policy.py tldw_Server_API/tests/Embeddings/test_embeddings_policy_toggle.py tldw_Server_API/tests/Embeddings/test_embeddings_policy_strict_mode.py tldw_Server_API/tests/Embeddings/test_embeddings_unsupported_provider.py` -> 45 passed, 754 warnings.
- `python -m compileall -q` on touched policy, endpoint, and test files -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Embeddings/embedding_policy.py tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py -f json -o /tmp/bandit_embeddings_policy_coord.json` -> 0 findings, no errors.

### Task 6: Add Orchestrator Prepare And Execute Phases

- [x] Add `orchestrator.py` with fakeable protocols:

```python
from __future__ import annotations

from typing import Protocol


class EmbeddingCache(Protocol):
    async def get(self, key: str) -> list[float] | None:
        raise NotImplementedError

    async def set(self, key: str, value: list[float]) -> object:
        raise NotImplementedError


class EmbeddingExecutor(Protocol):
    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        raise NotImplementedError
```

- [x] Implement `EmbeddingRequestOrchestrator.prepare(raw_input, context) -> PreparedEmbeddingRequest`.
  - It calls input normalization, provider resolution, policy enforcement, and execution-plan construction.
  - It returns normalized token totals for endpoint RG and billing.
  - It does not call cache or provider execution.
- [x] Implement `EmbeddingRequestOrchestrator.execute(prepared) -> EmbeddingExecutionResult`.
  - It reads cache entries using current `get_cache_key` semantics.
  - A full cache hit skips executor calls.
  - A partial cache hit calls executor only for missed texts and preserves original response order.
  - It writes only postprocessed canonical float vectors to cache.
  - It validates provider vector count equals miss count.
  - It applies dimensions postprocessing before cache writeback.
  - It returns headers for fallback provider and dimensions policy.
- [x] Preserve raw-text boundaries:
  - `PreparedEmbeddingRequest` may hold `NormalizedEmbeddingInput`.
  - `EmbeddingExecutionPlan` must not hold `NormalizedEmbeddingInput.texts`.
  - `repr(prepared.execution_plan)` must not contain input text.
- [x] Add `tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py`.
- [x] Cover prepare-only token accounting, full cache hit, partial cache hit, vector-count mismatch, redacted plan representation, fallback model mapping, and base64-independent cache values.
- [x] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py
```

Expected result after implementation: orchestrator unit tests pass without importing FastAPI.

Task 6 completed and reviewed. Added regression coverage from quality review for provider/model-coherent fallback after partial cache hits, non-retryable provider errors, rate-limit exhaustion preserving retry-after, base64 dimensions forcing reduction, and malformed vector-container rejection. Spec and quality re-reviews approved after fixes.

Verification:
- `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` -> 12 passed, 36 warnings.
- `python -m compileall -q tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` -> passed.
- `python -m bandit -r tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embedding_orchestrator_coord.json` -> 0 findings, no errors.
- Direct import check confirmed importing `tldw_Server_API.app.core.Embeddings.orchestrator` leaves `fastapi` absent from `sys.modules`.

### Task 7: Wire The Feature-Flagged Endpoint Path

- [x] In `embeddings_v5_production_enhanced.py`, extract the current create-handler body into `_create_embedding_legacy` with the same request, body, user, background task, provider header, and response objects.
- [x] Keep `create_embedding_endpoint` as the public route and make it choose:

```python
if env_flag_enabled("EMBEDDINGS_ORCHESTRATOR_ENABLED"):
    return await _create_embedding_with_orchestrator(
        request=request,
        embedding_request=embedding_request,
        current_user=current_user,
        background_tasks=background_tasks,
        x_provider=x_provider,
        response=response,
    )
return await _create_embedding_legacy(
    request=request,
    embedding_request=embedding_request,
    current_user=current_user,
    background_tasks=background_tasks,
    x_provider=x_provider,
    response=response,
)
```

- [x] Implement `_create_embedding_with_orchestrator` so endpoint-owned work stays at the boundary:
  - Check `EMBEDDINGS_AVAILABLE`.
  - Increment and decrement `active_embedding_requests`.
  - Run `_check_backpressure_and_quotas`.
  - Build `EmbeddingRequestContext` from request metadata, not raw input.
  - Call `orchestrator.prepare(embedding_request.input, context)`.
  - Reserve RG tokens using `prepared.normalized.total_tokens`.
  - Call `orchestrator.execute(prepared)`.
  - Commit RG tokens using result usage units in the `finally` block.
  - Log usage through `log_llm_usage`.
  - Apply response headers from `EmbeddingExecutionResult.response_headers`.
  - Build `CreateEmbeddingResponse` using existing `EmbeddingData` and `EmbeddingUsage` schemas.
- [x] Implement `_embedding_domain_error_to_http(exc)` mapping:
  - Input errors and prefix/dimensions/unknown-provider errors to `400`.
  - Policy deny to `403`.
  - Unsupported provider to `501`.
  - Rate limit to `429` with `Retry-After` when provided.
  - Missing credentials to existing `503` body shape with `error_code: missing_provider_credentials`.
  - Malformed provider response to `502`.
  - Provider unavailable, fallback exhausted, circuit breaker, and internal execution failure to `503`.
- [x] Add `tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py`.
- [x] Add endpoint tests:
  - Flag unset calls legacy path.
  - Flag true calls orchestrator path.
  - Input error maps to current `400` shape.
  - Token-limit error preserves top-level `{"error": "input_too_long", "details": [{"index": 0, "tokens": 3}]}` shape.
  - Missing credentials preserves `503` detail dict with `error_code`.
  - Rate-limit error includes `Retry-After`.
  - Response headers from result are applied.
- [x] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py
```

Expected result after implementation: feature flag tests pass, and the old path remains the default when the flag is unset.

Task 7 completed and reviewed. Added feature-flag routing for the orchestrator path while preserving legacy as the default; extracted `_create_embedding_legacy`; added endpoint orchestrator boundary handling for availability, active request metrics, backpressure, ResourceGovernor token reservation/commit, usage logging, response headers, and response schema construction; added endpoint cache/executor adapters with BYOK, batching, adapter-first execution, credential touch, and legacy HTTP/domain error parity; added endpoint parity coverage. Spec review approved. Quality review initially found cache/credential and adapter provenance risks; fixes added provider preflight before cache reads, post-cache credential touch, adapter-mode cache bypass, adapter provenance propagation, provider HTTP 4xx parity, and fallback missing-credential skip behavior. Quality re-review approved.

Verification:
- `python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py::test_orchestrator_full_cache_hit_touches_resolved_provider_credentials` -> 1 passed, 58 warnings.
- `python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py` -> 31 passed, 470 warnings.
- `python -m compileall -q tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/orchestrator.py tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py` -> passed.
- `python -m bandit -r tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py tldw_Server_API/app/core/Embeddings/request_types.py tldw_Server_API/app/core/Embeddings/orchestrator.py -f json -o /tmp/bandit_embeddings_task7_after_quality_fixes2.json` -> 0 findings, no errors.

### Task 8: Add Dual-Path Parity Coverage

- [ ] Extend `test_embeddings_orchestrator_endpoint_parity.py` with a helper that sends the same request twice:
  - Once with `EMBEDDINGS_ORCHESTRATOR_ENABLED` unset.
  - Once with `EMBEDDINGS_ORCHESTRATOR_ENABLED=true`.
  - Provider execution, cache, credentials, metrics, and RG are patched with deterministic fakes for both runs.
- [ ] Add parity cases:
  - Single string numeric embedding response.
  - Batch string response preserving indexes.
  - Single token-array response.
  - Batch token-array response with `encoding_format: "base64"` and `dimensions`.
  - HuggingFace dimensions reduce, pad, and ignore policies.
  - Full cache hit skips provider execution.
  - Partial cache hit calls provider for misses only.
  - OpenAI primary fallback to HuggingFace with `X-Embeddings-Provider` and `X-Embeddings-Fallback-From`.
  - Explicit `x-provider` suppresses fallback.
  - Provider vector-count mismatch maps to `502`.
- [ ] Assert parity for:
  - HTTP status.
  - Response JSON.
  - `X-Embeddings-Provider`, `X-Embeddings-Fallback-From`, `X-Embeddings-Dimensions-Policy`, and rate-limit headers when present.
  - Usage fields.
  - Cache write values as float vectors.
- [ ] Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py
```

Expected result after implementation: old-path and orchestrator-path outputs match for all parity cases, with expected differences limited to internal call counts asserted by the tests.

### Task 9: Compatibility Shims, Notes, And Security Verification

- [ ] Add short inline migration comments to endpoint wrappers for:
  - `_split_provider_model`
  - `_resolve_model_and_provider`
  - `_validate_dimensions_request`
  - `adjust_dimensions`
  - `decide_and_apply_l2`
  - `resolve_fallback_chain`
  - `map_model_for_provider`
  - `create_embeddings_batch_async`
- [ ] Confirm comments name the new owner and removal condition without user-facing deprecation messaging.
- [ ] Run focused Embeddings verification:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Embeddings_isolated/test_request_types.py \
  tldw_Server_API/tests/Embeddings_isolated/test_input_normalizer.py \
  tldw_Server_API/tests/Embeddings_isolated/test_provider_resolution.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_policy.py \
  tldw_Server_API/tests/Embeddings_isolated/test_embedding_orchestrator.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_characterization.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_v5_unit.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_dimensions_policy.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_token_arrays.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback.py \
  tldw_Server_API/tests/Embeddings/test_embeddings_fallback_model_map.py \
  tldw_Server_API/tests/Embeddings/test_batch_rate_headers.py
```

Expected result: all selected tests pass. If an unrelated pre-existing test fails, record the exact failing test, error, and reason in the Backlog task before continuing.

- [ ] Run import compilation on touched Python files:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m compileall -q \
  tldw_Server_API/app/core/Embeddings/request_types.py \
  tldw_Server_API/app/core/Embeddings/input_normalizer.py \
  tldw_Server_API/app/core/Embeddings/provider_resolution.py \
  tldw_Server_API/app/core/Embeddings/embedding_policy.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py
```

Expected result: command exits `0` with no output.

- [ ] Run Bandit on touched production code:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Embeddings/request_types.py \
  tldw_Server_API/app/core/Embeddings/input_normalizer.py \
  tldw_Server_API/app/core/Embeddings/provider_resolution.py \
  tldw_Server_API/app/core/Embeddings/embedding_policy.py \
  tldw_Server_API/app/core/Embeddings/orchestrator.py \
  tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py \
  -f json -o /tmp/bandit_embeddings_orchestrator.json
```

Expected result: command exits `0`, or any findings are confirmed pre-existing/non-actionable and documented with file, line, severity, and reason.

- [ ] Run whitespace validation:

```bash
git diff --check
```

Expected result: command exits `0` with no output.

- [ ] Update the Backlog implementation task with modified files, verification output, known skips, and final summary.
- [ ] Commit in reviewable slices. Recommended slice order:
  - Characterization tests.
  - Request types, normalizer, resolver, and policy extraction.
  - Orchestrator core.
  - Endpoint feature flag and parity tests.
  - Verification notes.

## Review Checkpoints

- [ ] After Task 1, review characterization results before extraction. Confirm each behavior is marked as contract or compatibility.
- [ ] After Task 5, review pure module APIs for over-coupling to FastAPI or endpoint globals.
- [x] After Task 7, review endpoint RG/billing ordering. No cache read or provider call should happen before the endpoint reservation decision.
- [ ] After Task 8, review parity failures as design feedback, not only test failures. If parity requires behavior changes, split them into a separate design decision.

## Rollback And Rollout

- The old endpoint path remains the default while `EMBEDDINGS_ORCHESTRATOR_ENABLED` is unset or false.
- The orchestrator path is opt-in for parity tests and manual validation.
- The adapter-registry flag remains separate: `LLM_EMBEDDINGS_ADAPTERS_ENABLED` controls adapter routing only.
- If cache keys cannot exactly match the old helper, namespace orchestrator cache entries with `cache_namespace="orchestrator:v1"` until parity proves exact compatibility.
- If Task 1 confirms the old path writes pre-adjustment dimension vectors, keep the orchestrator's approved postprocessed cache semantics namespaced while the flag is opt-in.
- Do not flip the default in this implementation PR.

## Handoff Prompt

Plan complete and saved to `Docs/superpowers/plans/2026-06-24-embeddings-request-orchestrator-implementation-plan.md`. Two execution options:

1. Subagent-Driven (recommended): use `superpowers:subagent-driven-development` to run independent task slices with review checkpoints after characterization, pure extraction, endpoint wiring, and parity verification.
2. Inline Execution: use `superpowers:executing-plans` to execute each checklist item in the current session with the same review checkpoints.

Which approach?
