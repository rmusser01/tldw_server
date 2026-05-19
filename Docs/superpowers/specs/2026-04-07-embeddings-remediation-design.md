# Embeddings Remediation Design

- Date: 2026-04-07
- Project: tldw_server
- Topic: Remediate confirmed Embeddings review findings and implement approved API hardening
- Mode: Design for implementation planning

## 1. Objective

Implement the confirmed Embeddings findings and the approved follow-on improvements without broadening into unrelated subsystems.

The remediation must:

- eliminate silent wrong-result behavior
- make media embedding failure modes truthful
- normalize embeddings/media-embeddings response semantics where current behavior is misleading
- add regression coverage for every corrected contract

## 2. Scope

### In Scope

- Embeddings API request handling and response behavior:
  - `tldw_Server_API/app/api/v1/endpoints/embeddings_v5_production_enhanced.py`
  - `tldw_Server_API/app/api/v1/schemas/embeddings_models.py` if needed for schema alignment
- Media embeddings endpoint behavior:
  - `tldw_Server_API/app/api/v1/endpoints/media_embeddings.py`
- Immediate support code required to fix the validated defects:
  - `tldw_Server_API/app/core/Embeddings/ChromaDB_Library.py`
  - `tldw_Server_API/app/core/Embeddings/jobs_adapter.py`
  - any directly touched cache-key helper path used by embeddings endpoint logic
- Targeted tests in:
  - `tldw_Server_API/tests/Embeddings/`
  - `tldw_Server_API/tests/LLM_Adapters/unit/` only if required by the cache-key fix

### Out of Scope

- Embeddings ABTest/Evaluations
- broad worker redesign
- unrelated response normalization outside embeddings/media embeddings
- performance-only tuning unless required for correctness

## 3. Confirmed Issues To Fix

1. Token-array decode failure is converted into `""` and can still produce a successful embedding response.
2. `generate_embeddings_for_media()` conflates embedding-generation failures with post-generation storage failures and can invoke fallback model generation for storage errors.
3. `generate_embeddings_batch()` can queue some jobs and still return HTTP 500, producing misleading failure semantics after side effects have already occurred.
4. Endpoint-level embeddings cache identity is too coarse when backend endpoint/base URL can change behavior.

## 4. Approved Behavior Changes

### 4.1 Embeddings Endpoint Validation

- Token-array decode failure must return HTTP 400.
- The request must not proceed with synthetic or provider-backed embedding creation after decode failure.
- The validation error should remain stable and testable, using a single explicit error detail for decode-invalid token input.
- The success response shape for valid requests remains unchanged.

### 4.2 Media Generation Failure Classification

- Fallback-model generation is allowed only for embedding-generation failures.
- If primary embedding generation succeeds but persistence into Chroma fails, the endpoint must surface a storage failure immediately.
- Storage failures must not be reinterpreted as provider/model failures.
- The surfaced failure should preserve storage provenance in the returned error message or status payload so tests can distinguish it from upstream provider failure.

### 4.3 Media Batch Submission Semantics

- Batch job submission semantics become explicit.
- Full success remains `202 Accepted`.
- Partial success also returns `202 Accepted`, but with a truthful body indicating partial acceptance.
- Full failure before any enqueue continues to use true failure status codes.
- This remediation does not promise full end-to-end retry idempotency for every client/network scenario; it fixes the misleading HTTP failure contract after partial side effects.

### 4.4 Cache Identity Semantics

- Cache identity must include backend-sensitive execution context whenever request routing can differ for the same provider/model/text/dimensions tuple.
- This includes provider endpoint/base URL identity for paths where `api_url` or equivalent override changes actual execution behavior.
- Cache identity must not include secrets or per-request noise; normalize on stable execution-affecting backend identity only.
- Cache hit rate is secondary to avoiding cross-backend result reuse.

## 5. API Contract Design

### 5.1 `POST /api/v1/embeddings`

- Invalid token-array decode becomes a validation error with HTTP 400.
- Existing successful response structure remains compatible.
- Existing numeric/base64 output semantics remain unchanged.

### 5.2 `POST /api/v1/media/embeddings/batch`

The response contract is expanded to support truthful partial acceptance.

Recommended response shape:

- `status`: `"accepted"` or `"partial"`
- `job_ids`: queued job identifiers
- `submitted`: count of successfully queued jobs
- `failed_media_ids`: list of media IDs that failed to queue
- `failure_reasons`: concise, machine-readable or at least stable reason strings

Response-model rule:

- Keep one response model for both full and partial acceptance.
- `failed_media_ids` and `failure_reasons` should be present with empty-list defaults on full success so clients do not have to branch on missing keys.

Compatibility rule:

- Existing clients that only read `job_ids` and `submitted` should continue to work.
- New fields are additive; the key behavior change is that mixed outcomes are no longer represented as HTTP 500 after side effects.

## 6. File Responsibilities

- `embeddings_v5_production_enhanced.py`
  - tighten token-array validation
  - fix cache-key identity inputs for endpoint-controlled execution
  - preserve existing success payloads
- `media_embeddings.py`
  - split generation and storage error handling
  - implement explicit partial-success batch semantics
  - align response models with approved API behavior
- `embeddings_models.py`
  - update schemas only if required for externally visible response consistency
- targeted tests
  - prove the new contracts and guard against regression

## 7. Testing Strategy

Required regression coverage:

- token-array decode failure returns HTTP 400 at endpoint level
- token-array decode failure does not call downstream embedding creation
- successful decode paths still succeed
- storage failure after successful primary generation does not trigger fallback
- storage failure returns a storage-classified error, not a provider-classified error
- actual generation failure still may use fallback where intended
- mixed batch enqueue returns explicit partial success instead of HTTP 500
- full enqueue success still returns accepted semantics
- cache-key identity differs when backend endpoint/base URL differs

Verification should prefer focused test slices over broad integration runs unless a changed contract specifically requires broader confirmation.
Do not require the known slow real-provider integration path to pass if the corrected contracts are already covered by focused endpoint/unit tests.

## 8. Risk Management

Primary risks:

- accidental breaking change for batch clients expecting only all-success or all-failure outcomes
- hidden coupling between cache key format and cache observability/metrics
- tests that currently encode the buggy behavior

Mitigations:

- additive response fields for partial success
- keep unchanged success payloads where no bug is being corrected
- update or replace tests that currently codify incorrect behavior
- verify changed semantics with endpoint-level tests first

## 9. Success Criteria

The remediation is successful when:

- token decode failures cannot silently produce embeddings
- storage failures are surfaced as storage failures
- mixed batch enqueue results are truthful after partial side effects
- cache identity no longer aliases backend-distinct execution paths
- every corrected behavior has direct regression coverage
