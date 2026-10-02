# Reverse Image Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver separate source, similarity, and identification searches through Unified MCP and authenticated REST, including durable image uploads and configured self-hosted backends.

**Architecture:** One server-owned async service handles validation, authorization, acquisition, quotas, provider calls, normalization, and cleanup. REST and MCP share that service; existing ingestion, storage, DB, egress, profile permissions, and Resource Governor provide the supporting infrastructure. Lens and Bing share a SerpApi adapter file; there is no fan-out, fallback, extra inference, or new generic plugin framework.

**Tech Stack:** Existing FastAPI, Pydantic, httpx, Pillow, pytest/Hypothesis, Loguru, storage/DB abstractions, and Redis Resource Governor. No new runtime dependency is planned.

**Spec:** [Approved design](Docs/Design/2026-09-05-reverse-image-search-tools-design.md), recovered from `836d2648b7` and corrected under **TASK-13409** on 2026-10-01.

## Global Constraints

- Tools: `web.image_sources`, `web.image_similar`, `web.image_identify`; module `image_search`, department `research`, disabled by default through `MCP_ENABLE_IMAGE_SEARCH_MODULE` or explicit module configuration.
- One configured backend per call, independent defaults; no implicit provider selection, fallback, pagination, queued searches, or AI-overview follow-ups.
- Exactly one non-null `image_url`, positive integer `upload_id` (`MediaFiles.id`), or strict `image_base64` plus `mime_type`. Unknown fields, null source fields, boolean integers, and malformed original inputs are rejected without sanitizer repair.
- Static JPEG/PNG/WebP only; 3 MiB and 16 million pixels service ceilings, intersected with selected method limits before bounded acquisition. No resize, re-encode, or method switch after a limit failure.
- `limit`: explicit 1–25; omitted means `min(10, effective_backend_max_results)`. URL 8,192 characters, backend ID 64, language 16, query 500; query only for similar/identify when supported.
- End-to-end deadline 60 seconds; decoded upstream response 2 MiB; public domain response 256 KiB; at most 20 warnings and five evidence entries per identification candidate.
- User allowance initially 10/minute; backend concurrency initially three. REST and MCP share accounting. Multi-worker/instance deployments require Redis fail-closed; single-worker memory mode must be explicit and application-scoped.
- Backend grants default empty/deny; trusted user/role identity only. Unknown and inaccessible backend IDs are indistinguishable. Private collections are explicitly shared with their grant holders and credential-scoped to one collection.
- `private_submission=disabled` by default. Any public staging requires `staged`, including local recipients; custom `allow_private_images=false` independently denies private input. Trusted non-forwarding local byte submission can remain enabled without external submission.
- Staging tokens have at least 128-bit entropy and ten-minute expiry; repeated reads work until revocation. Distributed staging shares both records and bytes; revoke in cleanup and sweep at startup/every minute.
- Preserve existing MCP envelope limits and non-image ingestion behavior. Search results are untrusted data; never persist raw image arguments, signed URLs, upload IDs, hashes, credentials, or provider bodies in logs/hooks/audits.
- Regular tests make no paid/external calls. Live checks are opt-in and missing credentials mean unverified, not passed.
- Before runtime edits, create linked Backlog implementation tasks; TASK-13409 covers design/planning only. Inspect the dirty checkout, use the worktree skill at execution, and preserve unrelated changes. Never stage the entire working tree.

## File structure and dependency order

Paths below are repository-relative. Existing files are narrow integration points, not candidates for broad refactoring.

| File(s) | Responsibility |
| --- | --- |
| `tldw_Server_API/app/core/Image_Search/contracts.py`, `config.py` | Strict shared domain models, normalized results, backend configuration, grants, and method selection. |
| `Image_Search/images.py` | Integrity checks and bounded binary resolution; existing reference lookup/read boundary reused without generation-config coupling. |
| `Image_Search/admission.py` | Shared Resource Governor request accounting and backend lease lifecycle. |
| `Image_Search/providers/serpapi.py`, `tineye.py`, `custom_http.py` | Three focused adapter files for four provider kinds; bounded response handling and provider-specific normalization. |
| `Image_Search/staging.py`, `service.py` | Transient lifecycle and single-call orchestration. |
| `tldw_Server_API/app/core/DB_Management/image_search_staging.py` | Staging-record DAL using existing DB sessions; new SQL stays here and in the established migration path. |
| `tldw_Server_API/app/core/Ingestion_Media_Processing/image_upload.py` | Focused validate/store image branch, invoked from existing `persistence.py`. |
| Existing media request/response schemas and `/media/add` | Add file-only image type and typed `original_file_id`; retain outer guards and envelopes. |
| `tldw_Server_API/app/api/v1/schemas/image_search_schemas.py`, `endpoints/image_search.py` | Shared model exports, authenticated operations/discovery, bearer-token staging reads. |
| `tldw_Server_API/app/core/MCP_unified/modules/implementations/image_search_module.py` | Three strict tools and sanitized inventory resource; existing web evaluation metadata. |
| Existing gateway subjects/runtime, MCP execution security/hooks/reporting, server wiring | Domain extraction, trusted per-hop authorization, opt-in registration, sensitive-value exclusion. |
| `tldw_Server_API/tests/Image_Search/`, existing ingestion/files/MCP/RG tests | Deterministic contract, integration, regression, and property tests. |
| `Docs/Development/REVERSE_IMAGE_SEARCH.md` | Setup, upload/file-ID flow, provider limitations, policies, custom protocol, deployment requirements. |

`Image_Search/` paths in the table expand under `tldw_Server_API/app/core/`. Add package `__init__.py` only where needed for imports. Tasks 1–2 feed acquisition; tasks 3–4 feed providers/orchestration; adapters and staging feed transport integration; the final stage verifies the complete upload-to-search slice. No subagent dispatch is required to execute this plan inline.

## Stage 1: Contracts and durable input

**Goal:** Establish strict shared semantics and a real owned-image upload path.
**Success Criteria:** Invalid original arguments fail identically; image ingestion creates readable owned originals with numeric file IDs and no model/search side effects.
**Tests:** Request/default/grant/result contracts; integrity tests; upload/discovery integration; partial-failure compensation and non-image regressions.
**Status:** Not Started

### Task 1: Strict contracts, configuration, and normalization

**Files:** Create `Image_Search/contracts.py`, `config.py`, `images.py`; create `tests/Image_Search/test_contracts.py`, `test_config.py`, `test_images.py`, `test_results.py` under the server tests directory.

**Interfaces:** Produce these names in `contracts.py`; all later tasks use them, not parallel DTOs:

```python
Operation = Literal["sources", "similar", "identify"]
SubmissionMethod = Literal["public_url", "direct", "staged"]
PermissionCheck = Callable[[str], Awaitable[Literal["allow", "ask", "deny"]]]

@dataclass(frozen=True)
class SearchContext:
    user_id: str
    roles: frozenset[str]
    request_id: str
    check_url: PermissionCheck

@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int
    max_pixels: int
    mime_types: frozenset[str]
    max_width: int | None = None
    max_height: int | None = None

@dataclass(frozen=True)
class ResolvedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    public_url: str | None

@dataclass(frozen=True)
class BackendSelection:
    backend_id: str
    provider: str
    operation: Operation
    method: SubmissionMethod
    limits: ImageLimits
    limit: int

class ImageSearchError(Exception):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.message = message
```

`ImageSearchRequest` is a strict Pydantic model for the exact section-3 fields. Define typed `Score`, `Warning`, `CollectionScope`, `SourceResult`, `SimilarResult`, `Evidence`, `IdentificationResult`, `ImageSearchResult`, `BackendCapabilities`, `BackendInventory`, and `BackendConfig` from spec sections 4–7; use operation-specific result validation rather than permitting mixed result variants. `ImageSearchResult` carries the nine success-envelope fields from the spec. `BackendConfig` carries fixed endpoint/credential reference, grants, collection, forwarding/private flags, operations/options/method limits, and maxima. `BackendCapabilities` is the validated provider-fact snapshot only; `BackendInventory` is the sanitized caller-visible entry with backend/provider IDs, collection, operation/options, effective result/default limits and permitted submission methods. Never serialize a configuration object as discovery.

Produce `validate_image(data: bytes, declared_mime: str | None, limits: ImageLimits) -> ResolvedImage` in `images.py`; `select_backend(operation: Operation, request: ImageSearchRequest, context: SearchContext, backends: Mapping[str, BackendConfig], defaults: Mapping[Operation, str], capabilities: Mapping[str, BackendCapabilities], private_submission: Literal["disabled", "direct", "staged"], staging_available: bool) -> BackendSelection` in `config.py`; `normalize_result(operation: Operation, payload: Mapping[str, Any], *, selection: BackendSelection, collection: CollectionScope) -> ImageSearchResult` in `contracts.py`.

- [ ] Add these executable contract tests, plus parameterized cases for null/conflicting sources, booleans, unknown fields, unsupported MIME, string bounds, and missing evidence:

```python
import pytest
from pydantic import ValidationError
from tldw_Server_API.app.core.Image_Search.contracts import ImageSearchRequest

def test_inline_control_character_is_not_repaired():
    with pytest.raises(ValidationError):
        ImageSearchRequest.model_validate(
            {"image_base64": "aG\x00k=", "mime_type": "image/png"}
        )

def test_omitted_limit_survives_validation():
    request = ImageSearchRequest.model_validate({"upload_id": 1})
    assert "limit" not in request.model_fields_set
```

- [ ] Run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Image_Search/test_contracts.py -q`; expect import failure before implementation, then the specific validation assertions to drive red/green.
- [ ] Implement strict source validation on original values. The decisive guard is:

```python
present = {key for key in ("image_url", "upload_id", "image_base64") if key in raw}
if len(present) != 1 or any(raw[key] is None for key in present):
    raise ValueError("Exactly one non-null image source is required")
if "image_base64" in present:
    base64.b64decode(raw["image_base64"], validate=True)
```

Use `ConfigDict(extra="forbid", strict=True)`, bounded field types, before/after validators, and encoded-size precheck before allocating decoded bytes. Keep `limit=None` internally with omission preserved; resolve it only after selecting the authorized backend. Check integrity/frame count, MIME, decoded dimensions and pixel product with Pillow before full pixel loading; treat decompression warnings as rejection. No inferred identity from visual-match titles.
- [ ] Implement grant preflight before image reads, deterministic method selection and effective-limit intersection. Normalize allowlisted typed fields only; preserve order/score scales/date precision, reject nonfinite scores, distinguish valid empty from all-malformed nonempty results, and enforce serialized response bounds by dropping whole entries/links rather than cutting identifiers. Add Hypothesis cases for exactly-one source, result bounds, finite scores, and capability intersections.
- [ ] Run the four task test files to green; commit only task files plus the linked implementation-task record with `feat(image-search): establish strict contracts and backend policy` after applicable checks.

### Task 2: File-only image ingestion and numeric file references

**Files:** Create `core/Ingestion_Media_Processing/image_upload.py`; modify its `persistence.py`, `api/v1/schemas/media_request_models.py`, `media_response_models.py`, and `api/v1/endpoints/media/add.py` only where dispatch requires it. Extend `tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py`, `tests/Files/test_files_reference_images_endpoint.py`; create `tests/Image_Search/test_image_upload.py`.

**Interfaces:** Consume `validate_image` and `ImageLimits`. Produce `persist_original_image(*, user_id: str, filename: str, data: bytes, mime_type: str, media_db: MediaDbLike, storage: StorageBackend, limits: ImageLimits) -> tuple[int, int]` in `image_upload.py`; tuple order is `(media_id, original_file_id)`. Imports use existing `MediaDbLike` in `DB_Management/media_db/runtime/validation.py` and `StorageBackend` in `Storage/storage_interface.py`.

- [ ] Add a failing schema test:

```python
from tldw_Server_API.app.api.v1.schemas.media_response_models import MediaItemProcessResult

def test_original_file_reference_is_an_explicit_integer_field():
    field = MediaItemProcessResult.model_fields["original_file_id"]
    assert field.annotation == int | None
```

Add route tests using existing media fixtures for the exact multipart values `media_type=image`, `keep_original_file=true`, `perform_analysis=false`; reject URL-only input, missing/incompatible form values, invalid static images, and storage/DB failures. Spies on inference, embeddings, claims and search callbacks must remain unused even with generic hooks enabled. Use injected storage rather than real remote calls.
- [ ] Run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Image_Search/test_image_upload.py -q`; expect missing field/image-type rejection before changes.
- [ ] Add the `image` literal, typed optional field, and isolated image dispatch before text/no-content gates or analysis hooks. Use original byte storage and existing media creation/transaction/cleanup helpers. Derive the **integer row ID**, not the returned insertion UUID:

```python
media_db.insert_media_file(
    media_id, "original", storage_path,
    original_filename=filename, file_size=len(data), mime_type=detected_mime,
)
row = media_db.get_media_file(media_id, "original")
if row is None or row["storage_path"] != storage_path:
    raise ImageSearchError("storage_unavailable", "Original image was not registered")
original_file_id = int(row["id"])
```

Here `media_id`, `storage_path`, and `detected_mime` are locals established by successful owned-media creation, storage, and `validate_image` in this function. The repository's `insert_media_file` returns a UUID string; never expose that as `upload_id`. Verify durable availability before reporting success. Track only newly created records/blobs for compensation; preserve pre-existing deduplicated originals and report no usable ID after failure.
- [ ] Run the new test file and both extended regression files to green, including cross-user discovery, soft deletion/trash, deduplicated ownership, remote storage doubles, and unchanged PDF/document originals. Commit as `feat(media): retain validated image originals for search`.

## Stage 2: Safe acquisition and shared admission

**Goal:** Prove permissions and quota accounting hold before any provider is contacted.
**Success Criteria:** Every URL hop is authorized and connection-safe; REST/MCP share user buckets and backend leases without distributed fallback.
**Tests:** Redirect deny/ask, DNS/egress restrictions, bounded reads, real execution context, quota mixing/cancellation/outage/crash expiry.
**Status:** Not Started

### Task 3: Binary acquisition and production domain authorization

**Files:** Modify `Image_Search/images.py`, gateway `profiles/subjects.py` and `gateway/profile_runtime.py`; use narrow seams in MCP `tool_execution/security.py`, `dependencies.py`, and server request-context construction. Modify `core/Image_Generation/reference_images.py` only for shared owned lookup/read boundary; put new lookup SQL in the existing media-files repository. Create `tests/Image_Search/test_acquisition.py`; extend MCP `test_profile_permission_rules.py`, `test_gateway_policy_simulation.py`, `test_protocol_scope_enforcement.py`.

**Interfaces:** Consume task-1 types. Produce `resolve_image(request: ImageSearchRequest, selection: BackendSelection, context: SearchContext, *, client: httpx.AsyncClient, media_db: MediaDbLike, storage: StorageBackend) -> ResolvedImage` in `images.py`; produce `require_url_permission(url: str, context: SearchContext) -> None`. Trusted server wiring supplies `SearchContext.check_url`; client-supplied roles, callbacks, policy snapshots, or approval markers are never authority.

- [ ] Add this failing shared-extractor test and gateway runtime/simulation/explanation checks for deny and ask using existing profile/grant fixtures:

```python
from mcp_unified.profiles.subjects import extract_permission_rule_subjects

def test_image_url_is_a_domain_permission_subject():
    url = "https://blocked.example/image.png"
    subjects = extract_permission_rule_subjects("web.image_sources", {"image_url": url})
    assert ("domain", url, None) in subjects
```

Add an httpx mock transport recording requests: first response redirects to a denied domain; assert only the first host is contacted. Repeat with ask/no grant, ask/valid scoped grant, missing trusted context, expired/revoked grant, excessive redirects, own managed/staging URL, forbidden IP/port, DNS rebinding, deceptive MIME, oversized chunked body, and truncated image.
- [ ] Run the new acquisition file and extractor test to red. The extractor currently omits `image_url`; isolated callback injection is not sufficient for the production test.
- [ ] Add `"image_url"` to the existing `DOMAIN_ARGUMENT_KEYS`. Preserve subject length/count guards; reject over-limit arguments safely rather than silently excluding their subject. For each source/redirect, use:

```python
async def require_url_permission(url: str, context: SearchContext) -> None:
    decision = await context.check_url(url)
    if decision == "ask":
        raise ImageSearchError("permission_required", "URL permission is required")
    if decision != "allow":
        raise ImageSearchError("permission_denied", "URL permission was denied")
```

Build the callback from authenticated, freshly resolved profile/domain policy and existing approval grants, not raw transport metadata. Trace gateway delegation through the actual selected backend and server `effective_policy_resolver`; if a transport cannot provide verified per-hop policy, fail closed for that path instead of assuming the initial gateway check covers redirects. Reuse the current compiler/evaluator and grant store; do not create a duplicate rule parser or a new ad-hoc signed-policy transport.
- [ ] Stream URL/file reads with caps before buffering; disable automatic redirects, permit at most five manually checked hops, and use the existing egress/pinned-address HTTP-client mechanism for connection-time address safety. A DNS precheck followed by an unpinned fetch is not sufficient. Reject credentialed/private source URLs and own managed/staging paths; preserve public-origin provenance. Reuse owned lookup and storage paths, but not image-generation size config or a read helper that buffers an oversized object first. Return `provider_refetches_url` later for URL submission; do not fetch result thumbnails/pages.
- [ ] Run task tests plus reference-image and profile/gateway regressions to green. Review production callback wiring explicitly before committing `feat(image-search): acquire images with per-hop authorization`.

### Task 4: One shared Resource Governor admission guard

**Files:** Create `Image_Search/admission.py`, `tests/Image_Search/test_admission.py`; modify `Config_Files/resource_governor_policies.yaml`. Use existing `Resource_Governance/governor.py`, `governor_redis.py`, and application governor lifecycle; do not change all other modules' fallback behavior.

**Interfaces:** Consume `SearchContext`. Produce `admit(governor: ResourceGovernor, context: SearchContext, backend_id: str) -> AsyncContextManager[None]` via `@asynccontextmanager`; `check_governor_topology(governor: ResourceGovernor, *, single_worker: bool) -> None` is async and checks capabilities/real Redis before accepting distributed operation. Both names live in `admission.py`.

- [ ] Start with this cancellation regression; extend the fake with controllable admission/outage decisions and use existing memory/fake-clock governor fixtures for ceiling tests:

```python
import asyncio
import pytest
from tldw_Server_API.app.core.Image_Search.admission import admit
from tldw_Server_API.app.core.Image_Search.contracts import SearchContext
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

class RecordingGovernor:
    def __init__(self):
        self.released = []

    async def reserve(self, request, op_id=None):
        handle = "backend" if "streams" in request.categories else "request"
        return RGDecision(True, None, {}), handle

    async def commit(self, handle_id, actuals=None, op_id=None):
        return None

    async def release(self, handle_id):
        self.released.append(handle_id)

async def allow_url(url):
    return "allow"

@pytest.mark.asyncio
async def test_cancelled_call_releases_backend_lease():
    governor = RecordingGovernor()
    context = SearchContext("1", frozenset(), "request-1", allow_url)
    with pytest.raises(asyncio.CancelledError):
        async with admit(governor, context, "lens_primary"):
            raise asyncio.CancelledError
    assert "backend" in governor.released
```

Add memory-governor tests mixing REST/MCP calls against one user entity and multiple users against one backend entity; two distinct backend IDs must not share their concurrency ceiling.
- [ ] Run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Image_Search/test_admission.py -q`; expect missing guard/import before implementation.
- [ ] Configure explicit policies:

```yaml
image_search.requests:
  requests: {rpm: 10, burst: 1.0}
  scopes: [entity]
  fail_mode: fail_closed
image_search.backend:
  streams: {max_concurrent: 3, ttl_sec: 90}
  scopes: [entity]
  fail_mode: fail_closed
```

Use `RGRequest(entity=f"user:{context.user_id}", categories={"requests": {"units": 1}}, tags={"policy_id": "image_search.requests"})` once and a separate `RGRequest(entity=f"service:image_search:{backend_id}", categories={"streams": {"units": 1}}, tags={"policy_id": "image_search.backend"})` for the retained lease. Confirm the governor's actual policy-scope resolution with tests; entity prefixes must remain distinct. Reserve the lease before consuming the request allowance where possible; release it if request admission fails. Commit a successful user reservation once; no refund after ambiguous submission or second charge on safe retry.
- [ ] Enforce the selected governor's real shared capabilities, not merely `REDIS_URL` presence. The existing factory can fall back to memory during import/connectivity failure; reject that result in distributed mode. Redis outage and lease-renewal failures must fail closed before further chargeable work. Hold the lease through bounded cleanup; release in `finally`. Bound cleanup to ten seconds so a 90-second lease covers the 60-second deadline, including crash reclamation margin; renew only if lifecycle changes legitimately require it. Explicit single-worker mode shares one application governor, never one per adapter or transport.
- [ ] Run task tests plus Resource Governance regression tests with injected Redis/fake clock. Include real Redis integration using existing fixtures when available; report unavailable infrastructure rather than substituting a memory test as proof of distributed enforcement. Commit as `feat(image-search): share quota accounting and backend leases`.

## Stage 3: Provider contracts

**Goal:** Provide evidence-backed adapters with bounded, sanitized responses.
**Success Criteria:** Supported matrix matches documented fixtures; unsupported operations/options fail before acquisition; custom declarations cannot grant authority.
**Tests:** Recorded/sanitized fixtures, request shape/authentication, malformed versus empty responses, one-search behavior, custom cache expiry and independent method limits.
**Status:** Not Started

### Task 5: SerpApi Lens/Bing and direct TinEye

**Files:** Create `Image_Search/providers/serpapi.py`, `tineye.py`; create `tests/Image_Search/test_serpapi.py`, `test_tineye.py`, and fixtures under `tests/Image_Search/fixtures/` named `lens_exact.json`, `lens_visual.json`, `bing_visual.json`, `tineye_matches.json`.

**Interfaces:** Each module produces async `search(operation: Operation, request: ImageSearchRequest, selection: BackendSelection, image: ResolvedImage, submission_url: str | None, *, client: httpx.AsyncClient, config: BackendConfig) -> Mapping[str, Any]`. It returns the operation-specific canonical provider payload consumed by `normalize_result`; `submission_url` is final public/staged URL, never an internal upload reference. `serpapi.py` dispatches using `config.provider`; no redundant Lens/Bing client base hierarchy.

- [ ] Recheck and pin [Lens](https://serpapi.com/google-lens-api), [Bing](https://serpapi.com/bing-reverse-image-api), [image upload](https://serpapi.com/image-api), and [TinEye](https://services.tineye.com/TinEyeAPI) contracts. Store minimal sanitized fixture fields with provenance/date, not credentials or full copyrighted payloads. Derive authentication, request paths, score/date semantics, and direct-image limits from those references; do not guess TinEye signing or confuse SerpApi with Serper.
- [ ] Start with this capability contract test, then add mock-transport tests that assert exact request counts and allowlisted output:

```python
from tldw_Server_API.app.core.Image_Search.providers.serpapi import LENS_SECTIONS

def test_lens_does_not_advertise_unsubstantiated_identification():
    assert set(LENS_SECTIONS) == {"sources", "similar"}
```

Lens identify must be rejected by selection with zero acquisition/search calls; TinEye similar/identify likewise. Bing identification must use explicit looks-like/name plus evidence; a visual-match title alone returns empty identification with the specified warning. Test provider-empty, nonempty/all-malformed, invalid top-level JSON, partial-invalid, finite scores, product strings, and precise/raw first-seen dates.
- [ ] Run both new adapter test files to red before adapter code.
- [ ] Map the pinned provider sections explicitly:

```python
LENS_SECTIONS = {"sources": "exact_matches", "similar": "visual_matches"}
BING_SECTIONS = {
    "sources": "pages_with_image",
    "similar": "related_content",
    "identify": "looks_like",
}
```

Verify actual section shapes against fixtures before enabling mappings. Lens query uses its documented refinement-capable visual/product route, not an unsupported exact-match parameter. Lens direct upload uses the documented image ID within its lifetime; Bing URL-only private inputs require permitted staging. TinEye uses documented URL/direct upload only for sources. Preserve match labels and score scale rather than inventing percentages.
- [ ] Stream and cap decoded response bodies at 2 MiB before JSON parsing, with credential requests never following redirects; sanitize exceptions/status text. Enforce selected method byte/pixel/dimension limits, including SerpApi's conservative 500 KB direct cap, Bing 4,000 pixels/side, and verified TinEye cap. Preparation and search share the enclosing deadline. Permit at most one transport-proven-unsent retry; no ambiguous timeout/5xx retry, follow-up URL, or pagination. Never log raw credentials/payloads.
- [ ] Run adapter/normalization tests to green, assert no CI external calls, and commit `feat(image-search): add bounded SerpApi and TinEye adapters`. Missing paid credentials leave live behavior explicitly unverified; do not enable Lens identification on speculation.

### Task 6: Configurable HTTP v1 and sanitized capabilities

**Files:** Create `Image_Search/providers/custom_http.py`, `tests/Image_Search/test_custom_http.py`; extend `config.py` and task-1 configuration tests; add fixture `custom_capabilities.json` and one typed result fixture per operation.

**Interfaces:** Produce `validate_capabilities(payload: Mapping[str, Any]) -> BackendCapabilities`, async `get_capabilities(*, client: httpx.AsyncClient, config: BackendConfig, now: float) -> BackendCapabilities`, and the task-5 `search` signature. Cache validated facts for five minutes per configured backend identity; endpoint or credential changes invalidate the entry. The operation's authorized inventory is built from current grants and fact snapshots, not cached access decisions.

- [ ] Add strict capability tests with this minimum v1 fixture and three maxima (5, 10, 40):

```json
{"schema_version":1,"operations":["sources"],"max_results":5,
 "input_limits":{"image_base64":{"mime_types":["image/png"],
 "max_image_bytes":1024,"max_image_pixels":100}},
 "options":{"sources":{"query":false,"language":false}}}
```

Test missing/flat/malformed limit maps, independent URL/base64 limits, boolean maxima, unsupported options, invalid version, invalid/expired cache and fixed-endpoint authentication without redirects. Omission resolves to five for this fixture; explicit ten fails before image reads. Endpoint data cannot replace configured collection/grants/private-image policy.

```python
import pytest
from pydantic import ValidationError
from tldw_Server_API.app.core.Image_Search.providers.custom_http import validate_capabilities

def test_custom_capability_version_is_not_coerced():
    with pytest.raises(ValidationError):
        validate_capabilities({
            "schema_version": 2, "operations": ["sources"], "max_results": 5,
            "input_limits": {"image_base64": {
                "mime_types": ["image/png"], "max_image_bytes": 1024,
                "max_image_pixels": 100,
            }},
            "options": {"sources": {"query": False, "language": False}},
        })
```
- [ ] Run the custom test file to red, then implement fixed `GET /v1/capabilities` and `POST /v1/{operation}` request construction using strict models. The body contains `schema_version=1`, random opaque `request_id`, exactly one compatible source, inline MIME when needed, and supported options/limit only; it must not contain internal `upload_id`, user IDs, roles or paths. Revalidate typed results and stamp trusted local provider/collection metadata. Unknown upstream reason/messages map to local safe codes.

```python
def validate_capabilities(payload: Mapping[str, Any]) -> BackendCapabilities:
    return BackendCapabilities.model_validate(payload)
```
- [ ] Refresh capability facts asynchronously on startup/expiry through existing lifecycle conventions, bounded by client/deadline policy. Listing returns cached facts only; it never makes one upstream request per user list call. No usable validated snapshot means backend unavailable; grants are always rechecked. Self-hosted private endpoints use explicit egress authorization without a general SSRF exemption; classify forwarding proxies as external.
- [ ] Run custom/configuration tests to green, including a non-forwarding byte backend and a URL-only backend under all three submission-policy settings; commit `feat(image-search): support configured HTTP backend contracts`.

## Stage 4: Lifecycle and transport integration

**Goal:** Connect the shared service safely to REST and MCP, including explicit public staging.
**Success Criteria:** Identical policy and input behavior; transient objects revoke across instances; enabling the module does not grant access.
**Tests:** Staging expiry/revoke/cleanup/topology/quota; transport parity and redaction; opt-in registration; private-identification capability discovery.
**Status:** Not Started

### Task 7: Transient staging with shared records and bytes

**Files:** Create `Image_Search/staging.py`, `DB_Management/image_search_staging.py`, `tests/Image_Search/test_staging.py`; register its migration through the existing AuthNZ/shared-DB migration path and extend service startup/scheduler wiring. New SQL, including DDL, stays in DB management; AuthNZ migration registration calls that DAL rather than embedding feature SQL. Use configured storage and existing database pool; no second task/job subsystem.

**Interfaces:** `StagedImage(token: str, url: str, expires_at: datetime)` is a frozen dataclass in `staging.py`. `StagingService.create(image: ResolvedImage, context: SearchContext, config: BackendConfig) -> StagedImage`, `revoke(token: str) -> None`, `read(token: str) -> tuple[bytes, str]`, and `sweep(now: datetime) -> int` are async. The service constructor accepts configured storage, the staging DAL, base URL, topology, quota, and deployment `private_submission`; it independently enforces publication gates using local `require_staging_permission(private_submission: str) -> None`. DAL methods are `create_record(token_digest: str, owner_id: str, request_id: str, storage_path: str, mime_type: str, byte_size: int, expires_at: datetime) -> None`, `get_live_record(token_digest: str, now: datetime) -> Mapping[str, Any] | None`, `revoke_record(token_digest: str) -> None`, and `expired_records(now: datetime) -> list[Mapping[str, Any]]`; run them through the established async DB pool/thread boundary.

- [ ] Add clock-controlled expiry/revocation tests, repeated reads, create-A/read-B/revoke-A/read-B, concurrent cleanup, quota exhaustion, broken storage and crash-sweeper recovery. Test `disabled` and `direct` deny `create` even when invoked directly with a self-hosted URL-only backend, before a storage write.

```python
import pytest
from tldw_Server_API.app.core.Image_Search.contracts import ImageSearchError
from tldw_Server_API.app.core.Image_Search.staging import require_staging_permission

@pytest.mark.parametrize("policy", ["disabled", "direct"])
def test_direct_submission_permission_is_not_publication_permission(policy):
    with pytest.raises(ImageSearchError) as raised:
        require_staging_permission(policy)
    assert raised.value.reason_code == "private_submission_denied"
```
- [ ] Run `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Image_Search/test_staging.py -q` to red.
- [ ] Use `secrets.token_urlsafe(32)`; store its SHA-256 digest, owner/call, expiry, MIME, size and owned transient storage key. Publish only the scoped token URL; never change a durable original ACL. Persist a live record only once the bytes are available; compensate failed record creation. Reserve quota atomically in the DAL so multiple instances cannot over-admit.

```python
def require_staging_permission(private_submission: str) -> None:
    if private_submission != "staged":
        raise ImageSearchError("private_submission_denied", "Public staging is disabled")
```

Call this inside `StagingService.create` before reserving quota or writing bytes, then enforce recipient grants, custom private-image consent, URL support and topology independently.
- [ ] Revoke the record before attempting blob deletion. Serving consults live authoritative records every time and returns the same 404 for expired/revoked/missing tokens; deletion lag must not extend access. Maintain retryable revoked cleanup records until physical deletion succeeds. Use no-store headers and existing access-log redaction seams; raw tokens must not appear in operational errors.
- [ ] Reject distributed records with local-only bytes at configuration/preflight. Hook an idempotent internal sweeper at startup and once/minute through existing service scheduling, independent of request cancellation. Run lifecycle tests to green with shared fake storage/DAL and existing real-DB fixtures; commit `feat(image-search): stage private images with scoped expiry`.

### Task 8: Shared service, REST, strict MCP tools and discovery

**Files:** Create `Image_Search/service.py`, `api/v1/schemas/image_search_schemas.py`, `api/v1/endpoints/image_search.py`, MCP `modules/implementations/image_search_module.py`; modify API `router_groups/content.py` and minimal-router selection only as needed, MCP `server.py`/module surface wiring, and `tool_execution/hooks.py`/`reporting.py` for scoped sensitive-value exclusion. Create `tests/Image_Search/test_service.py`, `test_endpoints.py`, MCP `test_image_search_module.py`; extend `test_protocol_preexec_validation.py`, `test_protocol_tool_hooks.py`, `test_tool_use_reporting_protocol.py`.

**Interfaces:** `ImageSearchService.search(operation: Operation, request: ImageSearchRequest, context: SearchContext) -> ImageSearchResult` and `backends(context: SearchContext) -> list[BackendInventory]` are async. The service is application-scoped with configured adapters, one governor, binary resolver dependencies and optional staging. Adapter callables have the task-5 signature and are injected by backend ID; no abstract adapter framework. `ImageSearchModule(config: ModuleConfig, *, service: ImageSearchService)` inherits `WebToolBase` solely for evaluation/structured-result plumbing and overrides sanitization. REST schema module re-exports shared request/result types rather than redefining them.

- [ ] Add REST/MCP tests passing the same raw cases from task 1 through real protocol validation and route handling. Include omitted `limit`, bool upload IDs/limits, null sources, unknown fields, MIME/source conflicts, and NUL base64. Use provider spies to prove rejected arguments never fetch/submit. Verify module disabled/enabled behavior and resource visibility under different backend grants.

```python
import pytest
from pydantic import ValidationError
from tldw_Server_API.app.core.MCP_unified.modules.implementations.image_search_module import ImageSearchModule

def test_module_sanitizer_cannot_repair_invalid_base64():
    module = ImageSearchModule.__new__(ImageSearchModule)
    with pytest.raises(ValidationError):
        module.sanitize_input({"image_base64": "aG\x00k=", "mime_type": "image/png"})
```

This focused sanitizer test needs no initialized module. The separate protocol/REST tests must use real initialized modules and request contexts.
- [ ] Run new service/endpoint/module tests to red. Implement the narrow sanitizer override without changing sibling web tools:

```python
def sanitize_input(self, input_data: Any, _depth: int = 0) -> Any:
    ImageSearchRequest.model_validate(input_data)
    return input_data
```

The contract's validation performs bounded checks; do not return a coerced dump or call the parent sanitizer. Execute validates again at the adapter boundary and uses the tool name to enforce operation-specific fields/capabilities.
- [ ] Order the service: strict validation → current grants/operation authorization → method/options/limits/private preflight → shared admission → bounded acquisition → permitted upload/staging → one provider search → typed normalization. Wrap queue/admission/acquisition/preparation/search in `asyncio.timeout(60)`; add a separate bounded cleanup budget of ten seconds. Revoke staging then release the backend lease in cancellation-resistant `finally`; sweeper handles failed cleanup. Clean up a staged object even if provider preparation fails. Return cleanup warnings without erasing valid results. Emit allowlisted metrics/context only.
- [ ] Register three POST paths and authenticated GET `/image-backends` under `/api/v1/research`, plus bearer-token-only GET `/image-staging/{token}`. Use existing AuthNZ dependencies and equivalent per-operation MCP/REST authorization; backend grants are additional, not a replacement for tool/RBAC permission. Map all reason codes/statuses from spec section 8, including 413 transport rejection and native auth/schema errors.
- [ ] Register three tool definitions with strict generated schemas, separate permissions, external-network classification and web evaluation metadata. Expose `image-search://backends` through the module resource API. Inventory is user-filtered, includes method-specific effective limits and resolved defaults, excludes internal URLs/grants/secrets, and reflects no private Bing-identify method when staging is forbidden/unavailable. Lens/TinEye identify stays unsupported. Existing envelope caps remain unchanged.
- [ ] Exclude image bytes/base64, upload IDs, complete URLs/tokens, credentials and raw provider bodies before generic hook/audit/report persistence; do not merely redact the provider logger. Tests inspect captured hook arguments, audit events, exception logs and staging access logs. Preserve safe operation/backend/count/latency metrics and regress ordinary web sanitization/reporting.
- [ ] Run all stage-4 tests and relevant protocol/router/profile regressions to green; commit `feat(mcp): expose separate image search tools and REST routes`.

## Stage 5: End-to-end verification and delivery

**Goal:** Demonstrate the approved feature end to end and document exact operational limits.
**Success Criteria:** Upload→discovery→REST/MCP search works without live providers; all policy/lifecycle regressions pass; no new Bandit findings; operator instructions match capabilities.
**Tests:** Full deterministic feature suite, relevant existing suites, property tests, topology/failure scenarios; optional benign paid-provider smoke checks.
**Status:** Not Started

### Task 9: Vertical slice, documentation and release gates

**Files:** Create `tests/Image_Search/test_upload_search_integration.py`, `Docs/Development/REVERSE_IMAGE_SEARCH.md`; update configured example settings/module YAML and existing MCP docs index discovered during execution. Update linked Backlog implementation records and this plan's stage statuses. No frontend changes.

**Interfaces:** Consume tasks 1–8 without another service/client abstraction. Test real `/media/add`, `/files/reference-images`, research routes and MCP protocol with injected adapters/storage/governor. Use existing authenticated API and MCP fixtures, not a separate fake endpoint implementation.

- [ ] Add the complete integration test: multipart static image upload → success with typed integer `original_file_id` in the file-ID namespace, separate from media `db_id` (their numeric values may coincide) → owned discovery yields same `file_id` → REST and MCP search that `upload_id` with one selected injected provider. Assert other-user discovery/search and deleted/trash originals fail, ingestion invokes zero inference/search hooks, and private policy denial causes zero reads/submissions. Start the upload/discovery portion with the established fixture and PNG helper; do not duplicate its authentication setup:

```python
from tldw_Server_API.tests.Files.test_files_reference_images_endpoint import (
    client_with_user, _png_bytes,
)

def test_uploaded_image_is_discoverable_by_numeric_file_id(client_with_user):
    upload = client_with_user.post(
        "/api/v1/media/add",
        data={"media_type": "image", "keep_original_file": "true", "perform_analysis": "false"},
        files={"files": ("sample.png", _png_bytes(), "image/png")},
    )
    assert upload.status_code == 200
    item = upload.json()["results"][0]
    assert item["status"] == "Success"
    file_id = item["original_file_id"]
    assert type(file_id) is int and file_id > 0
    discovery = client_with_user.get("/api/v1/files/reference-images")
    assert discovery.status_code == 200
    assert file_id in {entry["file_id"] for entry in discovery.json()["items"]}
```

For the complete search test, configure the same application service with the mocked custom direct-byte backend and shared single-worker governor from earlier task tests, grant the fixture user explicitly, and enable private consent. Exercise actual REST and MCP protocol paths using that returned `file_id`; retain provider call count and image-byte assertions. The upload/discovery test requires no configured search backend.
- [ ] Add failure-path integration tests for storage/DB compensation, quota mixing, provider timeout/cancellation, staging cross-instance reads/revocation, Redis outage, oversized decoded provider response, insufficient identification evidence, and all-malformed nonempty results. Expected failures must be structured failures, never successful empty searches. Run the full feature suite to red before making any integration fixes; fix the root shared seam with tests rather than a transport-specific bypass.
- [ ] Document provider setup and all three REST/MCP operations; show the upload form, numeric file-ID mapping and inventory-first selection. Include direct-method provider caps versus URL caps/transport envelope, denied default grants, one shared private collection, staged publication consent for any recipient, distributed Redis/shared-bytes requirements, cleanup/upstream retention limits, and private identification's Bing-plus-staging/custom-byte constraint. Provide exact custom v1 capabilities/request/typed-result examples. Explicitly state CLIP Retrieval is a possible future own-index adapter, not a bundled web-scale backend or an implementation of this v1 protocol.
- [ ] Run from the project venv:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Image_Search tldw_Server_API/app/core/MCP_unified/tests/test_image_search_module.py -q
python -m pytest tldw_Server_API/tests/Image_Generation/test_reference_images.py tldw_Server_API/tests/Files/test_files_reference_images_endpoint.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_persistence_original_storage.py tldw_Server_API/tests/Resource_Governance -q
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_scope_enforcement.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_preexec_validation.py tldw_Server_API/app/core/MCP_unified/tests/test_protocol_tool_hooks.py tldw_Server_API/app/core/MCP_unified/tests/test_tool_use_reporting_protocol.py -q
python -m bandit -r tldw_Server_API/app/core/Image_Search tldw_Server_API/app/core/Ingestion_Media_Processing/image_upload.py tldw_Server_API/app/core/DB_Management/image_search_staging.py tldw_Server_API/app/api/v1/endpoints/image_search.py tldw_Server_API/app/api/v1/schemas/image_search_schemas.py tldw_Server_API/app/core/MCP_unified/modules/implementations/image_search_module.py -f json -o /tmp/bandit_reverse_image_search.json
git diff --check
```

Also run the repository's configured formatter/linter and Bandit on the modified existing Python files enumerated by the feature's scoped diff; no broad lint rewrite. Record coverage on new core code and justify any gaps against the repository's >80% aim. Never substitute missing infrastructure or credentials with claims of passing integration/live tests.
- [ ] If explicitly configured, run opt-in live smoke checks using benign images and provider credentials without logging either. Confirm exact request shape, matching/evidence mapping and method limits for Lens/Bing/TinEye; record unverified providers separately. No live checks in default CI.
- [ ] Self-review scope and error mapping against all 14 verification categories in the spec; inspect staged diff, run required hooks without bypass, and commit only the feature/docs/task records. Complete Backlog records with commands/results and limitations. Remove only this plan when all five stages are complete, according to repository instructions. Creating/merging a PR is not automatic; an AI-authored PR requires the human requester to write the rationale-bearing Change summary before merge.

## Plan self-review and handoff

- Spec sections 1–5 map to tasks 1, 2, 5, 6 and 8; sections 6–8 map to tasks 3, 4, 7 and 8; verification/delivery map to task 9.
- All three approved latest findings have explicit changes and runtime regressions: shared `image_url` extraction plus real per-hop policy context (task 3), original-input REST/MCP parity without control stripping (tasks 1/8), shared request accounting and backend concurrency leases with Redis fail-closed (task 4).
- Owned upload support is not claimed until tasks 2 and 9 pass. No private identification support is claimed merely because Bing supports URL identification.
- Existing insertion UUID versus numeric `MediaFiles.id`, governor-factory memory fallback, smaller gateway subject limits, and missing trusted remote policy context are explicit integration hazards, not hidden assumptions.
- This is a plan, not executed code. Provider mappings/authentication remain contract-test-gated; regular verification uses mocks. Execute inline with the executing-plans skill unless the user chooses delegation; any delegated execution must obey the current authorization instructions.
