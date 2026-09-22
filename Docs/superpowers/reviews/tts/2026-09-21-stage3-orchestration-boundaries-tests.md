# Stage 3 — Orchestration, boundaries, efficiency, and the test estate

## Scope

Three subjects that stage 2 deliberately left alone, collapsed into one stage because each is too
small to carry a stage of its own:

1. **Orchestration** — `tts_service_v2.py` (4,357 LOC / 74 commits, the module's hot spot) and
   `adapter_registry.py` (1,615 / 56): how the service assembles adapters, and where it reaches
   through their interfaces.
2. **Boundaries** — the `core ↔ api/v1` seam, and the provider-limit tables that the routing and
   validation layers both read.
3. **The test estate** — the `tests/TTS` / `tests/TTS_NEW` split, which the audit brief identifies
   as itself an instance of the pattern under review.

The generic arc's stage-4 (data-source boundaries) is **omitted, not collapsed**: this module has
no persistence of its own. `grep -rn "SELECT |INSERT INTO|UPDATE .* SET"` over
`core/TTS/**/*.py` returns zero hits (recorded in stage 1) and there is no `DB_Management` import
outside the jobs worker's queue usage. There is nothing to review there.

## Code Paths Reviewed

- `core/TTS/tts_service_v2.py:TTSServiceV2.generate_speech (2021-2560)` — the ~540-line primary
  entry point; `:_generate_with_adapter (2561-2732)`; `:_try_fallback_providers (3904-4170)`;
  `:_get_fallback_adapter (3691-3768)`; `:_get_adapter (3401-3438)`;
  `:_generate_chunked_response (1041-1286)`; `:_convert_response_if_needed (3516-3566)`;
  `:_convert_pcm_to_format (892-913)`; `:_close_response_audio_stream (554-560)`;
  `:_close_request_adapter (562-580)`; `:_attach_response_metadata (657-706)`;
  `:_convert_request (2782-2966)`; `:_categorize_error (3871-3884)`;
  `:_resolve_circuit_breaker_key (1851-1876)`; `:__init__ (245-344)`.
- `core/TTS/adapter_registry.py:create_adapter_with_overrides (671-743)`;
  `:_initialize_adapter (745-900)`; `:MODEL_PROVIDER_MAP (1375-1470+)`;
  `:get_provider_for_model (1512-1529)`; `:get_adapter_by_model (1530-1545)`;
  `:_adapters (429)`; `:_adapter_specs`.
- `core/TTS/tts_validation.py:ProviderLimits.LIMITS (80-215)`; `:get_limits (~205-216)`;
  `:get_max_text_length (219-223)`; `:TTSInputValidator.MAX_TEXT_LENGTHS (287-314)`;
  `:MAX_TEXT_BYTES (316-319)`; `:_validate_text (773-800)`; `:_validate_parameters (915-945)`;
  `:validate_tts_request (1300-1340)`.
- `core/TTS/tts_config.py:ProviderConfig (49-...)`, specifically `max_retries (66)` and
  `timeout (65)`.
- `core/TTS/adapters/base.py:TTSCapabilities (59-79)` — the third holder of `max_text_length`.
- `core/TTS/tts_resource_manager.py:ConnectionPool (~210-306)`; `:ResourceManager.register_model (~872-...)`.
- `core/TTS/realtime_session.py:14`, `core/TTS/tts_jobs_worker.py:20,707,874` — the other two
  `api/v1/schemas` importers and the `_tts_metadata` readers on the core side.
- `api/v1/endpoints/audio/audio_tts.py:_extract_tts_metadata (168-...)`, `:194`, `:262`, `:933`,
  `:1260`, `:1436`, `:1444-1448`, `:1515` — the endpoint-side consumers of the metadata
  side channel (reviewed read-only; owner-only path, not proposed for modification here).
- `api/v1/endpoints/audio/audio_streaming.py:_allowed_formats_for (4011-4018)` — the only external
  consumer of `ProviderLimits.LIMITS["valid_formats"]`.

## Tests Reviewed

Located by import-grep. **Reachability, not measured coverage** — no suite was executed.

| Module | Importers | `tests/TTS` | `tests/TTS_NEW` | elsewhere | What it protects / does it downgrade risk |
| --- | --- | --- | --- | --- | --- |
| `tts_service_v2` | 35 | 5 | 17 | 13 | Broad surface coverage. `tests/TTS/test_tts_service_v2.py:33` hand-rolls `MockAdapter(TTSAdapter)` and drives the real registry+factory — good, but because `MockAdapter` inherits the *base* `validate_request`, the fallback-abort in tts-2 is unreachable. Does **not** downgrade tts-10 or tts-11. |
| `adapter_registry` | 27 | 11 | 10 | 6 | Best-covered file in the module. Downgrades the risk of changing `create_adapter_with_overrides` (tts-11). |
| `tts_validation` | 9 | 3 | 6 | 0 | Covers sanitization and per-provider rejection. No test asserts the two limit tables agree, so tts-12 is uncovered. |
| `tts_config` | 13 | 2 | 6 | 5 | Covers config parsing; nothing asserts a declared knob is consumed, so the dead `max_retries` in tts-12 is uncovered. |
| `tts_resource_manager` | 11 | 9 | 1 | 1 | `test_tts_resource_manager.py:222-230` covers `close_client`. Relevant to tts-1 (stage 2). |
| `voice_manager` | 11 | 0 | 8 | 3 | Entirely covered from `tests/TTS_NEW` — zero importers in `tests/TTS`. Illustrates tts-14: the two trees do not overlap, they partition. |
| `circuit_breaker` | 5 | 1 | 3 | 1 | Thin. Matters because tts-9's misclassification consequence routes through the breaker. |
| `tts_jobs_worker` | 5 | 0 | 2 | 3 | Thin, and one of the `_tts_metadata` readers (tts-10). |
| `gateway_config` | 13 | 1 | 8 | 4 | Well reached. |
| `gateway_catalog` / `gateway_execution` / `gateway_preflight` | 1 each | 0 | 1 each | 0 | **Single-importer modules** — one file's collection failure zeroes them. One of the gateway test files is currently an import error (see Validation Commands). |
| `phoneme_overrides` | 1 | 1 | 0 | 0 | Single importer. |
| `audio_converter` / `audio_utils` / `streaming_audio_writer` | 6 / 5 / 6 | 4 / 4 / 5 | 1 / 1 / 1 | 1 / 0 / 0 | Encode correctness only; no lifecycle assertions (stage 2, tts-7). |
| `tts_request_resolution` | 4 | 0 | 1 | 3 | Mostly reached from outside both TTS trees. |
| `utils` | 5 | 2 | 3 | 0 | Covers `parse_bool` itself; nothing asserts the bypassers agree (stage 2, tts-4). |

**No module in this package has zero importers.** Two *classes* do — see tts-14.

## Validation Commands

```
$ python3 - <<'PY'   # diff the two max_text_length tables inside tts_validation.py
provider                   LIMITS   MAX_TEXT_LENGTHS  DIFF
chatterbox                  10000              10000
default                      None               5000   <<< MISMATCH
dia                         10000              30000   <<< MISMATCH
echo_tts                     None                768   <<< MISMATCH
elevenlabs                   5000               5000
fish_s2                      5000               5000
higgs                        8000              50000   <<< MISMATCH
index_tts                    4000               4000
kitten_tts                   5000               5000
kokoro                    1000000            1000000
lux_tts                      5000               5000
neutts                       None               5000   <<< MISMATCH
omnivoice                    5000               5000
openai                       4096               4096
pocket_tts                   5000               5000
pocket_tts_cpp               5000               5000
qwen3_tts                    5000               5000
supertonic                  15000              15000
supertonic2                 15000              15000
vibevoice                   10000              15000   <<< MISMATCH
vibevoice_realtime           8192               8192
PY

$ grep -rn "max_text_length" tldw_Server_API/app/core/TTS tldw_Server_API/app/api/v1/endpoints/audio | grep -v __pycache__ | grep -v "tts_validation.py:[0-9]*: *\"max_text_length\""
core/TTS/tts_service_v2.py:1071:            max_len = getattr(caps, "max_text_length", None)
core/TTS/tts_validation.py:220:    def get_max_text_length(cls, provider: str) -> int:
core/TTS/tts_validation.py:223:        return limits.get("max_text_length", 5000)
core/TTS/tts_validation.py:779:        max_length = self.max_text_length_override or self.MAX_TEXT_LENGTHS.get(...)
core/TTS/tts_validation.py:780:        provider_max_length = ... self._get_provider_setting(provider, "max_text_length")
core/TTS/adapters/vibevoice_realtime_adapter.py:168:    MAX_TEXT_LENGTH = 8192
core/TTS/adapters/vibevoice_realtime_adapter.py:242:            max_text_length=self.MAX_TEXT_LENGTH,
core/TTS/tts_config.py: (ProviderConfig.max_text_length, Optional[int])
  -> ProviderLimits.get_max_text_length (219-223) has NO caller anywhere in app/.

$ grep -rn "ProviderLimits" tldw_Server_API/app --include='*.py' | grep -v __pycache__
core/TTS/tts_validation.py:77:class ProviderLimits:
core/TTS/tts_validation.py:922:                limits = ProviderLimits.get_limits(provider)      # speed bounds only
api/v1/endpoints/audio/audio_streaming.py:4013: from ... import ProviderLimits
api/v1/endpoints/audio/audio_streaming.py:4015: limits = ProviderLimits.get_limits(...)          # valid_formats only

$ grep -rn "max_retries" tldw_Server_API/app/core/TTS tldw_Server_API/app/api/v1/endpoints/audio | grep -v __pycache__
core/TTS/TTS-DEPLOYMENT.md:165:    max_retries: 3
core/TTS/tts_config.py:66:    max_retries: int = 3
core/TTS/tts_service_v2.py:945:  "max_retries": _pick_int(("segment_retry_max","segment_retries"), 2)   # unrelated: per-segment retry
core/TTS/tts_service_v2.py:1125:  max_attempts = max(1, int(retry_params["max_retries"]))                 # reads the line above
api/v1/endpoints/audio/audio_tts.py:683:  max_retries=3,                                                        # Jobs queue, not provider
api/v1/endpoints/audio/audio_jobs.py:297,327                                                                # Jobs queue, not provider
  -> ProviderConfig.max_retries (tts_config.py:66) has NO reader.

$ grep -rn "_tts_metadata" tldw_Server_API/app --include='*.py' | grep -v __pycache__
core/TTS/tts_service_v2.py:704:            target._tts_metadata = metadata          # WRITE
core/TTS/tts_service_v2.py:2055:            request._tts_metadata = metadata         # WRITE
core/TTS/tts_jobs_worker.py:707, :874                                                # READ
api/v1/endpoints/audio/audio_tts.py:169, :194, :262, :933, :1260, :1444, :1448, :1515 # READ
  -> 2 writes in core, 10 reads (2 core / 8 api).

$ grep -n 'adapter\._\|registry\._\|getattr(registry, "_\|getattr(adapter, "_' tldw_Server_API/app/core/TTS/tts_service_v2.py
576:        cached_adapters = getattr(registry, "_adapters", {})
1060:            caps = getattr(adapter, "_capabilities", None)
3731:            if registry and hasattr(registry, "_adapter_specs"):
3732:                specs = registry._adapter_specs

$ grep -rl "core\.TTS" tldw_Server_API/tests | wc -l ; \
  find tldw_Server_API/tests/TTS -name '*.py' | wc -l ; \
  find tldw_Server_API/tests/TTS_NEW -name '*.py' | wc -l
167
56
93

$ python -m pytest tldw_Server_API/tests/TTS      --collect-only -q -p no:randomly | tail -1
614 tests collected

$ python -m pytest tldw_Server_API/tests/TTS_NEW  --collect-only -q -p no:randomly | tail -3
1185 tests collected, 3 errors
ERROR tldw_Server_API/tests/TTS_NEW/property/test_echo_tts_chunking_properties.py - ModuleNotFoundError: No module named 'hypothesis'
ERROR tldw_Server_API/tests/TTS_NEW/property/test_tts_properties.py              - ModuleNotFoundError: No module named 'hypothesis'
ERROR tldw_Server_API/tests/TTS_NEW/unit/test_tts_gateway_properties.py          - ModuleNotFoundError: No module named 'hypothesis'

$ comm -12 <(find tldw_Server_API/tests/TTS -name '*.py' -exec basename {} \; | sort -u) \
           <(find tldw_Server_API/tests/TTS_NEW -name '*.py' -exec basename {} \; | sort -u)
__init__.py
test_elevenlabs_adapter.py

$ git log --since='24 months ago' --format=%ad --date=short -- tldw_Server_API/tests/TTS     | head -1; \
  git log --since='24 months ago' --format=%ad --date=short -- tldw_Server_API/tests/TTS     | tail -1
2026-09-07
2025-08-23
$ git log --since='24 months ago' --format=%ad --date=short -- tldw_Server_API/tests/TTS_NEW | head -1; \
  git log --since='24 months ago' --format=%ad --date=short -- tldw_Server_API/tests/TTS_NEW | tail -1
2026-09-09
2025-09-04
$ git log --oneline --since='24 months ago' -- tldw_Server_API/tests/TTS | wc -l      ; \
  git log --oneline --since='24 months ago' -- tldw_Server_API/tests/TTS_NEW | wc -l
125
202

$ grep -rn "skip\|skipif" tldw_Server_API/tests/TTS --include='*.py' -l | wc -l   # (marker occurrences, not files)
  TTS: 117 skip/skipif markers ; TTS_NEW: 25
$ ls tldw_Server_API/tests/TTS/conftest.py
ls: no such file
$ ls tldw_Server_API/tests/TTS_NEW/conftest.py
tldw_Server_API/tests/TTS_NEW/conftest.py     (22 fixtures; 7 custom markers at :47-57)
```

## Findings

---

### FINDING tts-10 — The core service returns metadata by writing a private attribute onto the API layer's Pydantic request object

```
axis:        encapsulation
class:       n/a
severity:    Medium
sites:       writes (core):
             core/TTS/tts_service_v2.py:_attach_response_metadata (704)
             core/TTS/tts_service_v2.py:generate_speech (2055)
             reads (core):
             core/TTS/tts_jobs_worker.py:707, :874
             reads (api, owner-only path, read-only for this review):
             api/v1/endpoints/audio/audio_tts.py:_extract_tts_metadata (168-172), :194, :262,
               :933, :1260, :1444, :1448, :1515
             the type the attribute is grafted onto:
             api/v1/schemas/audio_schemas.py:OpenAISpeechRequest, imported into core at
             core/TTS/tts_service_v2.py:25, core/TTS/realtime_session.py:14,
             core/TTS/tts_jobs_worker.py:20
canonical:   NONE. `TTSResponse` (adapters/base.py:244-277) already carries a `metadata: dict`
             field — but `generate_speech` is an `AsyncGenerator[bytes, None]`, so it has no
             return value to hang it on, which is why the side channel exists.
destination: n/a for the import (see below). For the side channel: give `generate_speech` an
             explicit out-parameter — an `Optional[dict]` the caller passes in and the service
             fills — or split metadata retrieval into its own awaitable. Either makes the data
             flow visible in the signature.
knowledge:   "how does a streaming generator return structured metadata alongside its bytes".
             Today the answer is an undeclared attribute on somebody else's DTO. It is invisible
             to the type checker (Pydantic models do not declare `_tts_metadata`; every read is a
             `getattr(..., None)`), invisible in the signature of the only function that produces
             it, and it makes the endpoint's completion detection fragile — `audio_tts.py:1444-1448`
             has to capture `metadata_before`, advance the generator, capture `metadata_after`,
             and compare them by **identity** (`is not`) to decide whether generation finished.
             That comparison is the whole tell: the caller cannot ask "are you done", so it
             watches a shared mutable slot.
impact:      Medium. It is not a live defect — it works — but it is the reason three core modules
             import an `api/v1/schemas` type, the reason the endpoint has an identity-comparison
             completion probe, and the reason nothing type-checks the metadata contract. Ten read
             sites across two layers means any change to the metadata shape is a ten-site,
             un-type-checked edit.
             On the import itself: this is the **mild** layering class the audit brief describes —
             schema-only imports, honest reading being that `OpenAISpeechRequest` is in the wrong
             package rather than that core is wrong to need it. Three sites is far too few to
             justify a ratchet test or a mass move; recorded, not actioned.
tests:       import-grep reachability, not coverage. `tts_service_v2` 35 importers,
             `tts_jobs_worker` 5. The endpoint-side reads are covered by `tests/Audio` (13 files
             import `core.TTS`). No test pins the `_tts_metadata` key set, so the contract is
             enforced by nothing.
effort:      moderate. The core-side change is contained, but 8 of the 10 readers are under
             `api/v1/endpoints/`, which is owner-only. Needs a design note:
             `Docs/Design/2026-09-21-tts-speech-metadata-channel-design.md`.
owner-only:  **yes** — any fix touches `tldw_Server_API/app/api/v1/endpoints/audio/audio_tts.py`.
             The core-side half (`tts_service_v2`, `tts_jobs_worker`) is not.
confidence:  confirmed (all 12 sites and the identity-comparison probe read directly).
```

---

### FINDING tts-11 — Every BYOK request, and every OmniVoice request, constructs and fully initializes a throwaway adapter

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       core/TTS/adapter_registry.py:create_adapter_with_overrides (671-743)
               :719  `adapter = adapter_class(config=provider_cfg)`
               :722  `initialized = await adapter.ensure_initialized()`
             callers that take this path:
             core/TTS/tts_service_v2.py:_get_adapter (3416-3419)  OmniVoice — unconditional
             core/TTS/tts_service_v2.py:_get_adapter (3420-3421)  any provider, when overrides present
             core/TTS/tts_service_v2.py:_get_adapter (3428-3436)  model-routed equivalents
             the teardown that makes it a throwaway:
             core/TTS/tts_service_v2.py:_close_request_adapter (562-580)
             where overrides originate:
             api/v1/endpoints/audio/audio_tts.py:814-819  `_resolve_tts_byok(...)` -> `tts_overrides`
             api/v1/endpoints/audio/audio_tts.py:1047, :1433  passed as `provider_overrides`
             the cached path this bypasses:
             core/TTS/adapter_registry.py:get_adapter (629-668), `_adapters` dict at :429
canonical:   n/a.
destination: n/a. The proportionate fix is a small keyed cache of override-adapters (key =
             provider + a hash of the override dict, excluding secrets), with the same idle
             eviction `tts_resource_manager` already applies to models — not a new abstraction.
knowledge:   n/a (efficiency axis).
cost-driver: `ensure_initialized()` -> `adapter.initialize()` + `get_capabilities()`, **once per
             request** instead of once per process. What it costs scales with the provider class,
             not with the text:
             • Remote providers — one extra `get_capabilities()` and, for ElevenLabs specifically,
               `initialize()` calls `_fetch_user_voices()` (`elevenlabs_adapter.py:271`), a live
               HTTPS round-trip to `/voices` **on every single request**. Scales linearly with
               request rate. The pooled httpx client itself is reused
               (`ConnectionPool._pools`, keyed by provider), so TLS is not re-established — but
               the voice-catalog fetch is pure added latency on the critical path, and it is the
               same call whose failure triggers tts-1.
             • Local model providers — `initialize()` loads weights. If a BYOK-style override is
               ever configured for a local provider, that is a multi-GB load and a CUDA
               allocation per request.
             • OmniVoice — takes this path **unconditionally** (`:3416`), overrides or not,
               because the supervisor is injected through the override dict
               (`_build_omnivoice_adapter_overrides`, 3441-3446).
             Compounding factor: `_close_request_adapter` then tears the instance down, so nothing
             is retained between requests and the cost is paid again immediately.
tests:       import-grep reachability, not coverage. `adapter_registry` is the module's
             best-reached file (27 importers, 11 `tests/TTS` / 10 `tests/TTS_NEW`), and
             `create_adapter_with_overrides` is directly covered. That makes a caching change
             cheap to verify. No test measures initialization count per request.
effort:      moderate. The cache is small, but the key must exclude credential material and the
             eviction must not resurrect tts-1's shared-client problem — sequence this **after**
             tts-1. `tests/TTS_NEW` already has OmniVoice supervisor fixtures (4 importers) to
             build on.
owner-only:  no for the fix (`app/core/**`). The override *origin* is in `api/v1/endpoints/`, but
             nothing there needs to change.
confidence:  confirmed (the construction, the `ensure_initialized`, the unconditional OmniVoice
             branch, the per-request teardown, and the ElevenLabs `/voices` call in `initialize`
             are all read directly);
             probable-risk (the magnitude — no profiling was run, and the local-provider case
             depends on a configuration this review did not observe in use).
```

---

### FINDING tts-12 — Four sources of truth for "how long may this provider's text be", two of them in the same file and disagreeing, one of them dead

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       source 1 — core/TTS/tts_validation.py:ProviderLimits.LIMITS (80-215), the
               `max_text_length` key of 18 provider entries. **Dead**: the only reader,
               `ProviderLimits.get_max_text_length (219-223)`, has no caller in `app/`;
               `ProviderLimits` is consumed only for `min_speed`/`max_speed`
               (tts_validation.py:922) and `valid_formats`
               (api/v1/endpoints/audio/audio_streaming.py:4015).
             source 2 — core/TTS/tts_validation.py:TTSInputValidator.MAX_TEXT_LENGTHS (287-314),
               21 entries. **This is the one that actually gates**, at `_validate_text (779)`.
               Plus `MAX_TEXT_BYTES (316-319)` for the `echo_tts` UTF-8 cap.
             source 3 — hardcoded in the two shim adapters:
               core/TTS/adapters/openai_adapter.py:670 (`> 4096`)
               core/TTS/adapters/elevenlabs_adapter.py:854 (`> 5000`)
               core/TTS/adapters/vibevoice_realtime_adapter.py:MAX_TEXT_LENGTH (168), used at :242
             source 4 — core/TTS/adapters/base.py:TTSCapabilities.max_text_length (67), populated
               per adapter by `get_capabilities()`, read by
               core/TTS/adapters/base.py:validate_request (372) and
               core/TTS/tts_service_v2.py:_generate_chunked_response (1071)
             the disagreements, sources 1 vs 2:
               dia        10000 vs 30000
               higgs       8000 vs 50000
               vibevoice  10000 vs 15000
               echo_tts    absent vs 768   (and MAX_TEXT_BYTES 767)
               neutts      absent vs 5000
             the same problem, one file over — a declared provider knob with no reader:
               core/TTS/tts_config.py:ProviderConfig.max_retries (66), documented for operators at
               core/TTS/TTS-DEPLOYMENT.md:165, read by nothing. (The `max_retries` at
               tts_service_v2.py:945/1125 is a *different*, unrelated per-segment retry count
               sourced from `extra_params`, and the ones in `audio_tts.py:683` /
               `audio_jobs.py:297` belong to the Jobs queue.)
             adjacent: the same four-source shape exists for model IDs —
               core/TTS/adapter_registry.py:MODEL_PROVIDER_MAP (1375-1470+) routes a model to a
               provider, while core/TTS/adapters/openai_adapter.py:SUPPORTED_MODELS (598) and
               core/TTS/adapters/elevenlabs_adapter.py:MODELS (190-210) independently decide
               whether that provider accepts it.
canonical:   `TTSInputValidator.MAX_TEXT_LENGTHS` is the de-facto correct copy — it is the one the
             request path reads, it is the most complete (21 entries vs 18), and it is the only
             one with the `echo_tts` byte cap beside it. Source 1's `max_text_length` key should
             be deleted outright; source 4 should be *derived* from source 2 rather than declared
             independently per adapter.
destination: n/a — no new module. Delete the dead key from `ProviderLimits.LIMITS`, have each
             adapter's `get_capabilities()` read `TTSInputValidator.MAX_TEXT_LENGTHS` when
             populating `TTSCapabilities.max_text_length`, and replace the two hardcoded literals
             with the same lookup.
knowledge:   "the maximum text a given provider will accept". Four independent declarations of one
             fact, in a file where two of them sit 70 lines apart and already disagree by up to
             6× (higgs: 8,000 vs 50,000). The change-amplification is concrete and already
             realised: someone raised `higgs` to 50,000 and `dia` to 30,000 in
             `MAX_TEXT_LENGTHS` and did not touch `ProviderLimits.LIMITS`, which still says 8,000
             and 10,000. The next person to raise a limit has a 50% chance of editing the dead
             table and shipping a change that does nothing — and no test would catch it, because
             nothing asserts the tables agree.
scenario:    (illustrating the *live* half, source 2 vs source 4) A 12,000-character request for
             `higgs`. `TTSInputValidator._validate_text (779)` allows it —
             `MAX_TEXT_LENGTHS["higgs"] = 50000`. It then reaches
             `TTSAdapter.validate_request (base.py:372)`, which compares against
             `self._capabilities.max_text_length` from the adapter's own `get_capabilities()`.
             Where those disagree, the request is accepted by one gate and rejected by the next,
             with a second, differently-worded error — and in the fallback loop
             (`_get_fallback_adapter:3757`) that second rejection is what decides whether the
             provider is skipped. A developer raising the cap edits `MAX_TEXT_LENGTHS`, sees the
             validator pass, and does not learn that the capability gate still rejects.
impact:      Medium. No user-visible defect is confirmed today — the divergent entries happen to
             fail closed (the dead table is stricter) — but the structure guarantees the next
             limit change is wrong, and the dead-knob half (`ProviderLimits.max_text_length`,
             `ProviderConfig.max_retries`) actively misleads: `max_retries` is documented to
             operators in `TTS-DEPLOYMENT.md:165` as a setting they can tune, and setting it does
             nothing at all.
tests:       import-grep reachability, not coverage. `tts_validation` 9 importers (3 `tests/TTS`,
             6 `tests/TTS_NEW`); `tts_config` 13. Existing tests assert individual provider
             rejections; none asserts cross-table agreement, and none asserts that a declared
             config field is consumed. Both gaps are trivially closable with one table-driven
             test each.
effort:      cheap. Delete one key from 18 dict entries; two lookups replace two literals; one
             parametrised test asserting the tables agree for every provider in
             `TTSProvider`. Deleting `ProviderConfig.max_retries` is a separate one-line change
             plus a `TTS-DEPLOYMENT.md` correction — or wire it up, but decide, because shipping
             a documented no-op knob is the worst of the three options.
owner-only:  no (`app/core/**` and `core/TTS/TTS-DEPLOYMENT.md`).
confidence:  confirmed (the table diff was computed and is recorded above; the dead readers were
             established by grepping all of `app/` for callers of
             `ProviderLimits.get_max_text_length` and `ProviderConfig.max_retries`, both zero).
```

---

### FINDING tts-13 — The service reaches into the registry's and the adapters' private state where public equivalents exist

```
axis:        encapsulation
class:       adoption-gap
severity:    Low
sites:       core/TTS/tts_service_v2.py:_close_request_adapter (576)
               `cached_adapters = getattr(registry, "_adapters", {})`
             core/TTS/tts_service_v2.py:_generate_chunked_response (1060)
               `caps = getattr(adapter, "_capabilities", None)`
             core/TTS/tts_service_v2.py:_get_fallback_adapter (3731-3732)
               `if registry and hasattr(registry, "_adapter_specs"): specs = registry._adapter_specs`
             the private state being read:
             core/TTS/adapter_registry.py:_adapters (429), mutated at :659, :866, :1206, :1241, :1285-1290
             core/TTS/adapters/base.py:_capabilities (307, set in ensure_initialized at 418)
canonical:   core/TTS/adapters/base.py:TTSAdapter.capabilities (304-307) — a public property that
             returns exactly `self._capabilities`, bypassed at tts_service_v2.py:1060.
             For the registry there is no public equivalent yet; the two reads want
             "is this adapter one of yours?" and "which providers have a registered spec?", both
             of which are one-line public predicates.
destination: n/a — add `TTSAdapterRegistry.owns(adapter) -> bool` and
             `TTSAdapterRegistry.has_spec(provider) -> bool` beside the existing public methods,
             and use the existing `adapter.capabilities` property.
knowledge:   the registry's cache representation. `_adapters` is a plain `dict[str, TTSAdapter]`
             mutated in six places (`:659`, `:866`, `:1206`, `:1241`, `:1285-1290`); the service's
             `_close_request_adapter` decides whether it owns an adapter — and therefore whether
             to close it — by scanning that dict's *values* by identity. If the registry ever
             keys differently, holds weakrefs, or shards per user, the ownership test silently
             starts returning the wrong answer and the service either closes a cached adapter it
             does not own or leaks one it does. The `hasattr` guard at :3731 is itself the tell:
             the service already knows it is reading something that may not be there.
impact:      Low on its own — all three reads are defensive (`getattr` with defaults, `hasattr`
             guard) so none can crash today. Filed because `_close_request_adapter` is the
             ownership test that decides whether `adapter.close()` runs, and `adapter.close()` is
             the trigger for tts-1. The two findings share a mechanism: if the ownership test ever
             answers wrong, tts-1's blast radius grows from BYOK requests to all requests.
             It clears the encapsulation drop rule — these are not DI, not DTOs, not thin
             adapters, and there is a concrete caller reaching in.
tests:       import-grep reachability, not coverage. `adapter_registry` 27 importers,
             `tts_service_v2` 35. The registry is well enough covered that adding two public
             predicates is low-risk; nothing currently asserts the ownership test's behaviour.
effort:      cheap. Two small public methods, one property swap, three call-site edits.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (all three reach-ins and the bypassed public property read directly).
```

---

### FINDING tts-14 — `tests/TTS` and `tests/TTS_NEW` are two live, undocumented, non-overlapping suites, and production code exists to satisfy the newer one

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       tldw_Server_API/tests/TTS      — 56 files, 18,937 LOC, 614 collected, 125 commits/24mo,
                                              first 2025-08-23, last 2026-09-07, **no conftest.py**,
                                              117 skip/skipif markers
             tldw_Server_API/tests/TTS_NEW  — 93 files, 31,322 LOC, 1,185 collected + 3 collection
                                              errors, 202 commits/24mo, first 2025-09-04, last
                                              2026-09-09, conftest.py with 22 fixtures and 7 custom
                                              markers (:47-57), 25 skip/skipif markers
             the production code that exists for the split:
             core/TTS/adapters/openai_adapter.py:OpenAITTSAdapter (589-801)
               — docstring: "Compatibility wrapper with extended OpenAI interface for TTS_NEW tests"
             core/TTS/adapters/elevenlabs_adapter.py:ElevenLabsTTSAdapter (714-1002)
               — docstring: "Compatibility wrapper with extended ElevenLabs interface for TTS_NEW tests"
             both registered as production at adapter_registry.py:329 and :334 (stage 2, tts-2)
             the one shared basename, testing two different classes of the same file:
             tests/TTS/test_elevenlabs_adapter.py:9        imports `ElevenLabsAdapter`
             tests/TTS_NEW/unit/adapters/test_elevenlabs_adapter.py:13 imports `ElevenLabsTTSAdapter`
             the security assertions that exist in exactly one tree:
             tests/TTS/test_elevenlabs_adapter.py:assert_sanitized_error (26-38)
               — `grep -rln "assert_sanitized_error"` matches this one file repo-wide
             the three currently-dead files:
             tests/TTS_NEW/property/test_echo_tts_chunking_properties.py
             tests/TTS_NEW/property/test_tts_properties.py
             tests/TTS_NEW/unit/test_tts_gateway_properties.py   — all `No module named 'hypothesis'`
             classes with zero direct test importers:
             core/TTS/adapters/qwen3_runtime_mlx.py:Qwen3MlxRuntime (18)
             core/TTS/adapters/qwen3_runtime_upstream.py:Qwen3UpstreamRuntime (11)
canonical:   NONE. No README, conftest comment, ADR, or design doc explains the split.
             `tests/TTS/adapters/README.md` documents the mock/integration convention for its own
             tree and never mentions `TTS_NEW`; `core/TTS/README.md:140-141` lists both suites
             side by side as things to run, with no rationale.
destination: n/a. The decision to make is which tree is canonical, recorded as an ADR — this is a
             decision, not a refactor, and ADR-001 governs. It is not a code-movement exercise.
knowledge:   "what is the expected behaviour of a TTS adapter". Two suites answer differently and
             neither is wrong:
             • `tests/TTS` is a **security/sanitization** suite — `assert_sanitized_error` (:26-38)
               asserts `exc.__cause__ is None`, `exc.__context__ is None`, and that raw credential
               markers never appear in `traceback.format_exception`. It drives real registry and
               factory objects (`test_tts_service_v2.py:18,33`) and asserts on private attributes
               (`adapter._status == ProviderStatus.NOT_CONFIGURED`,
               `test_openai_adapter_mock.py:47`).
             • `tests/TTS_NEW` is a **public-surface contract** suite — config-key mapping (:41-46),
               typed exceptions (:57), `supported_models` membership (:66-69); it asserts only
               public API (`adapter.provider_name.lower() == "openai"`, :51) and relies on
               `asyncio_mode = "auto"` (pyproject.toml:636) rather than explicit markers.
             They are complementary, and nothing says so. The concrete cost is not the duplicated
             LOC — it is that satisfying the second suite required adding two classes to
             production and then registering them as the real adapters (tts-2). The test estate
             is now shaping shipped behaviour, and neither suite's authors can see that from
             inside their own tree.
impact:      Medium. Three distinct consequences, all live:
             (a) **tts-2 cannot be fixed without resolving this.** Deleting the shim classes is the
                 right end state and is blocked on reconciling the suites.
             (b) **Coverage partitions rather than overlaps**, so a reader of either tree gets a
                 false sense of completeness. `voice_manager` has 8 importers in `TTS_NEW` and
                 **0** in `TTS`; `tts_resource_manager` has 9 in `TTS` and **1** in `TTS_NEW`;
                 `kokoro_adapter` 9/0; `qwen3_tts_adapter` 0/7; `echo_tts_adapter` 0/5. Only one
                 basename is shared between 149 files, and even that pair tests different classes.
             (c) **Silent erosion.** `hypothesis` is not installed, so three `TTS_NEW` files —
                 including the gateway property tests — collect as errors. `gateway_catalog`,
                 `gateway_execution`, `gateway_preflight` and `phoneme_overrides` each have exactly
                 **one** importer, so a single file's failure zeroes them. Separately,
                 `tests/TTS` carries 117 skip/skipif markers against `TTS_NEW`'s 25, so a large
                 share of its 614 collected tests never execute.
             Neither tree is abandoned — last commits 2026-09-07 and 2026-09-09, two days apart —
             so this is not a stalled migration that will resolve itself.
             `Qwen3MlxRuntime` and `Qwen3UpstreamRuntime` have **zero direct importers**: the two
             tests that nominally cover them (`tests/TTS_NEW/unit/adapters/test_qwen3_mlx_runtime.py:9-11`,
             `test_qwen3_upstream_runtime.py:7-8`) import only `Qwen3TTSAdapter` and stub the
             runtimes via `sys.modules` injection, so the concrete classes are never constructed.
             They are the only genuinely untested classes in the module — and even they are
             reachable, so **no part of this module is untested**.
tests:       this finding *is* the test review. Full per-module and per-adapter reachability is in
             the table above and in the per-adapter matrix sidecar.
effort:      expensive, and deliberately so — this is a decision with a 149-file blast radius.
             The cheap slices that should not wait for it: add `hypothesis` to the dev extra
             (one line in `pyproject.toml`, recovers 3 files); add a direct construction test for
             `Qwen3MlxRuntime` and `Qwen3UpstreamRuntime` (two small files); audit the 117
             skip markers in `tests/TTS` for ones that are now unconditionally skipping.
owner-only:  no (`tldw_Server_API/tests/**` and `pyproject.toml`).
confidence:  confirmed (all counts, dates, collection results, the shim docstrings, the registry
             registrations, and the absence of any explaining document were established by direct
             command output recorded above);
             assumption (that `TTS_NEW` was *intended* as a replacement for `TTS` — nothing states
             this, and the commit history is equally consistent with a deliberate two-suite
             strategy that was never written down. The recommendation is to write the decision
             down either way, not to assume which one it is).
```

---

## Dropped after review — recorded so they are not rediscovered

- **`stream_errors_as_audio` defaulting wrong.** Checked against ADR-011's "structured streaming
  failures are the default" clause. `tts_service_v2.py:312-320` parses the config with
  `parse_bool(..., default=False)` and the constructor logs a warning at `:328` when it is
  enabled. Correctly implemented; the 12 call sites downstream are the sanctioned escape hatch.
- **`generate_speech` taking an `OpenAISpeechRequest`.** Flagged, then narrowed: the transport DTO
  crossing into core is real but it is the *mild* schema-import class at 3 sites, and the
  actionable part of it is the metadata side channel. Folded into tts-10 rather than filed twice.
- **`core → api/v1` layering.** 3 sites, all `api/v1/schemas`, no `endpoints`/`API_Deps` imports —
  i.e. none of the "true inversion" class. Far below the threshold where the ratchet-test
  precedent (`tests/lint/test_endpoint_auth_deps_import_boundary.py`) would be proportionate.
- **`generate_speech` at ~540 lines.** A god method inside a god file. Not filed as its own
  finding because "this function is long" is not an axis — every concrete defect found inside it
  (tts-2's uncaught raise, tts-5's abandon paths, tts-11's per-request construction) is filed
  individually with its own scenario, which is the actionable form. The decomposition template if
  it is ever tackled is the shipped `core/DB_Management/media_db/` package split.
- **Raw SQL outside `DB_Management/`.** Zero hits. Not applicable.
- **`_provider_alias_tokens` / alias normalization (adapter_registry.py:127-136)** duplicated in
  spirit by `tts_service_v2._provider_aliases (3634-3657)`. Looked duplicated; is not — the
  registry normalizes a config key, the service builds an exclusion set that must also cover class
  names. `justified-divergence`.

## Suggested Refactor/Actions

1. **tts-12, dead-knob half** — delete the `max_text_length` key from the 18
   `ProviderLimits.LIMITS` entries; decide `ProviderConfig.max_retries` (wire it up or delete it
   and correct `TTS-DEPLOYMENT.md:165`). Add one parametrised test asserting every `TTSProvider`
   member resolves to the same limit through both surviving paths. Cheap, no design doc.
2. **tts-14, cheap slices only** — add `hypothesis` to the dev extra; add direct construction
   tests for `Qwen3MlxRuntime` and `Qwen3UpstreamRuntime`; triage the 117 skip markers in
   `tests/TTS`. Three small Backlog tasks; none of them requires the big decision.
3. **tts-13** — add `TTSAdapterRegistry.owns()` / `.has_spec()`, switch `tts_service_v2.py:1060`
   to the public `adapter.capabilities` property. Cheap. Do it **before** tts-1's fix so the
   ownership test being corrected is the public one.
4. **tts-11** — key a small override-adapter cache off provider + override hash, with the idle
   eviction `tts_resource_manager` already implements. Sequence **after** tts-1, since both touch
   adapter lifetime. Needs a short design note:
   `Docs/Design/2026-09-21-tts-override-adapter-caching-design.md`.
5. **tts-10** — replace the `_tts_metadata` side channel with an explicit out-parameter or a
   separate awaitable. **Owner-only**: 8 of 10 readers are in
   `api/v1/endpoints/audio/audio_tts.py`. Needs
   `Docs/Design/2026-09-21-tts-speech-metadata-channel-design.md` and an owner to carry the
   endpoint half.
6. **tts-14, the decision** — an ADR recording which suite is canonical and what the other is for,
   per ADR-001's workflow. Everything else about the split, including deleting the two production
   shim classes (stage 2, tts-2 part b), is downstream of that ADR and should not be attempted
   before it. This is the single highest-leverage item in the module and the slowest; start it in
   parallel with items 1-5 rather than after them.

Items 1-4 are `app/core/**` and `tests/**`. Item 5 is **owner-only**. Item 6 is an ADR.
Per the briefing, Backlog tasks are proposed here only — none were created, and no task file was
edited. Base branch assumed to be `dev` per `CONTRIBUTING.md:86,121`; nothing in this stage
depends on that choice.

## Pointers back to earlier stages

- Module shape, churn ranking, ADR-011 constraint check, and seed-cluster applicability:
  `2026-09-21-stage1-architecture-survey.md`.
- Findings tts-1 through tts-9, the adapter-family analysis, and the C8 seed-cluster resolution:
  `2026-09-21-stage2-adapter-family.md`. tts-2 and tts-14 are two halves of one problem and should
  be read together; tts-1, tts-11 and tts-13 all turn on adapter lifetime and should be sequenced
  as one thread.
- Raw inventories: `2026-09-21-stage1-source-inventory.txt`,
  `2026-09-21-stage1-churn-baseline.txt`, `2026-09-21-stage2-adapter-matrix.txt`.
