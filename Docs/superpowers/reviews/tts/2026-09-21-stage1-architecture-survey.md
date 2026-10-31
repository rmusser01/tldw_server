# Stage 1 — Architecture survey and inventory

## Scope

Establish the shape of `tldw_Server_API/app/core/TTS/` before reviewing it: what the files are,
which ones change, what the binding constraints are, and which of the audit's seed clusters
actually land here. No findings are expected to originate in this stage; it exists so stages 2
and 3 can prioritise by size × churn rather than by reading order.

## Code Paths Reviewed

- `core/TTS/adapters/base.py:TTSAdapter (279-570)` — the abstract adapter contract every provider
  implements: `initialize`, `generate`, `get_capabilities`, plus concrete `validate_request (360)`,
  `ensure_initialized (396)`, `convert_audio_format (434)`, `close (470)`, `stream_audio (551)`.
- `core/TTS/adapters/base.py:TTSRequest (99-241)` / `TTSResponse (244-277)` — the unified DTOs,
  both carrying dual-name compatibility fields (`audio_data`/`audio_content`,
  `duration`/`duration_seconds`) and a sentinel-based `__post_init__` supplied-field tracker.
- `core/TTS/adapter_registry.py:TTSProvider (92-124)` — 23 provider enum members, 2 of them
  (`ALLTALK`, `MOCK`) documented placeholders with no adapter.
- `core/TTS/adapter_registry.py:TTSAdapterRegistry.DEFAULT_ADAPTERS (328-...)` — provider → dotted
  class path table; the single place that decides which class is production.
- `core/TTS/adapter_registry.py:MODEL_PROVIDER_MAP (1375-1470+)` — model-id → provider routing
  table backing ADR-011's model-first selection.
- `core/TTS/tts_service_v2.py:TTSServiceV2 (239-4305)` — 4,357 LOC, 74 commits/12mo, the module's
  single hottest file and its orchestrator.
- `core/TTS/tts_validation.py:ProviderLimits (77-243)` and `:TTSInputValidator (245-...)` — the
  two provider-limit tables.
- `core/TTS/waveform_streamer.py:stream_encoded_waveform (9-62)`, `:encode_waveform_to_bytes (65-76)`
  — the module's designated shared audio-encoding helpers.
- `core/TTS/streaming_audio_writer.py:StreamingAudioWriter (22-350)`, `:AudioNormalizer (352-...)`.
- `core/TTS/utils.py:parse_bool (64-93)` — the module's designated scalar coercion helper.

### Module layout

| Area | Files | Role |
| --- | --- | --- |
| `adapters/` | 35 | 25 provider adapters + 4 Qwen3 runtimes + 6 sidecar/client/config support files |
| `backends/` | 4 | Fish-S2 transport split (base / commercial API / native HTTP) |
| `vendors/` | 7 | Third-party vendored runtimes (neuttsair, supertonic, supertonic2, kittentts) |
| root | 26 | service, registry, validation, config, resource manager, gateway (4), audio helpers (4), jobs worker, circuit breaker, realtime session |

### Size × churn (top 10, from the sidecars)

| File | LOC | Commits/12mo |
| --- | --- | --- |
| `tts_service_v2.py` | 4,357 | 74 |
| `adapter_registry.py` | 1,615 | 56 |
| `tts_validation.py` | 1,343 | 37 |
| `tts_config.py` | 649 | 35 |
| `voice_manager.py` | 1,624 | 31 |
| `adapters/kokoro_adapter.py` | 1,682 | 28 |
| `adapters/openai_adapter.py` | 805 | 25 |
| `adapters/vibevoice_adapter.py` | 1,534 | 22 |
| `audio_utils.py` | 817 | 19 |
| `adapters/elevenlabs_adapter.py` | 1,002 | 19 |
| `adapters/base.py` | 570 | 19 |

Full lists: `2026-09-21-stage1-source-inventory.txt`, `2026-09-21-stage1-churn-baseline.txt`.

The hot set is not the biggest set. `tts_service_v2.py` is both, but `adapter_registry.py`
(1,615 LOC / 56 commits) outranks `kokoro_adapter.py` (1,682 / 28) on churn at equal size, and
`tts_config.py` (649 / 35) is a small file that moves constantly. Stage 2 therefore weights the
registry and the two most-churned remote adapters (openai, elevenlabs) over the larger but
quieter local adapters.

## Tests Reviewed

Located by import-grep only, per the audit method — `grep -rl "core\.TTS" tldw_Server_API/tests`.
Detailed per-module and per-adapter reachability is recorded in stage 3; stage 1 records only the
top-level shape, because it changes what stage 2 is allowed to conclude.

- **167 test files** import `core.TTS`. 132 live under `tests/TTS` (55) and `tests/TTS_NEW` (77);
  **35 (21%) live outside both** — `tests/Audio` (13), `tests/Persona` (4), `tests/Audiobooks` (3),
  and ten other directories with 1-2 each. Path search of `tests/TTS*` would have missed a fifth
  of the coverage.
- Consequence for stage 2: no adapter in this module may be described as untested without an
  explicit per-name import-grep. Two are (see stage 3, finding tts-14).

## Validation Commands

```
$ find tldw_Server_API/app/core/TTS -name '*.py' -not -path '*__pycache__*' | wc -l
      72

$ find tldw_Server_API/app/core/TTS -name '*.py' -not -path '*__pycache__*' | xargs wc -l | tail -1
   42067 total

$ grep -rl "core\.TTS" tldw_Server_API/tests | wc -l
167

$ grep -rl "core\.TTS" tldw_Server_API/tests | sed 's|tldw_Server_API/tests/||' | cut -d/ -f1 | sort | uniq -c | sort -rn
  77 TTS_NEW
  55 TTS
  13 Audio
   4 Persona
   3 Audiobooks
   2 Watchlists
   2 VoiceAssistant
   2 Setup
   2 Config
   1 Workflows
   1 Storage
   1 Services
   1 Research_Workspace
   1 Infrastructure
   1 DB_Management
   1 Collections

$ grep -rn "from tldw_Server_API.app.api\|import tldw_Server_API.app.api" tldw_Server_API/app/core/TTS --include='*.py'
tldw_Server_API/app/core/TTS/realtime_session.py:14:from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
tldw_Server_API/app/core/TTS/tts_service_v2.py:25:from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
tldw_Server_API/app/core/TTS/tts_jobs_worker.py:20:from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest

$ grep -rn "SELECT \|INSERT INTO\|UPDATE .* SET" tldw_Server_API/app/core/TTS --include='*.py' | wc -l
0

$ grep -c "core/TTS" pyproject.toml     # BLE001 per-file-ignores membership
23
```

### Constraint check against `Docs/ADR/011-audio-api-semantics.md`

Read in full before any assertion in stages 2-3. Four clauses constrain findings here:

1. **Model-first routing** is binding. Nothing in this review proposes changing routing priority;
   finding tts-12 concerns the *tables* model-first routing reads, not the order it reads them in.
2. **Structured streaming errors are the default**; error-as-audio is "a configuration escape
   hatch, not the normal behavior". Verified at `tts_service_v2.py:312-320` — the
   `stream_errors_as_audio` config parse explicitly passes `default=False` and the constructor
   logs a warning at `:328` when it is on. **This is correctly implemented; not a finding.**
3. **`return_download_link=true` requires `stream=false`.** Not contradicted by anything below.
4. **Adapter init failures are retried after a cooldown, not permanent until restart.** Finding
   tts-1 in stage 2 defeats this clause in practice for one provider; it is filed as a correctness
   defect *against* the ADR, not a disagreement with it.

### Seed-cluster applicability

Per the audit briefing, only cluster **C8 (error → HTTP status mapping)** maps to this module.
Confirmed and expanded in stage 2 (finding tts-9). Clusters C1-C7, C9, C10 were spot-checked
against this module and have no sites here:

- C1 (base64 cursor padding): `grep -rn 'urlsafe_b64decode' core/TTS` → 0 hits.
- C2 (scalar/env coercion): applies in a *local* form — this module has its own canonical
  `parse_bool` and three bypassers. Filed as tts-4, scoped to the module, not to the repo cluster.
- C3-C7, C9, C10: no sites. Not investigated further.

## Findings

None. This stage is inventory. Findings begin in stage 2.

One item is recorded here rather than filed, because the briefing designates it noise:
23 of the module's files appear in the `pyproject.toml` `[tool.ruff.lint.per-file-ignores]`
BLE001 grandfather block (`pyproject.toml:1267-1290`), including every adapter this review
touches. Blind-except in those files is sanctioned policy and is not reported anywhere below.

## Suggested Refactor/Actions

None for this stage. The prioritisation it produced:

1. Stage 2 targets `adapters/` weighted by churn — openai, elevenlabs, kokoro, vibevoice first.
2. Stage 3 targets `tts_service_v2.py` (the god module) and the API boundary.
3. The three `core → api/v1/schemas` imports are the mild, known layering class (schema in the
   wrong package, not core misbehaving). Carried into stage 3 as part of finding tts-10 rather
   than filed as a layering finding of its own — three sites do not justify a ratchet test.
