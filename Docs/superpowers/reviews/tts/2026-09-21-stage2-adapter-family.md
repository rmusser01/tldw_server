# Stage 2 — The adapter family

## Scope

The 25 provider adapters under `core/TTS/adapters/` plus the 4 Fish-S2 backends, reviewed as a
*family* rather than one file at a time. The question this stage answers is: for each concern
that every adapter must handle — transport-error mapping, waveform encoding, byte accumulation,
scalar coercion, HTTP-client ownership, request validation — do the adapters agree, and where
they disagree, **which one is right**?

The per-adapter matrix backing this stage is `2026-09-21-stage2-adapter-matrix.txt`
(concerns A–H). Findings below cite it but do not repeat it.

Out of scope for this stage: `vendors/` (7 files, vendored third-party code, not ours to
restructure) and the provider-specific generation internals (prompt assembly, model kwargs) —
those are genuinely per-provider domain rules and are `justified-divergence` by construction.

## Code Paths Reviewed

- `adapters/openai_adapter.py:OpenAIAdapter (~100-586)` and `:OpenAITTSAdapter (589-801)` —
  including `_is_httpx_exception (47)`, `_is_http_status_error (52)`, `_is_timeout_error (58)`,
  `_safe_exception_label (65)`, `_bounded_network_error (70)`, `_bounded_existing_tts_error (80)`,
  `initialize (191-270)`, `_cleanup_resources (529-541)`, `validate_request (663-686)`,
  `generate (688-752)`, `generate_stream (753-790)`.
- `adapters/elevenlabs_adapter.py:ElevenLabsAdapter (~110-712)` and
  `:ElevenLabsTTSAdapter (714-1002)` — including the same six helpers at `49/54/60/65/71/81`,
  `initialize (252-292)`, `_fetch_user_voices (294-...)`, `_cleanup_resources (676-687)`,
  `fetch_voices (773-788)`, `get_voice_info (790-802)`, `clone_voice (804-820)`,
  `get_usage (822-843)`, `validate_request (847-886)`, `generate (888-...)`,
  `generate_stream (901-...)`.
- `adapters/qwen3_runtime_remote.py:RemoteQwenRuntime (54-404)` — `_coerce_bool (94-105)`,
  `_is_httpx_exception (126)`, `_is_http_status_error (130)`, `_is_timeout_error (135)`,
  `_parse_retry_after (141-151)`, `_normalize_http_status (153-187)`,
  `_normalize_http_status_error (189-194)`, `_normalize_remote_error (196-...)`,
  `initialize (240-251)`.
- `adapters/base.py:TTSAdapter.validate_request (360-394)`, `:convert_audio_format (434-466)`,
  `:ensure_initialized (396-432)`, `:close (470-479)`, `:_cleanup_resources (481-483)`.
- `core/TTS/waveform_streamer.py:stream_encoded_waveform (9-62)`, `:encode_waveform_to_bytes (65-76)`.
- `core/TTS/streaming_audio_writer.py:StreamingAudioWriter.write_chunk (108-194)`, `:close (196-231)`,
  `:_spill_wav_buffer_to_file (261-280)`, `:_finalize_wav (282-308)`.
- `adapters/luxtts_adapter.py:_stream_audio (533-570)`; `adapters/index_tts_adapter.py:_stream_audio_index_tts (404-467)`.
- `adapters/kitten_tts_adapter.py:_stream_audio (234-246)`.
- `adapters/{dia_adapter.py:477-483, higgs_adapter.py:494-500, kokoro_adapter.py:1100-1104 and
  1118-1131, vibevoice_adapter.py:1120-1126, chatterbox_adapter.py:1112-1120 and 1205-1214}` —
  the five stream-drain implementations.
- `adapters/audio_cpp_config.py:_as_bool (18-29)`, `:AudioCppConfig.from_provider_config (124-155)`,
  `:render_server_config (183-215)`; `adapters/audio_cpp_sidecar_supervisor.py:_as_bool (158-170)`.
- `core/TTS/utils.py:parse_bool (64-93)`; `core/testing.py:_env_truthy (23-27)`, `:is_truthy (30-32)`.
- `core/TTS/adapter_registry.py:DEFAULT_ADAPTERS (328-350)`,
  `:create_adapter_with_overrides (671-743)`, `:_initialize_adapter (745-900)`,
  `:unload_adapter (~1225-1270)`, `:shutdown_all (~1185-1210)`.
- `core/TTS/tts_resource_manager.py:ConnectionPool.get_client (226-280)`, `:close_pool (282-289)`, `:close_client (291-293)`,
  `:close_all (295-306)`, `:ResourceManager.get_http_client (865-867)`.

## Tests Reviewed

All located by import-grep (`grep -rlE "adapters\.<module>\b|adapters import [^#]*\b<module>\b"`),
never by path. These are **reachability counts, not measured coverage** — no suite was executed
for this review.

| Test file | Protects | Downgrades risk? |
| --- | --- | --- |
| `tests/TTS/test_elevenlabs_adapter.py:191-205` `test_cleanup_failure_log_sanitizes_exception_text` | That `_cleanup_resources` log output is sanitized | **No — it pins the defect.** It mocks `adapter.client` and asserts the `aclose()` path is taken. Fixing tts-1 requires editing this test, which is why tts-1's effort is "moderate" not "cheap". |
| `tests/TTS/test_tts_resource_manager.py:222-230` `test_close_client` | `ConnectionPool.close_client` closes *and* evicts from `_pools` | Partially — it proves the correct teardown helper works. It does not prove any adapter calls it. |
| `tests/TTS_NEW/unit/adapters/test_openai_adapter.py:731,772` | OpenAI `_cleanup_resources` sanitization | Yes for OpenAI — incidentally documents that OpenAI's cleanup does *not* aclose the shared client. |
| `tests/TTS/adapters/test_chatterbox_adapter_mock.py:517,681,757,873` | `waveform_streamer.stream_encoded_waveform` is invoked by chatterbox | Yes — this is the only test anywhere that touches the canonical encoder. Confirms the helper works; also confirms only one adapter reaches it. |
| `tests/TTS/adapters/test_openai_adapter_mock.py`, `tests/TTS_NEW/unit/adapters/test_openai_adapter.py` (12 direct importers total: 6 in `tests/TTS`, 4 in `tests/TTS_NEW`, 2 elsewhere) | OpenAI adapter surface | Partially. Neither suite constructs two adapters concurrently, so the `self.model` race (tts-8) is unreachable by the current tests. |
| `tests/TTS/test_elevenlabs_adapter.py`, `tests/TTS/adapters/test_elevenlabs_adapter_{mock,integration}.py`, +5 more (8 importers) | ElevenLabs adapter surface | Partially — good sanitization coverage, no lifecycle/pool coverage. |
| kokoro 9 importers, higgs 8, dia 6, vibevoice 8, chatterbox 6 (all concentrated in `tests/TTS`) | Local adapter generate paths | No for tts-3 — none of them measure accumulation cost, and all use short fixtures where O(n²) is invisible. |
| luxtts 2 importers, index_tts 1 importer | The two adapters with the unstarted-generator leak (tts-5) | No. Thinnest coverage in the family; neither test exercises `metadata_only`. |
| `tests/TTS_NEW/unit/adapters/test_qwen3_remote_runtime.py` + 2 more (3 importers of `qwen3_runtime_remote`) | `RemoteQwenRuntime` error normalization | Yes — the best-implemented error mapper is also the tested one, which strengthens the "promote this copy" recommendation in tts-9. |
| `audio_cpp` — 3 direct / 10 by class name, 6 in `tests/TTS_NEW` | audio.cpp config parsing | Unknown for tts-4: no test passes a non-canonical truthy string such as `"n"` to `_as_bool`. |

## Validation Commands

```
$ grep -rn "_is_http_status_error" tldw_Server_API/app --include='*.py'
tldw_Server_API/app/core/TTS/adapters/openai_adapter.py:52:def _is_http_status_error(exc: Exception) -> bool:
tldw_Server_API/app/core/TTS/adapters/openai_adapter.py:234:                        if not _is_http_status_error(e):
tldw_Server_API/app/core/TTS/adapters/openai_adapter.py:437:        if _is_http_status_error(e):
tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py:54:def _is_http_status_error(exc: Exception) -> bool:
tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py:499:            if _is_http_status_error(e):
tldw_Server_API/app/core/TTS/adapters/elevenlabs_adapter.py:649:        if _is_http_status_error(exc):
tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_remote.py:130:    def _is_http_status_error(self, exc: Exception) -> bool:
tldw_Server_API/app/core/TTS/adapters/qwen3_runtime_remote.py:200:        if self._is_http_status_error(exc):

  (Note: the briefing's fourth copy, core/LLM_Calls/error_utils.py:407, does NOT exist under that
   name — `grep -rn "_is_http_status_error" app/core/LLM_Calls` returns nothing. All three
   surviving copies are in this module. The line numbers 52/54/130 are exact, not drifted.)

$ grep -rn "def _safe_exception_label" tldw_Server_API/app/core/TTS --include='*.py' | wc -l
12

$ grep -rn "StreamingAudioWriter(" tldw_Server_API/app/core/TTS --include='*.py' | grep -v vendors | wc -l
18

$ grep -rn "waveform_streamer" tldw_Server_API/app --include='*.py' | grep -v __pycache__ | grep -v "waveform_streamer.py:"
tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py:1092: from ...waveform_streamer import stream_encoded_waveform
tldw_Server_API/app/core/TTS/adapters/chatterbox_adapter.py:1188: from ...waveform_streamer import stream_encoded_waveform
  -> 1 adapter, 2 call sites, out of 25 adapters.

$ grep -rn "all_audio += chunk" tldw_Server_API/app/core/TTS --include='*.py'
.../adapters/vibevoice_adapter.py:1125:            all_audio += chunk
.../adapters/kokoro_adapter.py:1103:            all_audio += chunk
.../adapters/kokoro_adapter.py:1126:            all_audio += chunk
.../adapters/higgs_adapter.py:499:            all_audio += chunk
.../adapters/dia_adapter.py:482:            all_audio += chunk

$ python3 -c '<bytes += vs bytearray += microbenchmark, 1200-byte chunks>'
chunks=  500 total=0.60MB  bytes+=:     2.4ms   bytearray+=:   0.05ms   ratio=   44.8x
chunks= 2000 total=2.40MB  bytes+=:    47.0ms   bytearray+=:   0.25ms   ratio=  186.4x
chunks= 8000 total=9.60MB  bytes+=:  1087.7ms   bytearray+=:   1.05ms   ratio= 1032.4x

$ python3 -c 'from ...TTS.utils import parse_bool; from ...adapters.audio_cpp_config import _as_bool; ...'
'y'          parse_bool(default=False)=True   _as_bool(default=False)=True
'n'          parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
'none'       parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
'null'       parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
''           parse_bool(default=False)=False  _as_bool(default=False)=False
'disabled'   parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
'maybe'      parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
'T'          parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE
'enabled'    parse_bool(default=False)=False  _as_bool(default=False)=True     <<< DIVERGE

$ python3 -c '<async generator aclose semantics probe>'
unstarted aclose  -> finally ran: [] resource still open: True
started   aclose  -> finally ran: ['finally-ran'] resource still open: False

$ grep -n "DEFAULT_ADAPTERS" -A 8 tldw_Server_API/app/core/TTS/adapter_registry.py | sed -n '3,9p'
    TTSProvider.OPENAI: "...adapters.openai_adapter.OpenAITTSAdapter",
    TTSProvider.KOKORO: "...adapters.kokoro_adapter.KokoroAdapter",
    TTSProvider.HIGGS: "...adapters.higgs_adapter.HiggsAdapter",
    TTSProvider.DIA: "...adapters.dia_adapter.DiaAdapter",
    TTSProvider.CHATTERBOX: "...adapters.chatterbox_adapter.ChatterboxAdapter",
    TTSProvider.ELEVENLABS: "...adapters.elevenlabs_adapter.ElevenLabsTTSAdapter",
    TTSProvider.VIBEVOICE: "...adapters.vibevoice_adapter.VibeVoiceAdapter",

$ grep -n "__enter__\|__exit__\|contextmanager" tldw_Server_API/app/core/TTS/streaming_audio_writer.py
  (no output — StreamingAudioWriter is not a context manager)
```

## Findings

---

### FINDING tts-1 — ElevenLabs cleanup closes the process-wide pooled HTTP client

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       core/TTS/adapters/elevenlabs_adapter.py:_cleanup_resources (676-687)  [the wrong copy]
             core/TTS/adapters/openai_adapter.py:_cleanup_resources (529-541)      [the right copy]
             core/TTS/adapters/qwen3_runtime_remote.py:RemoteQwenRuntime (54-404)  [no override; correct by omission]
             core/TTS/adapters/elevenlabs_adapter.py:initialize (263-267)          [acquires the shared client]
             core/TTS/adapters/openai_adapter.py:initialize (205-209)              [acquires the shared client]
             core/TTS/adapters/qwen3_runtime_remote.py:initialize (246-250)        [acquires the shared client]
             core/TTS/tts_resource_manager.py:ConnectionPool.get_client (226-280)  [the shared cache]
             core/TTS/tts_resource_manager.py:ConnectionPool.close_pool (282-289) / :close_client (291-293, alias)[the correct teardown]
             core/TTS/adapter_registry.py:_initialize_adapter (892-900)            [close-on-init-failure trigger]
             core/TTS/adapter_registry.py:create_adapter_with_overrides (734-743)  [close-on-init-failure trigger]
             core/TTS/adapter_registry.py:unload_adapter (1248-1251)               [operator-triggered close]
             core/TTS/tts_service_v2.py:_close_request_adapter (562-580)           [per-request close for BYOK]
canonical:   core/TTS/tts_resource_manager.py:ConnectionPool.close_pool (282-289), exposed as
             :close_client (291-293) — the only teardown that both closes the client AND evicts
             it from `_pools`.
destination: n/a — this is a deletion, not a new module. Delete the `aclose()` from
             elevenlabs_adapter.py:678-682 so it matches openai_adapter.py:529-541.
knowledge:   "who owns the lifetime of the per-provider pooled httpx client". The resource manager
             owns it (`ConnectionPool._pools`, keyed by provider, never evicted except by
             `close_client`/`close_all`). Two of three HTTP adapters encode that correctly;
             one encodes the opposite. Any future adapter copy-pasted from ElevenLabs inherits
             the wrong answer, and there is no assertion anywhere that would catch it.
scenario:    ElevenLabs is configured. Its `initialize()` fetches the shared client from
             `ConnectionPool._pools["elevenlabstts"]` (:263) and then calls `_fetch_user_voices()`
             (:271), which issues a live HTTP GET. A transient network blip or a briefly-invalid
             API key makes that raise. `adapter_registry._initialize_adapter` catches it and, in
             its `finally` at :892-900, calls `adapter.close()` -> `_cleanup_resources()` ->
             `await self.client.aclose()`. The *shared pooled client* is now closed. It is still
             sitting in `ConnectionPool._pools["elevenlabstts"]` because `aclose()` does not evict
             — only `close_pool` (282-289), reachable as `close_client` (291-293), does. ADR-011 says adapter init failures "can be
             retried after a configured cooldown"; on every retry, `ConnectionPool.get_client` (226-280) finds the
             cached entry and hands back the dead client, and `_fetch_user_voices` raises
             `RuntimeError: Cannot send a request, as the client has been closed.` **One transient
             failure becomes permanent until process restart**, defeating the ADR's cooldown
             clause. The same poisoning is reachable three other ways: an operator calling
             `unload_adapter` (:1248), an override adapter that fails to init (:734), and — on
             every single request in a BYOK deployment — `tts_service_v2._close_request_adapter`
             (:562-580), which closes any adapter not present in `registry._adapters`.
impact:      High. Provider-wide, persistent, silent (the failure surfaces as a generic network
             error on later requests, with no log line connecting it to the earlier cleanup), and
             it converts a recoverable fault into an unrecoverable one — the exact outcome ADR-011
             chose against.
tests:       import-grep reachability, not coverage. 8 files import `elevenlabs_adapter`;
             `tests/TTS/test_elevenlabs_adapter.py:191-205` is the only one touching
             `_cleanup_resources`, and it asserts the buggy call happens.
             `tests/TTS/test_tts_resource_manager.py:222-230` covers the correct
             `ConnectionPool.close_client`. Nothing tests the two together.
effort:      moderate. The code change is deleting five lines. The cost is that
             `test_cleanup_failure_log_sanitizes_exception_text` currently depends on the
             `aclose()` call and must be retargeted, and a regression test asserting
             "the pooled client survives adapter close" needs to be added.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (the divergence, the shared cache, the non-evicting close, and the four
             trigger paths are all read directly from the code);
             probable-risk (that this is actually firing in a live deployment today — that depends
             on whether ElevenLabs init has ever transiently failed there).
```

---

### FINDING tts-2 — The registered production adapters for OpenAI and ElevenLabs are classes written for the test suite, and they break the base `validate_request` contract

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       core/TTS/adapter_registry.py:DEFAULT_ADAPTERS (329)      -> OpenAITTSAdapter
             core/TTS/adapter_registry.py:DEFAULT_ADAPTERS (334)      -> ElevenLabsTTSAdapter
             core/TTS/adapters/openai_adapter.py:OpenAITTSAdapter (589-801)
               docstring (590-595): "Compatibility wrapper with extended OpenAI interface for TTS_NEW tests."
             core/TTS/adapters/elevenlabs_adapter.py:ElevenLabsTTSAdapter (714-1002)
               docstring (715-720): "Compatibility wrapper with extended ElevenLabs interface for TTS_NEW tests."
             core/TTS/adapters/base.py:TTSAdapter.validate_request (360-394)   [the contract: tuple[bool, str|None]]
             core/TTS/adapters/openai_adapter.py:validate_request (663-686)    [returns None / raises]
             core/TTS/adapters/elevenlabs_adapter.py:validate_request (847-886)[returns None / raises]
             core/TTS/adapters/chatterbox_adapter.py:485                       [conforming consumer: `is_valid, error = await self.validate_request(...)`]
             core/TTS/adapters/audio_cpp_adapter.py:193                        [conforming consumer]
             core/TTS/tts_service_v2.py:_get_fallback_adapter (3755-3768)      [the runtime type-sniff and the uncaught raise]
canonical:   core/TTS/adapters/base.py:TTSAdapter.validate_request (360-394) — the declared
             contract, honoured by 23 of 25 adapters.
destination: n/a. Two separate fixes, both subtractive: (a) make the two overrides return the
             base contract's tuple and move the raise-style checks into `generate()` where the
             shims already call them; (b) reconcile the two test suites so the shim classes can be
             deleted and the plain `OpenAIAdapter`/`ElevenLabsAdapter` registered instead.
knowledge:   "what does an adapter say when it cannot serve a request". The base class answers
             "return `(False, reason)` so the caller can try the next provider". The two shims
             answer "raise". A caller cannot satisfy both, which is why
             `tts_service_v2.py:3758-3763` contains a three-branch runtime type-sniff
             (`isinstance(result, tuple)` / `result is None` / `bool(result)`) that exists purely
             to paper over the disagreement. Every future adapter has two contradictory precedents
             to copy, and the type annotation on the base method no longer tells the truth.
scenario:    A 4,500-character request arrives with no provider hint and fallback enabled.
             `_get_fallback_adapter` (3691) iterates `TTSProvider` in declaration order; `OPENAI`
             is first (adapter_registry.py:101). It resolves the adapter and calls
             `await adapter.validate_request(request)` at :3757 — **outside any try/except**
             (the `try` above it at :3739 wraps only `get_adapter`, and closes at :3753).
             `OpenAITTSAdapter.validate_request` (openai_adapter.py:670-671) raises
             `TTSTextTooLongError("Text exceeds maximum for OpenAI")` because 4,500 > 4,096. The
             exception propagates straight out of `_get_fallback_adapter`, so ElevenLabs (limit
             5,000) and Kokoro (limit 1,000,000) are never tried. The caller gets a failure that
             names OpenAI's limit for a request neither of the other providers would have
             rejected. With a base-contract adapter the loop would have logged `(False, ...)` and
             continued. Same shape for any voice not in `OpenAIAdapter.VOICES` (:681-683) and any
             speed outside 0.25-4.0 (:685-686).
impact:      High. It silently disables provider fallback — the feature ADR-011 spends a paragraph
             specifying — for the most common rejection reason (text length), and it does so only
             for the two providers most likely to be configured. Separately, shipping classes
             whose own docstrings say they exist for a test suite means the production behaviour
             of OpenAI and ElevenLabs is defined by whichever assertions `tests/TTS_NEW` happens
             to make; see tts-14 for why that cannot be cleaned up without reconciling the suites.
tests:       import-grep reachability, not coverage. openai_adapter 12 importers (6 `tests/TTS`,
             4 `tests/TTS_NEW`); elevenlabs_adapter 8 (5 / 2). `tts_service_v2` has 35 importers.
             `tests/TTS/test_tts_service_v2.py:33` defines its own `MockAdapter(TTSAdapter)` which
             inherits the *base* `validate_request`, so the fallback loop is only ever exercised
             against conforming adapters — the failure mode above is structurally unreachable by
             the current tests.
effort:      moderate for (a) — the two overrides are ~25 lines each and both already have a
             `generate()` that calls them, so the raise-style checks have a natural home;
             the existing tests assert on the raises and would move with them.
             expensive for (b) — blocked on tts-14.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (the registration, the contract break, the type-sniff, and the absent
             try/except are all read directly). The 4,500-character scenario is derived from the
             code path, not observed in a running system.
```

---

### FINDING tts-3 — Quadratic byte accumulation when draining an audio stream, in four adapters

```
axis:        efficiency
class:       divergent-copies
severity:    High
sites:       core/TTS/adapters/dia_adapter.py:_generate_complete_dia (477-483)
             core/TTS/adapters/higgs_adapter.py:_generate_complete_higgs (494-500)
             core/TTS/adapters/kokoro_adapter.py:_generate_complete_kokoro (1100-1104)
             core/TTS/adapters/kokoro_adapter.py:_generate_complete_kokoro_with_alignment (1118-1131)
             core/TTS/adapters/vibevoice_adapter.py:_generate_complete_vibevoice (1120-1126)
             correct copies, same repo, same module:
             core/TTS/adapters/chatterbox_adapter.py:1112-1120 and :1205-1214 (bytearray)
             core/TTS/waveform_streamer.py:encode_waveform_to_bytes (65-76) (bytearray)
             core/TTS/tts_service_v2.py:1136-1140 (bytearray)
             core/TTS/tts_jobs_worker.py:820 (bytearray)
             core/TTS/gateway_execution.py:443 (bytearray)
canonical:   core/TTS/waveform_streamer.py:encode_waveform_to_bytes (65-76) for the whole
             operation; `bytearray` for the accumulation idiom specifically. Five of nine sites in
             this module already do it right.
destination: n/a — a five-line change at five sites. `all_audio = b""` -> `bytearray()`,
             `return all_audio` -> `return bytes(all_audio)`.
knowledge:   "how to concatenate a stream of byte chunks". `bytes` is immutable, so `b += chunk`
             compiles to `b = b + chunk`: a fresh allocation and a full copy of everything
             accumulated so far, on every chunk. `bytearray.__iadd__` extends in place. Nine sites
             in this module make this choice; four get it wrong. The two that share a *file*
             (kokoro:1103 and kokoro:1126) are both wrong, and the one adapter that was written
             against the shared helper (chatterbox) is right — which is the whole argument for
             tts-6.
cost-driver: O(N·K) memcpy where N = total encoded audio bytes and K = number of chunks yielded
             by the adapter's streaming generator. It scales with **generated audio duration ×
             chunk rate**, not with text length or sample rate directly. Measured on 1,200-byte
             chunks (≈0.2 s of 48 kbps MP3): 500 chunks (0.6 MB) = 2.4 ms; 2,000 chunks (2.4 MB)
             = 47 ms; 8,000 chunks (9.6 MB) = **1,088 ms**, against 1.05 ms for `bytearray` — a
             1,032× gap, and the gap widens with length because the cost is quadratic. A ~25-minute
             Kokoro or VibeVoice generation lands in the 8,000-chunk range. This runs on the event
             loop thread with no `await` between iterations of the copy, so it stalls every other
             in-flight request on the worker.
tests:       import-grep reachability, not coverage. kokoro 9 importers, higgs 8, vibevoice 8,
             dia 6 — all concentrated in `tests/TTS`. None of them measure allocation or use
             fixtures long enough for the quadratic term to show; the defect is invisible to the
             suite by construction.
effort:      cheap. Five mechanical edits, no behaviour change, no new test needed beyond the
             existing generate-path assertions (a `bytearray` returned through `bytes()` is
             indistinguishable to every caller).
owner-only:  no (`app/core/**` only).
confidence:  confirmed (sites read directly; cost measured, output recorded above).
```

---

### FINDING tts-4 — Three copies of a bool coercer bypass the module's own `parse_bool`, and one of them gates an SSRF switch

```
axis:        correctness
class:       adoption-gap
severity:    Medium
sites:       core/TTS/adapters/audio_cpp_config.py:_as_bool (18-29)
             core/TTS/adapters/audio_cpp_sidecar_supervisor.py:_as_bool (158-170)  [byte-identical]
             core/TTS/adapters/qwen3_runtime_remote.py:_coerce_bool (94-105)       [byte-identical]
             call sites that consume the divergent semantics:
             core/TTS/adapters/audio_cpp_config.py:133  allow_remote_base_url  <- SSRF gate
             core/TTS/adapters/audio_cpp_config.py:150  managed
             core/TTS/adapters/audio_cpp_config.py:153  retain_request_artifacts
             core/TTS/adapters/audio_cpp_config.py:212  lazy_load
             core/TTS/adapters/audio_cpp_sidecar_supervisor.py:66  autoselect_port
             core/TTS/adapters/qwen3_runtime_remote.py:261,262,263,267  capability overrides
             correct adopter, same pattern, same layer:
             core/TTS/adapters/echo_tts_adapter.py:_coerce_bool (1200-1204) — delegates to parse_bool
canonical:   core/TTS/utils.py:parse_bool (64-93), itself built on core/testing.py:is_truthy (30).
             Already adopted by `tts_service_v2`, `voice_manager`, `tts_validation`, `tts_config`,
             `adapter_registry`, `vibevoice_realtime_adapter` and `echo_tts_adapter` — 7 importers.
destination: n/a. `from ..utils import parse_bool` and delete the three private copies, exactly as
             `echo_tts_adapter.py:1200` already does.
knowledge:   "what counts as false". `parse_bool` treats an unrecognised string as the caller's
             `default`; the three copies fall through to `bool(value)`, which is True for every
             non-empty string. They also disagree on the falsy set: `parse_bool` accepts
             `n / none / null / off / no / false / 0 / ""`, the copies accept only
             `0 / false / no / off`. Runtime-verified above: `_as_bool("n", False)` returns
             **True** while `parse_bool("n", default=False)` returns **False**. Same for
             `"none"`, `"null"`, `"disabled"`, and any typo.
scenario:    An operator disables the audio.cpp remote-URL escape hatch by writing
             `allow_remote_base_url = n` (or `none`, or `disabled`) in the provider's
             `extra_params`. `_as_bool` returns `bool("n")` = **True**, so
             `audio_cpp_config.py:141-144` calls `validate_base_url(..., allow_remote_base_url=True)`
             and the adapter is permitted to point at an arbitrary non-loopback host — the exact
             thing the flag exists to prevent, and the same string spelled the same way means
             False everywhere else in the TTS module. The operator has no feedback: there is no
             warning for an unparsed token, and the config loads cleanly. `managed` (:150) and
             `retain_request_artifacts` (:153) fail open the same way.
impact:      Medium rather than High only because the flag is operator-set rather than
             request-set, so it is not directly attacker-controlled — the failure is a
             misconfiguration that silently does the opposite of what it says. Were `extra_params`
             ever plumbed from a request body to this constructor, it would become High.
             ADR-026 (`security-outbound-egress-and-ssrf-policy`) governs the gate this weakens.
tests:       import-grep reachability, not coverage. `audio_cpp` has 3 direct importers and 10 by
             class name, concentrated in `tests/TTS_NEW`; `qwen3_runtime_remote` has 3;
             `core/TTS/utils` has 5. No test passes a non-canonical token such as `"n"` to any of
             the three copies, so the divergence is uncovered on both sides.
effort:      cheap. Three imports and three deletions; `echo_tts_adapter.py:1200` is the working
             precedent. Add one table-driven test asserting the four call sites agree with
             `parse_bool` on `{"n","none","null","off","disabled","maybe"}`.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (divergence executed and recorded above; call sites read directly).
```

---

### FINDING tts-5 — Two adapters build the audio writer before the generator starts, so `aclose()` on an abandoned stream leaks the writer and its temp files

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       core/TTS/adapters/luxtts_adapter.py:_stream_audio (533-570)
               writer constructed at :539, `finally: writer.close()` at :568-569 inside the inner
               `stream()` coroutine, `return stream()` at :570
             core/TTS/adapters/index_tts_adapter.py:_stream_audio_index_tts (404-467)
               writer at :417, normalizer at :422, `finally: writer.close()` and
               `self._cleanup_temp_paths(temp_paths)` at :464-466 inside `stream()`,
               `return stream()` at :467
             abandon paths that call `aclose()` without ever starting the generator:
             core/TTS/tts_service_v2.py:_generate_with_adapter (2646-2651)   [metadata_only]
             core/TTS/tts_service_v2.py:generate_speech (2310-2311)          [metadata_only]
             core/TTS/tts_service_v2.py:generate_speech (2055-2057)          [gateway metadata_only]
             core/TTS/tts_service_v2.py:_close_response_audio_stream (554-560)
             endpoint entry point: api/v1/endpoints/audio/audio_tts.py:1436 (`metadata_only=True`)
             correct copies in the same family:
             core/TTS/waveform_streamer.py:stream_encoded_waveform (9-62) — `async def`, so the
               writer is constructed on first `__anext__`, and `finally` always pairs with it
             core/TTS/adapters/kitten_tts_adapter.py:_stream_audio (234-246) — `async def`
             core/TTS/adapters/neutts_adapter.py:348-377 — writer built inside the running generator
canonical:   core/TTS/waveform_streamer.py:stream_encoded_waveform (9-62) — the one implementation
             where construction and teardown are both inside the generator body.
destination: n/a. Either make `_stream_audio` an `async def` generator so the writer is created
             lazily (the kitten/neutts shape), or route through `stream_encoded_waveform` (tts-6).
knowledge:   "when does a resource acquired for a generator get released". A Python async
             generator that has never been advanced has no frame, so `aclose()` returns without
             executing the body — including `finally`. Verified above: unstarted `aclose()` left
             the resource open, started `aclose()` released it. Any resource acquired *before* the
             first `yield` is therefore outside the generator's own cleanup guarantee. Nine of the
             eleven streaming adapters acquire inside; two acquire outside.
scenario:    `POST /api/v1/audio/speech` with the metadata-only shim
             (`audio_tts.py:1436` -> `generate_speech(..., metadata_only=True)`) routed to
             IndexTTS2. `adapter.generate()` runs far enough to write the speaker-reference WAVs
             into `temp_paths` and to construct the `StreamingAudioWriter`, then returns a
             `TTSResponse` whose `audio_stream` is the unstarted `stream()` generator. The service
             hits `if metadata_only:` at `tts_service_v2.py:2310`, calls
             `_close_response_audio_stream` -> `audio_stream.aclose()`, and returns. The `finally`
             at index_tts_adapter.py:464-466 never runs: `writer.close()` is skipped, so the
             PyAV container and its `output_buffer` are not released, and — if the WAV buffer had
             already spilled — the `tts_wav_pcm_*.pcm` temp file that `close()` unlinks
             (`streaming_audio_writer.py:224-231`) stays on disk; and `_cleanup_temp_paths` is
             skipped, so the speaker-reference WAVs stay on disk too. Every metadata-only request
             to IndexTTS2 or LuxTTS leaks. Nothing reclaims them until process exit.
impact:      Medium. Bounded by how often the metadata-only path is used, but unbounded in
             accumulation — temp files are never swept. Rated below tts-1 because it degrades
             rather than breaks, and because the two affected adapters are the least-deployed in
             the family.
tests:       import-grep reachability, not coverage. `luxtts_adapter` 2 importers,
             `index_tts_adapter` 1 — the thinnest in the family. `tts_service_v2` has 35 importers
             but no test drives `metadata_only=True` against a streaming adapter, so nothing
             exercises the combination.
effort:      cheap. Change `def _stream_audio` to `async def` and move the two constructions below
             the first statement, or adopt `stream_encoded_waveform`. A regression test is easy to
             state: call `_stream_audio(...)`, `await gen.aclose()` without iterating, assert the
             writer is closed and `temp_paths` are gone.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (generator semantics executed and recorded; both adapter shapes and all
             four abandon paths read directly).
```

---

### FINDING tts-6 — `waveform_streamer.py` is the designated waveform encoder and one adapter out of 25 uses it

```
axis:        duplication
class:       adoption-gap
severity:    Medium
sites:       canonical, 2 call sites total:
             core/TTS/adapters/chatterbox_adapter.py:1092, :1188
             second-tier shared helper (11 callers) that is itself a duplicate of the below:
             core/TTS/adapters/base.py:TTSAdapter.convert_audio_format (434-466)
             hand-rolled AudioNormalizer + StreamingAudioWriter sequences (18 construction sites):
             core/TTS/tts_service_v2.py:_convert_pcm_to_format (892-913)   <- verbatim copy of base.py:434-466
             core/TTS/adapters/dia_adapter.py:382-392
             core/TTS/adapters/higgs_adapter.py:408-418
             core/TTS/adapters/index_tts_adapter.py:417-422
             core/TTS/adapters/kitten_tts_adapter.py:234-246
             core/TTS/adapters/kokoro_adapter.py:1005-1016 and :1080-1098
             core/TTS/adapters/luxtts_adapter.py:539-543
             core/TTS/adapters/neutts_adapter.py:354-358
             core/TTS/adapters/pocket_tts_cpp_adapter.py:450-454
             core/TTS/adapters/qwen3_tts_adapter.py:818-822 and :978-982
             core/TTS/adapters/vibevoice_adapter.py:955-960
             core/TTS/adapters/echo_tts_adapter.py:721-726 and :932-937
             core/TTS/waveform_streamer.py:43-44 (the canonical one)
canonical:   core/TTS/waveform_streamer.py:stream_encoded_waveform (9-62) and
             :encode_waveform_to_bytes (65-76). Docstring: "Shared helpers to progressively encode
             and stream waveforms". Zero ambiguity about intent; near-zero adoption.
destination: n/a for the bulk of it — adopt the existing helper. The one genuine consolidation is
             deleting `tts_service_v2._convert_pcm_to_format (892-913)`, which is
             `base.py:convert_audio_format (434-466)` retyped with the `async` removed, and
             calling the base method (or `encode_waveform_to_bytes`) instead.
knowledge:   "how to turn a numpy waveform into container bytes in the requested format" — four
             coupled decisions: normalise to int16 before encoding; chunk before writing so the
             stream is actually progressive; the PCM special case (`if fmt == PCM: return only the
             first write, not first+final`, because PCM has no trailer); and close the writer in a
             `finally`. Eighteen sites each re-decide all four. They have already drifted:
             `kitten_tts_adapter.py:234-246` writes the **entire** array as a single
             `write_chunk`, so its "stream" emits one buffer and streaming is decorative;
             `luxtts_adapter.py:539` and `index_tts_adapter.py:417` get the `finally` placement
             wrong (tts-5); `base.py:434` carries a `source_format` parameter that appears only in
             the signature and the docstring and is never read in the body — a dead knob eleven
             callers are still passing.
impact:      Medium. This is the change-amplification engine behind three other findings on this
             page: tts-3 (accumulation), tts-5 (cleanup placement) and the kitten single-chunk
             regression are all instances of the same decision being made 18 times. Adding a new
             output format, or fixing the PCM trailer rule, currently means finding 18 sites.
             Adopting the helper collapses the surface to one.
tests:       import-grep reachability, not coverage. `streaming_audio_writer` has 6 importers
             (5 `tests/TTS`, 1 `tests/TTS_NEW`). `waveform_streamer` has exactly **one** —
             `tests/TTS/adapters/test_chatterbox_adapter_mock.py` (4 patch sites at :517, :681,
             :757, :873) — i.e. the only adapter that adopted it is the only one that tests it.
             Adapter-side: kokoro 9, vibevoice 8, higgs 8, dia 6, chatterbox 6, echo 5, qwen3 7.
             Coverage is broad enough that migration is verifiable per adapter.
effort:      moderate, and it is naturally staged per adapter rather than big-bang. The dead
             `source_format` parameter is a separate cheap deletion. The `tts_service_v2`
             duplicate is a cheap one-file change. The 16 adapter migrations are individually
             cheap and individually testable; do the well-covered ones (kokoro, vibevoice, higgs,
             dia) first and leave `luxtts`/`index_tts` until tts-5 is fixed.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (18 sites enumerated; helper and its single adopter verified; the
             `source_format` parameter confirmed unused by grepping the method body).
```

---

### FINDING tts-7 — `StreamingAudioWriter`'s open → write → finalize → close protocol is enforced by a docstring

```
axis:        sequential-coupling
class:       n/a
severity:    Medium
sites:       core/TTS/streaming_audio_writer.py:StreamingAudioWriter.__init__ (30-106)
             core/TTS/streaming_audio_writer.py:write_chunk (108-194) — the protocol is stated in
               the docstring at :117-121: "the expected call pattern is one or more
               write_chunk(audio_data=...) calls followed by a final write_chunk(finalize=True)
               call; do not combine audio_data and finalize=True in a single call"
             core/TTS/streaming_audio_writer.py:close (196-231) — releases the PyAV container, the
               BytesIO buffer, AND unlinks the `tts_wav_pcm_*.pcm` spill file at :220-231
             core/TTS/streaming_audio_writer.py:_spill_wav_buffer_to_file (261-280) — creates that
               file via `tempfile.mkstemp` once the in-memory WAV buffer exceeds
               `max_in_memory_bytes`
             18 call sites each re-implementing the protocol: see tts-6's site list
             the two that get it wrong: luxtts_adapter.py:539-569, index_tts_adapter.py:417-466 (tts-5)
canonical:   NONE. `StreamingAudioWriter` defines no `__enter__`/`__exit__` — verified, the grep
             for `__enter__|__exit__|contextmanager` in that file returns nothing.
destination: n/a — add `__enter__`/`__exit__` to `StreamingAudioWriter` itself (about six lines,
             `__exit__` delegating to the existing `close()`), so `with StreamingAudioWriter(...)`
             becomes expressible and the 18 hand-written `try/finally` blocks collapse.
knowledge:   the writer's state machine. Four orderings are wrong and all four are expressible:
             (1) never calling `write_chunk(finalize=True)` — the container is never muxed shut,
             so the output is a truncated/headerless file; (2) passing `audio_data` together with
             `finalize=True` — explicitly forbidden by the docstring, silently ignored by the code
             at :127 (`audio_data` is dropped on the finalize branch); (3) calling `write_chunk`
             after `finalize` — `self.container` is `None` at that point; (4) skipping `close()` —
             leaks the container, the buffer, and a temp file on disk. None of the four is
             prevented by the type system or by the API shape.
impact:      Medium. This clears the axis drop rule on all three counts: invalid orderings are
             expressible, nothing enforces the sequence (no context manager, no state guard), and
             there is a **caller that actually gets it wrong today** — tts-5 is the proof, not a
             hypothetical. The severity is Medium rather than High because the common orderings
             are correct in 16 of 18 sites and the failure is a resource leak rather than wrong
             audio. Note the cost is not purely memory: `close()` is the only thing that unlinks
             the spill file, so skipping it leaves bytes on disk.
scenario:    (shared with tts-5) `metadata_only` request to IndexTTS2 — writer constructed,
             generator never started, `finally: writer.close()` never reached, PyAV container and
             the `tts_wav_pcm_*.pcm` spill file both leaked.
tests:       import-grep reachability, not coverage. `streaming_audio_writer` has 6 importers
             (`tests/TTS` 5, `tests/TTS_NEW` 1). They cover encode correctness per format; none
             asserts anything about lifecycle — there is no test that a writer left unfinalized
             produces bad output, and none that `close()` removes the spill file.
effort:      cheap for the enabling change (add the two dunders; nothing existing breaks, since
             `with` is opt-in). moderate to migrate the 18 sites, and it should ride along with
             tts-6 rather than being a separate pass — most of those sites are going to be deleted
             by tts-6 anyway.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (absence of the context-manager protocol verified by grep; the spill-file
             unlink read at :220-231; the mis-ordering caller confirmed in tts-5).
```

---

### FINDING tts-8 — `OpenAITTSAdapter.generate` mutates shared adapter state per request on a registry-cached singleton

```
axis:        correctness
class:       n/a
severity:    Low
sites:       core/TTS/adapters/openai_adapter.py:generate (694-732)
               :696-698  `old_model = getattr(self, "model", None)`; `self.model = ...`
               :730-732  `finally: if old_model is not None: self.model = old_model`
             readers of the mutated field, inside the same call:
             core/TTS/adapters/openai_adapter.py:_generate_complete payload (224)
             core/TTS/adapters/openai_adapter.py:344, :352
             the cache that makes the instance shared:
             core/TTS/adapter_registry.py:_adapters (429), populated at :659 and :866,
               read at :624 and :1308
             the identical-branch dead code:
             core/TTS/adapters/openai_adapter.py:718-729 — `if _is_httpx_exception(e): wrapper_error = X`
               / `else: wrapper_error = X`, where both X are the same four-line
               `TTSGenerationError(..., details={"error_type": type(e).__name__})`
canonical:   n/a. The adapter already has a per-call carrier for this: `TTSRequest.model`, which
             `_select_model`/`_canonical_model_id` read without mutating anything.
destination: n/a — thread the resolved model through the payload builder instead of through
             `self`. The comment at :696 ("temporarily overriding self.model") shows the author
             knew this was a workaround.
knowledge:   "which state on an adapter is per-instance and which is per-request". Adapters are
             cached one-per-provider (`adapter_registry._adapters`, a plain dict, entries created
             once at :659 and reused for the life of the process), so *every* field on `self` is
             shared across all concurrent requests to that provider. `self.model` is the only one
             any adapter writes during `generate()`.
scenario:    Two concurrent `POST /api/v1/audio/speech` calls to the same process, one with
             `model="tts-1"` and one with `model="gpt-4o-mini-tts"`, both routed to the cached
             OpenAI adapter. Request A executes :698 setting `self.model="tts-1"`, then awaits
             inside `super().generate()`. The loop switches to B, which sets
             `self.model="gpt-4o-mini-tts"`. A resumes, reads `self.model` at :224 while building
             its payload, and sends **B's model** to OpenAI. A is billed at B's rate and gets B's
             voice quality, with no error anywhere. Secondary defect at :730-731: the restore is
             guarded by `if old_model is not None`, so when the adapter was constructed without a
             configured model the override is never rolled back and leaks into every subsequent
             request on that instance.
impact:      Low. Requires concurrent differing-model requests to the same provider in one
             process, and the wrong outcome is a wrong model rather than a failure or a leak.
             Filed because it is silent, because it costs money, and because the fix is small.
             The identical if/else at :718-729 is separate and purely cosmetic — a vestigial
             branch that reads as if it discriminates and does not.
tests:       import-grep reachability, not coverage. 12 files import `openai_adapter`. None
             constructs two adapters or issues two overlapping `generate()` calls, so this is
             structurally unreachable by the current suite.
effort:      cheap. Pass the resolved model into the payload builder; delete the `self.model`
             round-trip and the `finally`. Collapse the identical branches at :718-729 in the same
             edit. A regression test is a two-task `asyncio.gather` with different models against
             one adapter, asserting each captured payload carries its own.
owner-only:  no (`app/core/**` only).
confidence:  confirmed (the mutation, the readers, the registry cache, the guarded restore, and
             the identical branches are all read directly);
             probable-risk (that interleaving actually occurs — it depends on an `await` landing
             between :698 and :224, which `super().generate()` does contain, but the window is
             narrow).
```

---

### FINDING tts-9 — Seed cluster C8: the sanitized transport-error mapper is written three times across the HTTP adapters, and the best copy is the least-used

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       core/TTS/adapters/openai_adapter.py:_is_httpx_exception (47-49), :_is_http_status_error (52-55),
               :_is_timeout_error (58-62), :_safe_exception_label (65-67),
               :_bounded_network_error (70-77), :_bounded_existing_tts_error (80-...),
               :_handle_http_status_error (437-...)
             core/TTS/adapters/elevenlabs_adapter.py:_is_httpx_exception (49-51), :_is_http_status_error (54-57),
               :_is_timeout_error (65-68), :_safe_exception_label (60-62),
               :_bounded_network_error (71-78), :_bounded_existing_tts_error (81-...),
               status branches at :649
             core/TTS/adapters/qwen3_runtime_remote.py:_is_httpx_exception (126-128),
               :_is_http_status_error (130-133), :_is_timeout_error (135-139),
               :_parse_retry_after (141-151), :_normalize_http_status (153-187),
               :_normalize_http_status_error (189-194), :_normalize_remote_error (196-...)
             core/TTS/backends/fish_s2_commercial_api.py:_is_timeout_error (361)
             core/TTS/backends/fish_s2_native_http.py:_is_timeout_error (365)
             `_safe_exception_label`, 12 copies (11 byte-identical, 1 variant):
               adapter_registry.py:79, streaming_audio_writer.py:17,
               adapters/{dia_adapter.py:97 (variant: accepts None -> "unknown"),
               echo_tts_adapter.py:53, elevenlabs_adapter.py:60, index_tts_adapter.py:50,
               luxtts_adapter.py:49, neutts_adapter.py:45, openai_adapter.py:65,
               pocket_tts_adapter.py:49, pocket_tts_cpp_adapter.py:40, supertonic_adapter.py:45}
canonical:   NONE for the transport-error mapping. The briefing's two candidates were checked and
             **neither serves this case**: `api/v1/utils/http_errors.py:145` returns an HTTP status
             for a FastAPI boundary (wrong layer — adapters must not depend on `api/`, and the
             adapters need a `TTSError`, not a status int); `core/exceptions.py:1104` maps a
             *core exception to* a status, i.e. the opposite direction from "httpx exception to
             domain error". The briefing also lists a fourth `_is_http_status_error` at
             `core/LLM_Calls/error_utils.py:407` — that symbol does not exist at any line in
             `core/LLM_Calls/`; all three surviving copies are inside this module.
             **BEST COPY: `qwen3_runtime_remote.py:_normalize_remote_error (196-...)`** — it is the
             only one that (a) carries a complete status table (401/403/429/400/408/504 + default)
             in one place rather than inline at each raise site, (b) parses `Retry-After` into the
             `TTSRateLimitError` (`_parse_retry_after`, 141-151), and (c) handles
             `core.http_client`'s own `CoreNetworkError` and `RetryExhaustedError` alongside raw
             httpx — which matters because all three adapters call `afetch`/`apost`/`astream_bytes`
             and can therefore receive those wrapper types, not just httpx exceptions.
destination: `core/TTS/adapters/_transport_errors.py`, single responsibility: **translate a
             transport-layer failure (httpx exception, `CoreNetworkError`, `RetryExhaustedError`,
             or an HTTP status + headers) into a sanitized `TTSError`, retaining no URL, header,
             body or credential-derived text.** That responsibility is the reason all three copies
             exist — every one of them carries a "without retaining credential-derived URLs"
             docstring — and it is narrow enough not to become a junk drawer. Seed it by moving
             `qwen3_runtime_remote`'s implementation, parameterised on `provider_key` and the
             configured timeout. This must not go in `Utils.py` or `http_client.py`.
knowledge:   two things that will diverge. First, **the status table**: which HTTP codes map to
             auth vs rate-limit vs timeout vs generic provider error, and therefore what the
             caller and the circuit breaker see. It exists once in full (`qwen3`) and twice
             scattered inline (`openai:437`, `elevenlabs:649`); adding 402 or 503 handling today
             means three edits and will get one. Second, **the sanitization invariant** — the
             whole family exists to keep provider URLs and credentials out of logs and error
             payloads; that is a security property currently asserted by convention across three
             files. `_bounded_existing_tts_error` has already drifted: `openai_adapter.py:80+`
             routes through the `auth_error()`/`rate_limit_error()` factories in `tts_exceptions`,
             `elevenlabs_adapter.py:81+` constructs `TTSAuthenticationError` directly with a
             different message shape, so the two providers return differently-worded auth failures
             for the same 401.
impact:      Medium. The consequence of a misclassification is concrete and traceable through
             `core/exceptions.py:796-879`, where the TTS exception classes carry their default
             status codes: a 429 that is classified as a generic `TTSProviderError` instead of
             `TTSRateLimitError` surfaces to the client as a 500-class provider failure rather
             than a 429 with `Retry-After`, and — because `tts_service_v2._categorize_error (3871)`
             and the circuit breaker key off the exception type — it is also counted as a hard
             provider failure, so repeated upstream throttling trips the breaker and takes the
             provider out of rotation instead of backing off. Only `qwen3_runtime_remote` extracts
             `Retry-After` at all, so OpenAI and ElevenLabs already cannot return it.
             Not rated High because no *currently reachable* input is known to be misclassified —
             this is drift risk plus a missing feature, not a live defect. (The audit's anchor bug,
             the double-escaped `r"HTTP\\s+(\\d{3})"` regex, does **not** appear in this module;
             none of the three copies parses status codes out of message text.)
             `_safe_exception_label` × 12 is filed here rather than separately because it is the
             same security invariant ("log a non-sensitive exception identifier") copied into
             every adapter; it is a one-line function and would be a Low on its own, but it belongs
             in the same destination module and should move with it.
tests:       import-grep reachability, not coverage. `qwen3_runtime_remote` 3 importers,
             `openai_adapter` 12, `elevenlabs_adapter` 8. Notably `tests/TTS/test_elevenlabs_adapter.py`
             defines `assert_sanitized_error` (:26-38) asserting `exc.__cause__ is None`,
             `exc.__context__ is None` and that raw markers are absent from
             `traceback.format_exception` — and `grep -rln "assert_sanitized_error"` matches that
             one file only. The sanitization invariant is well tested for exactly one of the three
             copies; promoting one implementation would let that helper cover all of them.
effort:      moderate. Not mechanical — it is a real module extraction with a security invariant
             to preserve — but the target already exists as working, tested code, so this is
             "move and parameterise", not "design". Do it after tts-1 and tts-2, which touch the
             same two files.
owner-only:  no (`app/core/TTS/**` only). The proposed module lives under `core/`, deliberately
             not under `api/v1/utils/`.
confidence:  confirmed (all sites enumerated and read; the two briefing-suggested canonicals
             checked and rejected with reasons; the fourth briefing site verified absent);
             probable-risk (the circuit-breaker consequence — it follows from
             `_categorize_error` keying on exception type, but no end-to-end trace was run).
```

---

## Dropped after review — recorded so they are not rediscovered

- **`_is_timeout_error` divergence.** `elevenlabs_adapter.py:65-68` omits `asyncio.TimeoutError`
  where `openai_adapter.py:58-62` and `qwen3_runtime_remote.py:135-139` include it. Since Python
  3.11 `asyncio.TimeoutError` **is** the builtin `TimeoutError`, so the two are equivalent on
  every Python this repo runs. `justified-divergence` — cosmetic only. It should still move with
  tts-9, but it is not a defect.
- **Per-provider generation internals** (prompt assembly, model kwargs, speaker handling in
  `dia`, `higgs`, `vibevoice`, `chatterbox`, `echo_tts`). These look duplicated in outline and are
  not: each encodes a different upstream runtime's API. `justified-divergence`.
- **Blind `except Exception` across the adapters.** 23 of this module's files sit on the BLE001
  grandfather list at `pyproject.toml:1267-1290`, including every adapter cited above. Sanctioned
  policy; not reportable.
- **`ProviderStatus` / `TTSCapabilities` as public dataclasses crossing the adapter boundary.**
  Intentional DTOs at a deliberate boundary — the encapsulation drop rule excludes these.

## Suggested Refactor/Actions

Ordered by (risk removed) ÷ (effort), not by severity.

1. **tts-3** — five-line `bytearray` change at five sites. No design doc, no ADR. One Backlog
   task. Do this first; it is the cheapest thing on the page with a measured payoff.
2. **tts-4** — three imports, three deletions, following `echo_tts_adapter.py:1200`. Add the
   table-driven agreement test. One Backlog task. Flag ADR-026 in the description since it touches
   the egress gate.
3. **tts-1** — delete the `aclose()` at `elevenlabs_adapter.py:678-682`; retarget
   `test_cleanup_failure_log_sanitizes_exception_text`; add a "pooled client survives adapter
   close" regression test. One Backlog task. **This is the one to do before any release** — it is
   the only finding here that can take a provider permanently offline.
4. **tts-5** — make the two `_stream_audio` methods `async def`. One Backlog task, pairs naturally
   with tts-7's context-manager addition.
5. **tts-8** — thread the model through the payload; collapse the identical branches. One Backlog
   task, cheap, low risk.
6. **tts-2 (part a)** — make the two `validate_request` overrides return the base tuple and move
   the raises into `generate()`. Needs a short design note because it changes observable adapter
   behaviour for two providers: `Docs/Design/2026-09-21-tts-adapter-validate-contract-design.md`.
   Part (b), deleting the shim classes, is blocked on tts-14 and should not be attempted here.
7. **tts-7 + tts-6** — one staged effort, not two. Stage 1: add `__enter__`/`__exit__` to
   `StreamingAudioWriter` and delete the dead `source_format` parameter from
   `base.convert_audio_format`. Stage 2: delete `tts_service_v2._convert_pcm_to_format` in favour
   of the base method. Stage 3-5: migrate adapters to `waveform_streamer`, best-covered first
   (kokoro, vibevoice, higgs, dia), leaving luxtts/index_tts until tts-5 lands. This needs
   `Docs/Design/2026-09-21-tts-audio-encoding-consolidation-design.md` and
   `IMPLEMENTATION_PLAN_tts-audio-encoding.md` with those stages — it is the only item on this
   page large enough to warrant the design-first treatment.
8. **tts-9** — extract `core/TTS/adapters/_transport_errors.py` seeded from
   `qwen3_runtime_remote`. Needs a design note (`Docs/Design/2026-09-21-tts-transport-error-
   normalization-design.md`) because it consolidates a security invariant across three files, and
   an ADR entry is *not* required — ADR-011 does not speak to error taxonomy internals. Sequence
   after tts-1 and tts-2 to avoid three concurrent edits to the same two adapters.

All items are `app/core/**`; none touches an owner-only path. Per the briefing, Backlog tasks are
proposed here only — none were created, and no task file was edited.
