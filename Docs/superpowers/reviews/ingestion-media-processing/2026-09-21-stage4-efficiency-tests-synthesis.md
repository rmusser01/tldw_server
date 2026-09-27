# Stage 4 — Efficiency, the Audio sub-package, and synthesis

## Scope

The two things stages 1–3 did not cover: the `Audio/` sub-package (33 files, ~22k LOC, the largest
and second-hottest part of the module), and Axis 5 — where ingestion actually spends money. Closes
with the cross-stage synthesis and the proposed Backlog tasks.

## Code Paths Reviewed

Audio — resampling and buffering:
- `Audio/Audio_Buffered_Transcription.py:BufferedTranscriber._resample (534-541)` and its two
  callers at `:384-386` and `:752-754`.
- `Audio/Audio_Transcription_Lib.py:_resample_audio_if_needed (336-355)`,
  `:_resample_audio_without_librosa (358-382)`.
- `Audio/Audio_Streaming_Unified.py:_resample_audio_if_needed (282-308)`,
  `:_ParakeetRNNTStreamer._resample_if_needed (1375-1404)`.
- `Audio/Audio_Transcription_Qwen3ASR.py:_maybe_resample (192-210)`,
  `Audio/Audio_Transcription_VibeVoice.py:_maybe_resample (179-196)`.

Audio — model caches:
- `Audio/Audio_Transcription_Lib.py:whisper_model_cache (2783)` + `whisper_model_cache_lock (2784)`,
  held at `:2871-2901`.
- `Audio/Audio_Transcription_Qwen3ASR.py:_MODEL_CACHE (50)` / `_MODEL_LOCK (51)`,
  `_ALIGNER_CACHE (54)` / `_ALIGNER_LOCK (55)`, held `:273-315`.
- `Audio/Audio_Transcription_VibeVoice.py:_MODEL_CACHE (52)` / `_MODEL_LOCK (53)`, held `:435-475`.
- `Audio/Audio_Transcription_Nemo.py:_model_cache (55)`, `:_get_model_cache_key (269)`,
  `:load_canary_model (442-514)`, `:load_parakeet_model (518-585)`,
  `:_load_parakeet_standard (588-616)`, `:_load_planned_nemo_model (408-433)`.
- `Audio/Audio_Transcription_Parakeet_ONNX.py:_onnx_model_cache (57)`, reads `:835`, writes `:999`,
  `:1070`.
- `Audio/Audio_Transcription_Parakeet_MLX.py:_mlx_model_cache (43)`,
  `:load_parakeet_mlx_model (309-…)` read at `:343-345`, write at `:414`.

Audio — async discipline:
- `Audio/Audio_Streaming_Unified.py:ParakeetStreamingTranscriber.process_audio_chunk (1715-1949)`
  (`async def` at :1715; blocking calls at :1871, :1877, :1911, :1916),
  `:CanaryStreamingTranscriber.process_audio_chunk (1974-2086)` (blocking at :2003, :2045),
  `:WhisperStreamingTranscriber.process_audio_chunk (2185-2262)` (blocking at :2214, :2238) and its
  sync helper `:_transcribe_audio (2264-2301)` with `self.model.transcribe(...)` at `:2279`,
  `:Qwen3ASRStreamingTranscriber._transcribe_via_vllm (2450-2521)` (`sf.write` at :2475; the HTTP
  call IS correctly offloaded at :2502).
- Contrast: `Audio/Audio_Streaming_Parakeet.py:_transcribe_chunk (263-294)` wraps the *same two*
  transcription functions in `asyncio.to_thread` at `:268`, `:277`, `:285`.

Audio — ffmpeg:
- `Audio/Audio_Transcription_Lib.py:_find_ffmpeg (4486-4536)`, `:validate_audio_file (4585-4600)`,
  `:convert_to_wav (4740-4930)` incl. `:4755-4760`, `:4792-4799`, `:4826-4842`, `:4894-4911` and
  the correctly-factored `:_run_ffmpeg_command (4845-4881)`.
- `Audio/Audio_Files.py:download_youtube_audio (1600-1700)` incl. the inline ffmpeg discovery at
  `:1620-1634`.

Audio — the adapter layer:
- `Audio/stt_provider_adapter.py:_require_benchmark_mode (395-405)` and its four callers
  (`:868`, `:1051`, `:1295`, `:1475`); the three inline divergent copies at `:1668-1672`,
  `:1956-1960`, `:2569-2573`; `AudioCppAdapter._validate_plan_request (2327-2351)`;
  `SttProviderAdapter._transcribe_planned_batch (771-794)`; `:_build_local_plan (550-638)`;
  `:transcribe_batch` ×9 (`726`, `944`, `1194`, `1369`, `1557`, `1870`, `2213`, `2484`, `2808`).
- `Audio/Audio_Streaming_Parakeet.py:is_transcription_error_message (29-42)` vs
  `Audio/stt_execution_contract.py:is_transcription_error_message (106-111)` vs
  `Audio/Audio_Transcription_Lib.py:is_transcription_error_message (1771-1778)`.

Efficiency, non-Audio:
- `persistence.py:_with_media_db_session (134-148)` and its call sites at `:1042`, `:3887`, `:3902`,
  `:4306`, `:4344`, `:6053`, `:6303`; the pre-check loop at `:4131`.
- `persistence.py:_source_hash_precheck (4242-4304)` — the `LIKE` at `:4271` / `:4292`.
- `persistence.py:schedule_media_add_embeddings (2399-2569)` — `EmbeddingsJobsAdapter()` at `:2455`
  inside the loop at `:2429`.
- `persistence.py:add_media_orchestrate` — `StorageQuotaService()` + `await .initialize()` at
  `:3334-3335` inside the loop at `:3288`; contrast the singleton `get_storage_quota_service()` at
  `:4984`.
- `PDF/PDF_Processing_Lib.py:_ocr_pdf_pages (1552-1675)` — render loop `:1608-1648`, drain
  `:1645-1668`.
- `persistence.py:add_media_orchestrate` — `TempDirManagerCls(cleanup=...)` at `:2833-2836`;
  `input_sourcing.py:TempDirManager.__exit__ (56-69)`; the permanent-storage step at `:3281`.

## Tests Reviewed

By import-grep: 46 files in `tests/Audio/`, plus `tests/STT/`, `tests/AudioJobs/`,
`tests/Streaming/`, `tests/VoiceAssistant/`, `tests/Benchmarks/`, `tests/TTS_NEW/`,
`tests/Setup/`, `tests/Resource_Governance/`, `tests/Workflows/`, `tests/DB_Management/`,
`tests/Utils/` — eleven trees for `Audio_Transcription_Lib.py` alone (finding -11).

| Test file / group | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/Media_Ingestion_Modification/test_parakeet_mlx.py` | Parakeet-MLX transcription; `:201` explicitly monkeypatches `install_parakeet_mlx` to a function that fails the test if called | Partly — it does **not** exercise the cache with two different `model_path` values, which is finding -14 |
| `tests/MediaIngestion_NEW/unit/test_audio_formatting.py` | `Audio_Files.format_transcription_with_timestamps` only | No, for the `Audio_Transcription_Lib` twin (see folded items) |
| `tests/Audio/` streaming tests | WebSocket framing, session lifecycle | No — none measures event-loop latency or asserts non-blocking behaviour, so finding -15 is invisible to them |
| `tests/STT/`, `tests/Benchmarks/` | `stt_provider_adapter` planning and batch execution | Partly — they cover the happy path of `plan_batch_execution`; none asserts that all nine adapters agree on which `mode` values are rejected (finding -20) |
| `tests/Media_Ingestion_Modification/test_ocr_structured_output.py` | the `_ocr_pdf_pages` page loop end-to-end | Yes for correctness; no for memory (finding -19) |

**Structural gap:** no test in the module asserts a *resource* property — not peak memory, not
connection count, not event-loop responsiveness, not thread-safety under concurrency. Every finding
in this stage is therefore invisible to the suite, which is why several have survived in files with
high churn.

## Validation Commands

```
$ grep -c "Audio_Transcription_Lib" \
    .../Audio/Audio_Transcription_Nemo.py .../Audio/Audio_Transcription_Parakeet_MLX.py
0    0            # neither imports it -> Audio_Streaming_Parakeet's sys.modules guard can be False

$ grep -n "async def process_audio_chunk" .../Audio/Audio_Streaming_Unified.py
1715, 1974, 2185     # all three are `async def`
$ sed -n '2264p;2279p' .../Audio/Audio_Streaming_Unified.py
    def _transcribe_audio(self, audio_np: np.ndarray) -> str:
            segments_raw, info = self.model.transcribe(

$ grep -n "Lock\|threading" .../Audio/Audio_Transcription_Nemo.py \
    .../Audio/Audio_Transcription_Parakeet_ONNX.py .../Audio/Audio_Transcription_Parakeet_MLX.py
(no output — zero hits in all three)

$ grep -n "_require_benchmark_mode" .../Audio/stt_provider_adapter.py
395:def _require_benchmark_mode   868:  1051:  1295:  1475:      # 4 of 9 adapters call it

$ grep -rn "media_processing_" tldw_Server_API/app | grep -v endpoints | grep -v schemas
input_sourcing.py:45:    def __init__(self, prefix: str = "media_processing_", *, cleanup: bool = True)
(the only hit — no reaper for this prefix exists anywhere in app/)

$ grep -n "keep_original_file" .../persistence.py
2835:  cleanup=not form_data.keep_original_file or form_data.media_type in {"pdf","document","ebook"},
3281:  if form_data.keep_original_file and form_data.media_type in ["pdf","document","ebook"]:
4535:, 4598:   (yt-dlp "keep_original" option only)

$ python -m pytest tldw_Server_API/tests/unit/test_ocr_types.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_adapter.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py -q
======================== 24 passed, 7 warnings in 0.93s ========================
```

## Findings

### FINDING ingestion-media-processing-13 — resampling fails open and the caller then relabels the sample rate anyway

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       THE BUG:
               Audio/Audio_Buffered_Transcription.py:BufferedTranscriber._resample (534-541) —
                 `except ImportError: logger.warning(...); return audio` at :540-541
               caller Audio/Audio_Buffered_Transcription.py:384-386 —
                 `audio_data = self._resample(audio_data, sample_rate, 16000)` then
                 **unconditionally** `sample_rate = 16000`
               caller Audio/Audio_Buffered_Transcription.py:752-754 — identical pattern
             THE SAME FAIL-OPEN, WITHOUT THE RELABEL (less severe, same family):
               Audio/Audio_Streaming_Unified.py:_resample_audio_if_needed (282-308) —
                 `return audio` at :308 on any exception, at the original rate
               Audio/Audio_Transcription_Qwen3ASR.py:_maybe_resample (192-210)
               Audio/Audio_Transcription_VibeVoice.py:_maybe_resample (179-196)
             THE CORRECT COPIES:
               Audio/Audio_Transcription_Lib.py:_resample_audio_if_needed (336-355) — librosa, then
                 a linear-interpolation fallback that actually resamples
               Audio/Audio_Transcription_Lib.py:_resample_audio_without_librosa (358-382) —
                 scipy.signal.resample_poly, then linear-interp fallback
               Audio/Audio_Streaming_Unified.py:_ParakeetRNNTStreamer._resample_if_needed (1375-1404)
                 — torchaudio with a per-(in_sr,out_sr) cache
canonical:   `Audio/Audio_Transcription_Lib.py:_resample_audio_if_needed (336-355)` is the correct
             copy and should be promoted: it is the only one whose fallback path still returns audio
             at the requested rate.
destination: a new `Audio/audio_resample.py` owning ONE responsibility — sample-rate conversion with
             a guaranteed post-condition (`returns audio at target_sr, or raises`). Five call sites.
             Not `Utils/Utils.py`.
knowledge:   "what a resampler is allowed to do when its backend is missing." Three of the six
             copies answer "return the input unchanged"; three answer "convert it anyway".
scenario:    Deployment without `librosa` (a heavy optional dep; the module's own code treats it as
             optional at :537-541). A client streams 48 kHz audio into the buffered transcriber.
             Line 385 calls `_resample`, which logs "librosa not available, returning original
             audio" and hands back the **48 kHz** array. Line 386 then sets `sample_rate = 16000`.
             From that point every downstream computation is wrong by exactly 3×: the chunk
             boundaries derived from `self.chunk_samples_at_16k` (:388-392) each cover one third of
             the intended audio, and the model is fed 48 kHz samples as though they were 16 kHz.
             Output is 3×-speed garbage with every timestamp short by a factor of three. The only
             signal is a WARNING log line; the request returns 200 with a transcript.
impact:      High. Silent, total corruption of the output on a configuration the code explicitly
             anticipates, with the lie (`sample_rate = 16000`) written one line after the function
             that refused to make it true. The `return audio` fallbacks in the other three copies
             are the same mistake with a smaller blast radius.
tests:       `tests/Audio/` and `tests/STT/` cover the buffered transcriber, but no test runs it
             with librosa absent. Import-grep reachability only.
effort:      cheap — make `_resample` raise (or fall back to linear interpolation, as
             `Audio_Transcription_Lib.py:346-355` already does) and delete the unconditional
             relabel at :386 and :754. One test: monkeypatch the librosa import to raise, assert the
             call raises rather than returning 48 kHz data tagged 16 kHz.
owner-only:  no
confidence:  confirmed — `_resample`'s ImportError branch and both callers' unconditional
             `sample_rate = 16000` read at the cited lines.
```

### FINDING ingestion-media-processing-14 — three of six STT model caches are unsynchronised, and one is unkeyed so it returns the wrong model

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       CORRECT — keyed and lock-protected:
               Audio/Audio_Transcription_Lib.py:whisper_model_cache (2783) /
                 whisper_model_cache_lock (2784), held across construction at :2871-2901
               Audio/Audio_Transcription_Qwen3ASR.py:_MODEL_CACHE (50) / _MODEL_LOCK (51) and
                 _ALIGNER_CACHE (54) / _ALIGNER_LOCK (55), held :273-315
               Audio/Audio_Transcription_VibeVoice.py:_MODEL_CACHE (52) / _MODEL_LOCK (53),
                 held :435-475
             KEYED BUT UNLOCKED — check-then-act race:
               Audio/Audio_Transcription_Nemo.py:_model_cache (55), key builder
                 :_get_model_cache_key (269). load_canary_model (442-514): read :477,
                 `from_pretrained` :495, write :509. load_parakeet_model → _load_parakeet_standard
                 (588-616): read :565, `from_pretrained` :603, write :613.
                 _load_planned_nemo_model (408-433): read :412, write :433.
                 Also mutates process-global `os.environ['NEMO_CACHE_DIR']` at :492 and :600 from
                 worker threads.
               Audio/Audio_Transcription_Parakeet_ONNX.py:_onnx_model_cache (57): read :835,
                 writes :999 and :1070.
             UNKEYED AND UNLOCKED — returns the wrong model:
               Audio/Audio_Transcription_Parakeet_MLX.py:_mlx_model_cache (43) — a bare
                 `Optional[Any] = None`, not a dict. Read at :343-345
                 (`if _mlx_model_cache and not force_reload: return _mlx_model_cache`),
                 written at :414. The function signature accepts `model_path`, `cache_dir` and
                 `force_reload` but only `force_reload` affects the lookup.
canonical:   `Audio/Audio_Transcription_Qwen3ASR.py:269-315` is the clearest correct copy — keyed
             dict, module lock, re-check inside the lock, store inside the lock. The five OCR
             backends' `_load_transformers` (stage 2, finding -6) use the identical correct shape,
             so the repo has six working examples of it.
destination: `Audio/model_utils.py` already exists in this package and already owns model-adjacent
             helpers. Add `cached_model(key, factory)` there — one responsibility, six callers.
knowledge:   "load a multi-gigabyte model once per process, keyed by the parameters that change it."
             Six answers: three right, two racy, one that ignores its own parameters.
scenario:    (unkeyed) Request A calls `load_parakeet_mlx_model(model_path="/models/parakeet-tdt-0.6b-v3")`;
             the model loads and is stored at :414. Request B calls
             `load_parakeet_mlx_model(model_path="/models/parakeet-tdt-1.1b")`. Line 343 sees a
             truthy `_mlx_model_cache`, logs "Using cached Parakeet MLX model" and returns **the
             0.6b model**. B's transcript is produced by the wrong model and reported upstream under
             the 1.1b name. The same applies to a differing `mlx_dtype`. No error, no warning beyond
             a DEBUG line.
             (racy) Two concurrent Canary requests both miss the read at :477 before either reaches
             the write at :509, so `from_pretrained("nvidia/canary-1b-v2")` runs twice and two
             ~1–3 GB models are resident simultaneously. On a GPU box this is an OOM; on CPU it is a
             doubled load time and doubled RSS. `_ocr_pdf_pages`-style thread pools and the STT
             batch adapters both dispatch concurrently, so the precondition is ordinary.
impact:      High. The unkeyed cache is a correctness bug that returns wrong output under a
             supported call; the two races are a resource bug on a path that routinely runs
             concurrently. All three are in files with no lock at all — `grep -n "Lock\|threading"`
             returns nothing for any of the three.
tests:       `tests/Media_Ingestion_Modification/test_parakeet_mlx.py` exercises the MLX path but
             never with two different `model_path` values; nothing tests concurrent loads anywhere.
effort:      cheap — the MLX fix is to make `_mlx_model_cache` a dict keyed on
             `(model_id, dtype, cache_dir)`, which is the one-liner the other five files already
             use. Adding `threading.Lock` to Nemo and Parakeet-ONNX is mechanical. Consolidating
             into `model_utils.py` is moderate and can follow.
owner-only:  no
confidence:  confirmed — all six caches read; the absence of any lock in the three files verified
             by grep.
```

### FINDING ingestion-media-processing-15 — three WebSocket transcribers run model inference directly on the event loop, while their sibling in the same package offloads the same calls

```
axis:        efficiency
class:       divergent-copies
severity:    High
sites:       BLOCKING (all `async def`, no `to_thread` / `run_in_executor`):
               Audio/Audio_Streaming_Unified.py:ParakeetStreamingTranscriber.process_audio_chunk
                 (1715-1949) — `transcribe_with_parakeet_mlx(...)` / `transcribe_with_parakeet(...)`
                 at :1871, :1877 (partial) and :1911, :1916 (final)
               Audio/Audio_Streaming_Unified.py:CanaryStreamingTranscriber.process_audio_chunk
                 (1974-2086) — `transcribe_with_canary(...)` at :2003 and :2045
               Audio/Audio_Streaming_Unified.py:WhisperStreamingTranscriber.process_audio_chunk
                 (2185-2262) — `self._transcribe_audio(...)` at :2214 and :2238;
                 `_transcribe_audio` is a plain `def` (2264-2301) whose :2279 is
                 `self.model.transcribe(...)` (faster-whisper)
               Audio/Audio_Streaming_Unified.py:Qwen3ASRStreamingTranscriber._transcribe_via_vllm
                 (2450-2521) — `sf.write(str(tmp_path), audio_np, ...)` at :2475 (minor: disk I/O
                 only; the HTTP call at :2502 IS correctly offloaded, with a comment at :2489
                 explicitly flagging the concern — so the file write was simply missed)
               Audio/Audio_Transcription_External_Provider.py:transcribe_with_external_provider_async
                 (195-418) — `sf.write(buffer, audio_data, sample_rate, format="WAV")` at :286;
                 in-memory BytesIO, CPU-only, lowest severity of the five
             CORRECT SIBLING, SAME TWO FUNCTIONS:
               Audio/Audio_Streaming_Parakeet.py:_transcribe_chunk (263-294) — `asyncio.to_thread`
                 at :268, :277, :285
             OTHER CORRECT COPIES (checked, no finding):
               Audio/Audio_Streaming_Unified.py:StreamingDiarizer (834-1143) — offloads at :937,
                 :960, :1067
               audio_batch.py:163-164 — `run_in_executor` around `process_audio_files`
               persistence.py:4612-4614 — same
               PDF/PDF_Processing_Lib.py:process_pdf_task (1445-1543) — `asyncio.to_thread` at :1497
canonical:   `Audio/Audio_Streaming_Parakeet.py:_transcribe_chunk (263-294)` — the same package, the
             same two transcription functions, wrapped correctly.
destination: n/a — this is a discipline fix, not a consolidation. The lasting fix is a lint rule.
knowledge:   n/a
cost-driver: **Whole-process event-loop stall for the duration of one model decode.** Scales with
             (a) the decode time of the configured model on the configured device — a
             faster-whisper `small` decode of a 5 s chunk is tens to hundreds of milliseconds on
             CPU, and a Canary/Parakeet decode is longer — and (b) the number of concurrent
             WebSocket sessions, because every other session's frames, keepalives and control
             messages queue behind it. The cost is not the decode itself (it has to happen) but that
             it is paid *serially across all connections* instead of in the thread pool.
             `process_audio_chunk` is called once per audio chunk per session, so the stall
             frequency is `sessions × chunks_per_second`.
scenario:    Two clients hold `/api/v1/audio/stream/transcribe` sessions with the Whisper backend.
             Client A's chunk reaches :2238; `self.model.transcribe` at :2279 blocks the single
             event loop for the decode. Client B's frames are not read, its keepalive is not
             answered, and every unrelated HTTP request served by the same worker also waits. With
             the Parakeet streaming endpoint — the same models, routed through
             `Audio_Streaming_Parakeet.py` — this does not happen, because that file wraps the
             identical calls in `asyncio.to_thread`.
tests:       none. No test in the module measures event-loop responsiveness; the streaming tests in
             `tests/Audio/` and `tests/Streaming/` assert framing and lifecycle only.
effort:      cheap — four `await asyncio.to_thread(...)` wrappers, following the shape at
             `Audio_Streaming_Parakeet.py:268`. Worth adding a small `tests/lint/` AST check that no
             `async def` in `Audio/` calls a known-blocking transcription entry point directly; the
             repo already has AST lint tests to copy (`tests/lint/`).
owner-only:  no
confidence:  confirmed — all three `async def` signatures and all blocking call sites read;
             `_transcribe_audio` confirmed as a plain `def` wrapping `model.transcribe`.
```

### FINDING ingestion-media-processing-16 — ffmpeg/ffprobe discovery is written three ways, and one of them corrupts the path with `str.replace`

```
axis:        correctness
class:       divergent-copies
severity:    Medium
sites:       Audio/Audio_Transcription_Lib.py:_find_ffmpeg (4486-4536) — the canonical resolver:
               project `Bin/ffmpeg.exe` → `FFMPEG_PATH` env (:4514-4517) → `app_dir/Bin/ffmpeg`
               (resolved relative to `__file__`, :4504-4509) → `shutil.which`
             Audio/Audio_Transcription_Lib.py:validate_audio_file (4585-4600) — ffprobe discovery
               done correctly: `shutil.which("ffprobe")`, sibling of the `_find_ffmpeg` result,
               then the literal `"ffprobe"`
             Audio/Audio_Transcription_Lib.py:convert_to_wav:4792 — **the drifted copy**:
               `ffprobe_cmd = ffmpeg_cmd.replace('ffmpeg', 'ffprobe') if 'ffmpeg' in ffmpeg_cmd
                else 'ffprobe'`
             Audio/Audio_Files.py:download_youtube_audio (1620-1634) — a third, independent ffmpeg
               resolver with a different order: hardcoded CWD-relative `'./Bin/ffmpeg.exe'` on
               `os.name == 'nt'` → `shutil.which` → three hardcoded Homebrew/usr paths → bare
               `'ffmpeg'`. **It never reads `FFMPEG_PATH`.**
             Related (same file, pure duplication, no bug): the conversion command tail at
               :4826-4842 and :4894-4911 is written twice verbatim, differing only in three
               literals (`10M`→`100M`, `50M`→`100M`, `-err_detect ignore_err`). The actual
               `subprocess.run`/`Popen` is already correctly factored into
               `:_run_ffmpeg_command (4845-4881)`, used by both at :4886 and :4916.
canonical:   `Audio/Audio_Transcription_Lib.py:_find_ffmpeg (4486-4536)` for ffmpeg;
             `:validate_audio_file:4585-4590` for ffprobe.
destination: a new `Audio/ffmpeg_locator.py` with one responsibility — resolving the ffmpeg and
             ffprobe executables — exporting `find_ffmpeg()` and `find_ffprobe()`. Three callers.
knowledge:   "where the ffmpeg/ffprobe binaries are." Three answers, and only one of them honours
             the `FFMPEG_PATH` environment variable that Docker deployments set.
scenario:    (a) `FFMPEG_PATH=/opt/ffmpeg/bin/ffmpeg`. `_find_ffmpeg` returns it (:4514-4517).
             Line 4792 runs `.replace('ffmpeg', 'ffprobe')` — which replaces **every** occurrence —
             yielding `/opt/ffprobe/bin/ffprobe`, a path that does not exist. The
             `subprocess.run` at :4793 raises `FileNotFoundError`, caught at :4818, and the user is
             told their file is corrupt: `ConversionError("Audio file 'x.mp3' is corrupted or
             invalid: ...")`. The file is fine; only the path arithmetic is wrong. Identical on
             Windows with `C:\ffmpeg\bin\ffmpeg.exe`. `validate_audio_file:4585-4590` gets this
             right — the two ffprobe resolvers are in the same file, 200 lines apart.
             (b) `FFMPEG_PATH=/opt/custom/bin/ffmpeg` with no ffmpeg on `PATH`. `convert_to_wav`
             succeeds; `download_youtube_audio` hands yt-dlp `ffmpeg_location='ffmpeg'` and the
             `FFmpegExtractAudio` postprocessor fails. The same server, the same config, two
             different answers to "is ffmpeg installed".
impact:      Medium. (a) mislabels valid media as corrupt on any deployment whose ffmpeg path
             contains "ffmpeg" more than once — which is most of them, since the binary is usually
             in a directory named after itself. (b) makes YouTube audio ingestion fail on exactly
             the Docker-style configuration `FFMPEG_PATH` exists to serve.
tests:       `tests/Audio/` and `tests/MediaIngestion_NEW/` cover conversion, but none sets
             `FFMPEG_PATH` to a path containing "ffmpeg" twice — which is why (a) survives.
effort:      cheap for the bug (one line: derive ffprobe from the ffmpeg path with
             `Path(x).with_name("ffprobe")`, or just reuse `validate_audio_file`'s approach);
             moderate for the three-way consolidation.
owner-only:  no
confidence:  confirmed — all three resolvers and the `str.replace` line read; the `FFMPEG_PATH`
             branch confirmed at :4514-4517.
```

### FINDING ingestion-media-processing-17 — `keep_original_file=true` on audio/video leaks the temp directory and keeps nothing

```
axis:        correctness
class:       n/a
severity:    Medium
sites:       persistence.py:add_media_orchestrate:2833-2836 —
               `TempDirManagerCls(cleanup=not form_data.keep_original_file
                                  or form_data.media_type in {"pdf","document","ebook"})`
             persistence.py:add_media_orchestrate:3281 —
               `if form_data.keep_original_file and form_data.media_type in ["pdf","document","ebook"]:`
               — the permanent-storage step, gated on the SAME set
             input_sourcing.py:TempDirManager.__init__ (44-48) — default prefix
               `"media_processing_"`; `:__exit__ (56-69)` — `shutil.rmtree` runs only
               `if self._created and self.temp_dir_path and self._cleanup`
             input_sourcing.py:52 — `tempfile.mkdtemp(prefix=self.prefix)`
canonical:   n/a
destination: n/a
knowledge:   "which media types keep their originals." Encoded twice, at :2835 and :3281, with
             opposite polarity, and the two encodings only agree by accident.
scenario:    `POST /api/v1/media/add` with `media_type=audio` (or `video`, `json`, `email`, `xml`)
             and `keep_original_file=true`.
             Step 1 — `:2835` evaluates `(not True) or ("audio" in {"pdf","document","ebook"})`
             = `False or False` = **False**, so `TempDirManager.__exit__` skips the `rmtree`.
             Step 2 — `:3281` requires `media_type in ["pdf","document","ebook"]`, which is False,
             so the file is never copied to permanent storage.
             Result: the `mkdtemp` directory under the system temp root, containing the full
             uploaded media, survives the request, and the "keep original" the caller asked for
             produced nothing durable. `grep -rn "media_processing_" tldw_Server_API/app` returns
             exactly one hit — the prefix definition — so no reaper exists anywhere. Every such
             request leaks one directory of the full upload size until the host is rebooted or the
             OS temp cleaner runs.
impact:      Medium. Unbounded disk growth proportional to (requests × upload size) on a documented
             API flag, plus a silently unfulfilled user request. Raised from Low because there is no
             cleanup path at all — not a delayed one, not a reaper, not a startup sweep — and the
             leaked content is user-uploaded media, so the disk fills with data that also should not
             be sitting in a world-readable temp root indefinitely.
             Note this is the one temp-file case in the module worth reporting: the other 27
             `tempfile.*` sites all use `TemporaryDirectory`/`NamedTemporaryFile` correctly, and
             `PDF/PDF_Processing_Lib.py:598`'s bare `mkdtemp` is properly cleaned in its `finally`
             at :1349-1418.
tests:       `tests/MediaIngestion_NEW/` and `tests/Media_Ingestion_Modification/` cover
             `/media/add`; nothing asserts the temp directory is gone afterwards.
effort:      cheap. Either extend the permanent-storage step at :3281 to the media types that can
             set the flag, or reject `keep_original_file=true` for types that cannot honour it.
             Either way, make `cleanup` depend on whether the original was actually moved, rather
             than re-deriving the type set. One test: call with `media_type=audio,
             keep_original_file=true` and assert the temp dir does not exist on return.
owner-only:  no
confidence:  confirmed — both expressions read, `TempDirManager.__exit__` read, and the absence of
             any reaper for the prefix verified by repo-wide grep.
```

### FINDING ingestion-media-processing-18 — the `/media/add` batch path opens and closes a database per item, twice per persisted item, and builds two clients inside loops

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       persistence.py:_with_media_db_session (134-148) — opens
               `create_media_database(client_id, db_path=db_path)` at :140 and calls
               `worker_db.close_connection()` at :148 on **every** invocation.
             Per-item in the pre-check loop (`for source_path_or_url in all_processing_sources:`
             at :4131): `_with_media_db_session` at :4306 and :4344.
             Twice per persisted item: the write at :3887 is immediately followed by
               `_enforce_chunk_consistency_after_persist` at :3902 → `_fetch_unvectorized_chunk_count
               (1029-1055)` → a second `_with_media_db_session` at :1042, solely to run one COUNT.
               Same shape per email child: loop at :5905 → `_with_media_db_session` at :6053 +
               check at :6068; loop at :6155 → `:6303` + check at :6318.
             Client re-instantiation inside loops:
               persistence.py:2455 — `EmbeddingsJobsAdapter()` inside `for result in results:` (:2429)
               persistence.py:3334-3335 — `StorageQuotaService()` + `await
                 quota_service.initialize()` inside `for result in results:` (:3288). The other
                 quota call site, :4984, correctly uses the module singleton
                 `get_storage_quota_service()`.
             Unindexable scan:
               persistence.py:_source_hash_precheck:4263-4281 and :4284-4302 —
                 `JOIN DocumentVersions dv ... AND dv.safe_metadata LIKE ?` with
                 `hash_fragment = f'%"source_hash":"{source_hash}"%'` (:4277, :4298). A
                 leading-wildcard LIKE on a JSON text column cannot use an index.
canonical:   n/a for the connection churn (the fix is to pass one session down). For the quota
             service, `get_storage_quota_service()` is the canonical accessor and :3334 bypasses it
             — that half is an adoption-gap.
destination: n/a
knowledge:   n/a
cost-driver: (1) **SQLite connection lifecycles.** One full open + pragma application + close per
             input item in the pre-check loop, before any processing starts: a 100-item
             `/media/add` pays 100 open/close cycles up front. Then `2 × (1 + n_children)` more per
             persisted document — a 50-attachment `.eml` costs 102 opens. Scales with
             `len(urls) + len(files)` and with attachment count.
             (2) **Full-table scan on `DocumentVersions`.** `O(rows × avg len(safe_metadata))` bytes
             scanned per non-URL audio/video item. This is the *fallback* branch, reached whenever
             the fast `source_hash` column lookup at :4247-4260 misses — i.e. on every genuinely
             new upload, which is the common case. Total = `n_items × full_scan`.
             (3) **Per-item client construction.** `StorageQuotaService.initialize()` is awaited
             once per Success result instead of once per process; `EmbeddingsJobsAdapter()` is
             constructed once per result.
             Negative result, stated explicitly: there is **no** unbounded SELECT — all four raw
             queries carry `LIMIT 1` (:4254, :4272, :4293, :4335) — and no unbounded `fetchall()`
             anywhere in the nine orchestration files. The whole-file `await file_obj.read()` calls
             at :5044, :5075 and :5268 are all preceded by a size cap (streaming `max_bytes` in
             `download_utils.py:244-256` for :5044; the 1 MB-chunk cap in
             `input_sourcing.py:382-394` for :5075/:5268), so they are bounded, not unbounded. The
             residual there is concurrency: `process_document_like_item` runs under
             `asyncio.Semaphore(_document_like_concurrency_limit())`, default 10 (:475), so worst
             case is ~10 × the per-type cap of resident `file_bytes` per in-flight request, with no
             global cap across requests.
tests:       `tests/MediaIngestion_NEW/`, `tests/Media_Ingestion_Modification/`,
             `tests/DB_Management/` cover the behaviour; none asserts connection counts or query
             plans.
effort:      moderate. Hoisting the two clients out of their loops is cheap and should be done
             immediately (:3334 should simply call `get_storage_quota_service()` like :4984 does).
             Threading one session through the pre-check loop is a signature change to
             `_source_hash_precheck` / `_url_precheck`. Replacing the `LIKE` needs a real index —
             either promote `source_hash` out of the JSON blob into a column on `DocumentVersions`
             (the `Media` table already has one, per `_media_has_source_hash_column` at :1178-1188)
             or add an expression index; that is a migration and needs a design note.
owner-only:  no
confidence:  confirmed for every call site and loop nesting (all read);
             probable-risk for the magnitude, which depends on DB size and backend and was not
             measured.
```

### FINDING ingestion-media-processing-19 — the PDF OCR page loop renders every page into memory before consuming any result

```
axis:        efficiency
class:       n/a
severity:    Medium
sites:       PDF/PDF_Processing_Lib.py:_ocr_pdf_pages (1552-1675) — the submit loop at :1608-1648
               renders `pix = page.get_pixmap(matrix=mat, alpha=False)` and
               `img_bytes = pix.tobytes("png")` (:1632-1634) for every page and calls
               `pool.submit(...)` (:1641 / :1643) for each, appending to an unbounded `futures`
               list; the drain loop `for fut in as_completed(futures):` does not begin until
               :1645, after the entire document has been rendered.
             Callers: PDF/PDF_Processing_Lib.py:process_pdf (478-1443) via the OCR branch at
               :1684+; reached from :process_pdf_task (1445-1543).
canonical:   n/a
destination: n/a
knowledge:   n/a
cost-driver: **Peak resident memory = page_count × PNG bytes at the requested DPI.** Both factors
             are caller-supplied and neither is capped: `ocr_dpi` defaults to 300
             (`process_pdf` signature) and is passed through from the API, and nothing bounds
             `page_count`. `scale = max(dpi, 72) / 72.0` (:1601), so bytes per page grow roughly
             with dpi². A 300-DPI letter page renders to ~2550×3300 px; as PNG that is commonly
             1–3 MB. A 500-page scanned PDF therefore holds ~0.5–1.5 GB of page buffers before the
             first `as_completed` result is read. With the default `concurrency = 1`
             (`runtime_support` defaults page concurrency to one, asserted by
             `tests/Media_Ingestion_Modification/test_ocr_runtime_support.py::
             test_load_ocr_runtime_profiles_defaults_page_concurrency_to_one`), all but one of
             those buffers sit idle in the executor queue the whole time.
             Secondary, probable-risk: with `concurrency > 1` the pool calls `backend.ocr_image`
             from several threads into one shared module-global torch model
             (`_load_transformers`, stage 2 finding -6). Those loaders are correctly
             double-check-locked, but HF `generate()` is not documented thread-safe for concurrent
             calls on one model instance. Not reproduced here; flagged because the knob exists.
scenario:    An operator uploads a 400-page scanned contract at the default `ocr_dpi=300` with a
             local backend. `_ocr_pdf_pages` renders all 400 pages to PNG before consuming a single
             OCR result; the worker's RSS climbs by ~1 GB and stays there for the duration of the
             run. In a container with a memory limit this is an OOM kill with no partial results,
             on a document size the API accepts without complaint.
tests:       `tests/Media_Ingestion_Modification/test_ocr_structured_output.py` exercises the loop
             end to end for correctness. No test uses a document large enough to show the memory
             profile.
effort:      cheap — bound the in-flight work: submit at most `concurrency + k` futures, drain, then
             submit more; or move the `get_pixmap`/`tobytes` into the worker and submit page indices
             instead of bytes (PyMuPDF docs are not thread-safe, so this needs the render to stay
             serialised under a lock — the simpler bounded-submission version is preferable).
owner-only:  no
confidence:  confirmed (the render-all-then-drain structure, read at :1608-1668, and the uncapped
             dpi/page_count); probable-risk (the per-page byte figures, which are estimates, and
             the concurrent-generate concern).
```

### FINDING ingestion-media-processing-20 — three STT adapters skip the shared benchmark-mode guard and accept a mode the guard exists to reject

```
axis:        correctness
class:       adoption-gap
severity:    Medium
sites:       CANONICAL, adopted by 4 of 9:
               Audio/stt_provider_adapter.py:_require_benchmark_mode (395-405) — rejects unknown
                 modes, then rejects `production-v1` specifically with
                 `"production-v1 is unsupported for local STT providers"` (:401-404).
                 Called at :868 (FasterWhisper), :1051 (Parakeet), :1295 (Canary), :1475 (Qwen2Audio).
             BYPASSED — inline copies of only the first half:
               Audio/stt_provider_adapter.py:Qwen3ASRAdapter.plan_batch_execution:1668-1672
               Audio/stt_provider_adapter.py:VibeVoiceAdapter.plan_batch_execution:1956-1960
               Audio/stt_provider_adapter.py:ExternalAdapter.plan_batch_execution:2569-2573
               All three write `if normalized_mode not in {"neutral-v1","production-v1"}: raise
               STTExecutionUnsupportedError(...)` and omit the `production-v1` rejection.
             A FOURTH VARIANT:
               Audio/stt_provider_adapter.py:AudioCppAdapter._validate_plan_request (2327-2351) —
                 rejects everything except `neutral-v1` (:2337-2340), with its own message string.
             RELATED, same file, same class of problem — the base class hardcodes its subclasses:
               Audio/stt_provider_adapter.py:SttProviderAdapter._transcribe_planned_batch (771-794)
                 is a concrete default whose body is
                 `if self.name not in {FASTER_WHISPER, PARAKEET, CANARY, QWEN2AUDIO}: raise ...`
                 — four subclasses rely on the base knowing their names, so adding a tenth local
                 provider means editing the base class.
               Audio/stt_provider_adapter.py:Qwen3ASRAdapter._transcribe_planned_batch (1844-1868)
                 and :VibeVoiceAdapter._transcribe_planned_batch (2187-2211) are a 25-line
                 copy-paste differing in exactly four lines (module name, function name, one kwarg,
                 the error string).
               `SttExecutionDescriptor(` is built by the shared `:_build_local_plan (550-638)` at
                 :608 and then hand-rolled again at :1804, :2135, :2396, :2700.
canonical:   `Audio/stt_provider_adapter.py:_require_benchmark_mode (395-405)`.
destination: n/a — the helper already exists in the right file. This is pure adoption.
knowledge:   "which STT benchmark modes a local provider may accept." Four answers in one 3,159-line
             file.
scenario:    `get_stt_provider_registry().get_adapter("qwen3asr").plan_batch_execution(
             model=None, language="en", task="transcribe", word_timestamps=False, prompt=None,
             hotwords=None, diarization=False, mode="production-v1")` returns a plan. The identical
             call against `faster_whisper` raises `STTExecutionUnsupportedError`. Qwen3-ASR is a
             local provider — its local branch loads a Transformers model on-box
             (`Audio_Transcription_Qwen3ASR.py:302`) — so it is exactly what the guard's message
             says is unsupported. Benchmark runs comparing providers under `production-v1` will
             therefore silently include Qwen3-ASR and VibeVoice while excluding the four providers
             that adopted the guard, making the comparison invalid rather than erroring.
             `ExternalAdapter`'s acceptance is arguably intentional (it is not a local provider);
             Qwen3-ASR's is not, and it has no compensating logic — it rejects hotwords and prompt
             unconditionally at :1673-1678, whereas VibeVoice (:1966) and External (:2579) gate
             those on `neutral-v1` only.
impact:      Medium. Wrong benchmark results rather than wrong transcripts, and the guard's own
             error text documents the intended policy that three adapters silently opt out of.
tests:       `tests/STT/` and `tests/Benchmarks/` import `stt_provider_adapter`; none asserts that
             all nine adapters agree on which `mode` values are rejected — which is the one test
             that would have caught this and is cheap to write as a parametrised loop over the
             registry.
effort:      cheap — replace the three inline blocks with `_require_benchmark_mode(mode)`, decide
             explicitly whether `ExternalAdapter` is exempt (and if so say so in the helper via a
             parameter, not by omission), and add the parametrised registry test.
             Separately cheap: invert `_transcribe_planned_batch` (771-794) so the base raises
             unconditionally and the four local adapters each get a one-line override — that
             removes the base class's knowledge of its own subclasses.
owner-only:  no
confidence:  confirmed — the canonical helper, its four callers, and all three inline copies read
             at the cited lines.
```

## Also verified — folded rather than filed separately

Each of these is real and was confirmed at the cited line, but is either dead code, already
mitigated, or too narrow to carry its own finding. Recorded so the next reviewer does not re-derive
them.

- `Audio/Audio_Streaming_Parakeet.py:is_transcription_error_message (29-42)` resolves the real
  implementation through `sys.modules.get("...Audio_Transcription_Lib")` and returns `False` when
  the module is absent (:33-34). That file imports only `Audio_Transcription_Nemo` and
  `Audio_Transcription_Parakeet_MLX` (:44-49), neither of which references
  `Audio_Transcription_Lib` (`grep -c` → 0 for both). So in a process that only serves the Parakeet
  streaming WebSocket, the guard at `:220` (also `:246`, `:305`) is always `False` and the sentinel
  string `"[Error] Parakeet transcription error"` is emitted to the client as a transcript segment.
  Genuine defect; folded because the fix is one import and it overlaps finding -15's file.
- `Audio/Audio_Transcription_Lib.py:format_transcription_with_timestamps (4975-5019)` reads only
  `Time_Start`/`Time_End` and collapses every segment to `[00:00:00-00:00:00]` for the
  `start_seconds`/`end_seconds` shape that `stt_provider_adapter.py:1428-1432` and
  `Audio_Transcription_VibeVoice._normalize_segments (232)` actually emit. The correct twin is
  `Audio/Audio_Files.py:format_transcription_with_timestamps (1470-1528)`, which falls back through
  three key spellings (:1509-1514). Folded because the `Audio_Transcription_Lib` copy has **zero
  callers repo-wide** — it is 45 lines of dead, wrong, public-named code shadowing the correct one.
  Deleting it is the whole fix.
- `Audio/Audio_Transcription_Qwen3ASR.py:_check_cancel (109-118)` and
  `Audio/Audio_Transcription_VibeVoice.py:_check_cancel (104-112)` are byte-identical apart from a
  docstring and do `bool(cancel_check())`, so an `async def cancel_check` is always truthy and
  aborts the transcription on the first checkpoint. `Audio/Audio_Transcription_Lib.py:_check_cancel
  (4452-4480)` handles this correctly via `inspect.isawaitable` (:4459). Folded into finding -20's
  family: `Qwen3ASR` and `VibeVoice` are a fork of each other — ten same-named functions whose
  diffs are a dropped docstring, the exception-tuple name, and a provider string in a log line
  (`_as_bool`, `_as_int`, `_as_str`, `_check_cancel`, `_resolve_settings`, `_resolve_audio_path`,
  `_load_audio`, `_maybe_resample`, `_get_torch_dtype`, `_resolve_device`, `_normalize_artifact`,
  `_transcribe_local`). That fork is the root cause; fixing `_check_cancel` twice is not.
- `Audio/Audio_Transcription_Parakeet_MLX.py:install_parakeet_mlx (267-300)` runs
  `subprocess.run([sys.executable, "-m", "pip", "install",
  "git+https://github.com/senstella/parakeet-mlx.git"])` at :291. It has **no production caller**;
  the only reference outside its own definition and a comment is
  `tests/Media_Ingestion_Modification/test_parakeet_mlx.py:201`, which monkeypatches it to a
  function that fails the test if invoked. Dead code, but a runtime `pip install` from a GitHub URL
  living in server code is a supply-chain surface that should be deleted rather than left dormant.
- `Audio/Audio_Transcription_Lib.py:_AUDIO_VALIDATION_CACHE (544)` is an unbounded dict keyed by
  resolved path, read at :4578 and written at :4622, never pruned. Grows for the process lifetime.
  Bounded in practice by distinct file paths seen; `functools.lru_cache` or a size cap is the fix.
- `Audio/Audio_Transcription_Nemo.py:492` and `:600` set `os.environ['NEMO_CACHE_DIR']` from inside
  a model loader that runs on worker threads — process-global mutation from a thread pool. Folded
  into finding -14, which touches the same loaders.
- `Audio/Audio_Files.py:_enforce_download_quota (166-191)` calls `asyncio.run(_check_quota())` at
  :191 from sync code. Safe **today** only because both callers of `process_audio_files` reach it
  through `run_in_executor` (a thread with no running loop). One direct call from an async context
  turns it into `RuntimeError: asyncio.run() cannot be called from a running event loop`.

## Explicit negative results

Hypotheses tested and **disproved** — recorded so they are not re-opened:

- `Audio/Audio_Transcription_Lib.py` has **zero** `async def`. No event-loop blocking is possible
  there despite its size.
- `Audio/Audio_Files.py` has one nested `async def` (`_check_quota`, :173-189) which awaits
  correctly. No blocking.
- `audio_batch.py:163-164` and `persistence.py:4612-4614` correctly offload the synchronous
  processing libraries via `run_in_executor`; `PDF/PDF_Processing_Lib.py:process_pdf_task:1497`
  uses `asyncio.to_thread`. The async boundary is right in all three.
- `Audio/Audio_Streaming_Unified.py:StreamingDiarizer (834-1143)` offloads all heavy work
  (:937, :960, :1067). Looked like finding -15; is not.
- `stt_provider_adapter.py:plan_batch_execution` ×9 is **not** copy-paste. The nine 12-line
  signatures are identical, but the bodies genuinely differ per provider and the shared skeleton was
  already extracted into `_build_local_plan (550-638)`. Only the *guard adoption* diverges
  (finding -20). Likewise `transcribe_batch` ×9: the first 27 lines are identical in 7 of 8 concrete
  adapters (91 duplicated lines, removable by one template method in the base), but everything below
  is real per-provider logic — FasterWhisper maps `task=="translate"` to `selected_lang=None`
  (:1075) and strips the Whisper metadata header (:1005) where Parakeet does neither; Canary reads
  the file with `soundfile` (:1409) and fabricates a zero-duration segment (:1428-1432). Filed as
  duplication only in the narrow 91-line sense, not as the 9× clone it first appears to be.
- Temp-file lifecycle in this module is otherwise sound: 27 of 28 `tempfile.*` sites use
  `TemporaryDirectory`/`NamedTemporaryFile` or a `finally` unlink correctly. `PDF/PDF_Processing_Lib.py:598`'s
  bare `mkdtemp` is cleaned in its `finally` at :1349-1418 (with retries), and
  `input_sourcing.py:TempDirManager` is a correct context manager. Only finding -17 is real — and
  it is a policy bug in the `cleanup=` argument, not a lifecycle bug.
- `defusedxml` is used at every XML entry point in the module. No `xml.etree` import exists outside
  a docstring. The only issue is the four-way divergence in what happens when it is missing
  (stage 3, finding -8 scenario B).
- `Docs/ADR/026` SSRF policy is honoured: `persistence.py:4915` calls
  `Security/url_validation.assert_url_safe` per item before download.

## Synthesis

The module's defects have one shape, repeated at four scales.

**A sibling set is created by copy-paste; the copies then drift; nothing detects the drift.** Nine
OCR backends (findings -1 to -7), nine STT adapters (-20), six model caches (-14), six resamplers
(-13), three ffmpeg resolvers (-16), three JSON salvagers (-4), 26 exception allowlists (-8), two
result-envelope types (-12). In every case the *first* copy is defensible and the *nth* is a
liability, because the decision the copies encode — what "true" means, what may be swallowed, how
long a temp file must live, whether the resampler may give up — is exactly the decision most likely
to be revised later.

Three structural facts keep the drift invisible:

1. **The tests are organised by history, not by module** (finding -11). Eleven test trees exercise
   one source file; four exercise the OCR backends; three of the nine OCR backends have no
   behavioural test at all. An engineer cannot see what their change covers.
2. **No test asserts a policy or a resource property.** Nothing checks which exceptions are
   swallowed, whether a task cancels, how many connections a batch opens, or whether the loop
   stalls. Findings -8, -13, -14, -15, -18 and -19 are all invisible to a green suite.
3. **The canonical helpers exist and are undiscoverable.** `core/testing.py:30 is_truthy` is the
   repo's truthy parser, living in a module whose docstring is about test-mode detection.
   `OCR/runtime_support.py` calls itself "Shared helpers for OCR runtime configuration" and no
   backend imports it. `result_normalization.MediaItemProcessResponse` has zero importers.
   `pyproject.toml`'s `media_processing` marker has one use. Each is an adoption gap caused by the
   helper being in a place nobody would look.

The repo already contains the template for the largest fix: `core/DB_Management/Media_DB_v2.py` was
a monolith with 121 commits of churn and no longer exists as a file — it is the `media_db/` package
(`api.py`, `constants.py`, `errors.py`, `legacy_content_queries.py`, `runtime/`). `persistence.py`
at 6,390 LOC / 98 commits is the same problem at the same layer with the same owners, and stage 3
names five seams along the same lines.

## Suggested Refactor/Actions

### Immediate (cheap, no design doc, each one line to a few lines)

| # | Change | Finding |
| --- | --- | --- |
| 1 | `delete=False` + `finally: os.unlink` in `dots_ocr.py:195` and `hunyuan_ocr.py:192` | -3 |
| 2 | Drop `asyncio.CancelledError` from `persistence.py:84`, `Audio_Streaming_Unified.py:80`, `Audio_Transcription_Lib.py:93` | -8 |
| 3 | Drop `HTTPException` from `persistence.py:101`; add `except HTTPException: raise` before the handler at `persistence.py:5079` | -8 |
| 4 | Add `ImportError` to `_UPLOAD_SINK_NONCRITICAL_EXCEPTIONS` and `_PERSISTENCE_NONCRITICAL_EXCEPTIONS` | -8 |
| 5 | Key `_mlx_model_cache` on `(model_id, dtype, cache_dir)`; add `threading.Lock` to the Nemo and Parakeet-ONNX caches | -14 |
| 6 | Delete the unconditional `sample_rate = 16000` at `Audio_Buffered_Transcription.py:386` and `:754`; make `_resample` raise or actually resample | -13 |
| 7 | `asyncio.to_thread` around the four blocking calls in `Audio_Streaming_Unified.py` | -15 |
| 8 | Derive ffprobe with `Path(x).with_name("ffprobe")` at `Audio_Transcription_Lib.py:4792` | -16 |
| 9 | `_require_benchmark_mode(mode)` at `stt_provider_adapter.py:1668`, `:1956`, `:2569` | -20 |
| 10 | `get_storage_quota_service()` at `persistence.py:3334`; hoist `EmbeddingsJobsAdapter()` out of the loop at `:2455` | -18 |
| 11 | Collapse `_resolve_json_prompt`'s three equal branches in `dolphin_ocr.py:231-247` | -2 |
| 12 | Delete the dead `format_transcription_with_timestamps` at `Audio_Transcription_Lib.py:4975-5019` and the dead `install_parakeet_mlx` at `Parakeet_MLX.py:267-300` | folded |

Every one of these needs the regression test named in its finding. Several would be the **first**
test in the module to assert the property in question.

### Backlog tasks to propose (not created — this audit is read-only; the Backlog is managed via its
MCP/CLI and task files must never be hand-edited)

| Task | Findings | Needs `Docs/Design/` + ADR? |
| --- | --- | --- |
| `Characterisation tests for dolphin / hunyuan / nemotron OCR backends` | gate for -1..-5 | no |
| `Ingestion immediate-fix batch (12 items above)` | -2,-3,-8,-13,-14,-15,-16,-18,-20 | no |
| `Adopt pytest.mark.media_processing across the 192 ingestion test files` | -11 | no |
| `Fix keep_original_file temp-dir leak for non-document media types` | -17 | no |
| `Bound page-image fan-out in _ocr_pdf_pages` | -19 | no |
| `Create app/core/Utils/coercion.py and migrate scalar/env coercion` | -1 | **yes** — repo-wide policy, write the design once at repo scope |
| `Define noncritical-exception policy in core/exceptions.py` | -8 | **yes** — interacts with the BLE001 grandfather list at `pyproject.toml:756` |
| `Unify the media process-result envelope` | -12 | **yes** — crosses the owner-only `api/v1/schemas/` boundary |
| `Move media dedupe SQL into core/DB_Management/media_db` | -10 | no |
| `Relocate document_upload_drafts storage into DB_Management` | -10 | **yes** — storage migration; SQLite-only today, no Postgres path |
| `Import ratchet: core -> api/v1, seeded at 107 files` | -9 | no — copy `tests/lint/test_endpoint_auth_deps_import_boundary.py` |
| `Extract persistence.py metrics and transcript-reuse seams` | stage 3 §6 | **yes** — follow the `media_db/` package split as the template |

Base branch assumed: `dev`, per `CONTRIBUTING.md:86,121`. Nothing here modifies
`tldw_Server_API/app/api/v1/**` except the envelope task, which is labelled owner-only.
