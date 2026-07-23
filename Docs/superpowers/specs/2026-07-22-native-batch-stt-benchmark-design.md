# Native Batch STT Benchmark Design

**Status:** Approved design

**Date:** 2026-07-22

**Backlog:** TASK-12985

**Scope:** Standalone, batch-only benchmarking of tldw_server STT provider adapters

## Summary

tldw_server should gain a small, standalone benchmark harness that compares
batch speech-to-text providers through the existing native
`SttProviderRegistry` and `SttProviderAdapter.transcribe_batch()` boundary,
extended with an optional immutable execution plan for benchmark calls.
The benchmark will use independently verified references, deterministic strict
and normalized WER/CER scoring, isolated provider/model worker processes,
separate cold-first-transcription and warm-adapter-transcription measurements,
incremental JSONL persistence, and reproducible regression and comparison
profiles.

The harness will not use Pipecat, an LLM judge, FastAPI, Jobs, or the
tldw_server Evaluations service. It will live with the repository's other
standalone benchmarks under `Helper_Scripts/benchmarks/`.

## Context

The initial reference was
[pipecat-ai/stt-benchmark](https://github.com/pipecat-ai/stt-benchmark/).
The review in this document reflects upstream `main` at commit `66f2cbf8`
(2026-07-18), inspected on 2026-07-22. Upstream behavior may change after that
revision.
That project is useful as an example of resumable per-sample execution,
provider/model registration, percentile reporting, and latency-versus-accuracy
analysis. Its central methodology does not fit this project:

- It streams PCM audio through a Pipecat pipeline with Silero VAD and Pipecat
  STT services.
- Its latency metric is speech-end-to-final-segment latency, which is a
  streaming voice-agent metric rather than a batch transcription metric.
- It generates reference transcripts with Gemini and computes "Semantic WER"
  with Claude.
- Its published result table is manually upserted into the README rather than
  being a report generated from a complete, immutable run artifact.

tldw_server already has a better execution boundary for this use case:
`SttProviderRegistry` resolves a native provider adapter, and every batch
adapter returns a normalized transcription artifact. The repository also has
an opt-in golden-audio test and mocked performance tests. These provide useful
test scaffolding but are not yet a trustworthy comparative benchmark.

## Review of the Pipecat Benchmark

### Useful ideas to retain

- Persist each completed sample immediately so interrupted runs can resume.
- Identify results by provider and concrete model, not vendor alone.
- Record transcription success rate separately from accuracy.
- Report median and tail percentiles rather than only means.
- Preserve per-sample hypotheses and show the worst errors.
- Keep provider construction lazy so unavailable optional dependencies do not
  prevent unrelated targets from running.

### Methodology and reproducibility risks

1. **LLM-dependent scoring**

   Semantic WER depends on a proprietary model, a long natural-language
   rubric, and the model's interpretation of semantic equivalence. A future
   model revision can change scores without any STT output changing.

2. **Reference provenance**

   The published dataset references were generated with Gemini and then
   human-reviewed. That can be useful, but it is not equivalent to using
   canonical corpus transcripts or a documented independent transcription
   protocol. The benchmark should preserve per-sample provenance and reviewer
   status.

3. **Configuration fairness**

   Provider factories contain different language hints, prompts, endpointing
   controls, VAD thresholds, and vendor-specific turn settings. Those may be
   valid production configurations, but results compare complete
   configurations rather than isolated model quality. A benchmark must publish
   every material setting and distinguish requested from actual model identity.

4. **Narrow workload**

   Published results use 1,000 English samples from a single Smart Turn
   training dataset. That is useful for short conversational turns but does not
   represent tldw_server's long-form podcasts, lectures, interviews, noisy
   media, compressed inputs, or technical vocabulary.

5. **Streaming-specific latency coupling**

   TTFS depends on Pipecat's VAD timestamps, finalization behavior, and
   provider integration. It does not answer the batch questions of cold first
   transcription, warm wall time, real-time factor, throughput, or long-form
   scaling.

6. **Published artifact traceability**

   The README table is treated as the source of truth and is updated one row at
   a time. The public summary does not establish a direct, machine-verifiable
   link to the full run configuration, raw per-sample results, hardware,
   network conditions, or uncertainty.

7. **Dependency and test surface**

   The package installs a large Pipecat extra set for all supported providers.
   Its current repository test tree contains one README-table-focused test
   module, leaving the core runner, timing, persistence, and LLM scorer with
   limited automated coverage.

These are reasons to reuse ideas rather than fork or adapt that implementation.

## Goals

- Compare every batch-capable tldw_server STT provider/model through the native
  adapter contract.
- Support a small stable regression profile and a larger comparison profile
  with the same manifest, runner, scorer, and result schema.
- Use independent references and deterministic scoring only.
- Report strict and normalized WER/CER without hiding meaningful recognition
  errors.
- Measure cold-first-transcription behavior separately from warm adapter
  transcription.
- Produce crash-safe, resumable, inspectable results.
- Capture enough environment and configuration metadata for an honest
  comparison.
- Remain English-first while making the manifest and scorer ready for
  language-specific profiles.
- Keep real-model execution opt-in and outside ordinary PR CI.

## Non-goals

- Streaming audio, VAD, TTFS, partial-result, or endpointing benchmarks.
- Diarization accuracy or speaker-attributed WER.
- FastAPI, authentication, Jobs, persistence APIs, or WebUI integration.
- Integration with the generalized Evaluations module in the first version.
- Distributed or parallel model execution.
- Automatic model installation or download.
- Fine-tuning or training workflows.
- Declaring one universally best model across different workloads and
  hardware.

## Architecture

The execution flow is:

```text
versioned manifest
    -> coordinator and target matrix
    -> isolated provider/model worker
    -> native SttProviderAdapter.transcribe_batch()
    -> normalized artifact and timings
    -> deterministic scorer
    -> incremental JSONL record
    -> JSON, Markdown, and terminal summaries
```

### Coordinator

The coordinator is the user-facing CLI process. It:

1. Validates the manifest before loading any model.
2. Resolves the selected profile and sample set.
3. Validates all provider/model targets without invoking registry fallback.
4. Creates or resumes a run directory.
5. Starts one isolated worker process per provider/model target, sequentially.
6. Monitors worker exit status and preserves partial results.
7. Builds summaries from the append-only result records.

Targets run sequentially in requested CLI order. Parallel model execution would
introduce CPU, GPU, storage, network, and thermal contention and is outside v1.

### Isolated target worker

Each provider/model combination runs in a fresh process. The worker:

1. Imports the provider lazily.
2. Resolves the requested adapter through `SttProviderRegistry`.
3. Records safe setup and environment metadata.
4. Runs the target's stable cold-probe sample as a cold first transcription.
5. Reuses the loaded adapter/model state for the remaining warm samples.
6. Announces the in-flight completion key to the coordinator before entering
   each synchronous adapter call.
7. Appends and durably flushes each completed sample record before starting
   the next.
8. Converts adapter exceptions, error sentinels, and empty artifacts into
   explicit result statuses.

Process isolation prevents optional-dependency failures, global model caches,
GPU allocations, and hard crashes from contaminating later targets.

### Adapter boundary

The benchmark calls the existing synchronous
`SttProviderAdapter.transcribe_batch()` method. It does not add parallel model
implementations or bypass production provider code.

The benchmark must fail closed:

- An unknown provider is a configuration error.
- A missing adapter is a target failure.
- The registry's defensive faster-whisper fallback must not be used for an
  invalid benchmark target.
- Project-defined transcription error sentinels such as `[Error: ...]` are
  adapter failures, never hypotheses. Adapters should raise a typed error at
  the native boundary; the worker also applies the existing centralized
  sentinel check as defense in depth.
- Requested provider/model identifiers and an allowlisted actual-execution
  envelope are recorded. Arbitrary adapter metadata is never copied into
  benchmark artifacts.
- A material requested/actual identity mismatch is visible in the result and
  prevents a baseline comparison unless explicitly corrected.

### Execution-contract preflight

The current `SttProviderCapabilities` object is not sufficient to decide
whether a configured target is local, network-backed, or eligible for
`neutral-v1`. Qwen3-ASR and VibeVoice can select HTTP backends from
configuration, and VibeVoice can fall back from HTTP to local inference.
Provider name alone must therefore never be used as an egress or backend
classification.

Every benchmark-capable adapter exposes one small, non-loading
`plan_batch_execution(...)` operation for the requested model and common
semantic settings. It returns an immutable, multiprocessing-serializable
execution plan plus a safe descriptor containing:

- Requested and resolved provider/model identity.
- Stable model revision, local artifact identifier, or an explicit
  `identity_unresolved` marker.
- Expected backend and whether backend fallback is possible.
- Audio egress as `none`, `loopback`, or `remote`, plus a non-secret opaque
  endpoint or region identifier for network-backed targets.
- Whether the requested task, language hint, prompt absence, hotword absence,
  diarization setting, and timestamp setting are honored.
- Expected device, compute type or dtype, and material decoding settings when
  discoverable without loading the model.
- Whether the model is available locally and whether executing the target
  would initiate a model download.

The coordinator creates and approves the plan before the target worker opens
audio for transcription, loads a model, or makes a network call. Manifest
validation may already have read the local file for hashing and duration. The
plan may carry sensitive runtime values in memory, but only its allowlisted
safe descriptor is hashed, logged, or serialized to disk. The coordinator
passes that exact plan to the worker. The worker verifies its safe descriptor
against `run.json`, then calls
`transcribe_batch(..., execution_plan=plan)`.

`execution_plan` is an optional backward-compatible adapter argument. Existing
production callers may omit it and retain current config-driven behavior. When
the benchmark supplies it, the adapter and underlying provider helpers must use
the resolved backend, endpoint, semantic settings, model path, fallback policy,
and no-download policy from the plan. They must not reread material execution
configuration or choose an unplanned backend. A plan mismatch fails before the
audio file is opened, closing the time-of-check/time-of-use gap between consent
and transcription. Network execution plans disable HTTP redirects so an
approved endpoint cannot forward audio to an unclassified destination.

Unknown egress fails closed. Any `loopback` or `remote` target requires
`--allow-network-targets` before its worker starts. A `neutral-v1` target must
pin one backend, prohibit backend fallback, and prove that the common semantic
settings are honored. A `production-v1` target may retain configured fallback,
but every sample records the actual backend. Mixed-backend production results
are reported by backend and as a complete-configuration outcome; they are not
eligible for model-quality ranking or performance gates.

`loopback` is reserved for literal localhost or loopback-address endpoints.
Hostnames or addresses that are not unambiguously loopback are classified
`remote`; the harness does not weaken privacy based on DNS inference.

The harness never installs or downloads models. Local measured targets must
pass preflight with their weights already available and with native
no-download behavior enforced by the supplied plan. The worker also enables
standard offline-library controls as defense in depth. A provider whose loader
cannot honor the no-download plan is unsupported by the benchmark until its
native loader exposes that control. This prevents network downloads from
silently becoming most of a `cold_first_transcription_seconds` measurement.

The descriptor is hashed with the benchmark/scorer implementation identity,
tldw_server Git commit, execution-relevant benchmark/adapter/provider source
fingerprints, relevant dependency versions, and safe target settings to produce
an `execution_contract_hash`.

After each call, the adapter returns a normalized, allowlisted actual-execution
envelope containing only provider, resolved model/artifact ID, backend/source,
device, compute type or dtype, and non-sensitive decoding identifiers. The
benchmark never serializes the provider's unrestricted metadata mapping; prompt
text, hotwords, headers, tokens, URLs, and unknown fields are discarded. A
mismatch is stored explicitly. Results may remain descriptive when identity
cannot be resolved, but unresolved identity or an
actual-versus-planned mismatch is never eligible for baseline gates. Exact hash
equality defines one resumable run generation; cross-run regression
compatibility compares the documented material fields so the implementation or
dependency version under test is allowed to differ.

No provider lifecycle hook is required in v1. Because some adapters load
weights lazily inside the first transcription, the cold measurement is named
`cold_first_transcription_seconds`; it is not presented as pure model-load
time. Explicit `prepare()` or `unload()` hooks can be considered later if they
become useful outside the benchmark.

Each run has one deterministic `cold_probe_sample_id`, shared by every
comparable target and stored in `run.json`. That gives every target the same
audio for its cold-first measurement. The probe is always excluded from warm
aggregates. On the first worker attempt its transcript is also the sample's
scored result. When a resumed worker finds that completion key already
present, it retranscribes the same probe only to restore warmed model state;
the new probe timing is stored in worker-attempt metadata and is not added as
another scored sample.

If the probe call fails, the worker may continue producing accuracy results,
but it does not report any call as warm yet. The first later successful
transcription is labeled `warmup_recovery` and excluded from warm-performance
aggregates; only subsequent successful calls are warm. If no warm-up succeeds,
the target has no warm-performance summary. Pending samples therefore retain
explicit and honest cold/warm classification whether the run finishes in one
process, recovers from a failed probe, or resumes across several.

### Comparison modes

Every target belongs to one named mode:

- **`neutral-v1`:** `task=transcribe`, the manifest language hint, no prompt,
  no hotwords, no diarization, and no requested word timestamps. This is the
  only mode eligible for model-quality ranking in v1. An adapter may retain an
  unavoidable provider default, but the report must expose it; a provider that
  cannot honor the common semantic settings is marked unsupported for this
  mode.
- **`production-v1`:** invokes the project's configured production adapter
  settings and permits prompts, hotwords, or provider-specific decoding
  options. These results compare complete configurations, not isolated model
  quality, and are labeled accordingly.

Targets appear in the same aggregate comparison only when they use the same
mode and common semantic settings. Provider-specific tuning is never silently
mixed into a `neutral-v1` leaderboard. A production target with sensitive
settings uses a user-supplied opaque `configuration_id`; prompt and hotword
contents are not serialized.

## Dataset Design

### One manifest, two profiles

Both profiles use one versioned JSONL manifest:

- **Regression:** a stable, stratified subset of approximately 40-100 samples
  intended for regular local validation and release checks.
- **Comparison:** hundreds or thousands of samples for model selection,
  dataset-slice analysis, and statistically useful comparisons.

The profile is a property of each sample, so there is no separate and drifting
fixture format.

### Hybrid English-first corpus

The first corpus should combine:

1. **LibriSpeech public pack**
   - Deterministically selected samples from `test-clean` and `test-other`.
   - Canonical transcripts and source metadata retained.
   - Dataset version and file checksum pinned.

2. **Optional Common Voice English import**
   - Used for greater accent, speaker, and recording-condition diversity.
   - Imported from a user-provided Mozilla Data Collective download.
   - Not mirrored by the repository or by a tldw_server-owned download host.

3. **tldw challenge pack**
   - Podcasts, lectures, interviews, technical jargon, proper nouns,
     compressed media, background noise, overlapping speech, varied accents,
     and several long-form samples.
   - Audio must be user-owned, redistributable, or kept in a private local
     corpus referenced by the manifest.
   - References require manual transcription and a second-person review.

The schema requires a BCP 47-compatible language tag from the beginning.
Future multilingual packs can add language-specific normalization profiles
without changing runner or result schemas. FLEURS is a suitable future
multilingual source, but multilingual data is not required for v1.

V1 uses a documented `bcp47-basic-v1` syntax check: a 2-8 ASCII-letter primary
language subtag followed by zero or more hyphen-delimited 1-8 ASCII
alphanumeric subtags. It canonicalizes case for comparison but does not claim
IANA-registry validation. Provider support is a separate execution-preflight
decision.

### Manifest record

Each JSONL record has this conceptual shape:

```json
{
  "id": "librispeech-test-other-1234-5678-0001",
  "audio": "public/librispeech/test-other/1234/5678/clip.flac",
  "reference": "the independently verified transcript",
  "language": "en",
  "normalization_profile": "en-v1",
  "duration_seconds": 8.42,
  "profiles": ["comparison"],
  "suite": "public-english-v1",
  "suite_visibility": "public",
  "annotation_profile": "librispeech-canonical-v1",
  "diagnostic_only": false,
  "source": {
    "dataset": "librispeech",
    "version": "openslr-12",
    "split": "test-other",
    "license": "CC-BY-4.0",
    "sha256": "hex digest",
    "reference_provenance": "canonical-dataset"
  },
  "tags": [
    "read-speech",
    "single-speaker",
    "challenging"
  ]
}
```

Required validation includes:

- Unique, stable sample IDs.
- Non-empty independent reference text.
- Syntactically well-formed BCP 47-compatible language tag and a known
  normalization profile. This is syntax/profile validation, not a claim that
  every provider supports the tag.
- Audio path containment under an explicit dataset root.
- Existing regular file, no symlink escape, and matching SHA-256.
- Positive duration measured from the checked audio file.
- When `duration_seconds` is declared, agreement with the measured duration
  within the greater of 100 ms or 1 percent.
- Known profile names and bounded tags.
- Stable suite and annotation-profile identifiers, with `suite_visibility`
  restricted to `public` or `private`.
- One consistent visibility for every record sharing a suite ID.
- Complete source, version, license, and reference provenance.

Absolute local paths are accepted at the CLI boundary through `--dataset-root`
but are not written into portable result artifacts. Records retain
manifest-relative paths. Performance metrics always use the measured duration,
not the declared metadata value. Duration probing occurs during validation,
outside every provider timing window, through one documented decoder path so
all targets use the same duration.

### Ground-truth policy

- Public benchmark samples use canonical dataset references.
- tldw challenge references are manually transcribed and independently
  reviewed.
- A model output can be stored as an unverified transcription candidate or a
  behavior snapshot, never as ground truth merely because a model produced it.
- Reference corrections require a manifest version change and a new hash.
- The report records the manifest hash and reference-provenance counts.

The tldw challenge pack uses a versioned annotation protocol. It defines
orthography, casing, punctuation, fillers, false starts, partial words,
unintelligible speech, non-speech events, numerals, abbreviations, proper
nouns, and reviewer/adjudication metadata. The protocol identifier is stored
as `annotation_profile`, and reference review status is retained in source
provenance without storing reviewer identities.

Overlapping speech has no unique linear word order. Until a deterministic
serialization or multi-reference scoring policy exists, overlap samples are
tagged and set to `diagnostic_only: true`. They appear in sample-level and
diagnostic slice reports but do not contribute to primary WER/CER, model
rankings, or regression gates.

Public reproducibility and private workload relevance are separate suites.
Primary pooled metrics are calculated per suite. The default report does not
combine public and private suites into one headline number; any later composite
must declare its suite weights explicitly.

## Deterministic Scoring

No LLM or external service participates in scoring.

### Raw text and exact match

The scorer receives the raw reference and hypothesis before any retention
policy is applied. Exact match replaces CRLF and bare CR with LF and makes no
other change. It intentionally remains sensitive to case, punctuation, and
whitespace. Raw text is persisted only according to the run's text-retention
mode.

### Strict profile

Strict WER and CER use this exact `strict-v1` preprocessing:

1. Apply Unicode NFC.
2. Replace every maximal run of characters for which Python
   `str.isspace()` is true with one ASCII space (`U+0020`).
3. Remove leading and trailing ASCII spaces.

WER returns an empty token sequence for an empty preprocessed string;
otherwise, it splits on `U+0020`. CER treats the result as a sequence of
Unicode code points, including the canonical internal spaces; it does not use
locale-dependent bytes or grapheme clusters. Case, punctuation, words,
fillers, symbols, and number forms otherwise remain unchanged.

### Normalized English v1 profile

The initial `en-v1` profile performs these transformations in order:

1. Unicode NFKC.
2. Map `U+2018`, `U+2019`, `U+02BC`, and `U+FF07` to ASCII apostrophe
   (`U+0027`).
3. Apply Unicode-aware `str.casefold()`.
4. Preserve `U+0027` when it occurs between two characters for which
   `str.isalnum()` is true; replace every other `U+0027` with `U+0020`.
5. Replace every remaining Unicode character whose general category starts
   with `P` (punctuation, including all dash punctuation) with `U+0020`, except
   for the preserved internal `U+0027`.
6. Replace every maximal `str.isspace()` run with one `U+0020`, then trim.

It deliberately does not:

- Remove filler words.
- Treat contractions and expansions as equivalent.
- Ignore singular/plural changes.
- Accept synonyms or paraphrases.
- Convert digits to words or words to digits.
- Ignore names, negations, dates, quantities, or units.

Those differences can change downstream meaning and should remain visible.
In particular, `we're` does not normalize to `were`, and `can't` does not
normalize to `cant`. Unicode NFKC may still perform compatibility mappings such
as full-width digits to ASCII digits; the profile performs no additional
number normalization.
Language-specific profiles can later replace `en-v1`; normalization must never
use an ASCII-only regular expression that destroys non-English text.

Normalized WER returns `[]` for an empty final string and otherwise splits on
`U+0020`. Normalized CER uses Unicode code points including canonical internal
spaces. The scorer implementation and every result record identify these rules
as `stt-score-v1`, with independent `strict-v1` and `en-v1` profile
identifiers.

### Edit counts and aggregates

The scorer returns substitutions, deletions, insertions, reference units, and
error rate for words and characters. Its dynamic-programming alignment uses a
deterministic operation priority when multiple minimum-cost paths exist:
match, substitution, deletion, then insertion.

For every target and suite, reports include:

- Per-suite pooled WER/CER: total edits divided by total reference units.
- Mean sample WER/CER.
- Sample p50, p90, p95, and p99.
- Exact-match rate.
- Successful-transcription, empty-output, and failure rates.
- Per-dataset and per-tag aggregates.

Every headline quality aggregate and quality gate is suite-specific, including
means, percentiles, exact-match rate, failure rate, and successful/empty-output
rates. Warm performance aggregates are also suite-specific; the single cold
probe remains a clearly labeled target-level observation. Optional cross-suite
views are informational and never gate.

All percentiles use linear interpolation over sorted values with the
zero-based index `h = (n - 1) * p`; the values at `floor(h)` and `ceil(h)` are
interpolated by the fractional part of `h`. A one-value population returns
that value. Reports never substitute a library's undocumented percentile
default.

Pooled normalized WER within each suite is the primary accuracy metric. Mean
and slice metrics remain visible so a large dataset or long samples cannot
hide a weak category. Diagnostic-only samples are excluded from primary
aggregates but always reported separately.

If a target fails or returns no usable hypothesis, the sample is recorded as a
failure and scored as an empty hypothesis for aggregate WER/CER. This prevents
selective failures from improving a target's accuracy. Invalid or empty
references are rejected during manifest validation instead of being scored.

## Timing and Resource Metrics

All elapsed timings use `time.perf_counter_ns()`.

### Required metrics

- Worker startup/import time.
- Registry and adapter setup time.
- `cold_first_transcription_seconds`.
- `warm_adapter_transcription_seconds` per sample.
- Total target wall time.
- Audio duration.
- Real-time factor:
  `processing_seconds / audio_duration_seconds` (lower is better).
- Throughput multiple:
  `audio_duration_seconds / processing_seconds` (higher is better).

The explicit RTF definition corrects an ambiguity in the existing mocked
performance test, which calls the inverse value "real-time factor."

The stable cold-probe sample participates in accuracy totals once but is
excluded from warm latency and RTF aggregates. A user-selected timing subset
may run multiple warm repetitions; the default accuracy run records one scored
transcription per sample.

These are end-to-end native adapter-call measurements. They include any audio
decode/resample work, provider preprocessing, model execution or HTTP request,
and adapter postprocessing performed inside `transcribe_batch()`. They are not
labeled pure model inference. One-repetition performance is descriptive.
Performance gates require at least three matching warm repetitions and report
the median and interquartile range using the scorer's documented percentile
interpolation. Cold-first results are also descriptive unless model-cache and
artifact-availability state are controlled and recorded.

### Optional best-effort metrics

- Process RSS before and after a transcription.
- Peak process RSS when supported.
- GPU/VRAM or unified-memory observations when a supported local tool exposes
  them without a new hard dependency.

Resource measurements are labeled best-effort and never silently compared
across incompatible operating systems or collection methods.

### Environment fingerprint

Each run records:

- tldw_server Git commit and dirty-worktree flag.
- Content fingerprints for the benchmark/scorer and the selected adapter and
  provider/loader source modules, so unrelated worktree edits do not invalidate
  resume.
- Manifest and target-configuration hashes.
- Scorer and normalization-profile versions.
- Python, `unicodedata.unidata_version`, operating system, architecture, and
  relevant package versions.
- CPU model, logical/physical core counts, and total RAM.
- GPU or Apple Silicon identity, driver/runtime, and visible memory when
  discoverable.
- Requested device and compute type.
- Deterministic sample-order seed and actual target execution order.
- Comparison mode and safe transcription settings including language hint,
  prompt/hotword presence and count, opaque `configuration_id` where required,
  and non-sensitive provider/model options.

Secrets are never serialized. API keys, bearer tokens, full environment
dumps, and sensitive config values are excluded. Prompt or hotword contents
and content-derived digests are excluded by default. This avoids leaking
low-entropy sensitive terms. `neutral-v1` prohibits prompts and hotwords;
`production-v1` uses the caller's opaque `configuration_id` for compatibility
checks without exposing those contents.

## Result Storage and Recovery

Runs are stored outside tracked source by default:

```text
.benchmarks/stt/<run-id>/
|-- run.json
|-- inflight.json
|-- results.jsonl
|-- summary.json
`-- summary.md
```

### `run.json`

Contains an explicit artifact `schema_version`, the immutable run identity,
compatible comparison fields, manifest hash, target matrix, environment
fingerprint, safe settings, the shared cold-probe ID, and worker-attempt
metadata. Each target includes its immutable `execution_contract_hash`.

Resume re-runs preflight and requires exact execution-contract equality. A
changed implementation, dependency set, target configuration, backend, model
artifact, device/compute contract, or egress classification refuses resume and
requires a new run directory. Run-level settings such as selected profile,
sample IDs, comparison mode, repetition policy, and text retention must also
match, as must the recorded hardware profile. Accuracy results from separate
hardware may still be compared later under the comparison policy, but they are
never pooled inside one resumed run.

### `inflight.json`

The coordinator atomically writes the active target, completion key, and
attempt ID after the worker announces a sample and before acknowledging that it
may enter `transcribe_batch()`. The file contains no transcript text. After a
terminal JSONL record is flushed and synced, the coordinator clears it.

If the worker or coordinator terminates abnormally, the coordinator or next
resume first checks whether that exact attempt already has a terminal JSONL
record. If so, it clears the stale in-flight state. Otherwise it appends one
`worker_crash`, watchdog `timeout`, or parent `interrupted` result for that
exact attempt. Failed keys are skipped on ordinary resume and receive one new
attempt only when the user supplies `--retry-errors`; there is no automatic
retry loop.

### `results.jsonl`

One append-only, schema-versioned record per sample attempt contains:

- Run and target IDs.
- Sample, repetition, and monotonically increasing attempt IDs.
- Requested and actual provider/model and execution identity.
- Status: `ok`, `empty`, `adapter_error`, `timeout`, `worker_crash`,
  `interrupted`, or `invalid_artifact`.
- Raw hypothesis and reference or their configured retained form.
- Strict and normalized edit counts and rates.
- Timing and optional resource measurements.
- Bounded, sanitized error type and message.

Each terminal JSONL record is flushed and `fsync`-ed before the next sample is
acknowledged. Persistence occurs outside the adapter timing window. This makes
the crash-recovery guarantee independent of Python file-buffer shutdown.

The completion key is derived from:

```text
manifest hash
+ provider
+ model
+ execution contract hash
+ sample ID
+ repetition
```

On resume, successful active keys are skipped. Failed keys remain visible and
are retried only with `--retry-errors`; a retry appends a higher attempt ID
rather than rewriting history. Reports reduce attempts deterministically by
selecting the highest attempt ID for each completion key, after rejecting
duplicate or non-monotonic attempt IDs. Only that active attempt contributes
to aggregates, while earlier attempts remain available as history. Summaries
are disposable views rebuilt from `run.json` and `results.jsonl`.

Every machine-readable artifact carries its own `schema_version`: `run.json`,
each `results.jsonl` record, and `summary.json`. `report` and `compare` reject
unsupported schema versions instead of guessing compatibility. `summary.md`
is a human-readable projection and does not require a separate schema.

Output transcripts can contain private or sensitive speech. The default output
directory remains untracked, is covered by a repository-tracked `.gitignore`
rule, and uses owner-only directory/file permissions where the platform
supports them (`0700` directories and `0600` files). The documentation warns
users to protect the artifacts, and the harness itself never uploads results.
When a selected native adapter sends audio to an external service, the
coordinator displays and records that egress classification and requires an
explicit `--allow-network-targets` flag before the worker starts. Unattended
runs cannot infer that consent from the presence of an API key.

The run records one text-retention mode:

- `full`: retain raw reference and hypothesis for every sample.
- `errors-only`: retain them only for any non-zero strict or normalized edit
  count, or non-`ok` status.
- `none`: retain neither; keep IDs, counts, timings, and bounded errors.

The default is `full` because benchmark diagnosis normally requires examples,
but the CLI warns when a selected suite has `suite_visibility: private`.
Retention changes inspection detail, not scoring, because scoring occurs before
the configured text is discarded. Generated JSON and Markdown summaries honor
the same retention mode and never reconstruct discarded text from the manifest.

## CLI

The first version exposes four subcommands:

```bash
# Validate references, files, checksums, and metadata.
python Helper_Scripts/benchmarks/stt_bench.py validate \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt

# Run or resume selected targets.
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt \
  --profile regression \
  --mode neutral-v1 \
  --text-retention full \
  --target faster-whisper=large-v3 \
  --target parakeet=parakeet-tdt-0.6b-v3-onnx

# Rebuild reports from incremental records.
python Helper_Scripts/benchmarks/stt_bench.py report \
  --run .benchmarks/stt/<run-id>

# Compare compatible runs or enforce a baseline.
python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline <baseline-summary.json> \
  --candidate <candidate-summary.json>
```

`--target` is repeatable and uses `provider=model`, avoiding ambiguity in model
IDs that themselves contain colons. A future target-config file may describe
larger matrices; v1 does not require one.

`run` performs execution-contract preflight for every selected target before
starting any worker. It reports unavailable local artifacts, unresolved
identity, unsupported language semantics, backend fallback, and egress
classification together so a matrix fails early. The harness exposes no model
download option.

The CLI uses the standard library `argparse` to match existing scripts under
`Helper_Scripts/benchmarks/`. It does not add a benchmark framework.

## Comparison and Regression Policy

Cross-target model/configuration comparisons require:

- Manifest content hash and selected sample IDs.
- Profile, suite membership, normalization/scorer versions, and Unicode data
  version.
- Comparison mode and common semantic settings.
- Repetition policy for performance metrics.

Provider/model identity is expected to differ in a cross-target comparison.
`neutral-v1` results are labeled model-quality comparisons;
`production-v1` results are labeled complete-configuration comparisons.

Same-target regression/baseline checks additionally require identical
provider/model identity, safe settings hash or opaque `configuration_id`, and
material adapter configuration. They also require resolved model-artifact
identity and matching effective backend and compute/dtype. Quality results
across different hardware or effective compute contracts may be compared
descriptively, with environment fingerprints visible, but are not eligible for
same-target regression gates. Performance thresholds are enforced only when
the candidate matches the baseline model artifact, effective backend and
compute/dtype, safe settings, hardware profile, collection method, target-order
policy, and a minimum of three warm repetitions. The implementation or
dependency version under test may differ and is shown explicitly; all other
material fields must match. Otherwise, performance differences are
informational.

Performance from network-backed targets is informational by default even when
the client hardware matches. Enabling a network-performance gate requires an
explicit opt-in plus a matching `network_collection_profile` that records a
non-secret endpoint/region identifier, client location/path label, concurrency,
and repetition policy. The report still labels the measurement as
network-dependent; it never presents remote-service latency as pure model
inference time.

Baselines define bounded, reviewable expectations rather than exact transcript
snapshots:

- Per-suite maximum absolute or relative normalized pooled WER/CER regression.
- Per-suite maximum failure-rate regression.
- Optional per-suite minimum exact-match rate.
- Eligible local hardware-matched, or explicitly profiled network-backed,
  per-suite maximum warm RTF or latency regression.

Strict scores remain diagnostic unless a specific dataset's formatting
contract makes them suitable for gating. Real-model regression runs are
opt-in, suitable for developer machines, GPU runners, dependency upgrades,
release candidates, and STT incident follow-up. Ordinary PR CI continues to
use deterministic fake adapters and small audio/manifest fixtures.

V1 point estimates do not establish statistical significance. Until paired
confidence intervals are implemented, all model rankings are labeled
descriptive and reports expose the paired per-sample deltas needed for later
analysis.

## Error Handling

- Manifest errors fail before any model loads and identify the sample and
  field.
- Execution-preflight errors fail before a target worker opens audio, loads a
  model, or makes a network request.
- Missing optional provider dependencies fail only that target.
- Adapter exceptions create a per-sample failure record and normally continue
  with later samples.
- Error sentinels and empty or malformed normalized artifacts are explicit
  failures.
- A hard worker crash preserves prior JSONL records and produces a
  sample-attributed crash result from the coordinator's persisted in-flight
  state, plus target-level attempt metadata.
- Interrupt handling stops scheduling new targets, asks the active worker to
  exit, and leaves a resumable run.
- Error messages are bounded and sanitized; secrets and complete environment
  values are never included.
- Report generation tolerates a truncated final JSONL line from abrupt process
  termination, reports it, and ignores only that incomplete line.

A worker-level watchdog can terminate a hung target. Per-sample hard timeouts
inside a reusable warm worker are not guaranteed in v1 because safely killing
an arbitrary synchronous GPU call would also destroy warmed model state. The
watchdog's `timeout` status applies to the persisted in-flight sample and ends
that worker attempt; it does not imply that the worker continued with warm
state. The result must describe this distinction rather than claim a timeout
guarantee it cannot provide.

## Integration with Existing STT Tests

The current opt-in golden test remains useful, with these changes:

- Import the benchmark's versioned scorer instead of maintaining a private
  ASCII normalizer and Levenshtein implementation.
- Use the regression manifest rather than provider-specific glob patterns.
- Resolve targets through `SttProviderRegistry` rather than hardcoding only
  faster-whisper, Parakeet, and Canary.
- Preserve adapter contract assertions such as artifact and segment shape.
- Treat generated adapter output as a behavior snapshot, not independent
  ground truth.
- Change `generate_stt_golden.py` to require a public/human reference or write
  an explicitly unverified candidate that cannot enter the scored manifest
  without review.

The existing mocked performance tests remain unit-level implementation checks.
Their artificial sleeps and mocked model results are not benchmark evidence
and must not be cited as real provider performance.

## Verification Strategy

### Unit tests

- Known WER/CER alignments for substitutions, insertions, and deletions.
- Empty reference/hypothesis behavior.
- Empty preprocessed strings produce empty token/code-point sequences.
- Strict versus normalized scoring.
- Exact `strict-v1` and `en-v1` Unicode mapping, category, whitespace, number,
  and non-English preservation cases.
- Internal-apostrophe cases prove that `we're` differs from `were` and `can't`
  differs from `cant`.
- Deterministic alignment tie-breaking and type-7-style percentile
  interpolation.
- Per-suite and diagnostic-only aggregate calculations and attempt reduction.
- Environment and actual-execution metadata allowlisting, including hostile
  adapter metadata containing prompt, hotword, token, header, and URL values.

### Property-based tests

- Identity produces zero edits.
- Normalization is idempotent.
- Repeated scoring is deterministic.
- Edit counts are non-negative and internally consistent.
- Pooled rates reconstruct from stored counts.

### Manifest tests

- Duplicate IDs.
- Missing or empty references.
- Invalid profiles and language tags.
- Unknown suites, inconsistent suite visibility, annotation profiles, and
  normalization profiles.
- Path traversal and symlink escape.
- Missing files and checksum mismatches.
- Measured/declared duration mismatch and malformed source metadata.

### Runner tests

Use fake native adapters to cover:

- Cold versus warm classification.
- Shared cold probe, probe-failure recovery, and resume warm-up without
  changing sample classification.
- Requested/actual identity validation.
- Bound execution plans for dynamic local/network backends, semantic settings,
  local model availability, no-download enforcement, and neutral-mode fallback
  rejection.
- A configuration change after planning cannot change the executed backend,
  endpoint, fallback policy, or semantic settings.
- Network plans reject redirects to a different endpoint.
- Resume refusal after an execution-contract change.
- Incremental persistence, in-flight recovery, and durable resume.
- Exceptions, recognized error sentinels, empty transcripts, invalid
  artifacts, and sample-attributed worker crashes.
- Retry attempt ordering and latest-attempt aggregation.
- Comparison-mode enforcement and external-target consent.
- Artifact schema rejection and local/network performance-gate eligibility.
- Deterministic sample order.
- Text-retention modes.
- Summary regeneration after partial completion.

### Opt-in real-model tests

The existing `stt_golden` profile exercises installed real adapters on the
small regression subset. These runs stay outside ordinary CI and are selected
explicitly by the developer or release process.

### Security and repository gates

- Run focused pytest tests for the scorer, manifest, runner, and report logic.
- Run `git diff --check`.
- Run Bandit on touched Python files before implementation completion.
- Do not claim real accuracy or performance without attaching the compatible
  manifest, run metadata, and raw result artifact.

## Initial Repository Shape

The implementation should begin with:

```text
Helper_Scripts/benchmarks/stt_bench.py
Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl
tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py
tldw_Server_API/tests/Benchmarks/test_stt_bench.py
Docs/Development/STT_Benchmark_Protocol.md
.gitignore
```

The first implementation should favor one importable CLI module and pure
functions over a new package hierarchy. Split modules only when the scorer,
worker, or reporting code becomes difficult to test independently.
The existing STT adapter module defines the immutable execution-plan contract
and error-sentinel enforcement at the shared native boundary. Affected native
provider/loader modules receive the minimum changes needed to consume that
plan, enforce backend/fallback/no-download choices, and return the allowlisted
actual-execution envelope. A provider remains unsupported by the benchmark
until those controls are enforceable. The benchmark does not build a parallel
provider registry or duplicate provider implementations.

Large public or private audio corpora are not committed. The repository may
include a tiny redistributable test fixture if one already exists or can be
added with explicit provenance and license.

## Deferred Enhancements

- Paired bootstrap confidence intervals for compatible comparison runs.
- A target-matrix configuration file.
- Provider lifecycle hooks that separate model preparation from first
  inference.
- Dataset import helpers for LibriSpeech, Common Voice, and FLEURS.
- Long-form boundary and timestamp metrics.
- Diarization error rate and speaker-attributed WER.
- Evaluations API/WebUI import of completed benchmark artifacts.
- Machine-readable export suitable for publishing a leaderboard.

These enhancements must build on the same manifest, result, and scorer
contracts rather than creating a second benchmark format.

## Acceptance Criteria

The eventual implementation is complete when:

1. A user can validate a hybrid English manifest and run at least two native
   provider/model targets without Pipecat or an LLM.
2. Each sample produces independently reproducible strict and normalized
   WER/CER counts.
3. Interrupted runs resume without duplicating completed scored keys; only the
   declared cold probe may be replayed without scoring to restore warm state.
4. Cold first transcription and warm RTF/throughput are reported with the
   documented definitions.
5. Reports include failures and dataset/tag slices rather than only successful
   global averages, and keep public/private suites and diagnostic-only samples
   distinct.
6. Invalid provider targets fail closed without faster-whisper fallback.
7. Compatible runs can be compared and hardware-mismatched performance gates
   are rejected.
8. Existing golden tests reuse the benchmark scoring contract.
9. Dynamic remote backends require preflight egress consent and execute the
   exact approved plan; neutral-mode targets cannot fall back across backends,
   and measured local targets cannot download models.
10. Recognized error sentinels are failures, and hard crashes or watchdog
    termination are attributed to the persisted in-flight sample.
11. A run cannot resume across an execution-contract change, and unresolved
    model/backend/compute identity is ineligible for regression gates; only
    allowlisted actual-execution fields are stored.
12. The English normalized profile preserves meaning-bearing internal
    apostrophes, records its Unicode data version, and follows a versioned
    challenge-corpus annotation policy.

## References

- [pipecat-ai/stt-benchmark](https://github.com/pipecat-ai/stt-benchmark/)
- [Pipecat benchmark service registry](https://github.com/pipecat-ai/stt-benchmark/blob/main/src/stt_benchmark/services.py)
- [Pipecat benchmark runner](https://github.com/pipecat-ai/stt-benchmark/blob/main/src/stt_benchmark/pipeline/benchmark_runner.py)
- [Pipecat semantic WER evaluator](https://github.com/pipecat-ai/stt-benchmark/blob/main/src/stt_benchmark/evaluation/semantic_wer.py)
- [LibriSpeech on OpenSLR](https://us.openslr.org/12/)
- [Mozilla Common Voice datasets](https://commonvoice.mozilla.org/en/datasets)
- [Mozilla Common Voice terms](https://commonvoice.mozilla.org/terms)
- [Google FLEURS publication](https://research.google/pubs/fleurs-few-shot-learning-evaluation-of-universal-representations-of-speech/)
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/stt_provider_adapter.py`
- `tldw_Server_API/tests/Audio/test_stt_adapters_golden.py`
- `Docs/Development/STT_Adapter_Golden_Tests.md`
- `tldw_Server_API/tests/Media_Ingestion_Modification/test_transcription_benchmarks.py`
