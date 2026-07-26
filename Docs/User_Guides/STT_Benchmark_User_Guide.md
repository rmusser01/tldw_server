# Native Batch STT Benchmark User Guide

Use this guide to compare tldw_server batch speech-to-text targets with
reproducible corpus, scoring, timing, and environment records. The shortest
safe path is a local `neutral-v1` run. Network targets and production
configuration comparisons are advanced opt-ins.

The
[native STT benchmark protocol](https://github.com/rmusser01/tldw_server/blob/dev/Docs/Development/STT_Benchmark_Protocol.md)
is authoritative for schemas, scoring, timing, privacy, and comparison
eligibility. This guide turns that protocol into an operator workflow.

## When to use this benchmark

Use it when you need to:

- compare supported batch STT models on the same authorized audio;
- measure strict and normalized WER/CER without subjective grading;
- record cold-first and warm adapter timings separately;
- detect a same-target regression after changing code, dependencies, models,
  or configuration;
- preserve enough metadata for another operator to assess the result.

The harness is batch-only. It calls tldw_server's native
`SttProviderRegistry` and `SttProviderAdapter` contract directly. It does not
use Pipecat, an LLM judge, FastAPI, the Evaluations service, or a running
tldw_server process. It does not download models or corpora.

## What the scores and timings mean

Every selected sample is scored deterministically before transcript retention
is applied:

- **Strict WER/CER** preserves case and punctuation after minimal Unicode and
  whitespace normalization.
- **Normalized WER/CER** applies the manifest's versioned normalization
  profile. English uses `en-v1`; this is deterministic text normalization, not
  semantic equivalence.
- **Exact match** changes only line-ending representation.
- **Failures and empty outputs** are scored as empty hypotheses. A provider
  cannot improve accuracy by failing on difficult samples.

Headline quality is per-suite pooled normalized WER. Public and private suites
are never silently pooled.

Timing is also split deliberately:

- **Cold-first transcription** is the native adapter-call duration for one
  deterministic probe in a fresh target process. It may include lazy loading,
  decode/resample, preprocessing, inference or HTTP, and postprocessing. It is
  not pure model-load time.
- **Warm transcription** is a later successful adapter call after the worker
  has established warm state.
- **RTF** is processing time divided by audio duration. Lower is better.
- **Throughput multiple** is audio duration divided by processing time. Higher
  is better.

Worker startup/import time is recorded separately. One warm repetition is
descriptive; performance gates require `--warm-repetitions 3` or greater and
at least three eligible warm observations in the gated suite.

## Prerequisites

Run commands from the repository root.

1. Activate the project virtual environment:

   ```bash
   source .venv/bin/activate
   ```

2. Install the tldw_server dependencies needed by the target you intend to
   test.
3. Install FFmpeg, including `ffprobe`.
4. Obtain an authorized corpus and independently verified references.
5. Preinstall every local model artifact. For a network target, provision and
   verify the separately managed provider before running the benchmark.
   Preflight rejects missing dependencies, missing local artifacts, and plans
   that would download weights.

Use the existing setup guides to prepare a local target:

- [CPU audio setup](../Getting_Started/First_Time_Audio_Setup_CPU.md)
- [GPU and Apple Silicon audio setup](../Getting_Started/First_Time_Audio_Setup_GPU_Accelerated.md)

### Choose an installed target

`--target` has the exact form `provider=model`. The provider must be registered,
and the model label is interpreted by that adapter. The benchmark CLI has no
target-discovery or non-executing dry-run command.

Choose a target that you have already installed and verified. For example,
depending on your hardware and setup:

| Hardware path | Illustrative target |
| --- | --- |
| CPU Parakeet ONNX | `parakeet=parakeet-tdt-0.6b-v3-onnx` |
| NVIDIA faster-whisper | `faster-whisper=large-v3` |
| Apple Silicon local CPU fallback | `parakeet=parakeet-tdt-0.6b-v3-onnx` |

These are examples, not an inventory of what is installed on your machine.
Use the exact provider and model/artifact label from your verified setup.
Unknown providers fail closed rather than falling back to another adapter.
Although `parakeet-mlx` is supported by other tldw_server STT paths, the
benchmark planner currently rejects it because it cannot prove immutable
device and dtype identity. Do not use it as a benchmark target until that
planner support exists.

Start with one target:

```bash
export STT_TARGET='parakeet=parakeet-tdt-0.6b-v3-onnx'
```

Replace that value if your installed target differs. Add another
`--target provider=model` only after the first target passes preflight.
Targets run sequentially in CLI order so they do not compete for model cache
or compute resources.

## Optional: user-managed audio.cpp server

The dedicated `audio-cpp` provider connects to a separately built, configured,
and operated `audiocpp_server`. tldw_server does not download, build, start,
restart, stop, or configure that process or its models. Follow the upstream
[audio.cpp server README](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md)
to build and start the server with the exact ASR model you intend to test; this
guide does not replace upstream build or server CLI instructions.

Before configuring tldw_server, verify the server manually:

- `GET /health` must succeed;
- `GET /v1/models` must list the exact intended model ID with `task=asr`.

Configure the provider under `[STT-Settings]` in
`tldw_Server_API/Config_Files/config.txt`:

```ini
audio_cpp_enabled = true
audio_cpp_base_url = http://127.0.0.1:8080
audio_cpp_default_model = REPLACE_WITH_EXACT_ASR_MODEL_ID
audio_cpp_timeout_seconds = 600
```

The corresponding environment overrides are
`STT_AUDIO_CPP_ENABLED`, `STT_AUDIO_CPP_BASE_URL`,
`STT_AUDIO_CPP_DEFAULT_MODEL`, and `STT_AUDIO_CPP_TIMEOUT_SECONDS`.
Environment values take precedence over `config.txt`. The base URL must be an
HTTP(S) origin; tldw_server derives `GET /health`, `GET /v1/models`, and
`POST /v1/audio/transcriptions` from it.

Ordinary OpenAI-compatible transcription requests use
`audio-cpp:<model>`. The aliases `audiocpp:<model>` and
`audio_cpp:<model>` are accepted. Exact selectors `audio-cpp`, `audiocpp`,
and `audio_cpp` use the configured default model. That default applies only to
ordinary requests: benchmark targets must always include the exact model:

```bash
export STT_AUDIO_CPP_TARGET='audio-cpp=REPLACE_WITH_EXACT_ASR_MODEL_ID'
```

Benchmark planning rejects selector tokens and selector-prefixed values in the
model portion. For example, do not use `audio-cpp=audio-cpp:<model>`:
`audio-cpp:<model>` is ordinary API selector syntax, not an exact server model
ID.

After setting `STT_MANIFEST` and `STT_DATASET_ROOT` as described in the next
section, run:

```bash
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT" \
  --profile regression \
  --mode neutral-v1 \
  --text-retention errors-only \
  --target "$STT_AUDIO_CPP_TARGET" \
  --allow-network-targets \
  --run 'audio-cpp-regression-v1'
```

`audio-cpp=<model>` supports `neutral-v1`; it does not initially support
`production-v1`. The network-consent flag is required even for a loopback
server. Supplying it authorizes audio transmission as soon as preflight
succeeds, so inspect the configured endpoint first.

Privacy warning: the run sends corpus audio to the configured server, and
`--text-retention errors-only` still retains transcript text for scored errors
or failed samples. Use `--text-retention none` when that text must not be
stored, and protect the run directory according to the corpus sensitivity.

V1 accepts only a regular `.wav` file that Python can validate as an
uncompressed PCM RIFF/WAVE container. Validation happens before network I/O.
The provider does not convert other formats, follow redirects, retry failed
transcriptions, fall back to another provider, or download models. It has no
authentication setting or TLS-verification-disable knob; deployments that
need authentication must use a suitable reverse proxy or the generic external
provider integration.

Persisted benchmark artifacts include the requested and resolved model labels,
the generic `audio_cpp_http` route/egress classification, and an opaque
endpoint ID. The discovered server backend, model family, and model mode remain
request-local normalized artifact metadata; they are not retained in benchmark
results. Model/artifact identity remains descriptive and unresolved, so
audio.cpp results are gate-ineligible and should be treated as descriptive
measurements.

Cold-first timing includes adapter discovery (`/health` and `/v1/models`), the
transcription request, and any server-side lazy loading. tldw_server cannot
reset the independently managed server: for a true server cold-start
measurement, restart `audiocpp_server` immediately before the run and leave
its lazy loading enabled. Warm calls reuse tldw_server's discovery cache and
the server's already-loaded model/session.

## Prepare the corpus and manifest

Keep corpus audio outside the tracked repository. A hybrid public/private
layout can look like:

```text
/data/stt-benchmark/
├── manifest.jsonl
├── corpus-notes/
│   ├── public-sources.md
│   └── private-authorization.md
├── public/
│   └── librispeech/
└── private/
    └── challenge/
```

Set reusable paths:

```bash
export STT_DATASET_ROOT='/data/stt-benchmark'
export STT_MANIFEST="$STT_DATASET_ROOT/manifest.jsonl"
```

Copy the checked-in manifest to use as schema documentation:

```bash
cp Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl \
  "$STT_MANIFEST"
```

The
[example manifest](https://github.com/rmusser01/tldw_server/blob/dev/Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl)
does not contain valid corpus metadata or audio. Replace or independently
verify every value, including IDs, references, language, profiles, suite,
visibility, annotation profile, diagnostic status, tags, paths, source
metadata, durations, and the all-zero checksums.

For each JSONL row:

- use an audio path relative to `STT_DATASET_ROOT`;
- use a canonical dataset reference or an independently reviewed human
  transcript, never model-generated text as ground truth;
- record the exact dataset release, split, license, and reference provenance;
- pin the SHA-256 of the individual audio file, not only its archive;
- keep `suite_visibility` consistent within a suite;
- put public reproducibility data and private workload data in different
  suites;
- mark ambiguous linear references such as overlapping speech
  `diagnostic_only: true`.

### Compute the checksum

Linux:

```bash
sha256sum "$STT_DATASET_ROOT/public/librispeech/example.flac"
```

macOS:

```bash
shasum -a 256 "$STT_DATASET_ROOT/public/librispeech/example.flac"
```

Windows PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 `
  "$env:STT_DATASET_ROOT\public\librispeech\example.flac"
```

Copy the 64-character hexadecimal digest into `source.sha256`.

### Measure the duration

```bash
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 \
  "$STT_DATASET_ROOT/public/librispeech/example.flac"
```

Copy the measured seconds into `duration_seconds`, or omit that optional field.
Validation measures duration independently and rejects disagreement greater
than 100 ms or one percent, whichever is larger.

### English first, multilingual ready

Use `language: "en"` with `normalization_profile: "en-v1"` for the first
English suite. A v1 run selects exactly one canonicalized language tag, so do
not mix `en` and `en-US` in the same selected profile.

Additional languages use the same manifest and runner format. Put each
language in an appropriate separately reported suite, set its per-sample BCP
47 language tag, and use `strict-v1` until a language-specific versioned
normalization profile exists. Do not present a pooled cross-language number as
though it were a single comparable headline metric.

## Validate before loading a model

```bash
python Helper_Scripts/benchmarks/stt_bench.py validate \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT"
```

Validation loads no provider model. It verifies schema, paths, checksums,
durations, suite consistency, and reference-provenance claims, then prints the
sample count and portable manifest hash. Validation cannot prove that a
license, authorization, or human-review claim is truthful; the corpus
maintainer remains responsible for those claims.

Do not continue until validation succeeds.

## Run a local regression benchmark

Choose a stable run identifier:

```bash
export STT_RUN_ID='english-local-regression-v1'
```

Then run:

```bash
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT" \
  --profile regression \
  --mode neutral-v1 \
  --text-retention errors-only \
  --warm-repetitions 3 \
  --target "$STT_TARGET" \
  --run "$STT_RUN_ID"
```

`neutral-v1` fixes common transcription semantics and prohibits backend
fallback. It is the only v1 mode eligible for model-quality ranking.

`run --run "$STT_RUN_ID"` treats the value as an identifier and creates:

```text
.benchmarks/stt/english-local-regression-v1/
```

If you omit `--run`, the CLI creates and prints a collision-resistant run ID.
Save that ID if you may need to resume.

To compare multiple installed models in one environment, repeat `--target`:

```bash
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT" \
  --profile comparison \
  --mode neutral-v1 \
  --text-retention errors-only \
  --warm-repetitions 3 \
  --target 'parakeet=parakeet-tdt-0.6b-v3-onnx' \
  --target 'faster-whisper=large-v3' \
  --run 'english-local-model-comparison-v1'
```

Only use targets that are actually installed on the same machine. Target order
is recorded and later matters for performance comparison eligibility.

## Inspect and rebuild reports

A completed run retains:

```text
.benchmarks/stt/<run-id>/
├── .coordinator.lock
├── run.json
└── results.jsonl
```

`run.json` and append-only `results.jsonl` are authoritative.
`inflight.json` exists only while a call is active or when crash recovery is
needed; it is cleared after successful persistence. The `report` command
creates or refreshes the disposable `summary.json` and `summary.md`
projections.

Rebuild a report by passing the run directory, not the run ID:

```bash
export STT_RUN_DIR=".benchmarks/stt/$STT_RUN_ID"

python Helper_Scripts/benchmarks/stt_bench.py report \
  --run "$STT_RUN_DIR"
```

Inspect:

- per-suite pooled normalized and strict WER/CER;
- exact-match, empty-output, success, and failure rates;
- cold-first duration separately from warm median and interquartile range;
- RTF and throughput;
- diagnostic-only samples and dataset/tag slices;
- planned versus actual execution identity and eligibility warnings.

Do not choose a model from one timing observation or from a single pooled
number that hides suite and failure-rate differences.

## Compare two compatible runs

`compare` takes `summary.json` paths:

```bash
export STT_BASELINE='.benchmarks/stt/english-baseline-v1/summary.json'
export STT_CANDIDATE='.benchmarks/stt/english-candidate-v1/summary.json'

python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline "$STT_BASELINE" \
  --candidate "$STT_CANDIDATE"
```

Without `--policy`, comparison is descriptive. It can compare different
targets when the remaining quality identities are compatible. It still
rejects partial summaries.

Before comparing, confirm that both runs have:

- the same manifest hash and selected sample IDs;
- the same profile, suites, and sample language;
- matching scorer, normalization, sample-profile, and Unicode identities;
- the same mode, seed, repetition policy, and common semantic settings;
- the same target-matrix size;
- corresponding targets in the same CLI order;
- complete, successfully regenerated summaries.

For same-target policy gates, also require compatible provider/model,
resolved artifact, backend, compute/dtype, safe settings or production
configuration identity, hardware, and collection methods. Performance gates
also require matching target order and enough eligible warm observations.

Implementation and dependency changes may be the subject of a regression
comparison and remain visible in the result.

## Add an optional same-target regression policy

Use a policy only when the baseline and candidate represent eligible runs of
the same target. For example, save this as
`/data/stt-benchmark/release-policy.json`:

```json
{
  "schema_version": 1,
  "suites": {
    "public-english-v1": {
      "max_normalized_pooled_wer_absolute_regression": 0.01,
      "max_failure_rate_absolute_regression": 0.0
    }
  }
}
```

Here `0.01` allows at most a one-percentage-point absolute increase in pooled
normalized WER. Choose thresholds from your release policy, not from this
illustration.

```bash
python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline "$STT_BASELINE" \
  --candidate "$STT_CANDIDATE" \
  --policy '/data/stt-benchmark/release-policy.json'
```

Exit codes are suitable for automation:

- `0`: comparison completed and all requested eligible gates passed;
- `1`: at least one eligible regression gate failed;
- `2`: invalid or incompatible input.

Do not treat exit code `2` as a quality regression. Fix the compatibility or
artifact problem first. Policy gates require compatible complete runs.

## Resume, retry, and retention

Rerun the identical `run` command with the same `--run` identifier to resume.
Completed keys are skipped. Resume requires exact run and execution-contract
equality; changes to the manifest, selection, mode, retention, implementation,
dependencies, model/backend, hardware, egress, or material settings require a
new run.

Failed keys remain visible and are skipped by default. Add `--retry-errors` to
the otherwise identical command to append a new attempt:

```bash
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT" \
  --profile regression \
  --mode neutral-v1 \
  --text-retention errors-only \
  --warm-repetitions 3 \
  --target "$STT_TARGET" \
  --run "$STT_RUN_ID" \
  --retry-errors
```

Retries do not rewrite history. Reports use the highest attempt for each
completion key while retaining earlier attempts in `results.jsonl`.

Use `--worker-watchdog-seconds N` when an adapter may hang. Only one
coordinator may operate on a run at a time; an overlapping invocation is
rejected before it changes attempt state or sends audio.

### Choose retention deliberately

| Mode | Retained transcript text |
| --- | --- |
| `full` | Every reference and hypothesis |
| `errors-only` | Text for non-zero edits or non-`ok` status |
| `none` | No reference or hypothesis; IDs, scores, timings, metadata, and bounded error context remain |

`full` is the default. The examples choose `errors-only` explicitly, but that
mode may still retain substantial private transcript text. Scoring occurs
before retention, and reports never reconstruct discarded text.

`.benchmarks/stt/` is ignored by Git, but Git-ignore is not access control.
Artifacts may contain private speech references, transcripts, source metadata,
and provider error context. Protect or encrypt them, restrict backups, and
expire or dispose of them according to the corpus authorization and retention
policy.

## Advanced: network and production-mode targets

Stop before adding `--allow-network-targets` and inspect:

1. the adapter's resolved endpoint;
2. whether it is literal loopback or remote;
3. provider privacy, training-use, and retention terms;
4. which audio is authorized to leave the local process;
5. the non-secret collection labels you will use.

An API key is authentication, not consent. Both loopback HTTP targets and
remote targets require `--allow-network-targets`. V1 has no safe preview:
successful preflight proceeds directly to sending audio. Redirects are
disabled, but you must still verify the configured endpoint outside the
benchmark.

`production-v1` preserves configured production behavior and requires an
opaque `--configuration-id`. It compares complete configurations, not isolated
model quality. A network run template is:

```bash
export STT_NETWORK_TARGET='external=external:REPLACE_WITH_CONFIGURED_PROVIDER'

python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT" \
  --profile regression \
  --mode production-v1 \
  --configuration-id 'stt-production-config-v1' \
  --text-retention errors-only \
  --warm-repetitions 3 \
  --network-collection-profile 'controlled-wired-network-v1' \
  --network-client-location 'replace-with-non-secret-location-label' \
  --target "$STT_NETWORK_TARGET" \
  --allow-network-targets \
  --run 'english-production-network-v1'
```

Replace every illustrative label after reviewing the actual adapter
configuration. Never put credentials in a run identifier, configuration ID,
network label, manifest, or command history. For the `external` adapter, the
model portion `external:<provider>` selects that named external-provider
configuration; the actual request model comes from the selected configuration.
The adapter does not resolve a concrete model artifact, so its results remain
descriptive and are not eligible for policy gates. Use a network-capable
native adapter with a resolved execution identity when you need an eligible
network regression gate.

Network performance is informational by default. A network performance policy
gate additionally requires matching non-empty network collection and client
location labels in both runs plus explicit comparison consent:

```bash
python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline "$STT_BASELINE" \
  --candidate "$STT_CANDIDATE" \
  --policy '/data/stt-benchmark/release-policy.json' \
  --allow-network-performance-gates
```

The result remains network-dependent. Do not present it as intrinsic model
latency.

## Troubleshooting

### Target preflight fails

- Confirm the target uses `provider=model`.
- Confirm the provider name is registered and the model label is accepted by
  that adapter.
- Confirm optional dependencies and model artifacts are already installed.
- Confirm the plan can run without downloading weights or changing backend.
- Return to the CPU or accelerated audio setup guide and verify the target
  outside the benchmark.

Unknown providers fail closed. Do not rename a target merely to reach a
fallback.

### Manifest validation fails

- Recompute the per-file SHA-256.
- Recheck the path relative to `STT_DATASET_ROOT`.
- Re-run `ffprobe` and correct or remove `duration_seconds`.
- Check for duplicate IDs, blank JSONL lines, inconsistent suite visibility,
  and placeholder source metadata.
- Confirm references are canonical or independently human-reviewed.

### Resume is rejected

Use the exact original arguments and environment. If anything material
changed, create a new run ID instead of mutating the old run.

### Compare returns exit code 2

Regenerate both reports, then work through the compatibility checklist.
Confirm that you passed `summary.json` files rather than run directories and
that targets pair in the same CLI order.

### A report is partial

Resume the run or explicitly account for its failed work. Partial summaries
are not accepted by descriptive comparison and are not eligible for policy
gates.

### Timing is unstable

- Use at least three warm repetitions.
- Keep target order identical between performance-gated runs.
- Avoid competing workloads and power-mode changes.
- Compare matching hardware and collection conditions.
- Report medians and interquartile ranges rather than one observation.

## Publication checklist

Before publishing or attaching results:

- [ ] Verify that every audio file and reference was authorized for this use.
- [ ] Record corpus release, split, license, provenance, and selection method.
- [ ] Include the exact manifest or portable manifest hash.
- [ ] Identify the tldw_server revision and disclose a dirty worktree.
- [ ] Identify concrete model/artifact, backend, compute settings, and target
      order.
- [ ] Include hardware and, when applicable, network collection context.
- [ ] State `neutral-v1` versus `production-v1`.
- [ ] Report public and private suites separately.
- [ ] Include failure/empty rates with WER/CER and timing.
- [ ] Include the policy used for any pass/fail claim.
- [ ] Confirm the comparison was complete and eligible.
- [ ] Remove or protect retained private text and error context before sharing.

Do not claim a universally best model from these point estimates. Never
publish a quality or performance claim based only on a copied Markdown table,
an unspecified provider/model, an unreviewed model-generated reference, or a
run that omitted its compatibility metadata.
