# Native Batch STT Benchmark Protocol

- **Protocol version:** v1
- **Scorer:** `stt-score-v1`
- **Scope:** batch transcription through tldw_server's native STT adapters

## Trust Boundary

This benchmark is deliberately small and deterministic:

- It does not use Pipecat.
- It does not use an LLM judge or semantic grading.
- It does not generate ground truth with an STT or language model.
- It does not download models or corpora.
- It does not call FastAPI, Jobs, or the Evaluations service.
- It calls `SttProviderAdapter.transcribe_batch()` through the native
  `SttProviderRegistry`.

References must come from a canonical dataset or an independently reviewed
human transcript. Model-generated text may be retained as an unverified
candidate, but it is not benchmark ground truth.

The implementation is
[`Helper_Scripts/benchmarks/stt_bench.py`](../../Helper_Scripts/benchmarks/stt_bench.py).
The checked-in
[`stt_benchmark_manifest.example.jsonl`](../../Helper_Scripts/benchmarks/stt_benchmark_manifest.example.jsonl)
is executable schema documentation, not a starter corpus. Every value in every
record is illustrative—including the ID, audio path, reference, language,
profiles, suite, annotation profile, tags, duration, source metadata, and
all-zero checksum. Preserve the schema shape, but replace or independently
verify every value before use.

## Manifest and Corpus

The input is a UTF-8 JSONL manifest plus an explicit dataset root. Each
non-blank line is one object with:

| Field | Meaning |
| --- | --- |
| `id` | Stable, unique lowercase identifier |
| `audio` | POSIX path relative to the dataset root |
| `reference` | Non-empty independently verified transcript |
| `language` | `bcp47-basic-v1` language tag |
| `normalization_profile` | `strict-v1` or `en-v1` |
| `duration_seconds` | Optional declared duration; checked against measured duration |
| `profiles` | One or both of `regression` and `comparison` |
| `suite` | Stable suite identifier |
| `suite_visibility` | `public` or `private`, consistent within a suite |
| `annotation_profile` | Versioned reference-annotation rules |
| `diagnostic_only` | Exclude from primary aggregates and gates when `true` |
| `source` | Dataset, version, license, reference provenance, SHA-256, and optional source metadata |
| `tags` | Bounded dataset slices such as `noisy` or `technical` |

Validation rejects duplicate IDs and JSON fields, unknown top-level fields,
blank lines, unsafe paths, symlink escapes, missing or non-regular audio,
checksum changes, invalid metadata, and duration disagreement greater than the
larger of 100 ms or one percent. Audio duration is measured once during
manifest validation, outside all provider timing windows.

Validation permits exactly `canonical-dataset` or `human-reviewed` as
`reference_provenance`, but it cannot prove that the claim is true. Corpus
maintainers must audit it independently. Model-generated candidates stay
outside the scored manifest. The golden-reference helper applies the same rule
when it emits a manifest row.

`bcp47-basic-v1` is intentionally a syntax rule, not an IANA registry lookup.
It accepts a 2-8 ASCII-letter primary subtag followed by zero or more 1-8
ASCII-alphanumeric subtags, separated by hyphens. Tags are compared in
lowercase. Provider language support is checked separately during execution
preflight.

V1 runs select exactly one canonicalized language tag, not merely one primary
subtag: do not mix `en` and `en-US` in the same selected profile. English
records use `en-v1`. Future multilingual records can use `strict-v1` until a
language-specific, versioned normalization profile is implemented; they do not
require a new manifest or runner format.

### Profiles, suites, and diagnostic records

- `regression` is a stable stratified subset, normally 40-100 samples, for
  regular local checks and releases.
- `comparison` is the larger selection used for model choice, slice analysis,
  and descriptive rankings.
- A record may belong to both profiles so the smaller set cannot drift into a
  separate fixture format.
- Public reproducibility and private workload relevance use distinct suites.
  Headline metrics are always per suite; public and private data are never
  silently pooled.
- Samples with ambiguous linear references, especially overlapping speech,
  must set `diagnostic_only: true`. They remain visible in sample and
  diagnostic-slice reports but do not affect primary WER/CER, rankings, or
  regression gates.

### Manual dataset acquisition

The repository does not redistribute benchmark audio. Keep corpora outside
tracked source and record the exact source release and license you obtained.

For LibriSpeech:

1. Manually obtain the `test-clean` and `test-other` archives from the official
   OpenSLR LibriSpeech release.
2. Preserve the upstream directory structure and canonical `.trans.txt`
   references under the chosen dataset root.
3. Record `dataset: librispeech`, the exact release identifier, split, and
   upstream license on every row.
4. Select sample IDs deterministically and commit or archive that selection
   list with the private/public corpus metadata.
5. Compute and pin the SHA-256 of every selected audio file. Do not use an
   archive checksum as the per-sample checksum.

For Common Voice:

1. Manually obtain the desired English release from Mozilla Data Collective
   after accepting its current terms.
2. Record the exact downloaded release, language configuration, split, and
   license from that release's own metadata; do not copy the illustrative
   placeholders from the example manifest.
3. Use the provided canonical sentence as the reference and retain stable
   source clip identity in optional `source` fields.
4. Pin every selected clip's SHA-256. Do not mirror the download from this
   repository or a tldw_server-owned host.

For a private tldw challenge pack, use only audio that is user-owned,
redistributable, or authorized for this use. Store it in a private suite and
review the privacy implications before choosing text retention.

Useful local checks are:

```bash
shasum -a 256 /data/stt/path/to/clip.wav
ffprobe -v error -show_entries format=duration \
  -of default=noprint_wrappers=1:nokey=1 /data/stt/path/to/clip.wav
```

The benchmark's `validate` command is authoritative; hand-entered duration
metadata is never used to calculate performance.

## Challenge-Pack Annotation Profile

Use `annotation_profile: tldw-challenge-en-v1` only when the following rules
were applied:

1. **Orthography:** transcribe audible lexical content verbatim using standard
   English spelling. Do not repair grammar, summarize, or paraphrase.
2. **Casing:** use normal sentence casing. Use verified conventional casing
   for proper nouns, acronyms, product names, and technical identifiers.
3. **Punctuation:** add conservative punctuation needed for readable sentence
   boundaries. Do not encode uncertain prosody as punctuation.
4. **Fillers:** retain audible lexical fillers using the fixed forms `uh`,
   `um`, `er`, `hmm`, `mm-hmm`, and `uh-huh`. Do not remove them during review.
5. **False starts and repetitions:** retain every audible completed word in
   spoken order. Do not rewrite a correction; for example, transcribe
   `Tuesday I mean Thursday`.
6. **Partial words:** retain an intelligible fragment with one trailing ASCII
   hyphen, such as `trans-`. If the fragment is not intelligible, use the
   unintelligible rule instead.
7. **Unintelligible speech:** use exactly `[unintelligible]` for a bounded
   unintelligible span. Tag the sample `unintelligible` and make it
   `diagnostic_only`; do not guess from context.
8. **Non-speech events:** omit background events from the scored reference.
   Represent a material event in tags such as `music`, `noise`, or `laughter`.
   A diagnostic experiment that intentionally expects event labels must use a
   separate versioned annotation profile.
9. **Numerals:** write the words that were spoken (`twenty twenty-six`, not
   `2026`). Do not normalize equivalent number forms after transcription.
10. **Abbreviations:** letter-by-letter initialisms use space-separated
    uppercase letters (`F B I`). Spoken acronyms use their verified
    conventional spelling (`NASA`). Expand an abbreviation only when the
    speaker says the expansion.
11. **Proper nouns:** verify spelling from a reliable source available to the
    annotators. If it cannot be resolved from the audio or source context,
    apply the unintelligible rule rather than silently inventing a name.
12. **Review and adjudication:** one annotator produces the first transcript
    and a second person reviews it while listening to the audio. They resolve
    differences by consensus; unresolved material differences are adjudicated
    by a third reviewer or the sample becomes diagnostic-only. Store
    `reference_provenance: human-reviewed`, a non-identifying `review_status`,
    and an `adjudication` value in `source`. Never store reviewer identities.

A correction changes the manifest content and therefore its hash. Version the
corpus metadata and preserve the rationale; never edit a published reference
silently.

## Deterministic Scoring

Every sample is scored once before the selected retention policy discards any
text. Failures and empty hypotheses are scored as empty hypotheses so selective
provider failure cannot improve aggregate accuracy.

### Exact text

Exact match converts CRLF and bare CR to LF and makes no other change. It is
sensitive to case, punctuation, and whitespace.

### `strict-v1`

Strict WER and CER:

1. apply Unicode NFC;
2. replace each maximal Python `str.isspace()` run with one ASCII space;
3. trim leading and trailing ASCII spaces.

Nothing else is changed. WER splits on that ASCII space. CER operates on
Unicode code points, including canonical internal spaces.

### `en-v1`

Normalized English scoring:

1. applies Unicode NFKC;
2. maps Unicode curly/modifier/full-width apostrophes to ASCII apostrophe;
3. applies Unicode `str.casefold()`;
4. preserves an apostrophe only when it is between two alphanumeric
   characters;
5. replaces all other apostrophes and Unicode punctuation with spaces;
6. collapses `str.isspace()` runs to one ASCII space and trims.

It does not remove fillers, expand contractions, accept synonyms, ignore
negation, or equate digits with words. For example, `we're` and `were` remain
different. This is deterministic normalization, not semantic judging.

Each WER/CER stores substitutions, deletions, insertions, reference-unit count,
and rate:

```text
error rate = (substitutions + deletions + insertions) / reference units
```

The primary accuracy metric is per-suite pooled normalized WER. Reports also
show strict scores, normalized CER, sample means and percentiles, exact-match
rate, success/empty/failure rates, datasets, tags, and diagnostic-only slices.
Percentiles use linear interpolation at `h = (n - 1) * p`.

Every result records `stt-score-v1`, `strict-v1`, the sample normalization
profile, and Python's runtime `unicodedata.unidata_version`. Comparisons require
matching scorer/profile identities and Unicode version; the protocol does not
pretend that Unicode behavior is fixed independently of the recorded runtime.

## Native Target and Network Policy

`--target` is repeatable and has the exact form `provider=model`. The provider
must exist in `SttProviderRegistry`; the model label is passed to that native
adapter. Unknown providers fail closed and never use the registry's defensive
faster-whisper fallback.

The coordinator preflights the complete target matrix before starting a
worker. The adapter returns an immutable execution plan that pins backend,
model/artifact identity, device/compute settings, semantic settings, fallback
policy, egress, and no-download behavior. The same plan is passed to the
worker; an actual/planned identity mismatch is recorded and disqualifies
gating.

- `neutral-v1` uses `task=transcribe`, the manifest language, no prompt, no
  hotwords, no diarization, and no requested word timestamps. Backend fallback
  is prohibited. This is the only v1 mode eligible for model-quality ranking.
- `production-v1` preserves the configured production behavior and requires a
  caller-chosen opaque `--configuration-id`. It compares complete
  configurations, not isolated model quality. Prompt and hotword contents are
  never serialized.

Models and dependencies must already be installed. Preflight rejects a target
that is unavailable locally, would download weights, cannot honor the plan, or
cannot honor neutral semantics. The worker sets common offline-library controls
as defense in depth. There is intentionally no `--download` option.

Audio egress is `none`, `loopback`, or `remote`. Both loopback and remote
targets require the separate `--allow-network-targets` flag; an API key does
not imply consent. Only literal localhost/loopback endpoints count as
loopback. Network execution plans disable HTTP redirects, so an approved
endpoint cannot forward audio to another destination. Unknown egress and
unplanned fallback fail closed.

V1 has no non-executing plan or dry-run command. Without network consent,
preflight fails generically; with consent, successful preflight proceeds into
execution. Before supplying `--allow-network-targets`, inspect the selected
adapter's configured endpoint and provider privacy/retention terms outside the
benchmark. Treat the flag as consent to start sending audio, not as a request
to preview a plan.

For a controlled network measurement, also set non-secret
`--network-collection-profile` and `--network-client-location` labels. These
labels never remove the network-dependent caveat.

## Timing Protocol

All elapsed measurements use `time.perf_counter_ns()`. Targets run sequentially
in CLI order, one fresh process per provider/model, to avoid model-cache and
resource contention between targets.

- Worker startup/import and registry/adapter setup are recorded separately.
- `cold_first_transcription_seconds` is the native adapter-call duration for
  one deterministic probe in a fresh target process. It may include lazy model
  loading, audio decode/resample, preprocessing, inference or HTTP, and
  postprocessing. It is not pure model-load or pure inference time.
- The same probe is used for every target and contributes to accuracy once. It
  is excluded from warm aggregates.
- `warm_adapter_transcription_seconds` is a later successful native adapter
  call after that worker has established warm state.
- If the cold probe fails, the first later success is
  `warmup_recovery` and is excluded from warm aggregates; only later calls are
  warm.
- On resume, a completed cold probe may be replayed without scoring to restore
  warm state. Its recovery timing is attempt metadata, not a second cold score.

For processing time `P` and measured audio duration `A`:

```text
RTF = P / A
throughput multiple = A / P
```

Lower RTF and higher throughput are better. Persistence and report generation
are outside the adapter timing window.

The default one repetition is descriptive. Performance gates require
`--warm-repetitions 3` or greater in both runs, at least three eligible warm
observations in the gated suite, and otherwise compatible execution and
hardware metadata. Use repeatable `--timing-sample ID` to restrict additional
warm repetitions; every selected sample still receives its one accuracy call.
Report medians and interquartile ranges rather than treating a single timing as
stable.

## CLI Workflows

Run from the repository root with the project virtual environment active.

### 1. Validate

```bash
source .venv/bin/activate
python Helper_Scripts/benchmarks/stt_bench.py validate \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt
```

Validation prints the sample count and portable manifest hash. It loads no
provider model.

### 2. Run or resume

```bash
python Helper_Scripts/benchmarks/stt_bench.py run \
  --manifest /data/stt/manifest.jsonl \
  --dataset-root /data/stt \
  --profile regression \
  --mode neutral-v1 \
  --text-retention errors-only \
  --warm-repetitions 3 \
  --target faster-whisper=large-v3 \
  --target parakeet=parakeet-tdt-0.6b-v3-onnx \
  --run release-regression-v1
```

`--run` names the directory under `.benchmarks/stt/`. Rerun the identical
command with the same `--run` value to resume. Successful completion keys are
skipped. Failed keys remain visible and are skipped unless the resumed command
adds `--retry-errors`; a retry appends a new attempt rather than rewriting
history. Omitting `--run` creates a collision-resistant run ID, which is
printed on completion; pass that ID explicitly to resume it.

Use `--worker-watchdog-seconds N` to terminate a hung target process. The
persisted in-flight sample receives `timeout`; the worker does not continue
with warm state.

### 3. Rebuild a report

```bash
python Helper_Scripts/benchmarks/stt_bench.py report \
  --run .benchmarks/stt/release-regression-v1
```

This validates `run.json` and `results.jsonl`, rebuilds `summary.json` and
`summary.md`, and prints the terminal projection. Summaries are disposable
views; the append-only run records are authoritative.

### 4. Compare

```bash
python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline .benchmarks/stt/baseline/summary.json \
  --candidate .benchmarks/stt/candidate/summary.json
```

Without `--policy`, comparison is descriptive and emits paired per-sample
deltas and rankings. To enforce compatible same-target regression bounds,
provide a versioned JSON policy:

```json
{"schema_version":1,"suites":{"public-english-v1":{"max_normalized_pooled_wer_absolute_regression":0.01,"max_failure_rate_absolute_regression":0.0}}}
```

```bash
python Helper_Scripts/benchmarks/stt_bench.py compare \
  --baseline .benchmarks/stt/baseline/summary.json \
  --candidate .benchmarks/stt/candidate/summary.json \
  --policy /data/stt/release-policy.json
```

Exit code is 0 when all requested eligible gates pass, 1 when an eligible gate
fails, and 2 for invalid input or incompatibility.

## Artifacts, Privacy, and Recovery

Each run is stored under `.benchmarks/stt/<run-id>/`:

```text
run.json
inflight.json
results.jsonl
summary.json
summary.md
```

`run.json` holds immutable run, environment, target, and execution-contract
identity. `inflight.json` attributes a crash, timeout, or interrupt to one
sample without storing transcript text. `results.jsonl` is append-only:
terminal records are flushed and `fsync`-ed before another sample starts.
A truncated final line after abrupt termination is reported and ignored during
report repair; valid prior records are preserved.

Resume requires exact run and execution-contract equality. Changed manifest,
selection, mode, retention, implementation fingerprint, dependency identity,
model/backend, hardware, egress, or material settings require a new run.
Results use monotonically increasing attempt IDs; reports deterministically use
the highest attempt for each completion key while preserving older attempts as
history.

Supported `--text-retention` modes are:

- `full`: retain every reference and hypothesis;
- `errors-only`: retain text for non-zero edits or non-`ok` status;
- `none`: retain no reference or hypothesis, only IDs, scores, timings, and
  bounded errors.

Scoring happens before retention. Reports never reconstruct discarded text
from the manifest. `full` is the default, and the CLI warns when it is used
with a private suite.

Run identity records immutable per-suite counts for independently supplied
reference provenance. Each result records its sample's provenance, and report
generation reconciles and displays the counts so provenance drift cannot be
hidden by a regenerated summary.

These artifacts may contain private speech, transcripts, source metadata, and
provider error context. `.benchmarks/stt/` is ignored by Git, but ignore rules
are not access control. The harness creates run directories and files with
owner-only permissions (`0700` and `0600`) where supported and never uploads
artifacts. Protect, encrypt, expire, or delete run directories according to
the corpus's privacy and retention requirements before sharing a machine,
backup, issue, or CI artifact.

## Comparison Eligibility and Interpretation

Quality comparison requires the same manifest hash, selected sample IDs,
profile, suites, scorer/profile/Unicode versions, mode, common semantic
settings, and repetition policy. Different targets may then be compared
descriptively.

Same-target regression gates additionally require matching provider/model,
resolved artifact, backend, compute/dtype, and safe settings or opaque
configuration identity. Every policy gate, including quality-only gates,
requires matching recorded hardware and collection methods; hardware-mismatched
quality remains descriptive. Performance gates additionally require matching
target order and at least three warm repetitions. Implementation source and
dependency-version differences are allowed because they are often the subject
of the regression, but they remain visible.

Network performance is informational by default. A network performance gate
also requires `--allow-network-performance-gates` and matching non-empty
network collection and client-location profiles. It remains labeled
network-dependent.

Mixed-backend production results, unresolved model identity, planned/actual
execution mismatch, partial summaries, hardware-mismatched performance, and
one-repetition timing are not eligible for the corresponding gates. Do not
declare a universally best model from these point estimates. V1 rankings are
explicitly descriptive; they are not statistical significance claims.

## Reporting Results

Attach or archive, subject to privacy:

- the exact manifest or its portable content hash;
- corpus version/license/provenance records;
- `run.json`, `results.jsonl`, and regenerated summaries;
- the tldw_server revision and dirty-state disclosure;
- target order and safe execution descriptors;
- hardware and network collection context;
- any policy used for a pass/fail claim.

Never publish an accuracy or performance claim based only on a copied Markdown
table, a provider name without a concrete model/artifact identity, an
unreviewed model-generated reference, or a run that omitted its compatibility
metadata.
