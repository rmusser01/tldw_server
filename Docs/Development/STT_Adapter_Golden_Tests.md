# STT Adapter Golden Tests

The `stt_golden` pytest profile is an opt-in real-audio regression check for
tldw_server's native batch STT adapters. It uses the same manifest loader,
English-ready normalization, and deterministic WER implementation as the
standalone STT benchmark. It does not use Pipecat or an LLM judge.

The profile is intended for developer machines or controlled release runners
with the required models already installed. Ordinary test runs do not load a
model or contact a provider.

## Ground-truth rule

The regression manifest must contain independently sourced references:

- `canonical-dataset` for a transcript supplied by the source corpus; or
- `human-reviewed` for a transcript reviewed independently of the model under
  test.

Adapter output is only a behavior snapshot or unverified candidate. It must
never become a scored reference merely because a model produced it.

The manifest schema and scorer are owned by
`Helper_Scripts/benchmarks/stt_bench.py`. Every selected row must include the
`regression` profile. Audio remains outside the repository unless its license
and inclusion have been reviewed separately.

## Required environment

Set all four variables:

```bash
export TLDW_STT_GOLDEN_ENABLE=1
export TLDW_STT_GOLDEN_AUDIO_DIR=/srv/tldw_stt_golden
export TLDW_STT_GOLDEN_MANIFEST=/srv/tldw_stt_golden/regression.jsonl
export TLDW_STT_GOLDEN_TARGETS='["faster-whisper=large-v3","parakeet=parakeet-mlx"]'
```

The test skips only when `TLDW_STT_GOLDEN_ENABLE` is not truthy. Once enabled,
missing or invalid required settings fail the run so a release job cannot pass
without exercising a model.

`TLDW_STT_GOLDEN_TARGETS` must be a JSON array. Comma-separated or
shell-delimited target lists are rejected so provider/model boundaries remain
unambiguous.

Optional variables:

- `TLDW_STT_GOLDEN_MAX_NORMALIZED_WER`: per-sample upper bound; defaults to
  `0.20`.
- `TLDW_STT_GOLDEN_ALLOW_NETWORK=1`: allows a planned loopback or remote
  target. Without it, any route that sends audio outside the process is
  rejected before transcription.

Each target is resolved strictly through `SttProviderRegistry`. Before
transcription, model loading, or network egress, the test creates a neutral
execution plan and checks that:

- the requested provider exists and supports batch transcription;
- the selected local artifact already exists and will not download; and
- any loopback or remote route has the separate network opt-in.

The adapter must execute that exact plan. Silent fallback to another provider,
backend, model, or network route fails the normalized artifact contract.

## Running the profile

Run only the golden module:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Audio/test_stt_adapters_golden.py \
  -m stt_golden -v
```

The existing Makefile target can also be used after exporting the manifest and
target variables:

```bash
make stt-golden STT_GOLDEN_AUDIO_DIR=/srv/tldw_stt_golden
```

For every `regression` sample and configured target, the test:

1. revalidates manifest containment, audio checksum, and duration;
2. approves an exact no-download execution plan;
3. asserts normalized artifact, non-empty segment, and actual-execution shapes;
4. computes strict and selected-profile scores with `score_transcript`; and
5. enforces `TLDW_STT_GOLDEN_MAX_NORMALIZED_WER`.

These checks are regression evidence for the installed target configuration.
They are not comparative performance results; use the standalone benchmark for
cold-first/warm timing and cross-target reports.

## Candidate snapshots

`Helper_Scripts/Audio/generate_stt_golden.py` can transcribe one clip into a
candidate snapshot:

```bash
python Helper_Scripts/Audio/generate_stt_golden.py \
  --provider faster-whisper \
  --model large-v3 \
  --audio audio/whisper/en/clip1.wav \
  --language en \
  --base-dir /srv/tldw_stt_golden \
  --output candidates/whisper-clip1.json
```

Candidate generation uses the same strict registry and exact no-download
preflight. Add `--allow-network` only when the selected planned target may send
audio to loopback or a remote endpoint.

The output is explicitly labeled:

```json
{
  "artifact_type": "stt-transcript-candidate",
  "reference_status": "unverified_candidate",
  "candidate_text": "adapter output"
}
```

It deliberately has no `reference` field and cannot be loaded as a scored
manifest record.

## Writing a verified manifest row

When the reference came from a canonical dataset or independent human review,
the helper can write one target-neutral JSONL manifest row without running an
adapter:

```bash
python Helper_Scripts/Audio/generate_stt_golden.py \
  --audio audio/challenge/clip1.wav \
  --language en \
  --base-dir /srv/tldw_stt_golden \
  --output rows/challenge-clip1.jsonl \
  --sample-id challenge-clip-1 \
  --reference "The independently reviewed transcript." \
  --reference-provenance human-reviewed
```

`--reference` and `--reference-provenance` must appear together.
`--reference-provenance` accepts only `canonical-dataset` or
`human-reviewed`, and a stable `--sample-id` is mandatory. The output is one
compact JSONL row that round-trips through `load_and_validate_manifest`.

The helper supplies conservative private-regression metadata:

- suite `private-golden-v1`;
- annotation profile derived from the reference provenance;
- dataset `local-golden`, version `1`, license `user-supplied`; and
- an audio SHA-256 calculated from the selected file.

Review and replace those source fields when the corpus has more specific
dataset, version, or license metadata. Combining rows into the maintained
manifest is an explicit review step; the helper never appends automatically.

## When to run

Run the profile:

- before a release that changes STT adapters or model configuration;
- after model, CUDA, driver, or heavy dependency upgrades;
- after preprocessing or normalization changes; and
- when validating a new native registry target on controlled hardware.

Record the manifest hash, target strings, hardware, dependency versions, and
whether network execution was enabled alongside any reported result.
