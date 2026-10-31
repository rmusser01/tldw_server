# Native STT Benchmark User Guide Design

- **Status:** Approved with review corrections
- **Backlog:** TASK-12985.15
- **Implementation plan:** [`2026-07-25-native-stt-benchmark-user-guide-implementation-plan.md`](../plans/2026-07-25-native-stt-benchmark-user-guide-implementation-plan.md)
- **Audience:** tldw_server operators comparing supported batch STT adapters
- **Primary path:** Local models first; network targets are an advanced opt-in

## Purpose

Add a task-oriented user guide for running the native batch STT benchmark
without requiring readers to reconstruct a workflow from the protocol
reference. The guide should help an operator move from an authorized corpus to
a validated manifest, a reproducible local run, a report, and a compatible
baseline comparison.

The existing
[`STT_Benchmark_Protocol.md`](../../Development/STT_Benchmark_Protocol.md)
remains authoritative for schemas, scoring rules, timing semantics, privacy,
and policy eligibility. The new guide explains how to apply that protocol.

## Scope

The guide will:

1. Explain prerequisites and the benchmark trust boundary.
2. Show a safe external corpus layout and how to copy the example manifest.
3. Explain the fields an operator must replace or independently verify.
4. Show checksum, duration, and `validate` commands.
5. Explain how to choose an already-installed native `provider=model` target
   when the CLI has no discovery or dry-run command.
6. Provide a first local `regression` run with one operator-selected target,
   with optional second-target examples separated by hardware.
7. Show how public and private suites can coexist while retaining independent
   provenance, and use English examples without making the manifest or scoring
   architecture English-only.
8. Explain cold-first versus warm measurements and warm-repetition selection.
9. Show `report`, baseline/candidate organization, descriptive `compare`, an
   optional same-target regression policy, and exit codes.
10. Distinguish the identifier accepted by `run --run`, the directory accepted
    by `report --run`, and the `summary.json` paths accepted by `compare`.
11. Provide a concise comparison-compatibility checklist, including ordered
    target pairing.
12. Explain retention modes, resume, retry, and run artifacts.
13. Add an advanced section for network egress, `production-v1`,
    configuration identity, and network performance gates.
14. Provide troubleshooting, result interpretation, and a publication
    checklist.

The guide will not redistribute corpora, promise that optional models are
installed, duplicate the full manifest schema, document unsupported providers,
claim real-model benchmark results, or add a target-discovery command, corpus
downloader, or new documentation dependency.

## Information Architecture

Create
`Docs/User_Guides/STT_Benchmark_User_Guide.md` and link it from the
“Transcribe and generate speech” workflow in `Docs/User_Guides/index.md`.
Also link it from `Helper_Scripts/benchmarks/README.md` so the existing quick
reference leads to the full operator workflow.

The guide will use this order:

1. When to use the benchmark
2. What it does and does not measure
3. Prerequisites and choose an installed local target
4. Prepare a hybrid corpus and multilingual-ready manifest
5. Validate
6. Run a local regression benchmark
7. Inspect and rebuild reports
8. Compare two compatible runs
9. Add an optional same-target regression policy
10. Resume, retry, and retention
11. Advanced network and production-mode operation
12. Interpret results
13. Troubleshoot
14. Publication checklist

This order prioritizes the shortest safe local workflow while keeping
high-egress and release-gating features clearly separated.

## Command and Safety Contract

Every command must use options present in the current CLI help. Examples will
run from the repository root with the project virtual environment active.
Placeholders will be visibly operator-supplied and will not resemble working
credentials.

The guide must state that:

- the example manifest is schema documentation, not valid corpus metadata;
- models and dependencies must already be installed;
- the benchmark does not download models or corpora;
- target syntax is `provider=model`, model labels are adapter-specific, and
  concrete targets are illustrative rather than guaranteed to be installed;
- unknown providers fail closed;
- `run --run NAME` creates `.benchmarks/stt/NAME`, `report --run` accepts that
  directory, and `compare` accepts each directory's `summary.json`;
- `--allow-network-targets` is consent to send audio, not a dry run;
- an API key is not network consent, and operators must inspect the configured
  endpoint plus provider privacy and retention terms before opting in;
- performance gates for any network-dependent target, including loopback and
  remote execution, additionally require matching network collection metadata
  and explicit gate consent;
- only compatible complete runs are eligible for policy gates;
- descriptive comparison also rejects partial summaries;
- descriptive cross-target comparison and same-target regression gating have
  different eligibility requirements;
- compare exit code `0` means the comparison completed and requested gates
  passed, `1` means an eligible gate failed, and `2` means invalid or
  incompatible input;
- failures are scored as empty hypotheses and cannot improve quality;
- `full` is the default retention mode, `errors-only` may retain transcript
  text for errors and non-zero edits, and `none` still retains metadata and
  bounded errors;
- `.benchmarks/stt/` being Git-ignored is not access control, so private run
  artifacts require explicit protection, expiration, or disposal;
- retained transcripts may contain private content.

The compatibility checklist must cover the manifest and selected samples,
profile, suites, scorer/normalization/Unicode identities, mode, seed,
repetition policy, target-matrix size, common settings, and the fact that
targets pair by CLI order. Same-target policy gates must additionally explain
the stricter execution, hardware, collection, and configuration requirements.

The corpus workflow must show independent license and provenance records for
public and private suites, executable checksum and duration commands, and
per-sample language metadata. English is the first documented path, but the
guide must explain how additional languages are added as separately reported
suites without publishing a misleading cross-suite aggregate.

## Verification

Before opening the PR:

- compare all documented options with `stt_bench.py` subcommand help;
- verify semantic claims that help text does not expose—defaults, operand/path
  meaning, compatibility, eligibility, and exit codes—against the parser,
  implementation, and protocol;
- run repository tests covering CLI help and the example manifest;
- scan relative Markdown links in the new guide, index entry, and benchmark
  README with a standard-library check rather than adding a dependency;
- run `git diff --check`;
- record that Bandit is not applicable to documentation-only guide changes;
- inspect the final branch diff against `origin/dev`;
- include exact verification evidence in the PR body.

## Pull Request

The existing `codex/native-stt-benchmark` branch will be pushed and used for a
PR targeting `dev`. The PR will summarize the complete benchmark work plus the
new user guide. Because the repository treats materially AI-authored PRs as
merge-blocked until the human requester writes the rationale, the PR body must
include a clearly marked human `Change summary` placeholder and explain that
the requester must replace it in their own words before merge.
