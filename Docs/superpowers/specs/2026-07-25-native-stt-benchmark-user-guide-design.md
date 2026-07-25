# Native STT Benchmark User Guide Design

- **Status:** Approved
- **Backlog:** TASK-12985.15
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
5. Provide a first local `regression` run with native provider targets.
6. Explain cold-first versus warm measurements and warm-repetition selection.
7. Show `report`, baseline/candidate organization, and descriptive `compare`.
8. Explain retention modes, resume, retry, and run artifacts.
9. Add an advanced section for network egress, `production-v1`,
   configuration identity, and network performance gates.
10. Provide troubleshooting, result interpretation, and a publication
    checklist.

The guide will not redistribute corpora, promise that optional models are
installed, duplicate the full manifest schema, document unsupported providers,
or claim real-model benchmark results.

## Information Architecture

Create
`Docs/User_Guides/STT_Benchmark_User_Guide.md` and link it from the
“Transcribe and generate speech” workflow in `Docs/User_Guides/index.md`.

The guide will use this order:

1. When to use the benchmark
2. What it does and does not measure
3. Prerequisites
4. Prepare corpus and manifest
5. Validate
6. Run a local regression benchmark
7. Inspect and rebuild reports
8. Compare two compatible runs
9. Resume, retry, and retention
10. Advanced network and production-mode operation
11. Interpret results
12. Troubleshoot
13. Publication checklist

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
- unknown providers fail closed;
- `--allow-network-targets` is consent to send audio, not a dry run;
- performance gates for any network-dependent target, including loopback and
  remote execution, additionally require matching network collection metadata
  and explicit gate consent;
- only compatible complete runs are eligible for policy gates;
- descriptive comparison also rejects partial summaries;
- failures are scored as empty hypotheses and cannot improve quality;
- retained transcripts may contain private content.

## Verification

Before opening the PR:

- compare all documented options with `stt_bench.py` subcommand help;
- run repository tests covering CLI help and the example manifest;
- scan relative Markdown links in the new guide and index entry;
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
