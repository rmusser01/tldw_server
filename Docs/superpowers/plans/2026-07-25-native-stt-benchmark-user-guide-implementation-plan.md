# Native STT Benchmark User Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a local-first operator guide for preparing, running, comparing, and safely sharing native batch STT benchmark results, then open a draft pull request against `dev`.

**Architecture:** Keep `Docs/Development/STT_Benchmark_Protocol.md` authoritative and add one task-oriented guide that links to it for exact schemas and eligibility rules. Reuse the existing CLI and setup documentation without adding discovery commands, corpus downloaders, dependencies, or runtime behavior.

**Tech Stack:** Markdown, existing Python `argparse` CLI, pytest, standard-library link checks, official Backlog.md MCP workflow with documented CLI fallback, Git, GitHub CLI.

---

## File Structure

- Modify: `Docs/superpowers/specs/2026-07-25-native-stt-benchmark-user-guide-design.md`
  - Incorporate the approved review corrections before guide implementation.
- Create: `Docs/User_Guides/STT_Benchmark_User_Guide.md`
  - Provide the end-to-end operator workflow and safety guidance.
- Modify: `Docs/User_Guides/index.md`
  - Add the guide to the speech workflow.
- Modify: `Helper_Scripts/benchmarks/README.md`
  - Point the existing benchmark quick reference to the operator guide.
- Modify through the official Backlog.md MCP workflow, with the documented CLI fallback: `backlog/tasks/task-12985.15 - Add-native-STT-benchmark-operator-user-guide-and-open-dev-PR.md`
  - Record implementation, verification, branch, and PR evidence.

## Task 1: Revise and Lock the Design

**Files:**
- Modify: `Docs/superpowers/specs/2026-07-25-native-stt-benchmark-user-guide-design.md`

- [x] **Step 1: Add the approved operator requirements**

Require the guide to cover:

- selecting an installed native `provider=model` target without implying discovery support;
- exact `run`, `report`, and `compare` operand/path semantics;
- a compatibility checklist and CLI-order target pairing;
- descriptive comparison versus same-target policy gating;
- policy exit codes `0`, `1`, and `2`;
- endpoint/privacy review before network consent;
- exact text-retention behavior and artifact disposal;
- public/private hybrid suites and English-first, multilingual-ready manifests;
- one primary local target with optional hardware-specific second targets.

- [x] **Step 2: Add discoverability and verification requirements**

Require links from both `Docs/User_Guides/index.md` and
`Helper_Scripts/benchmarks/README.md`. Require semantic verification against
the parser/source and protocol in addition to CLI help.

- [x] **Step 3: Self-review the design diff**

Run:

```bash
git diff --check
git diff -- Docs/superpowers/specs/2026-07-25-native-stt-benchmark-user-guide-design.md
```

Expected: no whitespace errors, and every accepted review finding is visible
in the design.

## Task 2: Write the Local-First Operator Guide

**Files:**
- Create: `Docs/User_Guides/STT_Benchmark_User_Guide.md`

- [x] **Step 1: Write purpose, trust boundary, and target selection**

Explain that the harness is batch-only, uses native adapters, has no Pipecat
or LLM judge, and computes deterministic strict and normalized WER/CER.
Describe the no-download precondition, `provider=model` syntax, adapter-specific
model labels, and CPU/GPU setup links. Use one operator-supplied local target
as the primary path; mark concrete targets as examples rather than guaranteed
installations. State that unknown providers fail closed instead of falling
back to another adapter.

- [x] **Step 2: Write corpus and manifest preparation**

Show an external corpus layout and copying the schema-only example manifest.
Require independent references, checksums, durations, license/provenance, and
public/private suite separation. Use English examples while explaining
per-sample language metadata and multilingual extension without cross-suite
pooling. Include executable checksum commands for Linux, macOS, and Windows,
plus an `ffprobe` duration command, before validation verifies the completed
manifest.

- [x] **Step 3: Write validate and local run workflows**

Use shell variables for the dataset root, manifest, target, and run name. Show:

```bash
python Helper_Scripts/benchmarks/stt_bench.py validate \
  --manifest "$STT_MANIFEST" \
  --dataset-root "$STT_DATASET_ROOT"

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

Explain cold-first versus warm adapter timing, the three-observation minimum
for performance gates, and that cold-first is not pure model-load time.

- [x] **Step 4: Write report, comparison, and policy workflows**

Distinguish:

- `run --run "$STT_RUN_ID"` from
  `.benchmarks/stt/"$STT_RUN_ID"`;
- `report --run RUN_DIRECTORY`;
- `compare --baseline BASELINE/summary.json --candidate CANDIDATE/summary.json`.

Add the compatibility checklist, descriptive cross-target comparison, a
minimal same-target policy example, and exit-code meanings. State that policy
gates require compatible complete runs, descriptive comparison also rejects
partial summaries, and failed samples are scored as empty hypotheses so they
cannot improve quality metrics.

- [x] **Step 5: Write lifecycle, network, interpretation, and publication guidance**

Cover immutable resume arguments, `--retry-errors`, append-only artifacts,
retention modes, private artifact handling, and disposal. Keep network and
`production-v1` operation advanced and opt-in. State that API credentials are
not consent, require endpoint/provider privacy review before
`--allow-network-targets`, and explain the extra network performance-gate
metadata and consent. Finish with troubleshooting and publication checklists.

## Task 3: Add Discoverability Links

**Files:**
- Modify: `Docs/User_Guides/index.md`
- Modify: `Helper_Scripts/benchmarks/README.md`

- [x] **Step 1: Link from the user-guide workflow**

Add `STT_Benchmark_User_Guide.md` to the existing “Transcribe and generate
speech” entry without creating a second navigation category.

- [x] **Step 2: Link from the benchmark README**

Keep the existing compact quick reference and direct operators to the new guide
for target selection, corpus setup, retention, comparison eligibility, and
network safety.

- [x] **Step 3: Inspect link context**

Run:

```bash
sed -n '45,60p' Docs/User_Guides/index.md
sed -n '1,55p' Helper_Scripts/benchmarks/README.md
```

Expected: both links are concise and do not duplicate the protocol.

## Task 4: Verify Documentation and CLI Accuracy

**Files:**
- Verify: `Docs/User_Guides/STT_Benchmark_User_Guide.md`
- Verify: `Docs/User_Guides/index.md`
- Verify: `Helper_Scripts/benchmarks/README.md`
- Verify: `Helper_Scripts/benchmarks/stt_bench.py`
- Test: `tldw_Server_API/tests/Benchmarks/test_stt_bench.py`

- [x] **Step 1: Verify every documented CLI surface**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py --help
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py validate --help
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py run --help
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py report --help
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python \
  Helper_Scripts/benchmarks/stt_bench.py compare --help
```

Expected: every documented option exists. Cross-check defaults, path meaning,
eligibility, and exit codes against `stt_bench.py` and
`Docs/Development/STT_Benchmark_Protocol.md`.

- [x] **Step 2: Run focused benchmark documentation tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py::test_example_manifest_documents_and_validates_every_record \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py::test_cli_help_documents_every_subcommand \
  tldw_Server_API/tests/Benchmarks/test_stt_bench.py::test_compare_cli_uses_documented_exit_codes \
  -v
```

Expected: all selected tests pass.

- [x] **Step 3: Check all touched relative Markdown links with the standard library**

Run a short, read-only Python script that extracts relative Markdown links from
the new guide, user-guide index, and benchmark README; resolves each link from
its containing directory; ignores URL, email, and anchor-only targets; strips
fragments before checking; and fails for missing paths.

Expected: every relative target exists. Do not add a link-checking dependency.

- [x] **Step 4: Run static documentation checks**

Run:

```bash
git diff --check
rg -n 'YOUR_API|\bsk-[A-Za-z0-9]{16,}|TODO|FIXME' \
  Docs/User_Guides/STT_Benchmark_User_Guide.md \
  Docs/User_Guides/index.md \
  Helper_Scripts/benchmarks/README.md
```

Expected: no whitespace errors, working-looking credentials, or untracked
TODOs. Record Bandit as not applicable because this slice changes Markdown and
Backlog records only.

## Task 5: Finalize, Commit, Rebase, and Open the Draft PR

**Files:**
- Modify through the official Backlog.md MCP workflow, with the documented CLI fallback: `backlog/tasks/task-12985.15 - Add-native-STT-benchmark-operator-user-guide-and-open-dev-PR.md`

- [x] **Step 1: Update pre-PR Backlog evidence**

Use the official Backlog.md MCP workflow to record the following. If the MCP
workflow remains unavailable or times out, use the already documented
Backlog.md CLI fallback:

- design and plan paths;
- guide and link paths;
- exact verification results;
- Bandit not-applicable rationale;
- branch name;
- the remaining human-authored `Change summary` merge gate.

- [ ] **Step 2: Commit the documentation slice**

```bash
git add \
  Docs/superpowers/specs/2026-07-25-native-stt-benchmark-user-guide-design.md \
  Docs/superpowers/plans/2026-07-25-native-stt-benchmark-user-guide-implementation-plan.md \
  Docs/User_Guides/STT_Benchmark_User_Guide.md \
  Docs/User_Guides/index.md \
  Helper_Scripts/benchmarks/README.md \
  "backlog/tasks/task-12985.15 - Add-native-STT-benchmark-operator-user-guide-and-open-dev-PR.md"
git commit -m "docs: add native STT benchmark user guide"
```

- [ ] **Step 3: Update from `origin/dev` and reverify**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git status --short
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
```

Expected: the tree is clean, the branch is based on current `origin/dev`, and
the final diff includes the complete benchmark implementation and guide. If
the rebase changes or conflicts with benchmark or guide files, rerun the
focused tests and documentation checks before proceeding.

- [ ] **Step 4: Push and open a draft PR against `dev`**

Push `codex/native-stt-benchmark`, then create a draft PR with:

- the complete benchmark and guide summary;
- exact verification evidence;
- a clearly marked human `Change summary` prompt asking what changed and why
  these implementation choices were made;
- an explicit statement that pasted AI-generated text does not satisfy the
  merge gate.

- [ ] **Step 5: Verify, record, and commit PR state**

First confirm that the PR base is `dev`, the PR remains draft, and the rendered
body contains the required verification evidence and human `Change summary`
prompt. Then record the verified PR URL and draft state in `TASK-12985.15`,
mark the task complete, commit that Backlog update, and push it to the same
branch. Verify that the final task-record commit appears in the PR. Then run:

```bash
git status --short
git diff --stat origin/dev...HEAD
git diff --check origin/dev...HEAD
```

Expected: a clean tree and a whitespace-clean final PR diff that includes the
task closeout commit. Do not claim merge readiness until the requester supplies
the human-owned `Change summary`.
