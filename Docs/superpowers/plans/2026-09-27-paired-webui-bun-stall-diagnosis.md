# Paired WebUI Bun Stall Diagnostic Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans when implementing production changes. This task is one bounded diagnostic, with no production implementation.

**Goal:** Identify the pending child script during the recurring frozen Bun install, retaining evidence even when the run times out.

**Architecture:** Prepare an isolated image ending immediately before the existing install instruction. Run the unchanged ordinary install once in an owned container, with a 300-second timeout and private retained process snapshots. Diagnostic images cannot qualify or replace production artifacts.

**Tech Stack:** Cached Bun 1.3.2 Debian arm64, Docker, existing POSIX/coreutils tools, host Python in the project venv.

**Spec:** `AGENTS.md` retry/reassessment rules; `Docs/superpowers/reviews/2026-09-25-complete-app-wp1-acceptance.md` quiet-install rulings; current source acceptance review.

## Global constraints

- TASK-13376.10 tracks this separate investigation. No full candidate retry.
- Preserve Bun 1.3.2, `apps/bun.lock`, package manifests and production Dockerfiles.
- Use verified cached base `oven/bun@sha256:ff851006c8b322761d53593e7a78c92d09ec0a6bc09a55f81c9861e614761d9a` on local arm64.
- Keep raw stdout, command lines and working directories in private `/private/tmp/task13376-bun-stall-diagnosis-3ef013bd94`; never read process environment, cookies or provider credentials.
- Inspect only descendants of the owned timeout process, including children attached to every thread. Preserve PID start ticks to detect reuse.
- Run one ordinary `bun install --frozen-lockfile --cwd /app/apps`, without verbosity or script/concurrency changes; GNU timeout is 300 seconds with a 15-second termination grace.
- Retain diagnostic output and private recovery material; clean up only exact recorded owned container/image identities. Preserve existing images, volumes and unrelated PostgreSQL services.
- No native CI retry/cancellation, publication, merge or acceptance waiver.

## Stage 1: Reassess exhausted attempts
**Goal**: Establish the missing evidence and choose a different diagnostic angle.
**Success Criteria**: Prior outcomes, hypotheses and alternatives are explicit.
**Tests**: Read retained logs and verify current runtime inputs are unchanged from `3ef013bd94`.
**Status**: Complete

- [x] Prior ordinary installs exceeded quiet bounds; changing output verbosity sometimes completed in 85–90 seconds but did not establish a fix.
- [x] Earlier 300-second instrumented run stopped with an idle child named `node`, then discarded raw lifecycle output. Its traversal omitted thread-owned children.
- [x] Later all-thread identity probe completed in 90 seconds and proved children used Bun, but classified every script as unknown. It did not reproduce the stall.
- [x] Current source candidate again stopped after resolved/extracted 256 and 1090 seconds without output. Native jobs remain running; API reads are rate-limited.
- [x] Compare three alternatives: wait for native evidence; identify the real pending script in one isolated ordinary install; change install flags/runtime. Choose the second to fill the evidence gap without changing the product. The third has no established causal basis.

Bun documents project lifecycle execution and parallel scripts in its [install documentation](https://bun.sh/docs/pm/cli/install). An upstream [Linux workspace deadlock report](https://github.com/oven-sh/bun/issues/30515) describes an earlier resolution-stage stall with no script activity; it does not establish this project's cause. Existing extension prepare imports WXT before checking `SKIP_WXT_PREPARE`; the frontend worker-copy script performs synchronous file operations. Either workspace script, a trusted dependency script, or Bun's own scheduling could explain a pending child. Do not select a fix until the actual script and stalled phase are observed.

## Stage 2: Capture one isolated install
**Goal**: Retain the evidence omitted by earlier probes.
**Success Criteria**: One install returns or times out within 315 seconds; raw output and owned descendant snapshots survive either outcome.
**Tests**: Shell syntax, cached-image tool preflight, descendant traversal fixture, actual bounded run and exit/status checks.
**Status**: Complete

**Files:** Create private `Dockerfile`, `install-probe.sh`, `run-diagnostic.py`, `evidence/install.log`, `evidence/snapshots/` and `journal.json` under the private root above. No production source edit.

- [x] Verify at least 10 GiB free, base digest/architecture, current branch and unchanged production input diff.
- [x] Copy the production dependencies stage only through the final COPY before RUN; add the diagnostic ownership label and create a unique temporary image with `--network=none --pull=false --load`.
- [x] Preflight `timeout`, `cat`, `awk`, `cp`, `readlink`, `date`, `sleep` and `mkdir` in the cached image. Validate traversal with a known descendant fixture; reject missing tools or a failed fixture.
- [x] Start one container by exact created ID with the private probe mounted read-only and the evidence directory writable. Run `timeout --kill-after=15s 300s bun install --frozen-lockfile --cwd /app/apps`, redirecting both streams to `evidence/install.log`.
- [x] Every 10 seconds, walk the timeout descendants through all `/proc/<pid>/task/*/children`; retain cmdline, cwd, executable, stat, io and thread wait categories for each stable PID. Copy package.json only from a verified descendant cwd under `/app/apps/` to identify dependency scripts. Do not infer script identity from process comm.
- [x] Retain final install exit code and elapsed time. If no stall occurs, report that result without another attempt or root-cause claim.

The single install returned 0, reporting 3665 packages installed in 85.08 seconds;
monitoring finished in 92 seconds. Stable children included canvas and Puppeteer
24.36.0 `node install.mjs`, both actually using `/usr/local/bin/bun`. Puppeteer
was still active at 82 seconds, with increasing CPU and I/O counters and
731975680 bytes written. No stalled child was observed. This identifies
successful-run scripts, not the earlier stalled script or its cause.

## Stage 3: Classify the result and preserve qualification limits
**Goal**: Report a concrete finding or the exact remaining evidence gap.
**Success Criteria**: No invented cause; owned cleanup verified and native/full release gates unchanged.
**Tests**: Private evidence inspection, exact container/image ownership checks, unrelated container/image/volume inventories, `git diff --check`.
**Status**: In Progress

- [ ] If timed out, identify the last stable child from retained argv/cwd/package metadata and compare its CPU/I/O changes and waits. A pending script alone does not establish its internal cause.
- [x] Remove only the recorded owned diagnostic container and image after labels/identities match; retain host evidence and any cleanup recovery records on failure.
- [ ] Update the task and current acceptance review with source, platform, bound, outcome and native status. Bandit is inapplicable to tracked Markdown/task-only changes.
- [ ] Present any production correction separately for review before implementation. Keep the original candidate and ordinary workflow evidence tied to their exact sources.

**Ruling:** Use one separately tracked retained-evidence probe after reassessing the exhausted ordinary-build path — prior probes lost the stalled script's identity, so another full build or verbosity change would not answer the unresolved question. Cost if wrong: one bounded diagnostic install may fail to reproduce; stop with that limit rather than retrying or accepting diagnostic artifacts.
