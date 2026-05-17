# Next Dev Server Memory Leak Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture low-impact evidence for the high-RSS Next dev server, classify the memory growth shape, and produce a ranked root-cause hypothesis report before any frontend code fixes.

**Architecture:** This is an evidence-gathering implementation, not an application-code change. The worker creates a durable report, rediscovering the live Next process before sampling, then branches only after data separates idle growth, route/build-cache growth, request storms, extension traffic, or dev-tooling/native memory.

**Tech Stack:** macOS process tools (`ps`, `pgrep`, `lsof`, optional `vmmap`/`sample` only after approval), Next.js dev server, Bun/Node, shared WebUI code under `apps/packages/ui`, WebUI shell under `apps/tldw-frontend`, extension shell under `apps/extension`.

---

## Source Inputs

- Design spec: `Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md`
- Design task: `TASK-422`
- Plan task: `TASK-423`
- Evidence report target: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`

## File Structure

- Create: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`
  - Durable evidence report with process identity, samples, log/request correlation, memory-shape classification, hypotheses, and next step.
- Modify: `<evidence-task-file>`
  - Use Backlog.md to create a new evidence-gathering task before writing the report. Do not reuse `TASK-422` or `TASK-423` for implementation.
  - Record the exact task file path after creation, for example `backlog/tasks/task-<id> - Investigate-Next-dev-server-memory-leak-evidence.md`; use that exact path in every `git add`, `git status`, and `git diff --check` command.
- No application source files should be modified in this plan.

## Guardrails

- Do not restart, kill, or signal the high-RSS process before the low-impact baseline is captured.
- Do not attach an inspector, run `sample`, run `vmmap`, or send diagnostic signals until the baseline evidence is recorded and the user approves the more intrusive step.
- Do not patch React, Next, WXT, or shared UI code in this plan. If evidence points to a fix, create a follow-up Backlog task and plan.
- Treat any historical PID, port, cwd, or log path as stale until Task 1 rediscovery confirms it.
- If the process no longer exists, record that explicitly and pivot to a clean reproduction plan instead of pretending the original evidence is still live.
- If process-inspection commands such as `lsof`, `vmmap`, or `sample` are blocked by macOS permissions or sandboxing, request approval for the specific diagnostic instead of substituting a broader or more destructive workaround.
- Do not stage or diff the whole `backlog/tasks` directory. This repository may contain unrelated dirty task records; always use the exact evidence task file path.

### Task 1: Create Evidence Task And Report Skeleton

**Files:**
- Create: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`
- Modify: `<evidence-task-file>`

- [ ] **Step 1: Search for an existing evidence task**

Run:

```bash
backlog search "Next dev server memory leak evidence" --plain
backlog search "next-server RSS investigation" --plain
```

Expected: no active duplicate task, or a clear existing task to reuse.

- [ ] **Step 2: Create the evidence task if no duplicate exists**

Run:

```bash
backlog task create "Investigate Next dev server memory leak evidence" \
  --label performance --label webui --label extension --priority high \
  --desc "Capture low-impact evidence for the high-RSS Next dev server before any runtime code fixes. Produce Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md with process identity, baseline samples, memory-shape classification, correlations, hypotheses, and next step."
```

Expected: a new task id and exact task file path, referenced in the evidence report.

- [ ] **Step 3: Create the report skeleton**

Create `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`:

```markdown
# Next Dev Server Memory Leak Investigation

Backlog task: TASK-<new evidence task>
Design: Docs/superpowers/specs/2026-05-17-next-dev-server-memory-leak-investigation-design.md
Date: 2026-05-17

## Executive Summary

Status: In progress.

## Process Identity

| Field | Value |
| --- | --- |
| PID | TBD |
| Parent PID | TBD |
| Command | TBD |
| CWD | TBD |
| Port | TBD |
| Mode | TBD |

## Baseline Samples

| Timestamp | PID | RSS KB | CPU % | Open Files | Active TCP | Log Signal | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |

## Memory Shape

| Signal | Result | Confidence | Notes |
| --- | --- | --- | --- |
| RSS slope | TBD | TBD | TBD |
| JS heap | Not collected | Low | Do not infer without approved inspector/diagnostics. |
| Native/external | Not collected | Low | Do not infer without approved intrusive diagnostics. |
| File descriptors | TBD | TBD | TBD |
| Build/route cache | TBD | TBD | TBD |

## Correlation Findings

## Hypotheses

1. TBD

## Recommended Next Step

TBD

## Verification

- Low-impact baseline captured before intrusive diagnostics: pending.
- No runtime source changes: pending.
```

- [ ] **Step 4: Commit the report skeleton and task**

Run:

```bash
git add Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md "<evidence-task-file>"
git commit -m "docs: start next dev memory leak evidence report"
```

Expected: commit contains only the report skeleton and intended Backlog task metadata.

### Task 2: Rediscover Live Process Identity

**Files:**
- Modify: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`
- Modify: `<evidence-task-file>`

- [ ] **Step 1: Identify candidate Next dev processes**

Run:

```bash
ps -ef | rg "next-server|next dev|tldw-frontend|18021|8080"
```

Expected: one or more frontend dev server candidates. Record all plausible candidates and select the high-RSS one only after memory data confirms it.

- [ ] **Step 2: Capture process memory and command details**

For each plausible PID, run:

```bash
ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,command -p <PID>
```

Expected: one target process clearly has high RSS or is the active Next server child.

- [ ] **Step 3: Capture parent and child process tree**

Run:

```bash
ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,command -p <PID>,<PPID>
pgrep -P <PID>
pgrep -P <PPID>
```

Expected: report lists parent `bun`/`node` process, high-RSS `next-server` child, and any child PIDs.

- [ ] **Step 4: Capture cwd and serving port**

Run:

```bash
lsof -a -p <PID> -d cwd -Fn
lsof -nP -iTCP -sTCP:LISTEN | rg "<PID>|18021|8080|next|node"
```

Expected: report records actual CWD and port. If the port differs from historical `18021`, use the current value for all later browser/request checks.

- [ ] **Step 5: Record mode and artifact paths**

From command line and package scripts, classify mode:

```bash
sed -n '1,80p' <CWD>/package.json
ls -la <CWD>/.next/dev/logs
```

Expected: report records whether current command looks like default `next dev`/Turbopack or `next dev --webpack`, and records the process-specific log path.

### Task 3: Capture Low-Impact Idle Baseline

**Files:**
- Modify: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`

- [ ] **Step 1: Start a no-interaction sample window**

Do not open browser routes, reload pages, or restart the process during this window.

- [ ] **Step 2: Take five 15-second process samples**

Run this manually five times, 15 seconds apart:

```bash
date '+%Y-%m-%d %H:%M:%S %Z'
ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,command -p <PID>
lsof -p <PID> | wc -l
lsof -nP -p <PID> -iTCP | wc -l
tail -40 <CWD>/.next/dev/logs/next-development.log
```

Expected: report has five rows with RSS, CPU, open-file count, TCP count, and recent log signal. If the log file is absent, record `log missing` rather than searching the main checkout blindly.

- [ ] **Step 3: Calculate baseline slope**

Compute:

```text
rss_delta_kb = last_rss_kb - first_rss_kb
duration_minutes = elapsed_seconds / 60
rss_slope_kb_per_min = rss_delta_kb / duration_minutes
```

Expected: report classifies idle slope as stable, slow growth, or fast growth using the actual sample values.

- [ ] **Step 4: Decide whether low-impact evidence is enough**

If RSS is stable and no logs/request churn appear, proceed to Task 4. If RSS grows quickly while idle, proceed to Task 5 before touching browser workflows.

### Task 4: Browser And Request Correlation Sweep

**Files:**
- Modify: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`

- [ ] **Step 1: Confirm the current frontend URL**

Use the current port from Task 2:

```bash
curl -sf http://127.0.0.1:<PORT>/ >/tmp/tldw_next_probe.html
```

Expected: command succeeds or records a clear connection error. Do not switch to `8080` unless Task 2 proved that is the live port.

- [ ] **Step 2: Capture route-load samples**

For each route, open it once in the browser or with the approved browser automation tool, then take one memory sample:

```text
/
/chat
/knowledge
/settings
/media
```

Sample command after each route:

```bash
date '+%Y-%m-%d %H:%M:%S %Z'
ps -o pid,ppid,%cpu,%mem,rss,vsz,etime,command -p <PID>
tail -80 <CWD>/.next/dev/logs/next-development.log
```

Expected: report identifies route compilation events and distinguishes expected one-time cache growth from repeated growth on revisits.

- [ ] **Step 3: Revisit the highest-growth route**

Open the route with the largest RSS jump three more times, waiting 15 seconds between visits.

Expected: report says whether growth repeats on the same route or only occurred during first compilation.

- [ ] **Step 4: Check backend/API pressure**

If frontend route samples coincide with backend request churn, sample the backend command and recent logs if available:

```bash
ps -ef | rg "uvicorn|tldw_Server_API.app.main"
```

Expected: report notes whether the Next dev process appears to be generating request pressure. Do not broaden into backend leak debugging unless evidence supports it.

### Task 5: Memory-Shape Classification Gate

**Files:**
- Modify: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`

- [ ] **Step 1: Classify with low-impact evidence first**

Use the data already collected:

```text
RSS slope: stable / slow growth / fast growth
CPU: idle / active compilation / busy loop
Open files: stable / growing
TCP connections: stable / growing
Logs: quiet / repeated compile / HMR / request churn / errors
```

Expected: report includes a confidence level and explicitly marks JS heap/native memory as uncollected if no approved intrusive diagnostic ran.

- [ ] **Step 2: Ask user before intrusive diagnostics if needed**

If RSS grows but low-impact evidence cannot separate JS heap from native/tooling memory, ask:

```text
Low-impact sampling shows RSS growth but cannot classify JS heap vs native/tooling memory. May I run a short intrusive diagnostic such as `vmmap <PID> -summary` or `sample <PID> 5` against the Next dev server?
```

Expected: no `vmmap`, `sample`, inspector attach, or diagnostic signal is run until the user approves.

- [ ] **Step 3: If approved, run one intrusive diagnostic**

Use only one narrow diagnostic first:

```bash
vmmap <PID> -summary
```

or:

```bash
sample <PID> 5
```

Expected: report records whether memory looks like JavaScript heap, native/external/tooling allocations, file-backed mappings, or unresolved.

### Task 6: Hypothesis Ranking And Next-Step Decision

**Files:**
- Modify: `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`
- Modify: `<evidence-task-file>`

- [ ] **Step 1: Rank hypotheses**

Update the report with 3 to 5 hypotheses in this format:

```markdown
1. Hypothesis: <specific cause>
   Evidence for: <sample/log/request facts>
   Evidence against: <facts that weaken it>
   Confidence: High|Medium|Low
   Next action: <one concrete follow-up>
```

Expected: each hypothesis cites sampled evidence, not intuition.

- [ ] **Step 2: Select one next step**

Choose one:

```text
minimal reproduction script
route-specific code audit
request-storm cleanup audit
extension background lifecycle audit
dev-server/tooling isolation
```

Expected: report explains why this next step follows from evidence.

- [ ] **Step 3: Close the evidence task**

Update Backlog task with final summary and verification:

```bash
backlog task edit <TASK-ID> \
  --status Done \
  --final-summary "Captured Next dev server memory evidence in Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md. Summary: <one sentence>. Next step: <selected step>. No runtime source files changed." \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 --check-ac 5 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6
```

Expected: task reflects evidence status and no-code-change verification.

- [ ] **Step 4: Verify and commit**

Run:

```bash
git diff --check -- Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md "<evidence-task-file>"
git status --short -- Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md "<evidence-task-file>"
```

Expected: whitespace check passes and status shows only the evidence report plus intended Backlog task metadata.

Commit:

```bash
git add Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md "<evidence-task-file>"
git commit -m "docs: capture next dev memory leak evidence"
```

Expected: commit contains evidence artifacts only.

## Verification Summary For This Plan

- Plan artifact only; no runtime source files are touched by writing this plan.
- Bandit is skipped for the plan task because only Markdown and Backlog metadata change.
- Implementation will run docs-only `git diff --check` for report/task changes and will not claim a fix unless a later fix task exists.
