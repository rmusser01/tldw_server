# Next Dev Server Memory Leak Investigation Design

Backlog task: TASK-422
Date: 2026-05-17

## Problem

The suspected memory leak is in the WebUI/extension frontend development server, not the FastAPI backend. Host process evidence identified the large process as `next-server (v16.1.4)` launched from `.worktrees/buddy-selection-config-ux/apps/tldw-frontend` on port `18021`, with roughly 6 GB RSS. The Python FastAPI backend on port `8000` was much smaller at the time of the first pass. The observed dev server is not on the documented default WebUI port `8080`, so any later browser or request-correlation plan must use the observed `18021` target unless a fresh process check says otherwise.

The workflow that caused the growth is unknown, so the investigation must start with measurement instead of route or component guesses.

Because the target is a live dev-server process, the implementation plan must not hard-code the historical PID, port, or log path. It must first rediscover the active process, its process tree, its current working directory, and the port it is serving.

## Scope

In scope:

- Investigate the running Next dev server process and its immediate frontend development environment.
- Include the Next parent/child process tree, not only the single high-RSS child process.
- Correlate memory growth with idle time, file watching, HMR, route compilation, browser requests, extension background traffic, and API request pressure.
- Inspect frontend and extension code paths only after evidence points to a likely trigger.
- Produce a ranked root-cause hypothesis list and a concrete next investigation or fix plan.

Out of scope for the first pass:

- Fixing frontend code before root cause evidence exists.
- Restarting or killing the 6 GB process before capturing enough evidence from it.
- Treating the FastAPI backend as the primary leak target unless request correlation proves it is being driven by the frontend.
- Broad cleanup in the dirty main checkout.

## Investigation Approach

Use an observation-first sequence.

1. Rediscover the target process from the process table: PID, parent PID, command line, current working directory, child processes, observed port, and whether the command is using default Turbopack dev mode or an explicit Webpack fallback.
2. Capture process telemetry for the active `next-server` process tree every 10 to 15 seconds over a short idle window.
3. Inspect dev-server evidence from terminal output and the `.next/dev/logs/next-development.log` under the process working directory, not the main checkout unless they are the same directory.
4. Correlate memory changes with browser and request activity only after the idle baseline.
5. Run focused workflow probes if the idle baseline does not explain the growth.
6. Move into code audit only after identifying a route, request type, build loop, or extension background path.

This sequence prevents guessing between Next/Turbopack behavior, HMR churn, frontend polling, streaming cleanup, route compilation cache growth, or extension background messaging.

## Measurement Plan

Collect a small evidence table with one row per sample:

- Timestamp.
- PID and command line.
- Current working directory and serving port.
- RSS and CPU.
- Parent process, child processes, and elapsed runtime.
- Open file count when available.
- Listening ports and active connections when available.
- Recent dev-server log lines.
- Current browser route or extension surface if known.
- Backend request activity if the frontend is generating repeated API calls.

The first milestone is a memory slope, not a fix. The result should show whether RSS rises during idle, rises only during route loads, rises with request storms, or stays stable after the initial build cache has settled.

## Triage Branches

If RSS rises while idle with no browser requests:

- Investigate Next/Turbopack dev-server behavior, file watching, workspace symlinks, `.next` cache churn, repeated route compilation, and source-map or HMR loops.
- Check for self-watch loops or generated-output churn in `.next`, test output, docs output, or Playwright output directories before blaming application code.

If RSS rises only after route loads:

- Replay a small route set and isolate the triggering route.
- Inspect that route's server-side imports, dynamic imports, SSR-incompatible browser libraries, heavy shared UI imports, and route compilation output.
- Distinguish one-time route compilation/cache growth from unbounded growth after repeated visits to the same route.

If RSS rises with request storms:

- Identify the client poller, stream, SSE, WebSocket, or retry source.
- Audit cleanup, refetch interval, abort handling, and mount lifecycle in `apps/packages/ui` and the WebUI/extension shell that mounts it.

If RSS rises only with extension sidepanel or background traffic:

- Focus on WXT/background proxy behavior, stream ports, quick-ingest session runtime, extension route mounting, and background request fanout.

If RSS growth appears tied to dev tooling rather than app behavior:

- Compare the observed dev-server mode with a smaller isolation matrix before changing app code: current Turbopack dev mode, explicit Webpack dev mode, and production build/start when feasible.

## Guardrails

- Do not make application code changes during first-pass evidence gathering.
- Do not restart or kill the current high-RSS process until evidence capture is complete.
- Keep user-owned dirty worktree changes untouched.
- If fixes become necessary, create or update a Backlog task before editing runtime code.
- Prefer a dedicated worktree for implementation if the main checkout remains dirty.
- Treat any historical PID, port, or path as stale until the implementation plan rechecks it.

## Expected Output

The investigation pass should produce:

- Confirmed target process and command line.
- Idle baseline memory slope.
- Correlation notes for logs, requests, file watchers, route loads, and extension activity.
- Ranked root-cause hypotheses with evidence for each.
- A durable evidence artifact at `Docs/superpowers/reviews/2026-05-17-next-dev-server-memory-leak-investigation.md`, unless implementation planning identifies a stronger repo-local convention.
- A next step selected from:
  - minimal reproduction script,
  - route-specific code audit,
  - request-storm cleanup audit,
  - extension background lifecycle audit,
  - or dev-server/tooling isolation.

## Verification Strategy

For the design phase:

- Confirm the spec links to `TASK-422`.
- Confirm all accepted design sections are represented.
- Run a docs-only whitespace check.
- Record that Bandit is skipped because this phase changes only Markdown and Backlog metadata.

For the later implementation phase:

- Use `superpowers:writing-plans` to turn this design into a stepwise evidence-gathering plan.
- Add any scripts or test probes under a Backlog task before editing files.
- Use focused verification only after the observed trigger is known.
