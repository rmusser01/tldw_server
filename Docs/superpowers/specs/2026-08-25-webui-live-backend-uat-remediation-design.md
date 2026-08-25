# WebUI Live-Backend UAT Remediation Design

**Status:** Revised after pre-implementation review

**Date:** 2026-08-25

**Parent task:** TASK-13124

**Baseline:** `origin/dev@b1d0aed671dcf45bbe4211a9690022c083c99feb`

## Purpose

Remediate the product, development-runtime, and release-gate defects found by
the 2026-08-25 exhaustive WebUI UAT, then execute the complete Playwright
Tier-1, Tier-2, and Tier-3 projects against an isolated real backend. Every
newly confirmed defect found by those tiers must enter the same evidence,
Backlog, test-first, fix, and rerun loop before the effort can close.

This design covers the WebUI, the shared `@tldw/ui` package, frontend E2E
infrastructure, and narrowly related backend contracts only when live evidence
proves the backend is the source of a failure.

## Evidence Baseline

The initial UAT used the exact baseline commit above with Next.js 16.1.4, an
isolated single-user backend, and offline fallback disabled.

- Every one of the 131 inventoried WebUI routes was loaded and inspected.
- The all-pages run completed 215 of 218 scenarios successfully.
- The maintained Tier-1 project passed all 32 scenarios.
- Research Workspace real-backend coverage passed 6, failed 5, and skipped 1.
- Chat Cockpit real-backend coverage passed 12 and failed 3.
- The legacy 17-scenario real-server workflow suite failed all 17 cases, with
  the failures split between stale contracts, unavailable model selection,
  first-run UI, and possible product defects.
- The default webpack development server reached approximately 63 GB RSS and
  497% CPU, then stopped serving pages while backend health remained 200.
- The Turbopack comparison remained responsive, but reached approximately
  11 GB RSS after the same broad compilation/UAT workload. That is useful
  comparative evidence, not proof that Turbopack is free of excessive growth.
- Route transitions produced status-0 `Failed to fetch` request diagnostics and
  backend-unavailable UX while the backend remained healthy.

The initial UAT did not have configured commercial provider credentials.
Generation quality is therefore outside the evidence baseline. Readiness,
failure behavior, truthful skipping, and use of a configured local provider
remain in scope.

## Goals

1. Remove the confirmed causes of false error popups and stale recovery UI.
2. Make the supported default development runtime survive sustained route and
   live-workflow traversal without wedging the WebUI.
3. Correct the Research Workspace initialization race and Prompt Workspace
   responsive layout defect.
4. Restore tracked character/persona live-chat continuity or produce a
   deterministic readiness outcome when no usable model exists.
5. Remove the Settings console warning and align Settings wayfinding coverage
   with the current information architecture.
6. Replace the real-server workflow suite with the smallest honest release
   gate that preserves its unique coverage, instead of retaining stale or
   duplicated scenarios by default.
7. Restore deterministic Kanban route-error recovery coverage.
8. Run the full Tier-1, Tier-2, and Tier-3 inventories against a real backend,
   remediate newly confirmed defects, and rerun until the remaining outcomes
   are passing or explicitly justified skips.

## Non-Goals

- Adding or provisioning commercial LLM credentials.
- Judging generated answer quality without a configured provider.
- Rewriting unrelated frontend architecture or visual design.
- Changing backend API semantics solely to preserve a stale E2E assumption.
- Converting optional-capability absence into a false passing result.
- Running destructive Tier-4 administrator or Tier-5 specialized workflows as
  part of this effort.
- Hiding framework diagnostics by broadly filtering console errors.

## Work Decomposition

The parent task is `TASK-13124`. Initial implementation is divided into nine
reviewable children:

| Task | Responsibility |
| --- | --- |
| TASK-13124.1 | Default development runtime and dev-tools diagnostics |
| TASK-13124.2 | Navigation cancellation and backend-unavailable UX |
| TASK-13124.3 | Research Workspace creation/source-view ordering |
| TASK-13124.4 | Prompt Workspace responsive layout |
| TASK-13124.5 | Character/persona live-chat continuity |
| TASK-13124.6 | Settings form state and wayfinding coverage |
| TASK-13124.7 | Real-server workflow release gate |
| TASK-13124.8 | Kanban forced-error recovery coverage |
| TASK-13124.9 | Full Tier-1–3 execution and iterative remediation |

New Tier-1–3 findings become additional children of `TASK-13124`. A finding is
not folded silently into an existing task unless it has the same root cause,
same affected contract, and same verification boundary.

## Architecture and Behavior

### 1. Development runtime

The implementation will not choose a default bundler solely from the first
UAT comparison. It will run the same clean-worktree route compilation,
navigation-churn, warm-idle, and health probes against webpack and Turbopack.
The harness records initial, post-compilation, and post-idle RSS/CPU plus HTTP
responsiveness for the actual Next server process.

A candidate default passes when it:

- completes the complete route traversal without an empty response or hung
  navigation;
- stays responsive during a 20-minute warm idle and a second critical-route
  traversal;
- remains below 16 GB RSS for this workload; and
- grows by no more than 2 GB during the warm-idle interval after route
  compilation has completed.

These limits are UAT guardrails for the observed machine and workload, not a
general production memory SLA. If exactly one bundler passes, `bun run dev`
uses it and the other remains an explicitly named compatibility command. If
neither passes, the task does not relabel either one stable; it must contain the
failure with an explicit UAT runtime/profile or bounded restart strategy and
record the unresolved upstream/runtime limitation.

Tests and developer documentation will name the evidence-backed default and
explain that the red bottom-left number is the Next.js development-tools
indicator, not an application error counter. The change will not broadly
suppress HMR messages in application error filters.

Three failed attempts to establish a qualifying runtime trigger the required
architecture review rather than a fourth bundler/configuration guess.

### 2. Backend-unavailable event corroboration

The shared request layer remains responsible for recognizing explicit
`AbortError`, `REQUEST_ABORTED`, and abort-message failures as cancellation.
Those failures must not be logged as request errors or emit
`tldw:backend-unreachable`.

Status 0 plus `Failed to fetch` is ambiguous: it can mean a real outage or a
browser-cancelled request. The global WebUI listener will treat the event as a
candidate outage, force the canonical connection check, and only display the
backend-unavailable recovery UI if the connection store settles in a
disconnected/error phase. If the check establishes `CONNECTED`, the candidate
and any visible recovery UI are cleared.

The forced check is single-flight through the existing connection store. A
newer event may replace diagnostic detail, but an older asynchronous result
must not reopen UI after navigation or after a successful check. Retry uses the
same corroboration path. Genuine backend outages continue to surface the
original method, path, and sanitized message.

### 3. Research Workspace readiness ordering

Source saved views are a child resource of a server workspace. The saved-view
hook therefore receives an explicit readiness input in addition to
`workspaceId`. It does not list, mutate, or reconcile views until the parent
workspace is known to exist on the server.

The existing workspace upsert remains the owner of server creation. Successful
upsert publishes a narrow `workspaceExists` readiness state for the active
workspace identity; saved views do not wait for unrelated source/context
projection work. On workspace changes, existence readiness resets before child-
resource effects can run. Existing server workspaces become ready after the
upsert confirms existence and then load views normally. A failed parent upsert
exposes the existing workspace sync/recovery state and does not translate into
a misleading saved-view 404.

All asynchronous results remain guarded by workspace identity and generation,
so a late response for a previous workspace cannot mark the new workspace
ready or overwrite its views.

### 4. Prompt Workspace responsive layout

The Prompt Workspace must fit inside the existing WebLayout content boundary
instead of sizing itself relative to the viewport while global rails are also
present. The fix will identify the first container that owns width/overflow
incorrectly and correct that boundary, rather than adding arbitrary offsets to
individual controls.

The acceptance viewports are 1365x768, 1365x900, the existing narrow/mobile
breakpoint, and a wide desktop. Interactive regions may scroll within their
documented panel, but global navigation, page heading, tabs, form labels,
buttons, and editor fields may not obscure one another. The regression asserts
geometry and interaction, not a screenshot hash.

The focused reproduction must first prove the overlap on the clean current
bundle without a first-run tour or stale test artifact. If it does not, this
task corrects the audit/test setup rather than introducing speculative CSS.

### 5. Character and persona chat continuity

This task begins with request and persistence tracing for the three failing
live cases. The trace must separate:

1. usable-model selection and provider readiness;
2. character/persona identity carried into the request;
3. canonical chat/session creation;
4. streaming completion or truthful terminal error; and
5. persisted history reload.

The task will coordinate with active Character Chat branches before editing
overlapping files. A model-readiness failure is fixed in shared live-test
selection or user-facing readiness behavior, not by weakening session
assertions. A confirmed identity/session defect is fixed at the first boundary
that drops the canonical identifier. Tests cover real observable persistence;
mock call-count assertions alone are insufficient.

The isolated UAT environment will configure the repository's deterministic
OpenAI-compatible mock model service through the real backend for generation
wiring and persistence tests. This validates the WebUI-to-backend-to-provider
path without claiming model-quality coverage. An independently available local
provider may be tested additionally, but is not required for determinism.
No-provider environments must still terminate quickly with an explicit reason.

### 6. Settings form and wayfinding

The parent Ant Form owns initial values. `TldwConnectionSettings` consumes the
form-controlled value and must not add a competing `Form.Item.initialValue` for
`rememberApiKey`. State synchronization must continue to reflect persisted
device-versus-session selection and saving behavior.

The page heading remains `Settings`; `Setup & Recovery` remains the section
heading. Tests will assert that hierarchy and navigation behavior instead of
promoting the section label to a second page-level heading.

### 7. Real-server workflow gate

Before editing the legacy 17 cases, the task maps every scenario to maintained
Tier-1–3 or dedicated real-server coverage. Redundant cases are deleted rather
than repaired. Unique critical behaviors are moved into the maintained tier or
retained in a smaller dedicated live gate. The resulting gate uses current
user-observable contracts:

- dismiss or complete intentional first-run tours through a shared helper;
- use current roles, labels, placeholders, and stable test IDs;
- match action controls narrowly enough that breadcrumbs cannot satisfy them;
- use current route identities and page purposes;
- use current cleanup endpoint payload/identifier contracts; and
- select only an advertised, configured, chat-capable model.

Provider helpers must distinguish a broad model catalog from runnable models.
If no runnable model exists, provider-dependent tests skip before mutation with
a concrete readiness reason. Non-provider-dependent CRUD/navigation cases may
not skip merely because generation is unavailable.

The gate will not use `test.skip()` after partial mutation or catch assertions
to manufacture a pass. Each justified skip is counted and reported. Success is
measured by preserved behavior coverage, not by preserving the number 17.

### 8. Kanban route-error recovery

The existing generic all-pages forced-error mechanism will be traced from
fixture to route boundary. The preferred fix is to register Kanban with that
shared mechanism or its intended route boundary, not to add a Kanban-specific
production hook. The signal must be inert in non-test builds and must not
create a production query parameter that crashes the page.

Under the signal, the route throws inside the intended error boundary and
renders accessible recovery controls. Clearing the signal and invoking recovery
returns to the normal Kanban board. The test asserts the boundary behavior and
recovery, not implementation-specific source text.

## Tier-1–3 UAT Loop

### Environment

- Run from the isolated worktree and branch.
- Start an isolated backend using the repository virtual environment and
  disposable database paths/configuration.
- Start the WebUI with the supported default development runtime.
- Start the deterministic local OpenAI-compatible test service and configure it
  through the real backend for generation-dependent scenarios.
- Set `TLDW_E2E_ALLOW_OFFLINE=0` and explicit backend/WebUI URLs.
- Use a dedicated single-user API key that is not a real secret.
- Record backend health before each project and after any frontend failure.
- Use one Playwright worker initially to keep request ordering and memory
  evidence interpretable. Increase workers only after the serial gate is clean.

### Complete inventory

Before execution, `playwright test --list` records exact Tier-1, Tier-2, and
Tier-3 denominators. All listed scenarios run; grep-selected subsets do not
qualify as a full tier pass.

The inventory also records every scenario that intercepts or mocks an API
request. Such a scenario still counts toward complete tier execution, but is
reported as UI/contract coverage rather than live-backend evidence. Critical
behaviors represented only by mocks receive a focused live-backend counterpart
when the isolated backend supports the capability.

### Failure classification

Every failing or skipped scenario receives exactly one primary classification:

- **Product defect:** current supported behavior is wrong.
- **Gate drift:** the test asserts an obsolete route, label, selector, tour, or
  API contract while the current product behavior is correct.
- **Optional capability unavailable:** the environment lacks a documented
  provider, local engine, binary, device, or service required by the scenario.
- **Environment defect:** the isolated server/runtime cannot provide a required
  baseline capability for reasons outside application behavior.

Evidence includes the failing assertion, screenshot/trace where useful,
frontend console/page errors, failed requests, backend status/log correlation,
and a minimal reproduction. Ambiguous failures remain failures until resolved;
they are not converted to skips by assumption.

### Remediation cycle

For each confirmed defect:

1. Search Backlog for duplicates and inspect coordination locks.
2. Create or attach to one reviewable child task under `TASK-13124`.
3. Record the root-cause hypothesis and evidence.
4. Write the smallest behavior regression and run it to observe the expected
   failure.
5. Implement one root-cause fix.
6. Run the focused regression to green, then its adjacent suite.
7. Record modified files and verification on the child task.
8. Rerun the complete affected Playwright tier.

After three failed implementation attempts for one issue, stop and revisit the
architecture with the user instead of stacking a fourth speculative fix.

### Baseline synchronization

The branch remains pinned while an individual red-green task is in progress.
After the initial child fixes and before the final Tier-1–3 certification, it
is synchronized with the then-current `origin/dev`. Conflicts are resolved
without discarding either workstream, focused tests are rerun, and the final
full-tier results record the exact synchronized commit. This avoids claiming
latest-dev certification against a stale long-running branch.

### Exit criteria

The Tier-1–3 loop exits only when:

- all three complete inventories have executed with the isolated live backend
  and deterministic model service available and offline fallback disabled;
- mocked/intercepted scenarios are identified and are not misreported as live
  backend coverage;
- all product and gate-drift defects discovered in the run are fixed and their
  affected complete tiers have been rerun;
- remaining skips are limited to documented optional capabilities or evidenced
  environmental constraints; and
- there are no unexplained failures, hangs, page errors, or backend-unavailable
  popups while backend health is good.

## Testing Strategy

Every production behavior change follows red-green-refactor. Each regression
names the production mutation it catches and exercises real behavior at the
lowest practical boundary.

Focused gates include:

- frontend dev-config Vitest coverage;
- shared request/background-proxy and WebLayout connection tests;
- Research Workspace saved-view/reconciliation tests;
- Prompt Workspace component plus responsive Playwright coverage;
- Character Chat unit/integration and focused real-server scenarios;
- Settings form/unit and Tier-1 coverage;
- real-server workflow helper/unit and full suite coverage;
- all-pages Kanban forced-error coverage;
- complete Tier-1, Tier-2, and Tier-3 projects;
- the exhaustive route sweep and dedicated Research/Chat real-server suites;
- the development-runtime memory/responsiveness harness with recorded samples;
- frontend typecheck and lint for touched scope; and
- Bandit for any touched Python scope, or an explicit frontend-only skip note.

Optimized build verification is required if a change affects Next configuration,
bundler aliases, or production compilation. Development-only runtime behavior
must also be verified in the development server; a production build alone is
not evidence for HMR or dev memory behavior.

## Safety and Data Isolation

- The user's existing dirty checkout is not modified.
- UAT uses disposable backend databases and disposable content identifiers.
- Tests clean up only the content they create and never use broad deletion.
- API keys in commands and artifacts are fake dedicated test values.
- Logs and screenshots must not contain real provider credentials.
- Existing unrelated worktree changes and untracked files are preserved.

## Documentation and Backlog Records

Each child task records its root cause, plan link, modified files, focused red
and green commands, broader verification, known skips, and final summary. The
parent task links this design and the implementation plan, tracks the final tier
denominators, and closes only after all child tasks and final verification are
complete.

Developer documentation will be updated where the supported default dev
runtime or UAT invocation changes. User-facing documentation changes only if a
product-visible recovery or readiness contract changes.

## Delivery

Work is committed incrementally on `codex/webui-live-uat-remediation`, with one
reviewable commit per child task where dependencies permit. No pull request or
merge is created without a later explicit request. The final handoff includes
the branch, commits, Backlog task state, exact verification results, remaining
capability skips, and any integration risks.
