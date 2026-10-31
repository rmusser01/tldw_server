# C03 Persona Recipe report visibility

Associated task: TASK-13260.278.18.83.45. Root recorded the actual failure and approved this report-only unit before edits under the human's standing all-43 fix/validation authorization.

## Evidence and scope

The actual advertised offline `benign_basic` Recipe control validated and completed once, but Recipes showed only `completed` and empty model recommendation slots. The backend already emits Persona summary, per-target cases, selected trajectories, trace references, run ID, worker/job metadata, and dataset snapshot/hash. RecipesTab currently reads only generic recommendation/candidate/failure report branches. This is a missing report consumer, not a scoring or worker defect.

Own only `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx`, a new mounted report test module beside it, and this plan. Preserve existing services, backend, ordinary recipe branches, form behavior, polling, errors, and auth/owner boundaries. Display only known existing public report fields, as React text, without raw metadata/config/trace dumps or fabricated counts. Select the Persona branch by the returned run's recipe ID. No runtime/browser/provider calls, types/build, dependencies, task edits, or Git actions.

Pre-unit RecipesTab SHA256: `86988f3f9eb38836d9c1f1e2045782bef990a13570027287c9604e3b900c395e`; exact bytes retained in memory only.

## ADR assessment

No new ADR. Reuse [ADR-014](../ADR/014-evaluations-openai-compatible-schemas.md) for existing public response fields and [ADR-015](../ADR/015-evaluations-existing-evaluator-integration.md) for evaluator/consumer separation. This renders existing authoritative results with existing shared UI primitives; no API, persistence, scoring, transport, security, or module-boundary rule changes.

## Stage 1: Qualify the missing report consumer

**Goal**: Mount the real RecipesTab, services/hooks, QueryClient and UI primitives against external response fixtures.
**Success Criteria**: Unchanged source reaches a completed Persona report, then fails meaningful summary/target/trace/lineage assertions for the demonstrated missing view; fixture/setup failures remain separate.
**Tests**: Completed report, absent fields/zero counts, per-case selected/fallback result, supplied opaque IDs and escaped text, lifecycle/error recovery, ordinary report regression.
**Status**: Complete

## Stage 2: Render existing Persona results

**Goal**: Minimal local Persona report branch showing summary, target results/cases, trace references, and present run/worker/dataset lineage.
**Success Criteria**: Causal tests pass; no empty model recommendation cards for Persona, no raw metadata/secret dump, and ordinary recommendation/result branches retained.
**Tests**: New mounted bodies plus genuinely affected RecipesTab suites using normal UI/Web Vitest configurations.
**Status**: Complete

## Stage 3: Freeze and hand off

**Goal**: Scope lint/syntax, complete self-review, independent review and root-owned integration/live qualification.
**Success Criteria**: No new touched lint findings; source/tests frozen with hashes, independent review clear, root combined types/build and actual existing Recipe report visibility accepted.
**Tests**: Node20.19.5/Vitest4.0.18, NODE_OPTIONS8192MB, outer600, normal configured plugins; scoped ESLint baseline/current and TSX syntax. Bandit N/A (TypeScript/Markdown only). Root owns types/build/live; no recipe rerun by this agent.
**Status**: In Progress

## Bounded attempt ledger

Maximum three attempts per issue. Record each actual test command and causal versus fixture outcome. No broad passing replay. Read-only static setup corrections are not defect reds: an initial ESLint invocation ignored the external shared path; corrected by using the same normal Web flat-config rules from repository-root cwd. An initial report parser encountered the existing Next Pages-directory notice; corrected parsing the emitted final JSON line without changing rules/source.

### Attempt 1 — fixture interaction failure

Unchanged production UI command: new module 8 failed / 1 passed, Vitest14.24s / outer14.499s. Failures reached missing report sections, but failure DOM exposed generic textbox indexing changed the inherited Candidate models/Judge provider fields instead of the Advanced JSON editors. This is not accepted as qualified causal red; external fixture returned a completed report despite the wrong request shape. Production unchanged. Correct only to the two exact unnamed JSON editor textboxes and add an exact submitted-payload guard before every report assertion. Attempt2 will qualify that correction and missing report consumer.

### Attempt 2 — clean causal report red

Normal configured UI new module: 8 missing-view failures / 1 ordinary report pass, Vitest15.10s / outer15.320s, no setup/fixture errors or skips. All nine launches passed the exact SCENARIO/CONFIG submitted-payload guard. Missing summary/targets/traces/lineage/unavailable text is the causal failure. Production remained byte-identical to the retained baseline during both commands. Added the minimal local Persona renderer afterward; ordinary model result branches and all request/lifecycle/polling code remain unchanged. Attempt3 will run current new+genuinely affected suites in UI and Web with normal configs.

### Attempt 3 — current green and affected qualification

Same production/test bytes, normal configured UI then Web commands, new9 + existing RecipesTab.launch24 per config: UI33passed / 2files, Vitest37.76s / outer37.963s; Web33passed / 2files, Vitest36.98s / outer37.222s. No failures, skips or fixture errors. Existing Node20.19.5/Vitest4.0.18; NODE_OPTIONS `--max-old-space-size=8192`; outer600; maxWorkers2; each new case30s (below300 ceiling), inherited launch60s. This completes the three-attempt path (initial fixture failure, corrected causal red, current UI/Web qualification); no further unchanged replay.

## Static qualification and self-review

- TSX syntax parser PASS on both touched files. Root owns full UI/Web types and build; neither was run here.
- Scoped ESLint uses the normal Web flat-config rules without overrides. Production baseline/current0errors/35warnings:33inherited `no-explicit-any`,2inherited `exhaustive-deps`. Exact rule/severity/message/source-line fingerprints match after normalizing embedded shifted line references. New test0errors/0warnings. No new findings. Bandit N/A for TSX/Markdown.
- In-memory before/current reconstruction removes exactly the two-line returned-run guard,116-line local renderer block and two-line Persona ternary branch; it reproduces all pre-unit source bytes exactly. No form, payload, hook, service, polling, backend or ordinary result bytes changed.
- Self-review read every changed hunk, all nine complete new test bodies, connected typed RecipeRunRecord/RecipeRunReport, actual backend Persona summary/target/case/trace producers and service public report projection. Display only known existing fields as React text; no raw config, jobs errors, selected-candidate blobs or trace bodies rendered. Finite nonnegative numeric counts preserve0; absent fields are not replaced with fabricated values. Ordinary recommendation/candidate/failure controls and existing launch/error/apply branches pass.
- Mounted tests exercise real Antd/Card/Alert/JsonEditor, QueryClient, services, query hooks and Zustand store. Only external apiSend responses, translation and notification boundaries are fixtures. Every launch verifies exact supported JSON input/config payload. Report fixtures intentionally exercise two case outcomes and independent counters; they qualify existing result display, not evaluator computation, provider quality, HTTP/auth/owner enforcement or worker execution. Pending malformed/absent data, actual read-error→Try-again recovery without resubmission, escaped opaque trace IDs, secret extra-field exclusion and recipe-change retirement are covered.

## Frozen handoff

Production `apps/packages/ui/src/components/Option/Evaluations/tabs/RecipesTab.tsx` SHA256 `e2b8f3fdda2b319b6eabc395418604234f10a37480f0ebc485fdb3a7945cd538`.

New test `apps/packages/ui/src/components/Option/Evaluations/tabs/__tests__/RecipesTab.persona-report.test.tsx` SHA256 `bb8898348cd0fbf756dd700f3bc10e3ad75c9b98225ac959c34bd265d724e7d3`.

Source/tests frozen after current green. Independent review and root combined types/build/actual existing completed report visibility remain open. No runtime/browser/profile/model/provider/API requests, task edits, Git actions, proof files or prior passing-unit replay performed. The actual Recipe run is not rerun by this agent. Stage3 remains In Progress until root-owned gates are satisfied.
