# Persona/Buddy Assistant Maturity Roadmap Design

## Goal

Create a staged roadmap for the broader Persona/Buddy assistant system after
the Live Personas visual identity work, Persona Visual Packs foundation, and
visual-pack reuse/library epics have landed.

The roadmap should treat Persona Chat, Persona Live, Persona Buddy, Persona
Garden, wake/voice behavior, MCP persona tools, and future renderer/provider
work as one assistant runtime with multiple surfaces. The first implementation
target is a reliability and UX baseline, not new capability expansion.

## Current Verified Tracker State

Current GitHub tracker state as of 2026-05-10:

1. `#1388` Track Live Personas visual identity and runtime assistant effort:
   closed.
2. `#1389` Add internal `persona_visuals` MCP module and runtime visual
   overrides: closed.
3. `#1428` Track Persona/Buddy visual pack reliability and product hardening:
   closed.
4. `#1449` Epic: Persona/Buddy visual-pack reuse and libraries: closed.
5. `#1497` Research: Persona visual-pack renderer and provider adapter
   evaluation: closed by PR `#1506`.
6. `#635` Tracking: Persona Chat enhancement: open, broad, stale, and useful as
   a source bucket for Persona Chat quality work rather than as a precise
   implementation plan.
7. `#1391` Track character/persona CYOA-VN mode effort: open, but not the
   primary Persona/Buddy assistant tracker. It should remain separate from
   assistant-runtime planning except where compatibility boundaries matter.

The visual-pack roadmap is complete as scoped. Future visual work should start
from the `#1497` evaluation and be split into new targeted issues rather than
reopening the completed visual-pack epic.

Before any tracker mutation, Stage 0 should re-verify live GitHub state and
record the checked date. The issue state above is the design baseline, not a
permanent source of truth.

## Product Framing

Persona/Buddy is one assistant runtime with several surfaces:

1. Persona Chat is the text-first interaction surface for persona behavior,
   role adherence, memory/RAG grounding, and conversation continuity.
2. Persona Live is the session runtime for voice, wake, live connection state,
   approvals, tool status, streaming events, and recovery.
3. Persona Buddy is the visible companion shell and persona identity facet. It
   renders active visual packs when present and falls back to derived buddy
   summary when visuals fail or are absent.
4. Persona Garden is the authoring and configuration surface for profiles,
   assistant defaults, voice/wake setup, visuals, policies, and setup/testing.
5. MCP persona tools are the composability layer. Runtime actions must be
   bounded and scoped; durable actions must remain draft/review/explicit-save
   oriented.

This framing keeps visual packs in context but does not let visual work dominate
the broader assistant roadmap.

## Architecture Principles

1. Persona profile remains the source of truth for identity, defaults, policy,
   voice/wake settings, and durable user-owned assistant configuration.
2. Persona Live remains the runtime source for live state. New live behavior
   should extend `/api/v1/persona/stream` rather than create a second runtime.
3. Persona Buddy consumes resolved persona/live state and must degrade cleanly
   when visuals, live state, or selected persona context are missing.
4. Persona Chat quality work should reuse persona profile/session contracts
   rather than invent a parallel role-playing engine.
5. MCP actions must be scoped, auditable, duration-bounded for runtime effects,
   and explicit-review for durable mutations.
6. VN/CYOA work remains compatible but separate. Do not route Persona/Buddy
   assistant behavior through VN scene state or VN Play runtime assumptions.

## Roadmap Stages

### Stage 0: Current-State Audit

Goal: replace stale umbrella tracking with a concrete map of what the current
Persona/Buddy system already does, where it is brittle, and which flows are
known-good.

Deliverables:

1. Evidence-backed audit document covering Persona Chat, Persona Live, Persona
   Buddy, Persona Garden, wake/voice, MCP persona tools, visual packs, docs,
   unit/integration tests, and E2E coverage.
2. Current contract inventory covering REST schemas, stream payloads, MCP tools,
   DB/state ownership, frontend surfaces, docs, and test coverage for each
   Persona/Buddy flow.
3. Evidence table with at least: flow, user journey, source files, API/runtime
   contracts, tests, linked issues, known failures, severity, and recommended
   next issue.
4. Proposed issue tree that updates or supersedes `#635` with concrete scoped
   issues after the evidence table identifies real gaps.
5. Known-good flow and smoke/E2E candidate checklist for setup, chat, live
   voice, wake, Buddy display, visual fallback, MCP runtime trigger, and
   recovery.
6. Explicit separation note for `#1391` VN/CYOA work so future assistant work
   does not drift into VN runtime assumptions.

Audit questions:

1. Which user journeys are currently supported end-to-end?
2. Which flows have tests but weak product copy or diagnostics?
3. Which flows have UI affordances but no reliable backend/runtime contract?
4. Which flows are mature enough for expansion, and which need stabilization
   first?
5. Which current contracts are server-owned, client-derived, persisted,
   session-scoped, or MCP-triggered?

Acceptance for Stage 0:

1. A maintainer can point to one document that says what exists, what is flaky,
   and what should be next.
2. The document includes a contract inventory and evidence table with file,
   test, issue, and severity references for each audited flow.
3. `#635` has a documented migration recommendation, including which useful
   links/comments should be preserved before it is rewritten or closed.
4. No code is changed except docs/task records unless a separately approved
   follow-up issue exists.

### Stage 1: Reliability and UX Baseline

Goal: make existing Persona/Buddy flows dependable, understandable, and
recoverable before adding new persona intelligence or runtime expansion.

This is the first implementation target.

Scope boundary: Stage 1 only hardens existing reliability diagnostics, recovery,
copy, and test coverage. It should not add new persona capabilities, new
runtime triggers, new renderer behavior, or new Persona Chat intelligence.

Why this comes first:

1. Persona Live, Buddy, wake/voice, setup, MCP, and visual packs already have
   significant runtime surface area.
2. New Persona Chat quality work will be hard to evaluate if setup and live
   state are unreliable.
3. MCP/runtime expansion increases blast radius unless diagnostics, recovery,
   and E2E coverage are already solid.
4. Renderer/provider work depends on the Buddy/runtime fallback path being
   trustworthy.

Likely workstreams:

1. Setup and live-session recovery:
   - verify setup test detours into Live Session still work.
   - reduce redundant connection clicks after recovery detours.
   - ensure failed live tests explain what to fix and where.
2. Wake/voice diagnostics:
   - audit saved `voice_chat_trigger_phrases` and `wake_behavior` behavior.
   - confirm manual wake arming copy matches actual browser/extension limits.
   - expose rejection reasons and live detector state clearly.
3. Buddy shell reliability:
   - confirm selected persona context always wins over stale cached assistant
     selection.
   - keep visual fallback and derived buddy summary stable when active pack
     load/render fails.
   - make dormant/no-buddy states understandable.
4. MCP persona tool diagnostics:
   - ensure `persona_visuals` capability, runtime, durable draft, generation,
     and library failures return actionable reasons.
   - keep transient runtime triggers separate from durable changes in copy,
     logs, and tests.
5. E2E and regression coverage:
   - cover setup to live detour.
   - cover live connection failure and recovery.
   - cover wake activation rejection and success paths with mocked browser
     behavior.
   - cover Buddy fallback with no active pack, broken pack, and stale selected
     persona context.
6. Docs and help copy:
   - align user guides with current local/self-hosted behavior.
   - explain what is session-scoped, profile-saved, transient, or durable.

Acceptance for Stage 1:

1. First-time and returning users can recover from live setup failures without
   leaving the Persona workflow.
2. Wake/voice state, rejections, and browser limitations are visible and
   documented.
3. Buddy never blocks Persona Live controls and never renders stale persona
   identity when fresher context exists.
4. MCP persona tool failures are actionable and scoped.
5. Focused E2E/regression tests protect the highest-risk flows.

### Stage 2: Persona Chat Quality

Goal: make normal persona chat feel coherent, stable, and measurable.

This stage should decompose `#635` into concrete slices. The links currently on
`#635` point toward role-playing/persona behavior evaluations and should inform
the eval design, but they should not substitute for repo-specific acceptance
criteria.

Before adding LLM-as-judge recipes or subjective scoring, Stage 2 should start
with human error analysis on representative traces. Each proposed eval should
name the failure mode it catches, the source traces or fixtures that motivated
it, and the deterministic checks that can run before any optional judge.

Likely workstreams:

1. Role adherence:
   - define how persona instructions layer with system prompts, character-card
     imports, chat context, policies, and tool behavior.
   - add regression tests for persona drift and instruction conflicts.
2. Memory/RAG grounding:
   - clarify when persona memory, conversation memory, notes, media, and RAG
     sources affect persona chat.
   - surface source/context state so users can tell what grounded the response.
3. Conversation continuity:
   - audit session-scoped versus persistent persona behavior.
   - define when persona state carries across conversations and when it does
     not.
4. Persona eval harness:
   - adapt the useful ideas behind role-adherence and role-playing eval links
     into local deterministic fixtures and optional LLM-as-judge evals only
     after representative traces and failure labels exist.
   - avoid vanity metrics; tie each eval to a failure mode.
5. Chat UX:
   - show effective persona, active policy/tool constraints, memory/RAG
     grounding, and status in a compact way.
   - avoid turning Persona Chat into a separate product surface disconnected
     from Persona Garden and Persona Live.

Acceptance for Stage 2:

1. `#635` is replaced by issues that each have clear behavior, tests, and UX
   acceptance criteria.
2. Users can understand what persona is active and what context is grounding
   the response.
3. Role adherence and memory/RAG behavior have repeatable evaluation coverage.
4. Persona Chat shares contracts with Persona Garden/Live instead of creating
   parallel persona state.
5. Optional LLM judges are backed by human-reviewed examples and deterministic
   baseline checks.

### Stage 3: Unified Runtime and MCP Expansion

Goal: let tools, live state, approvals, and Persona Chat participate in a
coherent assistant loop.

Before any new runtime trigger or MCP persona capability is implemented, this
stage requires a policy/security review that documents authorization scope,
duration limits, audit visibility, failure behavior, and the boundary between
transient runtime effects and durable persona mutations.

Likely workstreams:

1. Richer live state contract:
   - define canonical states for listening, thinking, speaking, tool running,
     approval needed, recovery, error, offline, and future tool sub-states.
   - document which states are server-owned, client-derived, or MCP-triggered.
2. Approval and tool-status visibility:
   - make pending tool plans and approval states visible in Persona Live, Buddy,
     and relevant Chat surfaces.
   - preserve destructive-action confirmation boundaries.
3. MCP capability discovery:
   - expand persona tool capability reporting beyond visuals where useful.
   - keep persona-scoped capabilities distinct from global MCP server status.
4. Scoped runtime triggers:
   - add runtime triggers only where they have bounded duration and clear user
     value.
   - do not let runtime triggers mutate saved persona state.
5. Audit/event visibility:
   - make tool-driven persona behavior inspectable enough for debugging without
     flooding the UI.

Acceptance for Stage 3:

1. Live state, tool state, and approval state have one documented contract.
2. Buddy and Live surfaces can explain what the assistant is doing without
   racing each other.
3. MCP-driven behavior is visible, bounded, and auditable.
4. Runtime/MCP expansion has an explicit policy/security review before
   implementation.

### Stage 4: Visual and Renderer Future Work

Goal: extend visual capabilities only after the broader runtime is stable.

This stage starts from
`Docs/Design/2026-05-10-persona-visual-renderer-provider-adapter-evaluation.md`
and should stay behind new targeted issues.

Likely workstreams:

1. Renderer capability registry:
   - expose supported renderers, feature flags, asset roles, manifest versions,
     and setup blockers.
2. Sprite atlas/sprite-sheet V1.1:
   - improve sprite performance inside the current safe raster model.
3. Non-sprite manifest V2:
   - add renderer-specific validation hooks and fallback requirements without
     breaking V1 `sprite_frames`.
4. Feature-gated Live2D spike:
   - use local fixtures, explicit licensing/setup gates, static fallback, and
     no automatic activation.
5. External MCP pack-provider contract:
   - allow providers to submit draft packs, archive previews, or generated
     candidates without runtime code or active-pack mutation.

Acceptance for Stage 4:

1. New renderer work cannot bypass review-before-activation.
2. Unsupported renderer packs fail during preview or capability checks, not at
   live runtime.
3. Optional renderer dependencies degrade without breaking Persona Live.

## Proposed Issue Tree

Create a new GitHub epic:

1. `Epic: Persona/Buddy assistant maturity roadmap`

Create one immediate child issue:

1. `Audit: Persona/Buddy current-state reliability and UX baseline`

Do not create the full child issue tree up front. Stage 0 should produce the
evidence table first, then create or update only the child issues justified by
observed gaps.

Candidate child issues after Stage 0 evidence:

1. `Sprint: Persona Live setup and recovery diagnostics`
2. `Sprint: Persona wake/voice diagnostics and copy alignment`
3. `Sprint: Buddy shell stale-context and fallback hardening`
4. `Sprint: Persona MCP tool diagnostics and audit visibility`
5. `Sprint: Persona Live/Buddy E2E reliability coverage`
6. `Epic: Persona Chat quality and evaluation`
7. `Research: Persona role-adherence and memory/RAG eval harness`
8. `Sprint: Persona Chat effective-context UX`
9. `Epic: Unified Persona runtime and MCP expansion`
10. `Research: Persona live-state contract v2`
11. `Epic: Persona visual renderer/provider follow-ups`

Issue hygiene:

1. Keep `#1391` open only for VN/CYOA work unless it is intentionally split.
2. Before rewriting or closing `#635`, preserve its useful reference links,
   comments, and role-playing/persona-eval notes in the Persona Chat quality
   issue or audit report.
3. Do not reopen `#1449`; future visual work should use new targeted issues.

## Testing Strategy

Stage 0 is documentation and issue decomposition only.

Stage 1 should prioritize:

1. Focused frontend tests around `sidepanel-persona.tsx`, Persona Garden setup
   components, Buddy shell, incoming payload hooks, and wake/voice controls.
2. Backend tests for `/api/v1/persona/stream` wake activation/rejection and MCP
   persona tool diagnostics.
3. E2E tests for mocked setup-to-live recovery, live failure recovery, Buddy
   fallback, and visual state override behavior.
4. Documentation checks through `git diff --check` for docs-only slices.

Stage 2 should add:

1. Deterministic persona prompt/contract tests where possible.
2. Human-labeled representative traces before optional subjective judges.
3. Optional eval recipes for role-adherence and grounding behavior only after
   failure modes are identified and deterministic baselines exist.

Stage 3 should add:

1. Contract tests for live state payloads and MCP runtime events.
2. Regression tests around approval/tool status propagation.

Stage 4 should add:

1. Renderer validation tests before runtime rendering tests.
2. Fixture-based renderer tests, never external-provider-dependent baseline
   tests.

## Risks and Mitigations

1. Risk: The roadmap becomes a vague mega-epic.
   - Mitigation: Stage 0 produces issue slices before implementation.
2. Risk: Persona Chat quality work invents a parallel persona engine.
   - Mitigation: keep persona profile/session contracts authoritative.
3. Risk: Runtime/MCP expansion bypasses user control.
   - Mitigation: bounded runtime effects and explicit review for durable
     changes.
4. Risk: Visual work distracts from reliability.
   - Mitigation: keep renderer/provider implementation in Stage 4.
5. Risk: VN/CYOA tracker work bleeds into assistant runtime.
   - Mitigation: keep `#1391` compatibility-only from this roadmap unless a
     future issue explicitly bridges the two.
6. Risk: The issue tree becomes an unverified issue flood.
   - Mitigation: create only the epic and Stage 0 audit first; let audit
     evidence justify the remaining child issues.

## First Concrete Next Step

Create and complete the Stage 0 audit issue:

`Audit: Persona/Buddy current-state reliability and UX baseline`

The audit should inspect the current code, docs, and tests for:

1. Persona Chat.
2. Persona Live.
3. Persona Buddy shell.
4. Persona Garden setup/profile/defaults/policies/visuals.
5. Wake/voice behavior.
6. MCP persona tools.
7. Visual-pack fallback and diagnostics.
8. Current E2E and unit coverage.

The audit should produce a written report and a proposed issue tree for Stage 1.
Only after that audit should implementation begin on reliability/UX fixes.

Minimum report shape:

1. Contract inventory table.
2. Evidence table with severity and source links.
3. Known-good flow checklist and smoke/E2E candidates.
4. `#635` migration recommendation with preserved references.
5. Stage 1 issue recommendations limited to reliability diagnostics and UX
   hardening.
