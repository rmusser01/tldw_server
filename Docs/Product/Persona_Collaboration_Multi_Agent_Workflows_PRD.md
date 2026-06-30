# Persona Collaboration / Multi-agent Workflows PRD

Status: Draft

Owner: Persona module / workflow orchestration

Tracking: #1926, split from #1902

Backlog: TASK-473

## Summary

Define the future Persona Collaboration / Multi-agent Workflows product contract: how multiple Personas can participate in one user-owned collaboration, coordinate plans, produce shared artifacts, request review, and execute tools without inheriting each other's memory, policy, scopes, or credentials.

The current Persona system is intentionally single-Persona per live session. It already provides Persona profiles, state docs, memory controls, scope and policy rules, live websocket turns, pending tool plans, confirmation flows, transcript export, and audit-friendly session persistence. Collaboration should build an orchestration layer around those primitives rather than replacing the single-Persona runtime.

This PRD does not make multi-agent collaboration a current Persona module completion blocker. It documents the final future slice moved out of `Docs/Product/Persona_Agent_Design.md`.

## Problem

Users may eventually want multiple Personas to work together: a researcher Persona gathers sources, an editor Persona critiques a draft, a cautious reviewer Persona checks policy and evidence, and a project Persona tracks decisions. That is useful, but it changes the product contract:

- more than one Persona may propose work,
- more than one policy/memory/scope boundary applies,
- shared artifacts need provenance from each participant,
- concurrent tool plans can conflict,
- users need review and arbitration instead of an opaque agent swarm.

If collaboration is added by letting Personas freely message or invoke each other, the system could leak Persona memory, combine permissions unintentionally, or execute tools without a clear accountable actor. The design needs a collaboration envelope with explicit participants, roles, turn policy, shared context, review gates, conflict handling, and audit.

## Goals

- Let a user create a collaboration with multiple Persona participants.
- Preserve each Persona's independent identity, state docs, memory, scopes, policies, tools, and provenance.
- Define participant roles such as lead, researcher, reviewer, critic, synthesizer, and observer.
- Support user-visible turn-taking and arbitration rather than uncontrolled concurrent chatter.
- Produce shared artifacts with per-Persona contribution provenance.
- Require review gates before external delivery, durable mutation, or privileged tool use.
- Evaluate policy, scope, memory, and tool access per Persona and per action.
- Provide audit records for participant messages, plan proposals, decisions, approvals, and tool outcomes.
- Keep V1 backend/contract-oriented and compatible with later UI.

## Non-goals

- No Buddy animation, Buddy runtime, or visual-pack work.
- No design-system backlog work.
- No implementation in this PRD slice.
- No replacement for single-Persona Garden live sessions.
- No implicit tool, memory, scope, or credential sharing between Personas.
- No autonomous scheduled collaboration; scheduled triggers belong to `Persona_Scheduled_Work_PRD.md`.
- No full Persona Tool Administration; tool grants and effective policy stay with `Persona_Tool_Administration_PRD.md`.
- No broad cross-app personalization memory writes; those belong to `Personalization_Memory_Layer_PRD.md`.
- No open-ended agent swarm or background self-improvement loop.
- No cross-user collaboration in V1.

## Current Contract Evidence

- `Docs/Product/Persona_Agent_Design.md` lists multi-agent and multi-Persona collaboration as future scope, separate from the implemented Persona Garden/live-session foundation.
- `tldw_Server_API/app/core/Persona/README.md` documents the current single-Persona feature set: profile catalog, live sessions, state docs, policy/scope rules, exemplars, websocket interaction, and policy-enforced tool plans.
- `tldw_Server_API/app/core/Persona/session_manager.py` tracks one `persona_id` per session, bounded turns, pending plans, preferences, and per-session ownership checks.
- `tldw_Server_API/app/api/v1/schemas/persona.py` defines `PersonaSessionRequest`, `PersonaSessionResponse`, `PersonaSessionSummary`, `PersonaSessionDetail`, and transcript export shapes for single-Persona sessions.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py` and `chacha/persona_state_store.py` persist Persona profiles, sessions, scope rules, policy rules, and Persona memory entries.
- `tldw_Server_API/app/api/v1/endpoints/persona.py` applies Persona policy evaluation before live tool calls and persists Persona session preferences and turns.
- `Docs/Product/Persona_Scheduled_Work_PRD.md` covers recurring background work and review-gated drafts, explicitly excluding multi-agent collaboration.
- `Docs/Product/Persona_Tool_Administration_PRD.md` keeps MCP Hub as the canonical tool governance source and defines Persona-specific effective access without transitive grants.
- `Docs/Product/Personalization_Memory_Layer_PRD.md` defines user-owned personalization memory boundaries and explicitly excludes multi-agent workflow behavior.
- `Docs/Product/Persona_Backed_Chat_Startup_PRD.md` and `Docs/Product/Workspace_Persona_Defaults_PRD.md` both exclude multi-Persona collaboration from their V1 scopes.

## Product Shape

V1 should model collaboration as an explicit user-owned object:

```json
{
  "id": "collaboration-id",
  "owner_user_id": "user-id",
  "title": "Literature review critique",
  "status": "active",
  "mode": "facilitated_turns",
  "participants": [
    {
      "participant_id": "participant-1",
      "persona_id": "researcher-persona",
      "role": "researcher",
      "memory_mode": "read_only",
      "tool_access_mode": "persona_effective_policy",
      "can_initiate_tools": true
    },
    {
      "participant_id": "participant-2",
      "persona_id": "reviewer-persona",
      "role": "reviewer",
      "memory_mode": "read_only",
      "tool_access_mode": "persona_effective_policy",
      "can_initiate_tools": false
    }
  ],
  "review_policy": {
    "user_review_required": true,
    "external_delivery_allowed": false
  }
}
```

Each participant references an existing Persona. The collaboration owns coordination state, shared artifact references, and audit events. It does not copy Persona profile content or grant one Persona access to another Persona's private memory or tools.

## Collaboration Modes

V1 should support one conservative mode:

- `facilitated_turns`: the orchestrator chooses one participant at a time, based on a visible plan and user-approved role order.

Later modes may include:

- `panel_review`: each participant independently comments on a shared artifact, then a lead summarizes.
- `debate`: two or more participants produce bounded arguments before synthesis.
- `parallel_research`: participants receive separate tasks and produce independent drafts for review.

Avoid a free-running mode in V1. Even in later modes, concurrency should be bounded by participant count, turn count, tool-call count, and artifact size.

## Authority And Isolation

Collaboration authority is the intersection of user ownership, collaboration settings, participant Persona policy, session policy, MCP Hub effective tool access, and explicit review gates.

Rules:

- A participant can only use its own Persona state docs, Persona memory, policies, scopes, and effective tool access.
- One Persona cannot inherit another Persona's tools, credentials, memory mode, or scopes.
- Shared context must be written into collaboration-owned artifacts or visible messages before another Persona can use it.
- Any tool plan records the proposing participant and evaluates that participant's policy.
- Any final action records the approving user and the participant whose policy authorized the action.
- `read_write` memory for any participant must be explicit and visible. It should not be inherited from the collaboration by default.

## Shared Context And Artifacts

Collaboration should communicate through explicit shared artifacts:

- collaboration brief,
- participant task assignments,
- participant messages,
- proposed plan,
- draft output,
- critique/review notes,
- final synthesis,
- decision log.

Shared artifacts must include provenance:

- collaboration ID,
- participant ID,
- Persona ID,
- source turn IDs,
- source tool-call IDs,
- review decision IDs,
- timestamps,
- redaction/safety flags where applicable.

Shared artifacts are not Persona memory by default. Promoting a shared artifact into Persona memory or personalization memory requires the relevant memory contract and explicit mode.

## Orchestration Model

The orchestrator should be a deterministic service, not another hidden Persona:

1. User creates collaboration and selects participants.
2. Service validates Persona ownership and availability.
3. Service builds a collaboration plan with participant roles, turn order, budgets, and review policy.
4. User reviews or accepts the plan.
5. Service runs one participant turn at a time or bounded parallel read-only turns where supported.
6. Participant outputs become visible collaboration events/artifacts.
7. Tool plans require the same confirmation and policy checks as single-Persona runtime.
8. Final synthesis is user-reviewable before external delivery or durable mutation.

Implementation can reuse Jobs for durable long-running collaboration runs. Live websocket interaction can be added later, but V1 should not depend on holding a browser socket open for a full collaboration.

## Turn And Conflict Policy

V1 should define strict budgets:

- max participants per collaboration,
- max turns per participant,
- max total turns,
- max tool plans per run,
- max concurrent tool calls,
- max artifact size,
- max wall-clock duration.

Conflict handling:

- Duplicate tool plans should be deduplicated or presented as alternatives.
- Conflicting proposed mutations should require user arbitration.
- Policy denials from one Persona do not block another Persona unless the action is shared or final.
- If a lead Persona and reviewer Persona disagree, the collaboration should present both positions and ask the user to choose.

## Memory Rules

Memory handling must be explicit:

- Default participant memory mode is `read_only`.
- A participant can retrieve only its own Persona memory and allowed personalization context.
- Collaboration shared artifacts can be referenced by all participants after they are visible in the collaboration event log.
- `read_write` Persona memory writes are participant-specific and require explicit mode.
- Broad personalization memory candidates follow `Personalization_Memory_Layer_PRD.md`.
- No participant may write to another participant's Persona memory.

## Tool Rules

Tool access follows `Persona_Tool_Administration_PRD.md`:

- Effective access is resolved per participant Persona.
- Tool calls record participant ID and Persona ID.
- Runtime approval is confirmation only and cannot broaden access.
- Missing credentials or missing grants are hard denials.
- Shared final actions require review if any participant's policy or collaboration review policy requires it.
- Revocation during a collaboration should stop future tool calls and emit safe notices.

## API Direction

Preferred V1 backend contracts:

- `POST /api/v1/persona/collaborations`
  - create collaboration with participant references, roles, review policy, and initial brief.
- `GET /api/v1/persona/collaborations`
  - list owner-scoped collaborations.
- `GET /api/v1/persona/collaborations/{collaboration_id}`
  - detail, participants, status, budgets, artifacts, and recent events.
- `POST /api/v1/persona/collaborations/{collaboration_id}/plan`
  - build or preview orchestration plan.
- `POST /api/v1/persona/collaborations/{collaboration_id}/runs`
  - start a run, preferably backed by Jobs for durability.
- `POST /api/v1/persona/collaborations/{collaboration_id}/review-decisions`
  - approve, reject, edit, arbitrate, or request more work.
- `GET /api/v1/persona/collaborations/{collaboration_id}/audit`
  - collaboration-scoped audit feed.

The implementation plan may choose a narrower read-only or draft-only first slice, but it should preserve these object boundaries.

## Data Model Direction

Preferred records:

- `persona_collaborations`
  - owner, title, status, mode, review policy, budgets, created/updated timestamps.
- `persona_collaboration_participants`
  - collaboration ID, participant ID, Persona ID, role, memory mode, tool access mode, ordering, status.
- `persona_collaboration_runs`
  - run status, job ID, plan version, started/finished timestamps, failure summary.
- `persona_collaboration_events`
  - typed event log for messages, plans, tool calls, approvals, denials, notices.
- `persona_collaboration_artifacts`
  - shared artifacts with provenance and lifecycle status.
- `persona_collaboration_reviews`
  - review decisions, reviewer user, target artifact/action, decision, rationale.

All Persona references should be reference-backed. Do not snapshot Persona profile content except for narrow audit summaries needed to explain historical decisions.

## UI Direction

This PRD is backend/contract-first. Future UI can add:

- collaboration setup wizard,
- participant role table,
- visible turn plan,
- shared artifact lane,
- participant contribution timeline,
- policy/tool/memory status per participant,
- review and arbitration queue,
- audit drawer.

The UI should not look like multiple independent chat windows fighting for attention. It should make the orchestration plan and current review decision obvious.

## Staged Delivery

### Stage 1: Contract And Storage Design

Goal: define collaboration objects, participant references, event log, budgets, and review semantics.

Deliverables:

- API schema proposal.
- DB migration design.
- Role and mode taxonomy.
- Review decision taxonomy.
- Budget and lifecycle rules.

### Stage 2: Read-only/Draft Collaboration Shell

Goal: create collaborations, participants, plans, and shared artifacts without autonomous execution.

Deliverables:

- Collaboration CRUD.
- Participant validation.
- Plan preview.
- Manual artifact/event append for test fixtures.
- Audit feed.

### Stage 3: Single-step Facilitated Turns

Goal: run one participant turn at a time through existing Persona prompt/context helpers.

Deliverables:

- Participant turn execution with Persona context isolation.
- Visible collaboration event output.
- Tests for ownership, missing Persona, deleted Persona, and memory isolation.
- No tool execution in the first turn-execution slice unless explicitly approved.

### Stage 4: Tool Plans And Review Gates

Goal: allow participant tool plans using existing Persona policy and MCP Hub effective access.

Deliverables:

- Tool-plan proposal per participant.
- Confirmation and review decisions.
- Policy/approval/credential failure handling.
- Revocation handling.
- Tests for no transitive grants.

### Stage 5: Jobs-backed Runs

Goal: support durable collaboration runs with retry, status, cancellation, and quotas.

Deliverables:

- Jobs integration for bounded runs.
- Run status and failure summaries.
- Idempotency keys.
- Cancellation and stale-run cleanup.
- Admin/owner visibility.

### Stage 6: Synthesis And Export

Goal: produce final shared artifacts with review and export controls.

Deliverables:

- Synthesis artifact generation.
- Provenance-rich final output.
- Export/delivery review gate.
- Transcript/export redaction rules.

## Validation Plan

- API tests for collaboration create/list/detail, participant validation, and owner scoping.
- Schema tests for budgets, roles, modes, review decisions, and event types.
- Runtime tests proving each participant uses only its own Persona context, memory mode, policy, scopes, and tool access.
- Tool tests proving no transitive grant, missing credential hard denial, approval-required behavior, and revocation behavior.
- Jobs tests for idempotency, retry, cancellation, stale-run cleanup, and owner quotas when Jobs-backed runs are added.
- Privacy tests proving hidden Persona state, raw tool output, secrets, and non-visible memory do not leak into shared artifacts.
- Export tests for provenance and redaction.
- Bandit on touched backend implementation paths when code is added.

## Risks And Mitigations

- Risk: multi-agent collaboration becomes an opaque agent swarm.
  Mitigation: V1 uses facilitated turns, visible plans, strict budgets, and user review gates.

- Risk: Personas inherit each other's permissions or memory.
  Mitigation: evaluate policy/memory/tool access per participant and require shared artifacts for cross-participant context.

- Risk: concurrent tools mutate the same resource.
  Mitigation: bounded concurrency, conflict detection, and user arbitration for mutations.

- Risk: collaboration duplicates scheduled work.
  Mitigation: V1 is user-started; scheduled triggers stay in the scheduled work PRD.

- Risk: collaboration creates hidden long-term memory.
  Mitigation: shared artifacts are not memory by default; memory promotion follows explicit Persona or personalization memory contracts.

- Risk: audit becomes unreadable.
  Mitigation: collaboration event log uses typed events, participant IDs, artifact IDs, and compact provenance.

## Acceptance Criteria

- Collaborations are explicit user-owned objects with reference-backed Persona participants.
- Participant role, memory mode, tool access mode, budgets, and review policy are visible.
- Each participant's policy, scopes, memory, and tool access are evaluated independently.
- No Persona can write to or read private memory from another Persona.
- Shared artifacts carry participant-level provenance and are not memory by default.
- Tool plans and privileged actions require existing Persona/MCP Hub policy checks and review gates.
- Runtime and Jobs-backed execution fail closed on revoked access, missing credentials, deleted Personas, or policy denials.
- Audit, review, export, and redaction requirements are defined and tested.

## Open Questions

- Should V1 start as draft-only collaboration planning before any Persona-generated turns?
- What participant roles should be built in versus user-defined?
- Should collaboration runs reuse Persona session rows, or should they stay in separate collaboration tables with links to participant turns?
- Should a lead Persona be required, or can the user be the only facilitator in V1?
- How should collaboration artifacts integrate with Workspaces once Workspace Persona Defaults are implemented?
