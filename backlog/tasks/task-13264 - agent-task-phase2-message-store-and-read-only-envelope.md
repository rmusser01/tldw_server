---
id: TASK-13264
title: 'agent_task phase 2: encrypted message store + read-only tool envelope'
status: In Progress
assignee:
  - '@robert'
created_date: '2026-09-27 00:00'
updated_date: '2026-09-27 00:00'
labels:
  - scheduled-tasks
  - automation
  - agent-task
  - security
dependencies: []
---

## Description (the why)

Implements chatbook ADR-184 (accepted 2026-09-27, rulings 1A + 2A) — the
phase-2 design from issue #2805: `agent_task` automations become executable
on the server. Phase 1 (ADR-077) left them unwired for two deliberate
reasons: `input.message` is redacted at rest (no usable prompt survives
persistence), and server-side tool use has no approval enforcement.

**Ruling 1A:** the raw message lives ONLY in a separate encrypted,
owner-scoped, TTL-bounded message store keyed by `message_ref`; the
scheduled-tasks DBs keep their `metadata_only` at-rest posture (backups and
exports never carry raw prompts). **Ruling 2A:** agent_task executes with a
read-only tool envelope; side-effecting calls terminate as a new
`approval_required` outcome carried back through the notification channel —
full queued escalation is a later step with its own task.

## Acceptance Criteria (the what)

- [x] An encrypted per-owner message store exists (separate DB file from
      the scheduled-tasks DBs) with store/resolve/purge, owner scoping,
      TTL with refresh-on-resolve, and corrupt-blob/wrong-key tolerance
      (unresolvable ref → None, never a leak)
- [x] agent_task preview creation persists the raw message to the store
      BEFORE the preview row, keyed by the redaction's `message_ref`; the
      persisted row carries only `metadata_only` metadata (pinned: raw
      sentinel absent from every response/repository surface)
- [x] A store failure (including unconfigured key) refuses authoring with
      `message_store_unavailable` (503) and persists nothing — fail-closed,
      same discipline as the execution-target bound
- [x] recurring_question previews never touch the store
- [x] The agent_task executor resolves the ref at dispatch (in memory
      only); an unresolvable ref is an honest failed run with a precise
      reason (no ref / unresolvable ref are distinct reasons)
- [x] The `approval_required` terminal outcome exists end-to-end (run
      status + `automation_run_approval_required` notification kind +
      consumer return); tool-requesting definitions terminate with it
      (error `tools_require_read_only_envelope`) instead of the phase-1
      skip. Runtime envelope execution (actually running side-effect-free
      tools in-loop) remains the open item
- [x] The consumer's `family_not_wired_for_execution:agent_task` skip is
      unreachable for agent_task (executor registered; dispatch sits
      behind the deployment certification gate by DESIGN — see notes)
- [ ] Env/config documented: `AUTOMATION_MESSAGE_ENCRYPTION_KEY` (falls
      back to `BYOK_ENCRYPTION_KEY`/secondary for rotation)

## Implementation Plan (the how)

1. `AutomationMessageStore` (core/Scheduled_Tasks/automation_message_store.py):
   per-owner sqlite file, `Security.crypto` envelope, dedicated key with
   BYOK fallback, fixed 30-day TTL refreshed on resolve (reference-liveness
   reclaim is a follow-up)
2. Service wiring: injectable store; agent_task preview-create writes
   store-then-row; refusal on failure; error-map entry
3. Executor: resolve ref → read-only envelope run; `approval_required`
   outcome; register executor; lift consumer skip; certification truth
4. Tests at each layer

## Implementation Notes

- Slices 1–2 (store + authoring wiring): merged via #3039.
- Slice 3 (executor + approval_required vocabulary): agent_task executes
  generation-only with its message resolved from the store at dispatch
  (missing ref and unresolvable ref are distinct failure reasons);
  tool-requesting definitions terminate `approval_required` with a
  pass-back notification instead of the phase-1 skip; `timed_out` added to
  the run-status Literal (latent drift — the consumer has written it since
  the TASK-13039 era but response validation would have rejected it).
- **Certification-gate finding (important):** agent_task dispatch sits
  behind the Phase-4D deployment-certification program
  (`execution_certification.py`): `current_agent_execution_stack_ready()`
  is hardcoded False pending a reviewed execution stack, and certification
  resolves draft_only at best without evidence receipts (build SHA,
  isolation profile). This is a deliberate security program — enabling
  agent_task execution in a deployment is an owner/operator action
  (evidence + reviewed stack install), not a code flip. The executor now
  exists so certified deployments dispatch immediately.
- Client follow-up needed: chatbook's `NotificationKind` Literal must add
  `automation_run_approval_required` (same latent-feed-parse class fixed
  for the four phase-1 kinds in chatbook PR #2215).
- Store-native re-encryption for BYOK-key rotation: follow-up (rotation
  note recorded in-code).

ADR required: covered by chatbook ADR-184 (accepted); no new server ADR.
