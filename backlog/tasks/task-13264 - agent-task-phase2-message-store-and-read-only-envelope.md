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
- [ ] The agent_task executor resolves the ref at dispatch (in memory
      only); an unresolvable ref is an honest failed run with a precise
      reason
- [ ] The read-only tool envelope executes agent_task runs; a
      side-effecting tool call terminates the run as `approval_required`
      with the attempted call recorded and the request carried in the
      result notification
- [ ] The consumer's `family_not_wired_for_execution:agent_task` skip is
      lifted and the execution-certification matrix reports agent_task
      execute availability only when store + envelope are operational
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

- Slices 1–2 (store + authoring wiring) landed first; slices 3–4 (executor,
  envelope, certification) follow on this task.

ADR required: covered by chatbook ADR-184 (accepted); no new server ADR.
