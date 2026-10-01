---
id: TASK-13245
title: Persist Workspace Persona opt-out and conversation provenance
status: In Progress
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-27 18:17'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13244
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
  - Docs/Design/2026-09-13-persona-workspace-choice-provenance-design.md
  - Docs/Design/2026-09-27-persona-workspace-strict-startup-refresh.md
  - >-
    Docs/superpowers/plans/2026-09-27-persona-workspace-strict-startup-implementation-plan.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Distinguish unset defaults from an explicit None choice, preserve creation-time assistant provenance, and define opt-in server resolution without changing legacy chat-create behavior. Issue 2950 Stage 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Explicit None survives restart and provisioning; opt-in inheritance persists bounded provenance atomically; stale versions and accepted retries cannot change identity.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 contract design underway in child TASK-13245.1; runtime not started. Corrected prior assessment: existing resolve_new_conversation_assistant already handles omitted identity inheritance and explicit-null opt-out;9 existing startup tests pass. Workspace chat create bypasses Sync v2; strict replay needs a DB-owned atomic path. Proposed design preserves legacy semantics and keeps Workspace Sync/Buddy/UI out of scope.

Stage 2 prompt/memory baseline: 13 passed, 9 failed with HTTP 503 missing_provider_credentials before mocked dispatch. Persona fixture patches chat.API_KEYS but runtime now uses provider_credential_runtime.load_server_config_snapshot. Diagnostic-only in-memory fixture supplying the same dummy credential via that loader yielded 22 passed. No runtime/test edits made. Repair the existing fixture and rerun ordinary suites during implementation; tracked in the staged plan and design validation record.

Design prerequisite TASK-13245.1 completed and published as draft PR #2958 (stacked on #2957). Independent review findings resolved at contract level. Parent remains In Progress: all runtime acceptance criteria are unchecked; requester approval is needed before slice 2A implementation.

Requester-requested second design review completed in TASK-13245.1/PR #2958. Verified/amended silent strict-selector downgrade on older servers (dedicated route), mixed-version cached-writer hazard (offline migration), lifecycle activation ordering, and unbounded permanent receipts (finite owner-scoped budget including tombstones). Independent re-review found no remaining material contract issues. Runtime untouched; revised contract and implementation verification gates still apply.

2026-09-27 delivery update: prerequisite resolver/choice and local startup-provenance stack merged normally into dev through PR #2963 at 10:40:42Z, merge commit 056d9adbb3f50243183ba8c6a3e9b367a21f1799 (head e5064376a67997901c2c33ddfe9e83b233c97bdf). TASK-13245.5 Done. All 70 exact-head hosted checks passed; requester-owned human summary and fresh review gates satisfied without admin bypass. Final integration 1095 passed/four known Bash>=4 skips/zero failures, including 734 Persona/Sync cases on official isolated SQLite/live-PostgreSQL fixtures; production Bandit clean across 14 files. Opt-out/provenance migrations are SQLite v69/v70 and PostgreSQL v73/v74. Canonical parity plan records merged 2A/local-2B delivery and exact evidence. Parent remains In Progress and its broad acceptance criteria remain unchecked: Stage 2C dedicated strict startup/versioned receipt/idempotency/send-time admission and Stage 2D broader validation are not implemented or certified by this merge. Tool-profile, provisioning/backfill and Research Workspace surface adoption remain separate open work; do not claim overall parity. Overarching issue #2950 remains open. Shared dirty checkout and other agents containers unchanged; heartbeat stopped.

2026-09-27: continued after normal integrated PR2963 merge. Created planning child TASK-13245.6 for Stage2C on latest audited server base 9668e1454b0b28b7a4de13e1a35496fa0b368c42 with native H1/H2. New design/plan explicitly requires receipt RLS in shared PostgreSQL, endpoint Persona guard before routing/credentials, deadlock-safe FK/replay locks and closure-fenced receipt-bound restore/Sync resurrection before route activation. Planning review/baseline underway; no runtime edits. Corrected stale completed Stage2A child TASK-13245.2 to Done based on integrated merge. Parent remains In Progress; Stage2C/2D and broader tool-profile/provisioning/Research work remain open.

2026-09-27 Stage 2C planning child TASK-13245.6 complete: current-dev source-backed refresh and five-stage plan independently reviewed; requester review required before runtime execution. Fresh unmodified official SQLite/live-PG baseline: 490 passed, one failed, one known SQLite parametrization skip, six warnings. Native PostgreSQL cascade retry leaks its enumeration read transaction after child failure; exact causal diagnostic recorded and separate prerequisite TASK-13245.7 filed To Do, not hidden or claimed fixed. Persona prompt/memory fixtures pass without modification. Canonical docs updated; no app/test/workflow diff against audited dev9668. Stage 2C/2D, profile/provisioning and Research adoption remain open; no broader parity certification.

Requester-approved TASK-13245.6 follow-up planning amendments cover strict transaction ownership/post-commit responses, backend-safe bounded text/body, immutable-owner admission and typed unavailable translation. Session preparation/preview/complete-v2 explicitly remain global-only; no Workspace session parity is claimed. Current refresh/executable plan and canonical documents carry the constraints and test gates; no Stage 2C runtime implementation in this amendment. TASK-13245.7 remains prerequisite; Stage 2C/2D and broader parity work remain open.

TASK-13245.7 local prerequisite repair Done in e1d05ddca0 (codex/persona-workspace-cascade-retry, dev35d6dd90d4 base). Hard/soft cascade enumeration and message-page reads now settle their own PostgreSQL transactions; outermost guard, caller work and admission closure unchanged. Final affected verification 714 passes/four SQLite-only driver skips/no failures across 15 files, official isolated SQLite/livePG. Multi-page image failure confirms earlier deletes durable and immediate retry; independent review gap closed, Ruff/compile/scoped Bandit pass. Repair is local, not a hosted CI or merged delivery. Completed task plan retained in implementation commit and retired from active tree. Parent remains In Progress; Stage2C strict receipts/admission, Stage2D and broader profile/provisioning/Research work remain open; create separate execution task before strict runtime edits.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
