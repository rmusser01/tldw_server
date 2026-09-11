# Buddy and Persona remediation implementation plan

Task: TASK-13226
Goal: Address the approved server/shared WebUI Buddy and Persona findings with explicit scope and verified behavior.
Spec: `Docs/superpowers/specs/2026-09-08-buddy-persona-ux-remediation.md`
ADR required: yes
ADR path: `backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md`
Reason: Independent content ownership, access-checked attachment and background interaction lifetimes cross storage and runtime boundaries.

## Stage 1 — Interface correctness (verified)

Files: AssistantSelect, RolePlaySetupDrawer, PlaygroundForm, PersonaGarden components and sidepanel-persona route.
Stage identity/settings; Apply only after successful validation/persistence. Correct shell geometry, tab focus, editing/live authority, input labels and ready-made previews. Tests exercise cancel/close/failure, keyboard navigation, narrow geometry and connected editing.

## Stage 2 — Server ownership and defaults (verified)

Files: Buddy DB repository, schema migration, Buddy services/schemas/router, chat creation.
Snapshot native art independently; expose bounded authenticated profiles/attachment APIs. Test source deletion, isolation, scope/version checks and omitted/None/override inheritance.

## Stage 3 — Shared Buddy management and interaction (verified)

Files: Common/PersonaBuddy, shell layouts, Buddy store/service, composer menu and workspace entry points.
Build preview-first management with separate Persona/artwork and explicit target. Keep one Buddy and drafts across routes. Integrate authorized transcript/reply/activity and existing decision boundaries. Add motion, keyboard positioning, reset and opted-in speech queue with named targets.

## Stage 4 — Ownership and integration verification (verified)

Exercise accepted work while navigating and losing a subscription against a real server boundary. Verify persistence, FIFO, exact acknowledgement, late-output fencing and access loss where supported. Resolve any gaps without claiming fixture checks prove backend execution.

## Stage 5 — Review and documentation (complete)

Run focused unit/integration tests, touched-file formatting/static checks and Bandit for Python. Perform one batched desktop/narrow, light/dark keyboard walkthrough, fix findings, then confirm. Review final diff against the traceability list; document evidence and explicit remaining limits. No full suite or unrelated Flashcards changes.

## Implementation and review evidence

Base: origin/dev at `6cd2745f69`; isolated branch `codex/buddy-persona-ux`.

| Reviewed issue | Final behavior and evidence |
| --- | --- |
| Persona selection takes effect before Apply | Controlled AssistantSelect and staged RolePlaySetupDrawer commit identity only on Apply; Cancel/Escape/Close, scene-save failure, parent failure rollback and standalone picker regression tests. |
| Obstructed Persona sections and coupled editing/live identity | Correct shared shell spacing, roving keyboard tab focus, persistent editing selector and frozen connected-session identity; Persona route and component regressions. |
| Late artwork previews and setup-heavy interaction | Ready-made gallery fetches actual protected artwork before enabling selection; imports/authoring remain secondary. Live transcript and composer precede expandable configuration. |
| Missing independent attachment | Schema 66 owns copied Buddy artwork/assets separately from source Personas; authenticated CAS attachment and scoped target resolution. SQLite plus live PostgreSQL 18 migration/CRUD/attachment/activity/acknowledgement/deletion verification. |
| Workspace defaults | Existing dev settings reused; server creation resolves omission/inheritance, explicit None and overrides without rewriting old conversations. |
| Buddy disappears on page change | Authenticated app-level IndependentBuddyHost sits outside replaceable WebUI route layouts and alongside shared extension routes. Real Persona→Notes navigation retains the selected conversation, draft and paused speech controls; one independent Buddy and no legacy duplicate. Logout, identity changes and demo mode clear private state. |
| Navigation mistaken for Stop | New accepted Buddy replies use fixed authenticated Chat admission and a process-owned FIFO. Tests cover detached admission/execution, concurrent conversation queues, Stop/publication fences, expired lease renewal, committed-result reconciliation and transient pre-dispatch failure. |
| Speech starts after cancellation or changes context | Shared useTTS and Buddy preflight lifetime fences, current-authority transcript reads, exact queued result IDs and conversation prefixes. Unmount/provider-delay/overlap regressions; no workspace microphone. |
| Keyboard and narrow usability | Explicit input labels, readable status tokens, draggable handle with arrow/Home alternatives and separate web/sidepanel positions. Real Next dialogs checked at 1365, 390, and 320px in light/dark themes; real drawer at 320, 390, and 480px in light/dark. |

The final combined shared frontend run passed 437 tests in 26 files. Another 40 WebUI app-layout/networking tests passed after the app-level lifetime fix. The real drawer walkthrough passed all six width/theme combinations and all three dismissal methods (Cancel, Escape, close), with correct identity on reopening and focus restoration. Nested picker Escape closes only the picker. Real Next navigation initially exposed whole-Host unmount and draft/selection loss between Persona and Notes; moving ownership above the replaceable layout fixed it. Both the full management walkthrough and a separate route/speech-control check now pass with zero page exceptions. Live PostgreSQL foundation review passed 34 tests with zero skips. Ledger/runtime review passed 23 tests, followed by 9 ledger tests including active-status filtering before pagination. Earlier broader backend runs passed 163 tests with PostgreSQL initially skipped; the later isolated PostgreSQL run supersedes those skips for Buddy coverage.

The real Next management walkthrough uses fixture API responses and actual bundled artwork. It verifies Cancel makes zero writes, workspace default None plus attachment Apply, the exact named reply target, no workspace microphone, dialog bounds and route continuity. The separate route check also preserves enabled/paused speech controls and proves only one Buddy is rendered. It is frontend evidence; backend continuation is proved separately through authenticated ASGI Chat integration with controlled provider responses.

The final rendered readability check covers 38 state/status instances across real light/dark themes. Buddy state badges were 10px and had 2.86–3.23:1 light-theme contrast; they now use 12px text and the readable theme foreground while retaining status backgrounds. The checked statuses now have at least 11.76:1 contrast. Current dev already applies readable foreground/background overrides to the older Ant Design Persona status tags; that earlier finding required verification, not another override. Evidence: output/buddy-ux/status-readability-{before,after}.json.

Frontend TypeScript reports 81 pre-existing diagnostics outside changed files. Focused ESLint reports zero errors (existing warnings retained). New and substantially changed UI files are formatted; small additions to large existing files preserve surrounding formatting. Python Ruff/Black/Bandit and diff whitespace checks passed for owned changes. No full repository suite was requested or run.

## Runtime boundaries

Normal Console routes share their browser-owned provider and do not themselves call Stop on navigation. Browser loss and credential changes do not have the accepted Buddy ledger guarantee. Legacy Persona Live remains connection-owned under ADR-046; its disconnect can cancel work. New Buddy replies use manual Send after optional dictation, and existing full Persona Live voice remains a separate workflow. Existing approval controls retain their authority; the new narrow reply adapter rejects slash commands and supplies no tools. Workspace attention represents persisted assistant results and accepted Buddy turn status, not invented progress for every runtime.

The local dependency link for diff resolved one environment-only diagnostic; the remaining 81 diagnostics are unchanged outside this work.

No packaged extension launch, real model-provider round trip, microphone capture or audible output was performed. The shared extension code is covered by frontend tests and the responsive WebUI walkthrough. Server restart ends unfinished Buddy work without credential persistence or automatic replay; multi-worker acceptance requires principal affinity as described in `Docs/Operations/Buddy_Turns.md`.

User guide: `Docs/User_Guides/WebUI/Buddy_And_Persona_Management.md`.
API: `Docs/API/Buddies.md`.
Operations: `Docs/Operations/Buddy_Turns.md`.

## PR verification

On the unchanged dev base `6cd2745f69`, the fresh PR run passed 437 shared-UI tests in 26 files and 57 targeted backend tests, with one PostgreSQL-environment skip. Earlier real PostgreSQL 18 verification remains recorded above. No full suite was run.

## PR review follow-up

Verified all 13 automated review comments and corrected ownership disclosure, target/save races, domain exception mapping, endpoint coverage, test synchronization and endpoint/core placement. Two redundant identity-reset suggestions were disproven against the canonical selection and connected-session models and received stronger regressions; the cross-feature integration test relocation was declined with explicit rationale. The independent review caught and prompted the asynchronous retarget fence.

The complete dispositions, focused regression/static results, generated docs/API checks and remaining CI limits are recorded in `Docs/Reviews/BUDDY_PERSONA_PR_FOLLOWUP_2026_09_08.md`. Follow-up directly implements ADR-005; no new architecture decision or schema change.
