# Chat availability without an offline outbox

TASK-13454.9, parent TASK-13454; latest dev `27ce976387`, integrated baseline `ca39b897de`.

Actual first visit at isolated WebUI port19764, with configured backend19765 absent, spent15seconds in readiness and then replaced the entire /chat route. Server settings recovery reached the real existing settings page. Two authorized independent source reviewers confirmed the gate and the existing offline Queue/reconnect behavior. Readiness is not authentication; preserve the independent signed-out removal, transport validation, H1 and draft owner fences.

## Stage 1: Reproduce route and offline mutations
**Goal**: Red checks for actual application/readiness wiring, full submit, actual compact callback and shared queue.
**Success Criteria**: Checks fail for hidden chat, offline enqueue/reset, and reconnect auto-dispatch; connected busy queue has positive coverage.
**Tests**: Rejected/stalled health, ready/degraded transition without editor remount; offline direct submit including image intent; shared real queue reconnect and held callback.
**Status**: Complete

## Stage 2: Keep the chat mounted and refuse offline dispatch
**Goal**: Reuse readiness diagnostics, existing queue blocked states and actual sender connection checks.
**Success Criteria**: /chat remains reachable through checks/retry; other routes gate normally; unavailable submission retains original input and attachments; queued requests pause for explicit review/run after reconnect; connected busy queue still works.
**Tests**: Stage1 checks green; signed-out/account boundaries; newline editing and accessible delivery controls.
**Status**: Complete

## Stage 3: Verify actual UI and regressions
**Goal**: Recheck fresh /chat and existing settings path from this worktree, with appropriate shared regressions and two source reviews.
**Success Criteria**: Actual absent-backend /chat exposes editor/setup action; typing survives readiness transitions; checks/type/extension compile/touched lint pass; reviewers clear.
**Tests**: CUA browser walkthrough, route/auth/readiness/queue/submit regressions. Bandit N/A for TS/docs-only scope.
**Status**: Complete

## Stage 4: Record evidence and bounded completion
**Goal**: Commit the reviewable unit with Backlog notes and honest acceptance limits.
**Success Criteria**: TASK13454.9 closes only verified scope. Parent remains open for unverified-owner first-setup draft preservation, cold multi-user offline history, actual provider/backend/package journeys and later stages.
**Tests**: Diff/staging review; no generated Next artifacts or dependency symlinks staged. No push/PR/merge request.
**Status**: Complete

Do not persist unowned drafts, fabricate ownership, replay unknown creation/provider requests or introduce another store/queue. Initial unconfigured draft-through-configuration is a separate integration: current owner fencing clears on authority change and settings Close reloads. Shell mounting alone does not close AC6 or cold Q1.

## Verification and limits

The qualified red checks reproduced route/offline mutation, interrupted compact context/dictation preparation, and already-connected restoration replay. The full-page queued document regression was requalified with a valid model and isolated mock; `/private/tmp/chat-readiness-held-document-red3.log` confirms the continuation resolves/sends without the final guard. Earlier fixture failures are excluded from readiness-red evidence.

Final 21 relevant suites: 290 tests pass (`/private/tmp/chat-readiness-no-outbox-final3-green.log`); WebUI typecheck and extension entrypoint compile pass. Touched lint has no added findings against ca39b897de. Bandit is not applicable: TS/TSX/JSON/tests/docs only, no Python production changes. Both independent read-only source reviewers clear the held-preparation and restored-queue findings. They did not run browser/package checks.

Actual isolated WebUI `/chat` at19764, absent backend19765: editor and connection recovery coexist, multiline text remains editable, Send is disabled, Retry and the subsequent failure preserve the same draft. Proof: `/private/tmp/chat-ux-readiness-draft-retained.jpg`. The recovery banner is compact and the composer fits the 1280x720 viewport. This is product WebUI evidence, not provider/storage acknowledgement, packaged extension/Chrome docking, VoiceOver, 320px/390px/zoom, or cold authenticated offline proof.

Pending work stays on the parent: truly unowned first-configuration draft retention, explicit greeting startup, scoped roleplay settings, responsive/accessibility/package journeys, Sync v2 and backend/provider closure. Readiness presentation does not fabricate ownership or preserve drafts through hard settings navigation. Restored requests are blocked for review, including uncertain former sending requests; already dispatched completions retain their existing handling.
