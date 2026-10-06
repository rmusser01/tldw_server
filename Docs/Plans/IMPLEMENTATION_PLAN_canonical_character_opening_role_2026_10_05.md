# Canonical native character opening role

TASK-13454.13, approved parent design Docs/Design/2026-10-04-chat-conversation-ux-remediation-design.md. Latest dev8aeddeee36 integrated at88a2d65aee. Both independent source reviewers traced factory callers and H1 before this plan.

## Stage 1: Qualify the actual role failure
**Goal**: Red-first actual non-Sync API create/messages/H1 regression for default and exact alternate zero; module startup compatibility.
**Success Criteria**: Failure is the named sender in the new root; raw content, qualified owner and existing metadata contracts are observed rather than inferred.
**Tests**: Existing authenticated API/real SQLite fixture, owner-bound H1 capture, module converter and seed-disabled/no-greeting/initial-message behavior.
**Status**: Complete

## Stage 2: Correct the shared seed insert
**Goal**: Use assistant and null parent only for the factory-generated greeting; retain exact accepted text and every caller-supplied initial message.
**Success Criteria**: One settled assistant root through existing H1 validation; no provider/user admission, new storage authority, old-row rewrite or Sync claim. Qualified UI/module labels remain usable; historical name immutability is not claimed.
**Tests**: Qualified red cases turn green; existing greeting, factory, H1 and module regressions remain green.
**Status**: Complete

## Stage 3: Verify the live owned journey and record completion
**Goal**: Actual UI opening save and known-ID reopen with the corrected isolated backend; independent final source reviews and local commit.
**Success Criteria**: One explicit create, raw owned assistant greeting, zero provider/user turn, preserved draft, correct character attribution; no repeat of the earlier uncertain create.
**Tests**: Relevant Python/frontend regressions, both existing type gates, touched Bandit/diff checks, live screenshots and official owned API reads. Remove only this completed plan after retaining its history.
**Status**: In Progress

Qualified red: actual default/alternate-zero H1 rows return Native Greeter as role; module seed returns the display name (3 failures,2 compatibility cases pass, /private/tmp/chat-canonical-opening-qualified-red.log). The implementation changes only the factory-generated greeting sender and explicit root; validation in progress.

Verification: focused11 and broad292 Python checks pass; actual module reopening adds one passing focused check. Bandit0 and no new Ruff findings. Both independent final reviews find no Critical/Important issues. Live one-create owned H1 proof with exact raw assistant root and zero provider requests: /private/tmp/chat-greeting-canonical-owned-proof.json. Cold reload preserves the known chat, correct speaker and exact focused draft: /private/tmp/chat-ux-saved-greeting-reloaded.jpg. Local implementation commit and final task recording are the remaining Stage3 actions. TASK-13454.11 owns the separate display-only H1 ID classification correction; both final type checks pass and touched frontend lint has no new diagnostics.
