# Shared UI static-root teardown — TASK-13817

The final verification repeat exposed the same real React root leak in two independent fixtures. RTL owns its rendered roots; AntD messages, notifications, confirmations, and waves use separately created roots through the installed RC renderer. Fix that shared resource boundary without changing real UI behavior, suppressing errors, or extending deadlines.

ADR assessment: no new ADR. This is test-only resource ownership and cleanup using the existing dependency renderer; no application API, persistence, or production rendering policy changes. ADR-059 governs task records.

## Stage 1: Prove ownership and isolation
**Goal**: Reproduce real static roots surviving RTL cleanup, including an effect cleanup and fresh reuse in subsequent cases.
**Success Criteria**: Behavior regression fails before the shared repair; real messages, notifications and queued confirmations are exercised.
**Tests**: New static-root lifecycle suite; current LocalChatList and RuntimeConfig fixtures.
**Status**: Complete

## Stage 2: Centralize actual renderer disposal
**Goal**: Track the actual installed native CJS renderer's containers, drain scheduled static modal rendering, perform real root unmount, and reset already-loaded notice singleton references.
**Success Criteria**: No eager AntD barrel import, fake UI, global motion changes, error suppression, or arbitrary deadline changes; retain instrumentation through teardown; remove superseded per-fixture teardown workarounds.
**Tests**: Red/green lifecycle suite, mocked AntD and cold guard compatibility, real modal/motion/visibility suites, relevant type/lint checks.
**Status**: Complete

## Stage 3: Review and final integration
**Goal**: Independent exact-head review and complete hosted validation before the authorized merge.
**Success Criteria**: All sixteen coverage shards, native complete merged report, seven required gates and executed checks pass on latest dev; tracking records reflect actual evidence.
**Tests**: Full hosted coverage and checks; verify merge state and remote dev after integration.
**Status**: In Progress
