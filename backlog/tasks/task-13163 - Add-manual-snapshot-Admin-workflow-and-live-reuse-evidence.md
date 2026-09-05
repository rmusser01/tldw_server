---
id: TASK-13163
title: Add manual snapshot Admin workflow and live reuse evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 04:59'
labels: []
dependencies:
  - TASK-13162
documentation:
  - Docs/Design/2026-09-04-llamacpp-manual-slot-snapshots.md
  - Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md
  - Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make manual cache preservation understandable and accessible to first-time administrators and repeat operators.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Admin panel implements enablement without implicit restart, slot inspection, save, compatibility reasons, confirmed restore and confirmed deletion.
- [x] #2 Operation status survives page reload, keyboard and narrow-screen flows work, and unsupported or unknown outcomes give explicit recovery guidance.
- [ ] #3 Pinned-build live save-stop-start-restore demonstrates cache reuse against a cold control; conversation and Pause/Resume semantics remain unchanged.
- [x] #4 Operator documentation records privacy, quiescence, retention, limitations and tested compatibility.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task3 of Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md: shared Admin UI and client, targeted UI tests, opt-in live harness and operator guide. ADR required yes; existing ADR043 governs. Do not claim complete or add supported build hashes without live cache-reuse evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented approved Admin snapshot workflow, typed client, durable historical operation recovery, explicit accessible confirmations, operator guide and opt-in disposable live harness under ADR-043. Task-scoped and final code reviews are clean after documented fixes. Verification: current backend 213 passed; live harness validators5 passed with1 actual live test skipped; current four-module UI63 passed; actual Admin app with mocked APIs passes2 light/dark narrow-screen browser flows. Changed-code static checks pass; inherited client532 lint warnings and unrelated frontend type errors documented. Live executable/model assets absent: pinned cache reuse, real Admin/Chatbook semantics and Pause/Resume evidence remain unverified. Production build allowlist empty. AC3 and full-acceptance DoD remain unchecked; keep In Progress. Guide documents numeric loopback and trusted-local-host boundary; testing lessons record source-derived wire and metric traps.

Live verification resumed with user-supplied executable/model paths. Both are regular files; executable is arm64 Mach-O, model exists (16 GiB). Preflight llama-server --version aborts with exit134 before model loading: libllama-server-impl.dylib requires __Z21mtmd_get_memory_usagePKc19mtmd_context_params, but libmtmd.0.dylib resolves to libmtmd.0.0.8145.dylib which does not export that symbol. otool/nm confirmed dependency mismatch (server library August14 versus mtmd February24). No inference or profile mutation occurred. Existing installation untouched; live acceptance remains blocked pending a consistent runtime bundle or approval for an isolated build.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Code implemented and independently reviewed through c6593deac1. Blocked acceptance: operator-supplied llama-server executable and GGUF model plus real runtime/client verification. No production build enabled; no PR, push or merge. Worktree and plan preserved for resumption.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
