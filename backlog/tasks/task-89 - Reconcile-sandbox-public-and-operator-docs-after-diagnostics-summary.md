---
id: TASK-89
title: Reconcile sandbox public and operator docs after diagnostics summary
status: Done
assignee: []
created_date: '2026-05-05 22:18'
updated_date: '2026-05-05 22:20'
labels:
  - sandbox
  - docs
dependencies: []
documentation:
  - Docs/API-related/Sandbox_API.md
  - tldw_Server_API/app/core/Sandbox/README.md
  - Docs/Sandbox/sandbox-runtime-capability-inventory.md
  - Docs/Sandbox/macos-runtime-operator-notes.md
  - Docs/Sandbox/sandbox-architecture-doctrine.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Align public and operator-facing sandbox documentation after the merged runtime diagnostics, macOS diagnostics, image-store cleanup, recovery summary, and status reason-code slices. Keep the change docs-only and do not alter runtime behavior or API schemas. Essential constraints: host-local runtimes such as seatbelt and worktree must remain documented as weaker than VM-grade isolation and not untrusted-eligible; vz_linux repair/recovery remains scoped to explicit admin-only ownership-checked flows; runtime discovery and admin diagnostics remain the source of truth.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sandbox API docs describe the current runtime discovery, admin runtime diagnostics, macOS diagnostics, image-store cleanup, recovery summary, and status reason-code surfaces without overstating guarantees.
- [x] #2 Sandbox README and operator notes point to the same source-of-truth docs and preserve host-local versus VM-grade distinctions for seatbelt and worktree.
- [x] #3 Capability inventory current gaps and maintenance guidance remain aligned with the merged diagnostics/recovery/status-reason slices.
- [x] #4 Verification covers documentation references and whitespace hygiene; Bandit is documented as skipped because this is docs-only.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Compare current public Sandbox API docs, Sandbox README, macOS operator notes, and runtime inventory against merged runtime/admin diagnostics behavior. 2. Patch only documentation drift: endpoint list/semantics, diagnostics response purpose, image-store cleanup/recovery status, run status reason codes, and host-local caveats. 3. Verify docs references/links and whitespace; record Bandit skip because no production code changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Docs-only reconciliation completed. Updated Sandbox API quick guide with admin runtime diagnostics, macOS diagnostics, image-store cleanup, VZ reconciliation repair, and status_reason_code guidance. Updated Sandbox README source-of-truth pointers, macOS operator notes cross-runtime diagnostics guidance, and inventory current-gap wording for host-local warnings. Verification: listed referenced docs successfully; verified documented admin endpoint route strings exist in sandbox.py; verified status_reason_code schema/taxonomy references exist; git diff --check passed. Bandit skipped because no production code changed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reconciled sandbox public/operator docs after the runtime diagnostics summary landed. The Sandbox API quick guide now documents admin runtime diagnostics, macOS diagnostics, image-store cleanup plan/mutation, VZ reconciliation repair, and status_reason_code semantics. The Sandbox README now points to the API guide and macOS operator notes as source-of-truth references. macOS operator notes now distinguish cross-runtime diagnostics from macOS helper/template diagnostics. The capability inventory now treats host-local docs/API warnings as covered and keeps future follow-up focused on UI/operator dashboard propagation. Verification: referenced docs exist; documented admin route strings exist in sandbox.py; status_reason_code exists in schema/taxonomy; git diff --check passed. Bandit skipped because only markdown/backlog docs changed.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
