---
id: TASK-13178
title: Expose synchronous email summarization in Service Prompts
status: Done
assignee: []
created_date: '2026-09-05 16:09'
updated_date: '2026-10-02 01:46'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/2887'
  - 'https://github.com/rmusser01/tldw_server/pull/3077'
documentation:
  - Docs/Design/email-summary-service-prompt.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the user-approved bounded email Service Prompts slice, including the provider/recursive form wiring and removal of the explicit-key-only analysis guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Email system instructions are editable in shared Settings and generic save/reset APIs.
- [x] #2 One owner-specific snapshot covers synchronous EML and supported containers; explicit empty/text values override storage, unset/reset use deployment defaults.
- [x] #3 Provider and recursive fields reach analysis; configured credentials and keyless providers remain handled by the shared analyzer.
- [x] #4 Disabled and nested-child analysis stay unchanged; corrupt overrides fail before upload processing; lookup connections close on their worker.
- [x] #5 Focused backend/UI regressions, OpenAPI validation, lint and Bandit pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Record approved design and baseline; add failing behavior tests; implement minimal registry, parser, adapter and analysis-guard changes; verify and independently review, then commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
CLI allocated TASK-13177 despite a main-checkout task already owning that number. This explicitly allocated replacement avoids that collision; archive only the newly created duplicate in this worktree.

Implementation and independent review complete. Added owner-scoped email instructions via existing registry/Settings; form provider and recursive options now reach analysis. Removed explicit-key-only gate, relying on shared configured/keyless credential resolution. Self-review reproduced JSON-shaped email body loss in three RED tests; input_is_literal_text=True fixed all three. Review follow-up added real key/model resolution coverage, real PST/OST traversal with libpff boundary substituted, and worker-local cleanup checks on success/corruption; six targeted checks passed. Final combined regressions running. Historical duplicate TASK-13177 was archived in this worktree only; active task uses explicitly allocated TASK-13178.

Final verification: 200 backend passed, 2 existing PST skips, 9 existing warnings; 197 shared frontend passed; 5 targeted WebUI Settings passed. Bandit zero findings across 4 production Python files; compilation and scoped Ruff lint/format passed. Email library baseline I001/SIM103 verified unchanged. Official OpenAPI export, generated TypeScript and fingerprint check passed; removing only 3 new email form fields reproduces dev fingerprint. Independent final review: no remaining findings. Temporary node_modules links removed. Full repo suite, full frontend typecheck, live provider/browser and actual libpff binary integration not run. Implementation complete; awaiting user integration choice.

User authorized push and PR creation. Opened PR #2887 against dev: https://github.com/rmusser01/tldw_server/pull/2887. Implementation commit a904a802b3f602785310e008dd4bcf238d5c6739. PR body records local verification, known skips and pending human Change summary before merge.

User requested latest-dev rebase, all PR review comments addressed and merge. Rebased cleanly onto dc0b7455f2 (documentation-only dev changes); range-diff confirms both implementation commits unchanged. Qodo posted zero bugs and four rule findings: endpoint architecture placement, missing return annotation, endpoint docstring, and form docstring. Reviewing dispositions; human Change summary still pending.

Qodo findings 2-4 addressed with JSONResponse return annotation and endpoint/form documentation. Finding 1 disposition: retain API-owned authenticated DB acquisition and worker lifecycle; existing core resolver owns decoding, validation and deployment defaults, so moving API dependencies or adding a callback wrapper would add coupling without relocating prompt semantics. Post-rebase and post-correction focused runs each passed 112 tests. Ruff lint/format, compilation, Bandit zero findings, official OpenAPI export/typegen/fingerprint checks passed; only public endpoint description changed in the schema. No new runtime behavior in review fixes.

2026-10-01 tracking closeout: GitHub verified PR2887 MERGED at 2026-09-05T17:20:18Z with merge commit 69c96ef715588f27d9cb953be13e75f3493c6ba9. Existing acceptance criteria and Definition of Done were already checked and implementation/review results recorded. Corrected stale In Progress status only; no source change or repeated tests. Bandit is not applicable to this tracking-only correction. Read-only merge receipt: /tmp/email_workstream_pr2887_status_20261001.json.

Published reviewed fixes as d4693fefe37bc75d9d9b54026c77c12de3200c76 on codex/email-followup-closeout-20261001; normal push verified. Draft PR3077 contains the implementation and this tracking closeout. Repair acceptance is locally complete; this new PR is not merged and its human-written Change summary and hosted CI are pending. Source receipt /tmp/email_followup_closeout_receipt_20261001.json binds ten exact source paths and frozen verification artifacts. Browser coverage checks the AuthNZ manifest entry plus selected-document content/render; the real AuthNZ guide HTTP probe is separate. Task-owned servers stopped after identity verification; shared installations and unrelated resources preserved. Completed owned plan retained at /tmp/email_followup_plan_completed_20261001.md after all stages are complete.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added user-editable synchronous email summarization system instructions through the existing Service Prompts storage, generic API and shared Settings. One authenticated owner snapshot covers EML and supported mail containers/recursive passes, with explicit prompt precedence, deployment defaults and safe connection cleanup. Repaired missing provider/recursive form wiring, delegated configured and keyless credentials to the existing analyzer, and preserved JSON-shaped email bodies as literal text. Disabled and nested-child analysis behavior remains unchanged. Verification and independent review are recorded in the linked design.
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
