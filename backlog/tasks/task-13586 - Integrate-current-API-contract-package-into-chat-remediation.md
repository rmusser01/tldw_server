---
id: TASK-13586
title: Integrate current API contract package into chat remediation
status: Done
labels:
- chat
- ux
- integration
dependencies:
- TASK-13593
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Integrate freshly fetched origin/dev590e7e5794224028c760a42bc5f031562ce7a6c3 into the isolated chat remediation branch after TASK13585. Preserve original request/path and audit snapshot semantics, resolve the proposed API ADR numbering collision without changing accepted rationale, and qualify the connected client/backend impact before the confirmed runtime-rail layout root. This follows the user requirement to work against latest dev and the existing all-ten root-cause queue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Current dev is included while chat/greeting/history/request behavior and audit entry-time metadata snapshots are preserved.
- [x] #2 Shared generated path package and normal WebUI/extension compilation and affected contract/audit tests are qualified without weakening guards or overwriting primary dependencies.
- [x] #3 Independent bounded reviews, introduced static/security checks and normal owned hooks pass; inherited/runtime/PG/package/AT/study limits and the remaining layout root are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: no new durable rule. Incoming Docs/ADR/065-frontend-api-contract-package.md is Proposed and already collides with accepted Docs/ADR/065-independent-notes-knowledge-provenance.md; preserve accepted rationale/path and renumber only the incoming proposed API record to a verified free ID with its current references. ADR049/050/059 continue to govern chat and task ownership. Upstream admin audit closure moved metadata copying from entry into deferred execution; investigate and preserve the original snapshot if causal. Highest task13585 across local/dev/open-PR files; root owns all edits/runners and primary checkout remains untouched.
Stage1 complete; Stage2 In Progress. Causal audit RED1/2 then GREEN2/2 preserves original top-level entry-time metadata before AuthNZ after-commit execution. Derived-route guard RED4/8 then GREEN10/10 fixes vacuous zero-route success using the existing TypeScript AST reader; actual generation2114 paths/3276 schemas includes existing Roleplay route. Proposed API ADR069 verified free across local/dev/open PR IDs; rationale unchanged and accepted Notes065 byte-equal. No primary dependency writes. Connected tests/types/security/static and independent review/hooks pending.
Independent source review confirmed two additional incoming root gaps: Docker builder has package manifest without committed type exports (no runtime failure inferred because Next ignores build type errors), and mandatory backend fingerprint gate does not compare the new committed paths artifact. Close using existing exporter canonical path output/check shared by Node generation and required Python-only CI/Makefile; explicit committed Docker leaf copies. Causal regression checks precede fixes; broader acceptance remains open.
Final frozen source: canonical generation/check2114 paths3276 schemas, actual CLI2114 routes49 fields with exact temporary snapshot restore, backend129/129, client305/305, parser10/10, Docker context5/5 and both normal client compilers exit0. Eight JS/TS lint0 against both parents; six Python Ruff0, three production files Bandit0 findings/errors. Both approved independent reviewers CLEAR. Schema/file/persistence boundaries remain doubles; no hosted CI/physical Docker/PG/current native build/package/AT/study inference. Stage3 normal owned hooks and merge recording pending.
Normal37-file owned hooks actual exit0 without bypass. Three own stages complete; own plan retires privately. Final metadata hooks and normal incremental merge recording follow.
TASK-13592 tracking collision correction: completed chat fixture dependency formerly TASK-13585 is now TASK-13593; upstream accounting retains TASK-13585.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Integrated current dev590e7e5 and qualified the shared API package against chat remediation. Restored original deferred audit snapshot; replaced vacuous route verification with actual union checks; reused the canonical exporter for committed path generation/required drift checks; selected both gates for package-only changes and supplied committed Docker declarations. Proposed API ADR069 preserves rationale and accepted Notes065. Final129 backend/305 client/10 parser/5 Docker-context checks, real2114-route CLI, both compilers, lint/Ruff/Bandit and two independent reviews pass. Normal37-file owned hooks pass. No physical Docker/PG/hosted CI/current native build/package/AT/study inference; rail overlap and broader master remain open.
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
