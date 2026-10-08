---
id: TASK-13533
title: Restore production WebUI bundle budget after Knowledge changes
status: In Progress
labels:
- frontend
- performance
- knowledge
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR3213 production Turbopack container and Onboarding builds exceed the unchanged 900 KiB route budget on the Mermaid QA route. Investigate the emitted/static graph and defer only genuinely optional preview code using existing React loading primitives, preserving rendering, keyboard, close/focus and the existing 600/900 KiB gates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Actual production build evidence identifies the over-budget route and existing cold optional import boundary without a raised budget or hidden required first-render bytes.
- [x] #2 The minimal repair preserves Mermaid rendering and preview open/close/reopen keyboard and focus behavior, with meaningful regressions and existing shared UI compatibility.
- [ ] #3 Production build, token parity and unchanged bundle gates pass with exact profile/inputs; affected native browser and extension evidence is accurate or precisely qualified.
- [x] #4 PR3213, task13514 and canonical workstream evidence accurately record this CI repair and remaining qualifications.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Duplicate search: bundle budget and Mermaid preview; related 2313/12698/12700 and webpack13260.278.10 are Done,13406 has unrelated M5 smoke exceptions. ID13533 exceeds dev13530 and max13532 across all open PR files at creation. Actual container job113248612983/run37758265587 head45925ab03d09e20f386324f153f1d0d5f922783a: route __debug__/mermaid-chat-cards900.0KiBgzip over900, shared595.7/600; log /private/tmp/knowledge-pr3213-container-webui.log SHA43bf1554754ba164dbdac03b149bebb2bb081340e7fd748d3f72f397d9a3366c. Onboarding job113249431163/run37758265846 same cause900.1/900, logSHA6faa1cbd1893f97f70366257ec873c7ba9ad65c9d85980031712a5b70ecff419. Prior supported webpack gate820.3/900 remains distinct; do not call it Turbopack proof. Read-only graph shows MermaidDiagramBlock statically imports and mounts a closed MermaidPreviewDialog; existing CommandPaletteHost has React.lazy plus first-use latch/retained mounted lifecycle. No product edit yet. ADR assessment:no new durable rule for reusing existing optional-module loading; assess existing governing UI lazy-boundary decisions before code.
Sole SDD implementer at BASE 3b4d624b42537bf60ce06f75a1fa411c3ab15151. Root cause: optional closed MermaidPreviewDialog statically imported/mounted by shared MermaidDiagramBlock; every Markdown/artifact caller routes through that shared component. Plan: focused RED cold-module/lifecycle regression; existing React.lazy/Suspense first-use latch retained mounting; affected shared tests/types, exact tracked Docker context production Turbopack build and owned native CDP QA. ADR required: no; searched Docs/ADR/README.md, no accepted lazy-loading decision; reuse existing CommandPaletteHost convention without new durable rule. TS-only touched product: Bandit N/A; scoped normal hooks required. Hosted amd64 gate remains pending regardless of local aarch64 result.
TASK13533 locally validated at BASE3b4d624: minimum shared optional React.lazy first-use retained-mount preview; inline renderer unchanged. RED expected0/received1 module request before action; focused13/13, affected51/51, final cold test1/1; both types0, scoped shared lint0errors/2 inherited any warnings, E2E lint0, product/new-test style0 and existing two full-file failures reproduce BASE. TS-only Bandit N/A. Actual Dockerfile production Turbopack/Bun1.3.2 frozen graph/token parity and unchanged600/900 gates pass595.0/874.5 local aarch64. Actual owned19260 runtime imagecb65a10a;111/111 browser script bytes match,0initial preview requests,4first-use chunks; actual SVG/keyboard/Escape/Close/completed focus/reopen reset passes0errors. Offscreen fixture prerequisite corrected via real scroll. Extension isolated development Chrome build0; new native extension preview unverified. Final product/config/lock unchanged; redundant test-only reconciliation build canceled in installation exit130, no replacement. All negative attempts/complete receipts preserved unique knowledge-pr3213-mermaid paths. Canonical Docs/Reviews review+JSON updated without dropping old fields; plan bounded subsection complete locally. TASK13533 stays In Progress for current hosted amd64 production statuses; independent review/publication root-owned;13512native/device and13532historicalCSV unchanged. Full report /private/tmp/knowledge-pr3213-mermaid-budget-report.md; no push/merge/cleanup.
Final exact9-file manual pre_commit hooks exit0; git diff --check0. Canonical source/receipt evidence preserved all old JSON fields; normal scoped commit with hooks required. Script runtime config digest failed before browser work; verified actual image manifestcb65a10a was used. New owned runtime ab664b3310139c03973d40ce46da031783bd8acb4aa3f9dc818fb1fda00ba3e4 and target A7EFD3A86099690ECF24C07ABCEE8254 remain for root-owned review/cleanup; localhost19260 bound only127.0.0.1. All current native checks0errors, no old service/profile/model mutation.
Independent scoped review approved0Critical/0Important with M1Minor: absence after import release could still be Suspense fallback. Fix round1 authorized only owning lazy test: wait for observable committed Modal mount including open=false before absence/reopen. Fault-inject removal of existing source-change setPreviewOpen(false), require meaningful RED, restore exact product bytes before GREEN. No production/build/browser/runtime changes, no51-suite rerun. Unique round1 receipts and normal scoped lint/style/hooks/commit; M2 inherited warnings/style remain qualified.
M1 round1: owning lazy test now waits for committed Modal marker even open=false after real import before absence/reopen. Fault-injected only source-change setPreviewOpen(false) removal: RED exit1 finds actual dialog at pre-reopen absence. Exact product bytes restored SHA7f920a86b2e77e89a9167cf4ea339de49a929a4ef898938c71a385cdf32cea15; focused GREEN1/1 exit0, scoped lint/style0. Unique round1 logs/report; original report SHA1ce6e725 remains exact. Canonical evidence extends only scoped M1 history; no product/config/budgets/builds/browser/runtime/51-suite/types repeats, M2 debt and all previous qualifications untouched. TS-test-only Bandit N/A; no durable ADR change. Normal scoped hooks/commit required; current hosted amd64 gates still pending.
2026-10-08 final local publication reconciliation: PR3213 https://github.com/rmusser01/tldw_server/pull/3213 product/test head1e895f8e8312d8c299c2f42dd8006bc1a314c80b contains all locally approved CI repairs. Fixture417 review Approved0Critical0Important (99 owning/59 native quota cases), five-matrix exact inventory3b4 review Approved0Critical0Important, optional preview692 and test-only1e895 round1 independently Approved0Critical0Important/newMinor0. Actual production Docker on local Linux/aarch64 passes unchanged595.0/600 shared and874.5/900 route gates; actual native111/111 scripts/focus/reset pass, new extension native preview qualified. Guard-removal RED now proves committed pending-source retirement and exact product bytes remain unchanged. This supersedes earlier independent-review-pending wording; current hosted amd64/seven exact-head required statuses, requester-written Change summary and merge remain gates. Original public429/default/broad/CSV/native/device qualifications remain. Fresh origin/dev2c5f19d is ancestor, MERGE_QUEUE unset; Qodo billing blocked and requester authorized merge after checks without it. No new durable ADR decision; governing066 and CI/shared-UI contracts unchanged. No new suite/build/runtime replay, no merge/cleanup.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Optional Mermaid preview now genuinely loads on first use and stays mounted for native close/focus. Independent original and M1 scoped reviews approve; guard-removal RED proves pending source retirement, restored owning1/1 passes with exact product bytes unchanged. Actual local Linux/aarch64 production/token/budgets595.0/874.5 and native111/111 scripts/focus/reset pass; hostedamd64/new-extension native preview remain qualified. Current seven exact-head required statuses and integration gates remain; no budget/dependency relaxation.
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
