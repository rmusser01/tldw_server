---
id: TASK-13188
title: Add manual snapshot Admin workflow and live reuse evidence
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-05 02:19'
updated_date: '2026-09-05 06:38'
labels: []
dependencies:
  - TASK-13187
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
- [x] #5 Cache-mode configuration and live verification are model-agnostic: strict swa_full option affects compatibility, arbitrary supplied models can be characterized, and unverified configurations never gain production support by inference.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Task3 of Docs/superpowers/plans/2026-09-04-llamacpp-manual-snapshots.md: shared Admin UI/client, targeted UI tests, opt-in live harness and guide. Approved continuation: add strict generic swa_full formatting/admission with effective-option identity tests; parameterize disposable harness cache mode/context and preserve failure metrics; document architecture evidence and RAM/restart implications; targeted TDD/static/security checks and supplied-model managed live test. ADR required: yes. ADR path: Docs/ADR/043-managed-llamacpp-manual-slot-snapshots.md. Reason: existing storage/runtime boundaries and fail-closed identity contract remain unchanged. No production build hashes or inferred support for untested architectures.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented approved Admin snapshot workflow, typed client, durable historical operation recovery, explicit accessible confirmations, operator guide and opt-in disposable live harness under ADR-043. Task-scoped and final code reviews are clean after documented fixes. Verification: current backend 213 passed; live harness validators5 passed with1 actual live test skipped; current four-module UI63 passed; actual Admin app with mocked APIs passes2 light/dark narrow-screen browser flows. Changed-code static checks pass; inherited client532 lint warnings and unrelated frontend type errors documented. Live executable/model assets absent: pinned cache reuse, real Admin/Chatbook semantics and Pause/Resume evidence remain unverified. Production build allowlist empty. AC3 and full-acceptance DoD remain unchecked; keep In Progress. Guide documents numeric loopback and trusted-local-host boundary; testing lessons record source-derived wire and metric traps.

Live verification resumed with user-supplied executable/model paths. Both are regular files; executable is arm64 Mach-O, model exists (16 GiB). Preflight llama-server --version aborts with exit134 before model loading: libllama-server-impl.dylib requires __Z21mtmd_get_memory_usagePKc19mtmd_context_params, but libmtmd.0.dylib resolves to libmtmd.0.0.8145.dylib which does not export that symbol. otool/nm confirmed dependency mismatch (server library August14 versus mtmd February24). No inference or profile mutation occurred. Existing installation untouched; live acceptance remains blocked pending a consistent runtime bundle or approval for an isolated build.

User approved isolated runtime repair. Created /private/tmp/llamacpp-snapshot-runtime.xEuJ3u using copied August14 executable/libraries and corrected versioned library links; original installation untouched. --version now succeeds: build10430 commit4c1a0af40. First disposable live run reached runner startup but failed in HTTP User-Agent version lookup because the existing venv lacks tldw-server package metadata. Retrying with supported TLDW_VERSION=0.1.41 from pyproject.toml, no dependency installation or code changes.

Isolated runtime copy verification completed: build10430 commit4c1a0af40. With TLDW_VERSION=0.1.41, live test ran save→stop→start→restore→warm request→separate cold request. Save/restore receipts complete with2770 tokens; artifact266502188 bytes. Live assertion failed because warm timings.cache_n=0 (<80% saved tokens). Result1failed5passed7warnings180.21s. Root cause of zero reuse not yet diagnosed; do not infer successful compatibility from receipts. Logs /private/tmp/llamacpp-snapshot-live-retry.log and XML /private/tmp/llamacpp-snapshot-live-retry.xml. Process check confirms no remaining disposable llama-server; original executable matches copied bytes and original mtmd link remains unchanged. Guide now records negative evidence and exact hashes. Production allowlist remains empty; AC3 unfulfilled.

User requested latest official macOS ARM64 download and retry. Downloaded b10816 archive into /private/tmp/llamacpp-latest-arm64.RlECpI; SHA256726ca8e7680203280b72029f92380aaf482e6a48ebe4a73fbe934ccc0bcf2de9 matches official asset digest. --version build10816 commit427291b5b. Same disposable test/model/settings: save/restore2770 tokens; warm and cold each cached0 processed2780. Result1failed5passed7warnings179.05s. Log /private/tmp/llamacpp-snapshot-live-b10816.log and XML /private/tmp/llamacpp-snapshot-live-b10816.xml. Latest upgrade alone does not resolve reuse; no cause attribution or production support claim. Test children exited; original installation untouched.

Diagnostic-only investigation isolated cause on b10816: direct native /completion and /slots (no tldw modules) reuses2770/10 tokens in same process but after restart reuses0/processes2780. Native restored.log explicitly forces full prompt reprocessing due to missing SWA/hybrid cache data; model n_swa1024. Pinned server-task.h566–574 clears checkpoint list; save/restore persists sequence/tokens, not that list. One-variable native --swa-full experiment restores2770 cached/10 processed; prompt processing199.455ms versus24414.949ms default. SWA allocation300→3200MiB atctx16384, separate nonSWA320MiB unchanged. Diagnostic script/logs /private/tmp/llamacpp-reuse-diagnosis.nqonY0 (baseline-fspl26xp and full-c6euz_74); all child processes stopped. No production changes made. Current tldw formatter/config gate lacks swa_full; implementing validated option and memory/restart UX plus fresh managed acceptance requires follow-up authorization. Production allowlist remains empty.

Approved model-agnostic extension implemented under existing ADR043: shared strict boolean swa_full at profile create/update/runtime/cleaning boundaries; formatter and snapshot admission accept it; actual command identity tests prove true invalidates prior cache-mode compatibility while false equals omitted. No model-name branching or production allowlist change. Harness accepts explicit cache mode/context for any supplied model and publishes negative metrics before reuse assertions. Updated operator guide and testing lesson. TDD observed initial failures, then targeted regression165passed1opt-in-skipped7warnings; Ruff/check-format eightfiles and Bandit fourproductionfiles clean. Independent read-only review found no findings. Real managed b10816 run with supplied Gemma and swa_full passed15tests156.95s: saved/restored2770, warm reused2770/processed10 versus cold0/2780; executable/model hashes unchanged. Evidence /private/tmp/llamacpp-snapshot-managed-full.log and .xml. Other architectures and real Admin/Chatbook/Pause-Resume acceptance remain unverified; keep AC3 and full DoD open.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Generic strict full-SWA cache option and configurable characterization harness implemented; no Gemma detection. Managed restart reuse verified for supplied b10816/Gemma/full-cache configuration (2770 reused,10processed versus cold0/2780). Targeted165passed plus live15passed; static/security checks and independent review clean. No production support gate opened. Task remains In Progress pending other-model evidence and real Admin/Chatbook semantics verification.
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
