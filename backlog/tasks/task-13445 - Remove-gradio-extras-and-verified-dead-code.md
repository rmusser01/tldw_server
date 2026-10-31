---
id: TASK-13445
title: Remove gradio extras and verified dead code
status: Done
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Batch 1 Stage 2. Plan: Docs/plans/2026-10-01-due-debt-sweep-implementation-plan.md. pyproject.toml:470-471 gradio group + line 480 'all' extras; metrics_logger.py:226; DB_Manager.py:1380-1397; legacy_maintenance.py:125; root plan-file index only (never delete other agents' plans).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Executed 2026-10-03. Removed gradio extras group + 'all' reference (pyproject); tomllib parse OK; pip install -e . --dry-run OK; 0 gradio refs remain. Deleted check_media_and_whisper_model stub chain (legacy_maintenance stub + DB_Manager wrapper + import; no external callers). Deleted commented Workflow Functions + Dead code FIXME blocks (DB_Manager tail). Cleaned stale Gradio docstring note (Video_DL_Ingestion_Lib). Drift note: metrics_logger.py gradio block already removed upstream before this task. Root plan index: Docs/Plans/2026-10-02-root-implementation-plan-index.md (13 files, none deleted). Verification: quick-launch suite 29 passed; Bandit 0 findings on touched paths.
Review fix round 1 (2026-10-03): deleted metrics_logger Gradio comment block (earlier 'already absent' claim was a case-sensitive grep miss - corrected); removed stub from __all__; converted broken test import to absence assertion (3 passed); removed doc entries in both Media_DB_v2.md copies. MCP boundary-test 5 failures verified pre-existing (no gradio refs in output; branch touches no .github files).
Renumbered 2026-10-04 from TASK-13400 when PR #3155 was rebased onto dev: the branch was cut from codex/post2970-uat-20260920, and dev had meanwhile given that id to an unrelated task. Branch commit messages and code references use the new id. The stage plan (Docs/Plans/2026-10-01-due-debt-sweep-implementation-plan.md) exists only on that unmerged branch.
Rebase onto dev (2026-10-04, PR #3155): Docs/Plans/2026-10-02-root-implementation-plan-index.md dropped. Dev's 304e438704 (docs: relocate root implementation plans) removed every root IMPLEMENTATION_PLAN_*.md, so the index listed 13 files that no longer exist.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Gradio packaging and verified dead code removed with green tests and clean Bandit.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
