---
id: TASK-2276
title: Backfill DeepSeek OCR backend ADR
status: Done
assignee: []
created_date: '2026-06-07 05:34'
updated_date: '2026-06-07 16:42'
labels:
  - docs
  - process
  - adr
  - ocr
  - deepseek
dependencies:
  - TASK-2275
references:
  - Docs/ADR/inventory/2026-06-07-deepseek-ocr-confirmation-audit.md
  - Docs/Design/DeepSeek_OCR_Backend.md
  - Docs/OCR/DeepSeek-OCR.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a bounded accepted ADR for the confirmed DeepSeek OCR backend decision from INV-026. Scope the ADR to the local Transformers-only deepseek backend, HuggingFace AutoTokenizer/AutoModel loading with trust_remote_code, default markdown prompt and Gundam sizing, safe output extraction, non-persistent-by-default result handling, CUDA/FlashAttention availability gates with env overrides, registry/API exposure, manual dependency setup, and caveats from the confirmation audit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one DeepSeek OCR backend decision.
- [x] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-026 points to the accepted ADR.
- [x] #3 Update Docs/Design/DeepSeek_OCR_Backend.md with a short ADR reference without changing the historical decision text.
- [x] #4 Keep caveats explicit: trust_remote_code risk, manual dependency install, default CUDA/FlashAttention gating with env overrides, no server mode, temporary output by default, and actual registry priority behavior.
- [x] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Started in isolated worktree `.worktrees/deepseek-ocr-adr-backfill` from `origin/dev`.
- Duplicate active `TASK-2276` files exist in `backlog/tasks/`, so Backlog CLI numeric task edits are unsafe for this task. This DeepSeek task record is maintained by exact file path in this branch.
- Plan: create ADR-024 from the TASK-2275 confirmation audit, update the ADR index and INV-026 inventory row, add a historical-source ADR reference to `Docs/Design/DeepSeek_OCR_Backend.md`, then verify docs and focused OCR tests.
- Verification: `git diff --check` exited 0. Reference scan across touched docs/task files found no absolute developer-machine paths or temporary Bandit report artifact names. Focused tests passed with 18 passed, 6 warnings: `source ../../.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/MediaIngestion_NEW/test_ocr_backend_deepseek.py tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py`.
- Bandit was not run because this task changed only documentation and Backlog metadata, not Python/code.
- Known skip: the live DeepSeek OCR PDF integration test remains intentionally gated by `DEEPSEEK_OCR_RUN_INTEGRATION=1`, CUDA, and local model dependencies.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created `Docs/ADR/024-deepseek-ocr-local-transformers-backend.md` for the bounded DeepSeek OCR local Transformers backend decision. Updated the ADR index, INV-026 inventory row, and source design reference so the accepted ADR is discoverable while preserving caveats for manual dependencies, `trust_remote_code=True`, default CUDA/FlashAttention gates with env overrides, no server mode, temporary output by default, actual registry priority behavior, and gated live-model tests.
<!-- SECTION:FINAL_SUMMARY:END -->
