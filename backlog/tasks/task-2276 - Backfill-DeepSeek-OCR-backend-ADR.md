---
id: TASK-2276
title: Backfill DeepSeek OCR backend ADR
status: To Do
assignee: []
created_date: '2026-06-07 05:34'
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
- [ ] #1 Add a new immutable ADR under Docs/ADR/ with the next available ADR number and one DeepSeek OCR backend decision.
- [ ] #2 Link the ADR from Docs/ADR/README.md and update Docs/ADR/inventory/2026-06-03-decision-inventory.md so INV-026 points to the accepted ADR.
- [ ] #3 Update Docs/Design/DeepSeek_OCR_Backend.md with a short ADR reference without changing the historical decision text.
- [ ] #4 Keep caveats explicit: trust_remote_code risk, manual dependency install, default CUDA/FlashAttention gating with env overrides, no server mode, temporary output by default, and actual registry priority behavior.
- [ ] #5 Record verification and Bandit applicability in this task.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
