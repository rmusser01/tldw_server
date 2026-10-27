# A05 configured remote OCR

Task: TASK-13260.278.18.83.45. Parent: IMPLEMENTATION_PLAN_uat_a05_ingest_progress_20261013.md. Frozen523ef323 plus preserved WIP, not latest dev.
ADR assessment: no new architecture decision; reuse existing configured remote OCR backend and ADR021 lifecycle/ADR022 processing. Existing Qwen vision service8010 confirmed model via public modelsHTTP200; no model install/launch/config change there. Original private SQLite profile/data/keys retained; only owned API child environment receives OCR mode/host/port/model, no .env rewrite or credential inspection/logging.

## Stage1: Confirm and review configuration
**Goal:** Bind real existing vision service without automatic downloads.
**Success Criteria:** Exact model present, review fixed owned18354/18339 stop/start guards, source/profile/port/state controls and no managed model start.
**Tests:** Existing backend contract review, operator-only guard/static checks, independent source review.
**Status:** Complete

## Stage2: Owned runtime and scanned source
**Goal:** One bounded API transition and actual image-only PDF extraction.
**Success Criteria:** Owned shutdown15s/startup120s, no force; original failures preserved. One scanned-PDF ingest meets Rowan semantic oracle or records exact failure.
**Tests:** Real normal chooser/wizard/provider/canonical media readback; no mocked OCR or text layer.
**Status:** In Progress

## Stage3: Acceptance accounting
**Goal:** Track OCR and related recovery scope truthfully.
**Success Criteria:** Actual processor/model/output/status mapped, pending gates remain open.
**Tests:** Task/plan current, scoped syntax/lint/security/source review, no unchanged replay.
**Status:** Not Started

Independent operator/source review CLEAR; one Minor wording correction distinguishes normal private configuration loading from credential inspection/logging, corrected. Seven startup admission/five config-delta operator-only checks pass, Ruff0/Bandit0/syntax0. First config preparation failed before writes because original OCR section absent; private clone adds only that section/priority, parsed original values identical. Actual one-page scanned fixture5820016bytes has one raster image/no text layer. Runtime transition and OCR remain unrun at this entry.

First owned stop consumed soleSIGTERM and failed15.011s/over-grace; no startup followed. Monitor50949 completed0 later; fresh18354/18339 absent/18906free; actual prior markerapiA05WebOwnerExitCode=-15/readyFalse/stateexited. No delay-cause/ASGIclean-shutdown claim. Reviewed NEW after-exit recovery CLEAR,10pure operator admission cases/Ruff0/Bandit0; unchanged config/model/lock/freeport/source/profile/cleanup guards, distinct consumed marker. ONE declared recovery startup initiated, API39164/owner39161, readiness unqualified while starting. No further shutdown/force/model action.

Recovery API39164/39161 canonicalready8.423s, exactownedparent/privatecwd/listener verified. Actual normal client OCR catalog identifies llamacpp remote/available/configured/Qwen model/managedstartfalse. FirstscannedPDFStartACK108ms/job10/batchcbf53fd0/3s returns existingMedia3; no newOCR qualification. New shared request-field cause tracked separately in IMPLEMENTATION_PLAN_uat_a05_ocr_toggle_20261013.md; no speculative dedup patch.
