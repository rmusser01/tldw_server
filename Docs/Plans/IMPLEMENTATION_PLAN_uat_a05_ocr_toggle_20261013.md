# A05 OCR request toggle
Task: TASK-13260.278.18.83.45; parent A05 ingest plan. Frozen523ef323 plus preserved WIP, not latest dev.
ADR assessment: no new architecture rule; existing ingestion contract/ADR022 governs, reuse enable_ocr on actual AddMediaForm/PDF processor.

## Stage1: Reproduce actual request contract failure
**Goal:** Prove the UI toggle never enables PDF OCR.
**Success Criteria:** Normal configured shared executor tests fail on missing enable_ocr for queue/process-only true/false; explicit advanced and per-item precedence covered.
**Tests:** New cases in existing quick-ingest-batch.test.ts, actual transport spies/poller, no fake UAT/model.
**Status:** Complete

## Stage2: Minimal shared mapping and checks
**Goal:** Bind PDF document.ocr to enable_ocr while retaining parser/overrides/authority/storage/cancellation.
**Success Criteria:** Causal greens and affected tests UI/WebUI, configured types/lint, independent actual-source review.
**Tests:** Complete service suite and connected routing/fallback tests, current configured types, scoped lint; TS Bandit N/A.
**Status:** Complete

## Stage3: Live validation
**Goal:** One declared scanned-PDF recovery on corrected client.
**Success Criteria:** One review/process result exposes actual OCR text/processor, preserved original job10 duplicate outcome, canonical owned saved identity; no overwritten failure or raw OCR acceptance without evidence.
**Tests:** Necessary production build/source binding and one bounded owned frontend replacement; real chooser/provider/output/readback. Broad A05 and engineering gates remain open.
**Status:** In Progress

Causalred5failed1passed53deselected; actual transport fields omit enable_ocr, no mocked liveOCR claim. Minimal7linePDF-only mapper implemented afterward; explicit advancedfalse and perfilefalse precedence preserved. Complete service+fallback UI59/Web59 pass, configuredWebtypes0/exact48bindingsrestored. Lintsource0errors30unchangedwarnings/test0errors23unchangedwarnings; TS BanditN/A, newcheckerRuff0/Bandit0. Independent actual2path source/newtestbody/checker review CLEAR. New38binding build+fixed74191/74189 transition operator deltas independentlyCLEAR/Ruff0/Bandit0/syntax0, unrun at this entry. Scanned firstjob10 duplicate/actual OCR unqualified retained; one declared live recovery after corrected build, no unchanged blind replay or overwrite.

Necessary38binding production build completed naturally0/137.870s, compiler/token-sync/bundle all0, buildBgHHUtfGqO0B5aQdiDogk; exact38bindings restored. Compiler skipped type validation; separate configuredWebtypes0 already recorded, not inferred from build. Fresh frozenHEAD/buildidentity/old74191parent74189/listener guards passed; ONE bounded frontend transition initiated, readiness unqualified while running. No API restart/model install/newwindow/publication.

Oldfrontend74191/74189 firststop over15/no startup; latermonitor27979 natural0/bothabsent/oldreadyFalse/actualintExit0. Single no-signal recovery independentlyCLEAR/7operatorguards/Ruff0/Bandit0; new44954/44952 ready0.763s/buildBgHH verified by actual next-data in existingoneTab. Original stop outcome preserved/no ASGI or cause claim. ONE corrected-client review-before-saving scannedPDF recovery StartACK104ms running; direct extraction/readback not yet qualified. Original job10 and source3 retained, no overwrite/newwindow/model action.

Corrected-client scannedPDF review recovery actualHTTP500/plainInternalServerError/UI180s/0saved/1failed; no OCR text/draft qualified. API/provider publichealthboth200 afterward. QuickstartNextproxyTimeout180000 and OCRHTTP180 alignment is hypothesis only; sourceexactcause not established. Owned-system-log bounded timestamp filters returned0request-window records; no oldCIlog read, no forced retry/source/timer patch. New slots metadata probe404/unsupported, not repeated. ONE declared provider-only SSE diagnostic prepared with same300DPI page/model/prompt/temp/max2048, total180 cap/one-use control, reasoningcounts only; synthetic event accounting selfcheck/Ruff0/Bandit0. This diagnoses performance/protocol only, never UI/UAT acceptance. Pending source review before model request.
