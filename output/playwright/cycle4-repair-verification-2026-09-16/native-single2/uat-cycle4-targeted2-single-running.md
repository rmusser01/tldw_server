# Targeted single-user recheck on committed setup correction

Product8097d672d5. New isolated configuration/database/browser with existing installed dependencies. API18402 PID29873 started2026-09-15T20:52:07 local; WebUI18482 PID29976. API process/start-time comparison is unchanged after ordinary Chat. No provider config edits outside visible wizard, no API restart.

## Running results

- UAT106 PASS bounded: actual local setup discovers/saves exact Gemma, first-chat200 Hello., completion200, wizard unmounts, durable selectedModel matches verified provider/model before browser authentication. Expected Restore media access prompt accepts private runtime key via normal UI.
- UAT107 PASS bounded: ordinary Chat200 using custom-openai-api/exact Gemma without API restart. Visible answer JUNIPER-42.
- UAT103 PASS bounded: two real replies200; second request has user/assistant/user exactly once. Normal reload settles to5rows, matching canonical5 (system +2user/2assistant), no duplicates. Saved conversation03fce409-a003-4bc6-8525-7555090d5fa2.
- Title becomes Reply with exactly JUNIPER-42. | tldw on reload.
- UAT111 PASS bounded: Note fa6e93c4-ef9d-463e-a012-ab3ae6098648 source message5ae6eef9-f6fe-431f-83f0-c9be9fce7a7c links through actual Notes menu to the same saved five-row conversation without false unsaved guard.
- UAT104 PASS bounded: actual processing at0:56 minimizes, Notes route settles, terminal job remains available when Quick Ingest reopens.
- UAT105 backend bounded PASS/frontend FAIL: job1 completed with Warning, saved media1/UUID5ea63b88-4105-4651-bc9d-d6f365e9bd6b and sanitized analysis warning. Source226chars/37words visible in Media; no analysis yet. Results incorrectly says generic unexpected error0succeeded/1failed/noOpenMedia. Existing task46 owns frontend warning projection correction.
- UAT115 explicit model selection remains stable and actual Media analysis is running;110failedproviderUI and060Review remain pending.
- UAT108/113 are being checked separately in preserved multi-user profile; not claimed by this single profile.
- Harness correction: first post-reload snapshot had4local rows; later canonical system row makes5. A stale4-of-4 hover timed out without changing app state; use settled5-row transcript.

This is targeted validation, not a full fresh workflow run/signoff. The earlier failed profile remains separately preserved.

Harness corrections: Chat More actions is a tooltip, not a menu; initial wait timed out after correctly opening it. Ingest dialog changes from Quick Ingest (1) to Quick Ingest during processing; the first named-dialog snapshot wait timed out before Minimize was clicked. Notes uses a named Notes list region rather than a Notes heading; a wait for a heading timed out after navigation already succeeded. None is a product failure or a passing check. Final snapshots use observed controls.

115 PASS bounded: exact selected model reached custom-openai-api/exact Gemma chat request200; versions201 saved grounded Markdown heading+5facts, version2 UUID10355894-3cf7-4b3e-84fe-848b385b7884.
110 PASS bounded: real unavailable Ollama stream/fallback502 shows local Failed to generate analysis, pageErrors[], no Runtime Error overlay; source/prioranalysis retained, workingmodel restored. Two HTTP-error console entries are expected negative-request evidence.
060Review pending: item selection straddled shared-source HMR/fullrefresh while108fix was being edited. Do not treat selection-reset snapshot as productfailure or Reviewpass. Browsercheckspaused; verified UATNext18482PID29976 andold18480PID23418 stoppedgracefully, unusedoldAPI18400PID16030 stopped. CurrentAPI18402 remains unchanged; profiles/evidenceretained.
