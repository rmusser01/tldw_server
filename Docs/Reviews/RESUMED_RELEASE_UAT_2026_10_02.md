# Resumed release UAT — 2026-10-02

Tracking: TASK13260, In Progress. UAT261 and its original acceptance criteria remain open. Full UAT is resumed across all43 families (A12/B9/C12/X4/S6), applicable database/auth cells, Web/extension surfaces, fresh state, clean installation, supported upgrades and D/L/U modes. No full-run or human acceptance is claimed.

## Open bugs

| Case | Reproduction and actual result | Expected result |
| --- | --- | --- |
| A08 Notes selection | With ControlB selected, select ControlA, immediately edit and Save. The PUT updates prior ControlB instead of intended ControlA. | Save the requested note or prevent editing until its identity loads. |
| A08 Notes export | Export the three tagged notes through unfiltered JSON export. All keywords become empty arrays; the filtered export retains tags. | Preserve tags in every supported export path. |
| A12 Unsaved connection | Edit Server URL without saving, navigate to Preferences and return. The edit disappears without keep/discard guidance. | Explicitly handle unsaved edits. |
| A12 Chat dispatch | Send in a new Saved Chat. Original plus sole declared recovery produced no completion request; recovery shows request_config_scope_changed. Two Sends exhausted. | Dispatch one intended turn or give actionable scope recovery. |
| S02 Account backup | Export a full account containing image-bearing Character2. Job reports completed, but manifest omits Character2 and ZIP contains its truncated JSON; raw bytes are not serializable. | Produce a valid, complete backup or fail honestly. |
| A05 TXT preservation | Ingest Rowan422-byte TXT and Larch129-byte TXT. Saved content loses the final LF in each case (421/128 bytes). | Preserve the declared exact source bytes. |
| A05 Media selection | Open Larch from /media?id=1. URL becomes id2 while Rowan stays selected/loading with repeated successful GETs. Sole planned terminal reload restores Larch. | Load the selected media without a reload. |
| S04 Research query | Query the selected Rowan/Larch sources. Browser times out; backend later completes a real Gemma call after retrieval fallback, but answer is unavailable to the user. | Return the cited result within the declared budget or expose a recoverable result. |
| S04 Workspace reload | Reload the populated research workspace once. UI creates/selects a new empty workspace while original server sources remain. | Retain the current workspace identity. |
| B01 Character export | Export TestBot JSON/PNG. Required v3 fields land at data.data inside a second envelope. | Produce one valid character-card envelope. |
| B02 Character Chat | Send Hello once as TestBot4 with saved generation settings. Native completion rejects409 unsupported_history_context_character_extensions; conversation has no messages. | Accept supported saved Character settings or explain the incompatibility before dispatch. |
| B04 TTS readiness | Providers/voices are empty and selected Kitten engine is disabled/catalog404. Filling text enables Play/Preview while readiness is Unknown without setup guidance. | Gate unavailable synthesis and show useful setup guidance. |
| C01 Prompt search | Search the exact hyphenated Prompt name. Server returns500; FTS reports no such column20261002 while local library still finds it. | Treat the query as user text and return results or a safe empty response. |
| C07 Repository output | Generate repository text in dark mode. Output textarea has pale text on white, contrast about1.18. | Legible themed output. |
| S03 Kanban detail | Create one board/two lists/three cards and open SELENE card2. Checklist200 response is a wrapper; frontend treats it as an array and crashes at .map. | Open the card detail with empty checklist/comments. |

All15 UAT bugs remain open. Other source-supported causes above remain investigation notes; the local S03 correction below still requires runtime qualification. Product repairs need their own official tasks, causal red/green, affected checks, Bandit and independent review in a separate repair source/runtime.

## S03 local source correction

TASK13260.280 corrects the shared checklist/comment client envelopes, loads checklist items, and maps API name fields to existing UI title/content fields for reads and writes. Checklist load errors now show an alert and block both button and Enter submission.

Causal regression tests reproduced the original crash and the review-found error/keyboard gaps. The final correction passed25 affected tests, formatting and independent source review. Lint has0errors/5inherited warnings; focused TypeScript has0touched diagnostics/3dependency diagnostics, so no full-project typecheck pass is claimed. Bandit cannot analyze TypeScript. The last test-only selector correction passed all4 component tests.

This is a local source correction in the repair worktree, with no push or frozen-runtime change. Fresh distinct-fixture corrected-runtime UAT remains pending browser recovery. The inherited first50-comment pagination limit remains unqualified.

## Partial results and limits

- Setup and Saved Chat returned ORBIT-742; client-local history survived reload. This does not certify server mirroring or all auth cells.
- Biology generation saved five supported cards. Selected study/practice/rerate/lapse/keyboard/end/reload controls were observed; completed ratings must not be replayed for timing. C remains unrated for separately planned coverage.
- World Book and dictionary library edits/preview were observed. Actual conversation context injection/transformation remains untested.
- Documentation search, exact selected document, no-results and clear behavior were observed. Chunking UI was not executed.
- OpenWebUI JSON, Rowan PDF/image PDF/DOCX, repository and controlled image inputs are prepared. They do not certify imports, ingestion, real vision/OCR or asset persistence. DOCX visual QA remains blocked after one LibreOffice source-load failure; no unchanged render retry.
- Catalog has43 families/426 variants. The713 collected registrations were list-only, not executed passes/skips. Backup adapters, complete versions/ownership/context mappings and final denominator remain unresolved;4146 was provisional. Connector404 and disabled Forum are blocked.
- Dependencies were reused. Clean installation, supported upgrades, remaining cells/surfaces/modes, full release and human UX acceptance remain pending.

## Repair PR continuation

The human requested all recorded root causes be addressed in one PR before further UAT. TASK13260.281 owns this repair work; see the root-cause repair plan. UAT execution is on hold while these repairs are addressed. Source tests and UAT acceptance remain separate.

## Current continuation

Frozen UAT source: /private/tmp/tldw-uat-resumed-20261002-413c/sqlite-single-source at dev413c2c9123509f17d96d514a7722e076598ea28f. Do not edit or replace it with repair/advancing PR source. Existing profiles/data and pending test inputs remain in that run directory.

Browser recovery remains pending from the earlier single request. No answer/time/heartbeat grants approval. Do not repeat the question, silently restart/restore/reauthenticate, or replay failed cases. Current browser/principal/drafts are unverified.

Preserve original action/recovery ceilings: no third A12 Send, B02 resend/settings-removal workaround, C01 reapply/Send/search replay, S03 Move/export/retry/reload, S04 original query/reload/late-answer measurement, duplicate export/import or media-selection replay to hide failures. Original budgets: startup120s, navigation/save15s, acknowledgment2s, generation180s, background600s, at most one declared recovery. Real providers only; official PG fixtures/restricted application roles.

PR2979 and Chatbook2950 are merged. PR3084 human Change summary is already published; do not reask. Final-source qualification, fresh required contexts, independent bound-workflow review and whole macOS natural-exit validation remain open. First-import investigation stays STOP/REASSESS; existing Mac red WIP is retained. Installer53/55 and consumer56 causal/final gates remain open.

At the user's direction, redundant private evidence files and repeated proof bundles were deleted. Keep bug lists and concise result/reproduction notes. Historical artifact links are obsolete; do not recreate them or retain new routine proof bundles. No new product fix or UAT acceptance resulted from cleanup.
