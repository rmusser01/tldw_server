# Resumed release UAT — 2026-10-02

Tracking: TASK13260, In Progress. UAT261 and its original acceptance criteria remain open. Live UAT remains on hold pending corrected-runtime readiness and the outstanding browser recovery direction. Reviewed source fixes are in PR3096, which is ready for review. Its scope remains all43 families (A12/B9/C12/X4/S6), applicable database/auth cells, Web/extension surfaces, fresh state, clean installation, supported upgrades and D/L/U modes. No full-run or human acceptance is claimed.

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

All 15 original UAT failures remain open for corrected-runtime qualification. Their source corrections are independently reviewed in [PR3096](https://github.com/rmusser01/tldw_server/pull/3096). Source regression results do not replace live UAT or human acceptance.

## Reviewed source corrections

| Task | Cases | Correction |
| --- | --- | --- |
| TASK13260.281.1 | S02, A05 TXT, C01, B02 | Encode and validate backup image bytes, fail incomplete exports, preserve plaintext whitespace, catch wrapped FTS errors and use literal fallback, allow recognized Character generation metadata while retaining authoritative sampling. |
| TASK13260.281.2 | A08 selection/export, A05 Media, A12 unsaved edits | Fence the previous Notes editor during detail selection, request export keywords on every page, retain the incoming Media URL during hydration, and use the existing route-leave guard for unsaved connection edits. |
| TASK13260.281.3 | A12 dispatch, S04 query/reload | Compare effective dispatch inputs instead of store object identity, retrieve evidence without generating an unused answer, and restore migrated workspace identity and sources under the current account lease before fresh initialization. |
| TASK13260.281.4 | B01, B04, C07 | Normalize one Character envelope for JSON/PNG, gate Play and Preview on actual provider/voice readiness with guidance, and apply existing foreground/background theme colors to repository output. |
| TASK13260.280 | S03 | Unpack checklist/comment response envelopes, hydrate checklist items, map canonical mutations, block submission after load failure, and fetch all comment pages. |

Original repair qualification: 353 backend cases on the integrated dev7117 source (12 suites, no failures/skips) and 241 frontend cases across 22 suites passed. Independent actual-source review has no unresolved Critical/Important findings. Bandit on all six touched Python production files has 0 findings/0 errors. Touched frontend lint has 0 errors; inherited warnings remain. The full frontend typecheck completes with existing unrelated dependency/React diagnostics and no touched-path diagnostics, so it is not a project-wide typecheck pass. Existing Python lint findings remain outside the changed lines.

The dark-mode color correction uses the original observed readability failure and source inspection plus existing repository component checks; no mirrored CSS-class test or new live visual pass is claimed. All corrections still need fresh, separately authorized corrected-runtime UAT. Qodo pagination coverage now verifies all 151 comments across pages; live runtime qualification remains pending.

## Qodo follow-up

Reviewed corrections distinguish informational Research warnings from source failures, invalidate stale restoration hints while retaining tombstones, offer a separate workspace explicitly, and preserve note content with malformed optional keywords. Both Chat entry points carry the provenance of model/tool choices so ordinary selector changes invalidate preparation while explicit overrides remain fixed. Ignored Chat/Studio generation controls and the dead ungrounded-answer branch are removed; standalone RAG generation remains available. Markers/helper documentation and public Character export/import regressions address the remaining server comments.

Affected checks: 54 backend cases, 17 Kanban cases and a final 19-case Chat caller/owner subset passed. Restoration/Research/RAG/Parameters suites passed; two stale Studio voice-catalog expectations were corrected and passed on focused rerun. Independent final source review is clear. Touched lint adds no errors; inherited findings remain. Final reviewed-source typecheck exited 2 with 23 inherited React/dependency diagnostics and 0 touched-path diagnostics; no global pass. No live UAT is accepted.

The consumer-side portrait/version fix is reviewed in local Chatbook commit 20948093434ac9af6a0c6c055586840baa4b651d, with 39 public portrait/manifest cases passing. Automatic approval review rejected its separate-repository push before execution; companion PR publication was asked once and is pending. Server corrections are published in 07df134ed3, with all inline replies and the summary-only reply posted. Six server threads are resolved; the Chatbook compatibility thread remains open. PR3096 awaits final-head required checks, fresh Qodo acceptance and the human-written Change summary already requested once.

## Partial results and limits

- Setup and Saved Chat returned ORBIT-742; client-local history survived reload. This does not certify server mirroring or all auth cells.
- Biology generation saved five supported cards. Selected study/practice/rerate/lapse/keyboard/end/reload controls were observed; completed ratings must not be replayed for timing. C remains unrated for separately planned coverage.
- World Book and dictionary library edits/preview were observed. Actual conversation context injection/transformation remains untested.
- Documentation search, exact selected document, no-results and clear behavior were observed. Chunking UI was not executed.
- OpenWebUI JSON, Rowan PDF/image PDF/DOCX, repository and controlled image inputs are prepared. They do not certify imports, ingestion, real vision/OCR or asset persistence. DOCX visual QA remains blocked after one LibreOffice source-load failure; no unchanged render retry.
- Catalog has43 families/426 variants. The713 collected registrations were list-only, not executed passes/skips. Backup adapters, complete versions/ownership/context mappings and final denominator remain unresolved;4146 was provisional. Connector404 and disabled Forum are blocked.
- Dependencies were reused. Clean installation, supported upgrades, remaining cells/surfaces/modes, full release and human UX acceptance remain pending.

## Repair PR continuation

The human requested all recorded root causes be addressed in one PR before further UAT. TASK13260.281 and its four repair children own this work in PR3096, rebased without conflicts onto dev 6c5da178 after PR3092. All eleven repair commits are unchanged. Focused integrated Jobs completion/WorkerSDK/Chatbook checks passed 41 with 9 PostgreSQL cases deselected; this does not qualify PostgreSQL. Existing repair review/Bandit remains applicable to unchanged source. Reviewed fixes are published; UAT execution remains on hold for runtime readiness and browser direction. Source tests and UAT acceptance remain separate.

Latest rebase: PR3096 now integrates dev25f11ffe (PR3077). All sixteen owned commits and the full owned patch are unchanged; only fifteen upstream paths differ, with no direct repair-path overlap. Standalone docs tracing/bounded discovery, strict ordinary smoke diagnostics, drawer regression and foreign task history are preserved. Independent review is clear. Documentation/drawer unit checks pass 10 cases; pure smoke classifiers pass 11 cases without browser/page/API or webserver use. Config syntax and scoped frontend/UI lint pass with only configuration warnings; the initial frontend invocation ignored two UI paths, corrected by explicit UI-base lint. No Docker/browser packaging, native, PostgreSQL or live-UAT acceptance. No new Python change for Bandit. Prior own-source checks remain unchanged; fresh current-head hosted and human/companion gates remain open. ADR required: no; existing decisions are preserved.

## Current continuation

Frozen UAT source: /private/tmp/tldw-uat-resumed-20261002-413c/sqlite-single-source at dev413c2c9123509f17d96d514a7722e076598ea28f. Do not edit or replace it with repair/advancing PR source. Existing profiles/data and pending test inputs remain in that run directory.

Browser recovery remains pending from the earlier single request. No answer/time/heartbeat grants approval. Do not repeat the question, silently restart/restore/reauthenticate, or replay failed cases. Current browser/principal/drafts are unverified.

Preserve original action/recovery ceilings: no third A12 Send, B02 resend/settings-removal workaround, C01 reapply/Send/search replay, S03 Move/export/retry/reload, S04 original query/reload/late-answer measurement, duplicate export/import or media-selection replay to hide failures. Original budgets: startup120s, navigation/save15s, acknowledgment2s, generation180s, background600s, at most one declared recovery. Real providers only; official PG fixtures/restricted application roles.

PR2979 and Chatbook2950 are merged. PR3084 human Change summary is already published; do not reask. Final-source qualification, fresh required contexts, independent bound-workflow review and whole macOS natural-exit validation remain open. First-import investigation stays STOP/REASSESS; existing Mac red WIP is retained. Installer53/55 and consumer56 causal/final gates remain open.

At the user's direction, redundant private evidence files and repeated proof bundles were deleted. Keep bug lists and concise result/reproduction notes. Historical artifact links are obsolete; do not recreate them or retain new routine proof bundles. No new product fix or UAT acceptance resulted from cleanup.

CI docs correction: the full-suite gap-verified-9 job failed two refresh checks because the canonical Chatbook guide changed without its Published copy. The supported refresh adds only that missing six-line paragraph. Both failures reproduced before regeneration; all 39 refresh/adjacent Chatbook docs checks then passed (4 warnings, 19.76s, exit 0). No code or test changes; Bandit N/A for prose. Current-head hosted checks and the existing human/companion gates remain open.

CI ingestion assertion correction: gap-verified-4 failed three sync cases because four expected-content assertions removed the final LF present in their Markdown fixtures. All three failures reproduced before edits. Local-directory create/change, detached-note reattachment and archive sync now require exact fixture content, retaining their existing identity/status/conflict checks. All 22 affected/adjacent plaintext cases pass (4 warnings, 20.85s, exit 0, no skips); independent assertion review is clear. Ruff check/syntax pass; format flags and 104 LOW Bandit assertion notices are unchanged from HEAD, with no new findings. An initial aggregate invocation stopped before collection due to plugin autoload and required only operator setup correction. Production and live UAT remain unchanged; final-head gates remain open.
