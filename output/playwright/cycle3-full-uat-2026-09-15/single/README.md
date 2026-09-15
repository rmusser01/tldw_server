# Cycle 3 single-user UAT evidence

This is an **in-progress capture**, taken from the file inventory at **2026-09-15 08:15:17 UTC**. It contains successful controls and open findings; it does not certify full workflow acceptance. Later evidence from the continuing browser run is outside this inventory.

The run uses frozen product revision `d40e17dc81`, fresh isolated single-user API18200 / WebUI18280, and the unchanged real llama.cpp provider on9099. Existing dependencies were reused. This certifies neither a clean-machine dependency installation nor multi-user acceptance. The [running tracker](../../../../Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md) owns current issue status.

## Successful controls captured

| Control | Retained evidence and limits |
| --- | --- |
| Fresh browser and real setup Chat | [Browser open](uat-cycle3-single-open.txt), [actual first Chat result](uat-cycle3-single-first-chat.json). The result records the configured model and real response. |
| Paste ingest, analysis and first Open in Media | [Exact fictional source](uat-cycle3-single-source.txt), [ingest requests](uat-cycle3-single-ingest-requests.txt), [completed ingest](uat-cycle3-single-ingest-complete.txt), [Media snapshot](uat-cycle3-single-media-source.txt), [screenshot](uat-cycle3-single-media-source.png). Explicit provider selection was required after UAT056; successful ingestion does not resolve the configuration or progress findings below. |
| Real saved Chat replies | [Initial requests](uat-cycle3-single-chat-requests.txt), [final requests](uat-cycle3-single-chat-final-requests.txt), [original server conversation](uat-cycle3-single-original-chat-server.json). Both replies persist; the duplicate conversation and mode change remain UAT062. |
| Chat-derived Note and backlink | [Save requests](uat-cycle3-single-note-save-requests.txt), [save result](uat-cycle3-single-note-save.json), [opened Note](uat-cycle3-single-note-open.txt), [backlink](uat-cycle3-single-note-backlink.txt). Persistence and origin work; loaded save-status text remains UAT063. |
| Reviewed Chat-derived Flashcard and persisted schedule | [Question/answer review dialog](uat-cycle3-single-card-review-dialog.txt), [saved server card](uat-cycle3-single-flashcards-server.json), [visible answer](uat-cycle3-single-card-answer.txt), [reloaded Study](uat-cycle3-single-study-reloaded.txt), [server card after review](uat-cycle3-single-flashcards-reviewed-server.json). The reviewed card has version 2, a review timestamp and the next due time. Manage's console error remains UAT055. |
| Separate Notes-to-Flashcards workflow preparation | [Notes start](uat-cycle3-single-notes-start.txt), [saved Biology Note](uat-cycle3-single-biology-saved.txt). These show preparation and a saved Note, not completion of generated-card acceptance. |

## Open findings represented

| Finding | Evidence and coverage |
| --- | --- |
| UAT055: Manage emits deprecated AntD List console error | [Console capture](uat055-cycle3-manage-console.txt). The earlier Study session-list repair remains distinct from this newly exercised Manage path. |
| UAT056: Ingest Review claims readiness before required provider validation | [Late validation](uat056-ingest-late-validation.txt) captures the return to Configure and actionable provider requirement. The earlier Ready to Process snapshot is referenced by the tracker and is outside this bounded copy. |
| UAT057: Duration estimate ignores analysis cost | [Completed ingest](uat-cycle3-single-ingest-complete.txt) reports the actual 50 seconds. The earlier approximately 3 second estimate is recorded in the tracker; its Review snapshot is outside this copy. |
| UAT058: Home/setup and Flashcards titles are blank | [Fresh browser open](uat-cycle3-single-open.txt) and [reloaded Flashcards](uat-cycle3-single-study-reloaded.txt) omit a Page Title. The tracker also records direct empty-title inspection; that inspection is not independently retained here. |
| UAT059: Disabled Reading Queue is shown as a temporary outage | The [ingest request inventory](uat-cycle3-single-ingest-requests.txt) supports the absence of a reading-list call. The Home wording and prerequisite trace are recorded in the tracker; their earlier snapshots are outside this copy. |
| UAT060: Media analysis shows raw Markdown | [Media screenshot](uat-cycle3-single-media-source.png) and [snapshot](uat-cycle3-single-media-source.txt) show literal Markdown markers in the generated analysis. The screenshot was visually inspected. |
| UAT061: Ingestion progress reports unconfirmed stages | The [ingest request inventory](uat-cycle3-single-ingest-requests.txt) supplies request context. The precise UI 50–55% versus server 20% comparison depends on processing snapshots/response391 recorded by the tracker, outside this copy. |
| UAT062: First saved standard Chat duplicates history and changes mode | [Visible saved Chat](uat062-first-saved-chat.txt), [final requests](uat-cycle3-single-chat-final-requests.txt), [original server conversation](uat-cycle3-single-original-chat-server.json), [duplicate server conversation](uat-cycle3-single-extra-chat-server.json). Both real replies exist; this is not a claim of lost answer content. |
| UAT063: Loaded saved Note has conflicting save status | [Opened Note](uat-cycle3-single-note-open.txt) shows No server save status yet alongside Version 1 / Last saved. |

## Retention and verification

`manifest.json` lists the exact 25 retained files, initial source modification times, SHA-256 hashes and byte sizes. Selection was limited to the top-level `/private/tmp/uat-cycle3-single-*` JSON, TXT and PNG files present at the inventory cutoff, plus the three explicitly named finding captures. All source originals remain intact. Embedded `.playwright-cli` references point to original capture artifacts and do not imply those files were copied.

The selected evidence was checked against 28 credential values from both private cycle3 runtime manifests and common JWT, Bearer-token, API-key, password and private-key patterns; no matches were found. The single PNG was visually inspected and shows the fictional Aster fixture and Media interface, with no credentials. Private manifests, scripts, broad log collections, databases and browser profiles are excluded.
