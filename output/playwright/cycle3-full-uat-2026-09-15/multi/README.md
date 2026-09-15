# Cycle 3 multi-user UAT evidence

Execution ended **2026-09-15 11:14:21 UTC** with failures and explicit coverage limits. This is not an all-pass acceptance result.

- [Final main matrix and chronology](multi-results.md): setup/admin creation, Alice workflows, same-browser account transitions and offline recovery.
- [Main capture manifest](main/manifest.json): 309 captures, including 41 valid JSON files and four visually inspected PNGs. The manifest also hashes the final report and records eight empty captures excluded from the bundle.
- [Bob/admin report](bob/REPORT.md) and [SHA256 manifest](bob/SHA256SUMS): independently reviewed Biology generation/study, owned/foreign API controls, and admin-owned delete/Trash/restore. This earlier sealed bundle is unchanged.
- [Running findings tracker](../../../../Docs/Reviews/FRESH_INSTALL_SINGLE_MULTI_UAT_TRACKER_2026_09_14.md): 32 open findings across both modes, including reopened055. Review addenda083/084 are recorded there without rewriting the earlier Bob report.

Application behavior was frozen at `d40e17dc81`. The documented `c10e1464fc` exception disables only persistent development caching for the existing isolated UAT build directory; [environment evidence](../environment/README.md) retains the comparison. Fresh configuration, databases and browser profiles reused installed dependencies and the real local llama.cpp model. Credentials and private runtime configuration are excluded.

The parent verified capture hashes, JSON parsing, and absence of fourteen known runtime credential values plus JWT/private-key markers. All four PNGs were visually inspected. This is a bounded credential check, not a guarantee that arbitrary secrets could never occur in a capture.

## Outcome limits

Exact Wikipedia ingestion stored zero articles; its dependent search and grounded Chat remain blocked. The deliberate Cedar Flashcard/study path is blocked by missing saved-message actions085. Multi-user mixed deck plus undecked session accounting076 was not executed and is not claimed passing. A one-card Alice session and Bob's five-card single-deck session do not establish that control.

Offline two-tab logout and Alice's queued Note recovery passed, with Bob's source GET correctly denied. Browser isolation failed separately: QA recent question metadata crossed accounts086, and browser Back restored Alice's full private Note text in Bob's Flashcard generation form064. No Bob generation/save was submitted using that text. Earlier Bob→admin Quick Ingest metadata exposure082 is retained in the separate bundle.
