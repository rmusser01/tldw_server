# Cycle 2 targeted repair evidence

These are targeted checks using the existing isolated single-user data on API18100/WebUI18180 and multi-user data on API18101/WebUI18181, with the real unchanged llama.cpp provider on9099. They do not replace the next full fresh single-user/multi-user workflow run. The running tracker records each finding and the scope of its verification.

- UAT047: Disconnect request inventories over4m13s and redacted healthy reconnection.
- UAT048: the [final targeted retest](uat048-boundary-final-report.md), including the repair frozen in `d40e17dc81`, passes the two-tab offline logout boundary. Notes stays at its app URL with only the native signed-out screen; reconnect automatically opens login. Bob sees only his own Notes, Alice's queued draft syncs with POST201, and an independent Bob GET of that new Alice note returns404. Offline remote token revocation is not claimed.
- UAT049: actual optional Help loader failure, cancelled reload preserving an unsaved field, and successful reconnect/reload/open.
- UAT050/051: unchanged path-like model request and current composer recovery. Original browser-only persistence evidence was disproved and corrected under052.
- UAT052: independent original-conversation and derived-artifact reads, visible Note origin/backlink, correct card question/answer, persisted review schedule. The old defective fixture was rated Again before reviewing the new card; historical bad content is not counted as repaired.
- UAT053/054: settled offline shell and automatic feedback rejection without a blocking modal. Deliberate offline requests still produce browser network diagnostics. Interrupting required initial Chat settings loading is a separate active-request failure control.
- UAT055: fresh Study page with a labeled native session list; the browser console reported0errors/0warnings.
- Environment: simultaneous generated Next caches exhausted disk space and caused two research-run500responses. After retiring only the multi-user generated cache, the independent research-run read recovered200. Future WebUIs run sequentially.

## UAT048 evidence sequence

- `uat048-final-*` files are the **intermediate retest with a remaining cross-tab failure**, despite their original filenames. They are retained unchanged. The [intermediate report](uat048-final-multi-report.md) records the other Notes tab's browser offline error before the cross-tab repair; its Settings screenshot shows an empty login form.
- `uat048-boundary-*` files are the **final targeted pass**. The [signed-out snapshot](uat048-boundary-notes-signed-out.txt) and [screenshot](uat048-boundary-notes-signed-out.png) show the corrected boundary. The later [title check](uat048-boundary-final-title.txt) verifies `Signed out | tldw`, with its own [snapshot](uat048-boundary-final-signed-out.txt) and [screenshot](uat048-boundary-final-signed-out.png). Both screenshots were visually inspected and show only native signed-out text, with no private content. The [ownership control](uat048-boundary-bob-foreign-note.json) records Bob's authenticated404 for Alice's newly synced note.

`manifest.json` records retained file hashes, byte sizes, and the two UAT048 evidence stages. Source snapshots were captured through the credential-redacting browser helper; all retained files were scanned against both test runtime manifests. No runtime credentials, databases, or provider secrets are included. Full fresh-cycle acceptance remains pending.
