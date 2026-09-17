# UAT225 — independent original-scenario native audit

## Verdict

**PASS for the remaining bounded native criteria.** TASK13260.163 AC3 requires repeating the original administrator suggestion reads and recording the outcome/limits. TASK13260.164 AC4 requires independent review/static checks plus those original native reads, with provider/worker and matrix limitations explicit. This packet supplies the native portion; the prior independent Stage B report supplies 173 focused + 200 adjacent passing PostgreSQL/SQLite tests, zero skips, and clean applicable static checks. No further UAT225 acceptance step is missing under those criteria. Parent can close the two tasks after retaining this result and updating the running tracker; this reviewer changed neither.

Real native generation is **not** required by these remaining criteria and was not performed. The unavailable worker is reported truthfully. This disposition does not release or certify the full fresh matrix, or resolve the separate Settings probe finding.

## Verified evidence

- All **36** entries in `native-manifest.json` match their SHA256 values. The two source receipts contain identical **3,669** backend file entries and revision `e1ccad4be7cf5b5c1b4c1e3b7405741a3ad19f0d`, at 11:32:05.953Z and 11:40:29.694Z. Replacement receipt identifies PID10113 starting 11:32:22.589Z and health200 at 11:32:32.189Z. The old owned PID89545 received SIGTERM and released its port; it was still alive during that health receipt, so these inputs do not prove its final exit.
- Safe identity events show the initial Alice session, followed by administrator **1 / uat_admin** at 11:35:24.425Z and repeatedly through the reload. The first helper attempt timed out on the wrong page before filling credentials; the preserved recovery used visible Settings logout and the normal login form. This is a retained harness precondition error, not successful authentication evidence by itself.
- The original two 503 responses from 09:16:33Z are preserved in `original-read-failure.json`. On the same original note, Graph/list/capabilities/runs each returned **200** at 11:37:16.736–.849Z. The list returned a valid empty result and source fingerprint; capabilities returned `generation_available=false`, `notes_graph_suggestions_worker_unavailable`, and the configured llama.cpp/Gemma disclosure.
- `reload.txt` explicitly records `await page.reload()`. The reload returned the Notes list. Visible note/Graph/Suggestions controls were then reopened; new Graph/list/capabilities/runs responses each returned **200** at 11:39:34.610–.705Z. The source fingerprint and capability revision match the first set. This proves fresh requests after reopening, not persistence of Graph selection through reload.
- The safe observer contains **52** events: 14 identity, 19 request and 19 response. All captured domain requests are GETs and all captured responses are successful. Every endpoint summary in `acceptance-result.json` matches the corresponding timestamp, URL, status and decoded body in `safe-events.json`.
- Seven original-note readbacks are decoded-body identical from 11:37:00.215Z through 11:39:54.173Z. Note `11c86b62-d045-4836-9e2b-f9b68b14650f` remains owner1/version1, with the same title/content/timestamps; tag14 remains version1 with the same portable ID. No captured note save, product decision, generation or model call is present.
- I viewed `reload-suggestions.png`: the original note and tag edge render, the panel discloses provider/model/data categories, says **“The suggestion worker is unavailable.”**, disables **Generate**, and reports no suggestions ready. The initial and reload PNGs are byte-identical; the independent evidence for the two request rounds is the distinct event stream, not image differences.

## Limits and separate observations

The reused PostgreSQL profile retains its privileged/BYPASSRLS qualification. This is an administrator read-path check, not a new restricted-role or multi-user native isolation test. The earlier metadata receipt establishes the original absent Sync/owner authority at 09:41Z; it is not a new post-restart database query or interpreter inspection. Backend file manifests establish source stability; this is not a clean whole-repository/frontend-manifest claim.

The safe observer recorded no pageerror and no failure in its captured domains. The console file nevertheless retains **two Settings `/openapi.json`404** entries, logout warnings, development/HMR messages and graph-library wheel-sensitivity warnings. No clean-console claim is made. The Settings probe is separately tracked as UAT230; it does not change the successful original suggestion-read responses. No defective zoom behavior is proven or disproven by these screenshots.

Local generation/publication/decisions and canonical retirement remain supported by the separate automated database suite using synthetic provider replies. There is no native worker generation, acceptance mutation, new provider setup, fresh profile, or full-matrix result in this packet. The audit performed no browser/runtime/service, production, task/tracker or git mutation and read no credentials or private login helper.

## Hash-bound record

`input-manifest.json` binds all 36 supplied inputs plus the supplied manifest. `verification.json` records the independently checked source/event/readback comparisons. The two `task-*-criteria.txt` files preserve the exact criteria read through the official CLI. `audit-manifest.json` binds this frozen audit. The separate Stage B review is `.tmp/uat225-b-independent-20260917/REVIEW225-B.md`, SHA256 `9c1d796970deeb1c7ecb748d830a5259747603e18dcc035ff7e9497c17938502`.
