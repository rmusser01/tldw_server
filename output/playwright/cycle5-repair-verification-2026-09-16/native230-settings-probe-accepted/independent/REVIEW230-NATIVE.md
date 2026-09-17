# UAT230 — independent native acceptance audit

## Verdict

**PASS for the remaining native portion of TASK13260.171 AC3.** The original authenticated same-origin Settings probe no longer appears during ordinary entry, reload, or logout/login, while Settings and Connection remain usable. Combined with the already completed independent 87-test/static review, this supports closing UAT230 after the parent retains the evidence and records the tracker outcome. No additional UAT230 native step is missing. This audit does not start or certify the fresh matrix.

## Source and input binding

All **24** allowlisted inputs match `native-manifest.json`. The four before/after source hashes match each other, the independently reviewed source manifest, and the actual Git blobs at committed revision `db2b0602acd2a738fb752613a4086956b3e93bf3`. This is the four-file frontend repair boundary, not a whole-workspace or backend-runtime certification.

## Observation and positive controls

The observer was attached at **12:05:06.909Z**, while the page was still `/notes`, before the recorded visible Settings button and **Edit server** link clicks. Its code observes request/response URL, method/status for `/openapi.json` and the four Billing read routes, successful auth/me ID/username, and pageerrors. It does not intercept responses, alter authentication or suppress requests. The Page-level event history survives the explicit `page.reload()`; each later history contains the previous history as an unchanged prefix.

| Settled round | Authenticated state and positive UI | Time since latest identity | OpenAPI/Billing events; pageerrors |
| --- | --- | --- | --- |
| Initial Settings, 12:05:58.841Z | Logged In; Test Connection shows Core reachable / RAG healthy | 21.390s | 0; 0 |
| Normal reload, 12:06:38.864Z | Logged In; connection results correctly reset to not checked | 23.205s | 0; 0 |
| Normal logout/login, 12:07:53.669Z | Logged In; repeated Test Connection shows Core reachable / RAG healthy | 27.698s | 0; 0 |

All rounds exceed the existing five-second optional-probe timeout. The complete observation is **166.760 seconds**, with **six unique identity events**, all administrator1 / uat_admin. The three normalized acceptance entries exactly match their raw CLI `### Result` objects. They are cumulative captures, not six identities per round. The absence is supported by authenticated Settings content and successful Connection controls; it is not an empty observer alone.

`reload.txt` contains the actual `page.reload()` command. `logout.txt` contains the actual Logout click and its resulting **Login Required**, visible Username/Password fields, and Login button. `login-snapshot.txt` is the later restored logged-in page; it is not the pre-login form. The private helper output is excluded and was not read. The next successful auth/me identity at 12:07:25.971Z and final UI provide the post-login evidence.

I viewed both PNGs. The initial image shows the same-origin server `http://127.0.0.1:18583`, Multi User (Login), Logged In and successful Core/RAG checks. The final scrolled image again shows Logged In and successful checks. Their cropped/scrolled viewports are sufficient for these controls, not a full-page layout certification. The body receipts also contain no Billing controls.

## Limits

This is the reused privileged PostgreSQL multi-user quickstart administrator case. It establishes no new restricted-role isolation, fresh-install, direct-backend native Billing discovery, or matrix result. Direct-backend capability/404/abort/logout/server-change behavior remains covered by the separate automated suite. The original two404 are retained in the earlier UAT225 native package; they are not erased or relabeled.

The observer is deliberately scoped to the affected page and endpoints. **Zero pageerrors is not a clean-console or all-network assertion**, and this packet makes neither claim. The repair is supported by the absence of precisely the unsupported probe while the intended UI is active. No provider/model, API fixture, browser storage or credentials were modified by this audit; no browser/runtime, source, task/tracker or git mutation was performed.

`verification.json` contains the independent comparisons and exact timings. `input-manifest.json` binds all supplied inputs plus their manifest; the task criteria are preserved through a read-only official CLI capture. `audit-manifest.json` binds this frozen audit. Earlier source review: `.tmp/uat230-independent-20260917/REVIEW230.md`, SHA256 `95cf578e38527750408e3239e5fff3ce1c367a171567077f057722b95031efaf`.
