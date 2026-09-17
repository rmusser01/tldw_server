# UAT237 independent native acceptance

**CLEAR for bounded AC1 and AC2 acceptance on the repaired PostgreSQL single-user profile.** Task TASK13260.179. Source revision: `6f6983b0620aae1f0892c6b0d3ae3bebfc105e02`. This review inspected retained evidence and source bindings; it performed no browser, process, database, product, Git, or task mutations.

## AC1 — disconnected Media recovery

The retained Disconnect action clicks the visible **Disconnect** button in Server Settings. Normal navigation reloads `/media`; the settled snapshot and a subsequent resnapshot show **“Add your credentials to use Media”**, with **Open Settings**, and no **“Failed to search media”** notifications. These are settled UI observations, not an assertion that every possible transient notification or network request was recorded. The observer was installed after the disconnected snapshots, so it cannot prove an absence of requests throughout the preceding interval.

The first settings helper waited for a multi-user **Logout** control and timed out in single-user mode. Two subsequent reconnect attempts failed on stale element references. All three remain in the packet as unsuccessful helper attempts. The successful helper uses the observed **Open Settings** label and waits for the API Key form. They are not counted as application defects or successful reconnects.

## AC2 — reconnect and real outage recovery

The normal credential form was used to save/test the existing profile's API key. Its private helper is hashed only; its contents and filled credential values are not reproduced. The settled settings UI reports **Core: reachable** and **RAG: healthy**. Returning to Media displays **Media Inspector** and `rowan-observatory-public-20260917`, establishing recovery of the existing catalogue.

A separate, real API outage preserves meaningful connected-failure feedback. The receipt records the owned API process **91770** exiting on **SIGTERM** at **23:03:54.945 UTC**. Media first shows readiness testing, then settles on **Unavailable / Backend readiness check failed**, with **Retry**, **Health & diagnostics**, and **Server settings**. Thus suppressing redundant credential-related search toasts does not hide this configured-server outage.

The API was restarted on the same profile, source and port **18702**, as PID **97807** at **23:04:21.465**. The visible **Retry** action ran at **23:04:40.339**; the capture finished at **23:04:40.918**, with Media Inspector and the original Rowan item visible. The observation includes RAG-health HTTP200 at **23:04:40.583**, and actual `/api/v1/media` HTTP200 responses at **23:04:40.861** returning item **1**, title `rowan-observatory-public-20260917`, and total **1**. Both page sizes20 and50 returned the same item. This verifies native catalogue retrieval through the Media search surface; it does not establish a new typed-query/filter scenario.

The unauthenticated health probe returned401. That only shows HTTP reachability; the authenticated UI and Media HTTP200 responses establish successful recovery. The safe process proof records PostgreSQL holders18859/18878 and the independent multi-user API91810 as retained. The interrupted receipt targets the single-user API, not PostgreSQL. This review did not independently resignal, restart, or inspect live processes.

## Source and process binding

The stopped and restarted backend receipts bind to the same immutable binding hash, repaired source root/revision, original profile working directory, and API port. The frontend receipt binds to the same source root and its Next.js working directory. The original profile, initialization and holder receipt hashes are checked against the binding. Preparation gate, completion, source manifest, reused-Python receipt and archive hashes are checked against it as well.

All12 files frozen by the existing independent Media review match the repaired source copy and its source manifest, including `ViewMediaPage.tsx`, `useMediaSearch.ts`, and their connection/outage tests. That prior review records193 Media/wizard tests plus37 downstream consumer tests passing with zero skips; these tests were not rerun here. Existing source coverage includes readiness gating, authority changes, transport-outage recovery, and unexpected-failure diagnostics. Its TypeScript/Bandit limitations remain unchanged.

The process proof was captured at **23:05:10.707**. Process receipts are mutable on exit: the audit records their bytes/status and hash at its own observation time, plus the proof's earlier hash. A later normal process exit may change a receipt without changing the source/profile binding. No full logs, private helper contents, credentials, or authentication headers are included in this report or audit projections.

## Limits

This accepts UAT237's two criteria for this targeted PostgreSQL single-user native run. The original defect was observed on SQLite single-user; this packet does not claim a fresh native SQLite rerun, PostgreSQL multi-user acceptance, a clean-OS install, or the full48-case matrix. The existing isolated profile/data and dependencies were reused. The outage exercised readiness failure and visible Retry, not every possible HTTP500 while an already-mounted catalogue search is active. No fabricated responses appear in the public retry helper or retained response evidence; the private credential helper was deliberately not inspected. Acceptance relies on the correlated real-process outage and real HTTP recovery observations, with the preceding implementation review as complementary evidence.
