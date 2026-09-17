# Independent native audit — UAT206 and UAT215

## Disposition

**Both bounded native gates pass.** Recommend closing TASK13260.144 (UAT206 worker startup) and TASK13260.154 (UAT215 quarantined-job representation), together with their already retained automated and independent source checks. **This does not accept the complete Study Pack result-display workflow:** UAT219 remains a separate observed failure.

The audit reads the 12 exact inputs in `input-manifest.json`, captures the original task criteria in `task206.txt` / `task215.txt`, and derives `receipt.json`. No live browser, runtime, provider, DB, product, task, or git operation was performed. Input originals remain unchanged.

## UAT215 — original job 2

At 2026-09-17 08:13:09.616 UTC the read-only API receipt establishes Alice/id2. At 08:13:09.635, the original `/api/v1/flashcards/study-packs/jobs/2` returns HTTP200, job ID2/status `failed`, `study_pack:null`, and the safe message `Study pack generation failed.` Logout returns200. The later authenticated catalogue at 08:18:06.043 independently lists the same ID2 as `failed`.

This satisfies the original native scenario without recreating or mutating job2. Safe terminal list/detail representation is directly observed. The prior independent source review (`.tmp/uat215-independent-20260917/REVIEW215.md`) records 13 passing, zero-skipped real SQLite/PostgreSQL Jobs tests and Bandit0; the task records the existing 21 frontend terminal-polling controls. Those tests were not rerun in this evidence-only audit. There is no native reopening of job2 in the UI in this allowlist; frontend polling recovery remains established by the retained automated controls. The task's pending native note specifically required the original job2 GET, now present.

## UAT206 — restored default starts and processes ordinary UI work

The normalized startup receipt identifies API PID76778, started08:12:33.057 UTC, and worker `study-pack-worker-76778` startup at local01:12:42.166 America/Los_Angeles (08:12:42.166 UTC). It records that neither STUDY_PACK_JOBS_WORKER_ENABLED nor STUDY_PACK_JOBS_WORKER_ID is present in exclusive environment overrides. No workaround override or alternative launcher was introduced by this audit. This attribution uses the parent's retained startup receipt; it is not an independent dump of the live inherited environment.

The actual UI click in `alice218-pack-submitted.txt` sends exactly one POST at08:13:54.396 for a new pack using Alice's existing note `b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8`; no provider/model override is in that request. Its response at08:13:54.424 is202 with **job5 queued**. The visible dialog says it is queued, then `Creating your study pack.` The captured identity is Alice/id2. Detail polling first reports running at08:13:57.520. The actual trace later returns to queued at08:14:32.761 and to running at08:14:46.526; that intermediate transition is preserved, and its cause is not established here.

At08:18:06.043, the separately authenticated Alice catalogue GET returns200 with **the same job5 completed**. This establishes that ordinary UI-origin queued work was processed to terminal completion by the reviewed default-worker runtime. AC3 expressly asks to record provider or persistence failures separately; it does not require pretending a separate result-read failure is fixed. Prior task checks cover explicit false, disabled route and sidecar startup controls; this native run exercises the restored default only.

## Separate UAT219 result-display failure and evidence limits

From08:15:20.218 through08:15:49.918, this capture records **20 HTTP500 detail responses** for job5 (safe generic internal-server-error bodies). The dialog says `Unable to check study pack progress. Checking again shortly.` and retains title/source with the submit button loading/disabled. There is no successful completed-job detail response or rendered generated pack in the allowlist. The catalogue proves completion status, not generated content quality, exact pack/card counts, or successful UI delivery. Parent identifies the independent response-schema defect as UAT219; these receipts themselves prove the500 boundary, not its internal exception stack. No clean-console or full end-to-end acceptance is claimed.

All three retained source manifests have revision **6d06aae9bd03364f1ce68940c41b14f12fb5febe** and identical3668 path/hash entries before startup, after health, and during the failure. Relevant exact hashes are in `receipt.json`; startup source SHA `ede2e5dac545c9cccb139aaa2cb1f15c995dcf6f0da47b9184c8d2fc32ff7afb` and UAT215 endpoint SHA `cc3f242d2c115a324db2f2ed180256c32408c1cb53a9515e7b795bea4183b107` match the reviewed repairs. These are retained source/launch attribution, not introspection of process memory.

The job5 API receipt reuses a job2-oriented prose scope label; its recorded GET path, authenticated identity and response unambiguously identify the catalogue and job5. Cumulative browser captures were counted from the final event list once, avoiding duplicate counts across pending/progress captures. An initial audit parser assumed the source `files` field was a mapping; it is a list and the parser was corrected without changing evidence.
