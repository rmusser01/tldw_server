# UAT205 independent review — correction required

TASK13260.143. Read-only review of author manifest d69d0b9a97821c3928e5e1b07494121dc28ba2cdd5459f0d9780025f7c2060b3. All three source/test hashes and snapshots matched before independent verification; exact copies retained under initial-review-snapshot/.

## Finding (P2): completed job without an available result remains pending forever

`StudyPackCreateDrawer.tsx` now derives pending solely from non-null jobId (line128) and disables Create/loading for that entire lifetime. The existing completion branch only handles a truthy result deck_id (line102). The API explicitly permits a completed job with no result: `flashcards.py::_study_pack_from_job_result` returns None for missing pack_id or an unavailable pack, and the response schema/client type permits null study_pack and optional/null deck_id. The real job hook stops polling for every completed status. Consequently this valid response leaves the drawer disabled with “Waiting for a status update” even though no further status polling will occur.

A nonmutating Vite loader appended an independent case to the author's real Drawer + real TanStack fixture. After queued→completed/null and fetchStatus=idle, the expected enabled button was still disabled. Result: **1 expected failure / 10 deliberately name-filtered tests**, 1.19s (`completed-missing-result-red.log`). This is an actual rendered/hook behavioral failure, not a class/string test. `probe.config.ts` and `completed-missing-result.probe.txt` retain the exact probe; no repository production/test source changed.

Recommended bounded correction: treat completed-without-usable-deck as terminal unavailable-result feedback, clear local jobId and preserve inputs for deliberate retry, without a success callback or navigation. Also control non-null pack with missing/null deck. Author and parent notified before any edit.

## Independent evidence on initial frozen bytes

- Default frontend config: **32 passed / 4 files / zero skips**, 3.07s (`focused.log`).
- Nonmutating exact old drawer replay: **5 expected failures / 5 passing controls**, 4.34s (`baseline-red.log`).
- ESLint both TSX paths: **0 errors / 0 warnings**.
- Full fresh compiler: **90 errors**, byte-identical to retained baseline (90); zero owned-path diagnostics. This is not a clean project type-check claim.
- Bandit: **0 findings / 2 TSX parse errors**; no TypeScript security assurance. No new credential, HTML, network, authorization or storage surface was found in the four locale additions/local pending derivation.
- Initial test command combined --root and repo-relative --config incorrectly; exited before collection. Preserved as `initial-command-path-error.log`, then corrected to run from the frontend cwd. It is reviewer command setup failure, not product evidence.

## Scope and preserved behavior

Original accepted/poll-gap/error controls meaningfully use real service hooks. Failed/cancelled clears jobId and retains inputs; normal completed deck invokes existing success/navigation once; rejected POST allows retry. The authority event→revision generationKey→keyed ImportExportTab source chain exists, and the late-POST keyed drawer control is useful component evidence. It does not constitute a native account-switch or full AuthNZ acceptance test. Existing close/reopen/new intent resets remain; this repair does not add durable tracking or remote job cancellation.

No browser/runtime/config/model/DB/task/tracker/git mutation performed. Final clearance is withheld pending the bounded terminal-result correction and fresh frozen verification.
