# UAT127 / TASK13260.67 — completed ingest reattachment stalls

## Finding

**Confirmed frontend effect-lifecycle defect.** The resumed wizard can successfully read a completed server job and discard that result forever. The reattachment effect keeps a started-signature marker after cleanup. Next's configured React StrictMode replays mount effects: the first setup starts the GET, cleanup marks that poll cancelled, and the second setup sees the retained signature and returns without a replacement poll. The first completed response is then ignored. The session stays processing, so a reload repeats the same sequence.

Current product inspected at repository HEAD `76f04209d39cac726cef74dc6e219221b0d9f381`; cycle5 controller records product freeze `ab527eb3b4` with later evidence/tracking commits. This investigation changed no repository files, browser state, server processes, or inference state.

## Native evidence and precise boundary

- `/private/tmp/cycle5-multi-native-ingest-submit.txt`: POST ingest jobs HTTP200 at 08:22:08.632Z, batch `cf5e70fb-6325-4340-92ef-f6f0258c3740`, one job ID2, file source.
- `/private/tmp/cycle5-multi-native-ingest-resume-wire.txt`: Minimize → Notes → reopen retained job2; observed GET200 processing at 08:22:55.712Z and no new ingest submission during the captured observer window.
- `/private/tmp/cycle5-multi-native-ingest-resume-terminal.txt`: reopen observed GET200 at **08:28:00.480Z**, outer `status: completed`, `completed_at: 08:23:07`, progress100, nested `result.status: Warning`, `media_id: 1`, media UUID, null error, duplicate analysis warnings.
- `/private/tmp/cycle5-multi-native-ingest-settled.txt` and `.png`: retained processing0/1, Results disabled, waiting for server results. Root reports this capture is >26 seconds after the terminal read and ordinary reload also remains stuck; this audit did not perform that reload.
- `/private/tmp/cycle5-multi-native-ingest-results.txt`: additional retained processing snapshot.

The job's temporary file source path and repeated warning text do not explain the stall. `quick-ingest-session-reattach.ts:128–154` reads outer job status and extracts the nested result; `:157–176` correctly derives completed for this saved-warning payload. `ingest-job-results.ts:131–144` deduplicates warning text and retains saved media navigation. The wizard's result/progress builders already support this shape. The failure is between an awaited, valid reattachment snapshot and applying it to wizard state.

## Causal code

All paths below are relative to `/Users/macbook-dev/Documents/GitHub/tldw_server2`.

- `apps/tldw-frontend/next.config.mjs:91`: `reactStrictMode: true`.
- `apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx:1093–1098`: effect skips a signature already in `activeReattachSignatureRef`, then records that signature before starting its async poll.
- `:1112–1113`: awaits actual reattachment, then discards it when the effect-local `cancelled` flag is true.
- `:1179–1184`: cleanup sets that flag and clears the timer, but leaves the signature recorded.
- `:1033–1037`: the other reset only runs when reattachment is no longer needed. A stuck processing session continues to need it, so this does not rearm mount replay.

This is a concrete StrictMode setup → cleanup → setup failure. A read can be visible in native network evidence without being accepted by the component. No failed-warning classification or new server submission is needed to reproduce it.

## Independent temporary reproduction

Root explicitly authorized a private, process-only Vite transform after initial source triage. It reuses the permanent saved-warning boundary test, actual mounted `QuickIngestWizardModal`, production session store/authority and production `reattachQuickIngestSession`, with the existing controlled `bgRequest` response and actual results component. The only behavioral variable is wrapping the mount in `React.StrictMode`; no product or repository-test bytes are transformed or written. The test file alone is transformed in memory to parameterize the wrapper and record safe state fields.

Artifacts:

- `/private/tmp/cycle5-uat127-strict-reattach-probe.config.ts`
- `/private/tmp/cycle5-uat127-strict-reattach-probe.log`
- `/private/tmp/cycle5-uat127-source-hashes.txt`

Command, from `apps/packages/ui`:

```sh
bun run test --config /private/tmp/cycle5-uat127-strict-reattach-probe.config.ts
```

Result: **1 passed, 1 failed, 45 deselected**, 2.97s.

| Same controlled completed/saved-warning response | Status reads | Persisted lifecycle | Step | Processing status | Results |
| --- | ---: | --- | ---: | --- | ---: |
| Ordinary mounted control | 1 | completed | 5 | complete | 1 |
| StrictMode mount | 1 | processing | 4 | running | 0 |

The StrictMode case fails its expected visible “Items saved with warnings” region. The ordinary control also proves warning text and Media navigation through the existing assertions. This is not a live HTTP test: transport is the existing controlled `bgRequest` boundary returning `{ok:true,data:{status:'completed',result:...}}`; native artifacts independently establish the actual HTTP200 body. UI shell mocks remain as in the permanent test, and a pre-retained real authority lease keeps the comparison focused on polling lifecycle.

## Smallest post-freeze correction

**Expected production owner: only `QuickIngestWizardModal.tsx`**, plus its existing session tests and task67. Reassess only if the permanent interacting test proves another boundary necessary.

Treat the signature as ownership of the currently active effect, not permanent evidence that polling once started. Cleanup must release that effect's signature (conditionally when it still matches), alongside its existing cancelled flag/timer cleanup, so a replacement setup can poll the same persisted job. Retain the cancelled and `operation.isCurrent()` checks: old reads must never apply after cleanup, account/server change, cancellation, or replacement session. Do not disable StrictMode, loosen authority guards, alter classification, restart ingestion, or cancel the server job to make progress.

The private reproduction above is RED only; this audit did not apply even an in-memory production fix or claim GREEN. Implementation should first retain a permanent actual StrictMode regression, then make the minimal lifecycle edit and verify it.

## Required bounded regression/verification set

1. **Terminal on first resumed read under StrictMode:** hydrate owned direct-job tracking, return the native-equivalent completed/saved-warning envelope, then assert step5, persisted complete1, warning visible, correct original item mapping and Media1 navigation. Include no-Strict control. Use the actual reattachment service, not only a mocked completed snapshot.
2. **Processing then terminal:** real service controlled GETs, one active continuing poll after replay; advance the existing interval and assert terminal projection stops subsequent polling. Include clean saved completion to establish Warning is incidental.
3. **Lifecycle:** actual modal owner Minimize → route/remount → reopen and hydrated normal reload preserve session/batch/job IDs, reach terminal Results, and make no new ingest POST/upload/cancel request. Use persisted file metadata, whose original browser File is absent after reload.
4. **Late-read negatives:** the abandoned first replay read resolving later cannot overwrite newer progress/results; cleanup clears local timers. Preserve existing cancelled-session, replacement-session, account/server ABA and changed-job-map tests. No foreign job cancellation or automatic mutation replay.
5. Run existing session/integration, button resume, direct reattach and UAT105 result/classifier tests; scoped lint comparison and relevant type baseline after implementation. Native rerun after source freeze in both modes must verify completed Results, saved-warning navigation, and no additional ingestion submission across minimize/resume/reload. No new inference is necessary if existing saved jobs remain available and authorized.

## Relation to prior issues

Keep **UAT127/task67 distinct**. UAT104/task45 repaired Minimize's failure to dismiss the modal; its retained test uses **extension-runtime** tracking and checks dismissal/identity/resume, without direct-job terminal reconciliation. UAT105/task46 repaired saved-warning projection and its actual direct reattachment control has no StrictMode wrapper. Existing polling-loop coverage at `QuickIngestWizardModal.session.test.tsx:2420` also mounts without StrictMode and mocks the snapshot. UAT082 authority protections must remain intact; they are not the cause proven by this controlled comparison.

References for prior scope: `output/playwright/cycle4-repair-verification-2026-09-16/ingest-minimize104/uat104-implementation-report.md` and `cycle4-small-ui-independent-review.md`. Minimize's native dismissal/no-resubmit observation can remain valid, while complete progress/results acceptance remains blocked by127.

## Frozen source hashes

SHA256, retained in `/private/tmp/cycle5-uat127-source-hashes.txt`:

- Wizard: `6c8e5a7d9b629daefd14126aff846277864fd82f098a362e99e113800e709a5d`
- Existing session test: `625ded3b53b0e5da4a6fe32cdeddf8adab481daef759887cb63e3899cf0e2f71`
- Reattach service: `d6f213552ff65a0b10dc7c9cdd0deac2513713dc332550567bba65ecc4cbeb9d`
- Result classifier: `93f08e1801015a09dfaad638f3a9eefded7fd30841cc33581dc3c65d4e2d38b8`
- Next config: `e12475ff9274f9447bd5c8b2c6380f868e6d0883f6dc163446fe2236bb7cd465`
- Private probe config: `4357fba854129c5fc2641651a863f09257c058a886284d55b0258776935f72cd`

No live runtime instrumentation proves the browser's exact internal effect trace; the native outcome plus identical source-level mounted reproduction establish a concrete repair target. Single-mode comparison was still being collected by root; this report makes no single-mode native failure/pass claim.
