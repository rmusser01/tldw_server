# Independent TASK13260.125 harness review

## Verdict

Clear within the approved private-harness scope. No material defect found in the preparation ownership, initialization proof, or startup gates. This review author did not implement the repair. Full-matrix execution remains separately gated.

Fresh verification: **23 Node tests passed, 0 skipped; 5 Python tests passed, 0 skipped.** Exact commands, logs and hashes are in `verification.json`, `node-tests.log` and `python-tests.log`. Both commands exited 0. All seven current files match both the frozen owned manifest and their review snapshots. Owned manifest SHA256: `bfbfbfb56793fa4ecb95c276d0f4da5415d7a78a9bd4e487a2be177490129d1c`.

## Reviewed behavior

- Source ownership compares canonical roots. Preparation reserves a new source in a private preparing record before helper writes; an existing binding or runtime directory is refused. A helper failure retains the binding and partial state. Completed preparation is required before initialization or launch. Serial scheduling remains the documented parent responsibility.
- The wrapper calls the initializer from the selected frozen root with `non_interactive=True`. It writes an exclusive mode0600, attempt-specific proof only after the coroutine returns normally. SystemExit(0), KeyboardInterrupt and other exceptions do not write it. The module-origin check occurs after module import and before calling main; it is not claimed to prevent import-time execution.
- Launcher completion additionally requires raw exit0, no forwarded interruption, completed proof status and exact attempt/preparation identity. Nonzero exit, missing proof, old attempt proof, changed preparation or incomplete proof cannot certify initialization. Both backend and frontend reject missing, legacy or mismatched success markers before port checks or spawning.
- Each initialization retry gets a fresh UUID proof path. Logs use time plus UUID and exclusive creation, preserving rapid failed attempts without overwriting. Partial data and old proof files are retained. The existing action process receipt records the latest attempt; it is not a per-attempt receipt archive. The retained attempt logs and normal-return proofs are separate. No automatic reset/cleanup was added.
- Backend/frontend commands, runtime environment construction, provider override stripping and PG receipt validation remain unchanged apart from the new prerequisite checks and process-result recording. The official PG holder, official fixture test, pytest isolation configuration and runtime `PROTOCOL.md` match their original snapshots byte for byte. Protocol SHA256: `31eb6a32988a6b03245c4a165c98b1bcbd0799909d331dc82e7f1f852c015c5b`.

## Verification limits

Before running the controls I inspected their boundaries. Node evaluates the actual launcher control flow in a VM with fake inspection, helpers, runtime environment, ports and child processes; real process preflight and dynamic imports throw if reached. Its filesystem writes stay in newly created disposable test directories. Python disables plugin autoload and repository conftest collection and supplies a fake initializer through import_module. No real launcher CLI action, application import, official fixture collection, database operation, archive/dependency copy, server, browser or provider call ran. Real `profiles` and `holders` directories remain absent.

The author’s retained original-launcher replay reports 19 failures and 4 passing controls; I inspected the diff and regression assertions but did not rerun that replay. The fresh 28 controls establish the repaired fake-boundary contracts, not successful dependency relocation, real initialization/authentication, port availability or native workflows. Normal return is intentionally a control-flow proof. The nine entrypoint fingerprints remain a partial check; the parent must retain a full source-archive manifest and runtime provenance later. No product, task, tracker, git or runtime changes were made by this review.
