# TASK13260.203 — UAT261 diagnostic launcher preparation review

## CLEAR for preparation only

Reviewed `matrix-uat261-capture.mjs` against the retained `matrix-upgrade.mjs`. The diff confines launches to backend/pg-single and exact original/upgrade run labels before it reads profile state; binds the diagnostic module and release gate by exact SHA-256; requires the original source commit; preserves the original profile/init/holder hashes in a new private diagnostic binding; requires an absent diagnostic root before any write; and checks the owned port before spawning.

The changed launcher starts only the capture app factory, sets a dedicated capture directory, and records attempt logs/receipts under a new diagnostic root. It does not add source writes, provider tuning, profile provisioning, broad process termination, or overwrite behavior. Signal handling applies only to the child it creates.

## Bounded checks

- `node --check` passed.
- Missing arguments exited with `Use backend|frontend CELL ORIGINAL_RUN UPGRADE_RUN`.
- Invalid action exited at the first diagnostic confinement guard with `Diagnostic is confined to the unchanged original PGsingle backend`.

Neither negative invocation reached runtime/profile/port/process launch code. The first attempted invalid-action receipt had a local review-directory creation race and is not used; the retained rerun is authoritative.

## Limits

This does not execute the launcher or review `uat261_capture.py`, its gate contents beyond the launcher’s checked fields, a native workflow, provider output, or credentials. The capture wrapper remains author-owned and unreviewed until frozen. This review is diagnostic preparation, not UAT261 native acceptance.

## Final helper amendment

Final helper hash: `f2fcf115eafb77c43f2d330526d59eb21f18a2a72e008676503f04389ab65e1f`.

One guard was added immediately after `releasedSource`: `checked.sourceCommit` must equal the diagnostic gate's fixed reviewed source commit. This closes the distinction between the released upgrade gate and the diagnostic gate’s exact source binding. Syntax remains valid; prior negative receipts and review bytes are retained under `prior/`. Verdict remains **CLEAR for preparation only**.
