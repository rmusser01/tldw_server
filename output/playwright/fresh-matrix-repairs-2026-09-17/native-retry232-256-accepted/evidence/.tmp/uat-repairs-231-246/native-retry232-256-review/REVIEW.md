# UAT232/256 independent native retry review

## Verdict

**Pass, with a stated UI-capture limit.** The retained retry execution is bound
to the original conversation and failed client-message identity, receives HTTP
200, and ends with exactly one successful SSE terminal marker and no error
terminal frame. A normal canonical reload reports seven rows, including the
expected new user then assistant rows, and the conversation record also reports
seven messages.

The immutable-startup receipt records the approved repair revision for backend
and frontend processes in both `pg-single` and `pg-multi`. Each process records
the same approved source revision, the approved immutable upgrade root, and a
receipt digest. This establishes the available source/binding/holder parity for
the retained startup evidence.

## Audit

Run from the repository root:

```sh
node .tmp/uat-repairs-231-246/native-retry232-256-review/audit.mjs
```

The generated [audit.json](audit.json) contains only check outcomes, safe
counts/types, and SHA-256 digests. It passed all 14 checks against eight
retained inputs.

## Evidence reviewed

- `native-targeted/pg-single/retry256-{started,observed,canonical-reload}.txt`
  for request identity, response status, SSE terminal state, and canonical
  persistence.
- `native-targeted/pg-single/retry256-{retry,canonical-reload}.js` plus the
  pre-retry and settled UI captures for artifact retention and traceability.
- `native-upgrade-preparation/retry256-startup-safe.json` for the revision and
  four-process immutable upgrade matrix.

## Coverage limits

- This is a read-only review of one retained `pg-single` retry and its normal
  reload. It does not establish all provider, cancellation, or multi-user retry
  behavior.
- The UI captures are retained and hashed, but this review deliberately does
  not reproduce their raw text or classify visible labels. The canonical reload
  and request/stream evidence are the independently machine-checked recovery
  proof.
- No browser, runtime, database, model, source, or test changes were made.
