# Independent fresh copy/preflight audit — parent TASK13260

## Disposition

**Clear for the bounded preparation evidence.** All four original archive manifests identify released revision `8f8774e6c868b304a96d95ab82e28389c129a78b` and contain exactly the same ordered tracked entries: **33,433 each (33,432 regular files with SHA256/size plus one preserved symlink)**. No archive entry mismatch or duplicate path was found. Canonical manifest-entry digest: `37e5b78a8201b03859a15e874b8dd30f7db1c48d7cff4288fab18e914f59bba2`.

Release gate is `RELEASED`, run `fresh-final-20260917`. Its **15 harness hashes match current exact files**. `complete.json` names the same revision/gate and the original 694,169,600-byte tar. An independent streaming SHA256 read matches its recorded hash: `8b76fd0b8c3b2a858821bfad9976be5602dd139e8a827dfba4a02f2b3d96873a`. No archive was created or extracted for this audit.

## Four preflights

| Cell | Original tracked entries | Source hashes matched to original manifest | Python origins | Frontend origins |
|---|---:|---:|---:|---:|
| sqlite-single | 33,433 | 9 | 7 | 4 |
| sqlite-multi | 33,433 | 9 | 7 | 4 |
| pg-single | 33,433 | 9 | 7 | 4 |
| pg-multi | 33,433 | 9 | 7 | 4 |

All four completed preflight JSON receipts were available at audit time. Each records its distinct cell archive root, the released revision and the shared copied Python interpreter/prefix. The three application package origins are beneath that cell's source root; uvicorn, FastAPI, Pydantic and psycopg origins are beneath the copied virtual environment. Next, React, ReactDOM, TypeScript and the Next CLI resolve inside each cell's copied node_modules. Normalized per-cell origins agree. These are retained preflight observations, not a second app import or launch performed by this reviewer.

## Dependency metadata

The Python copy receipt's seven original `.pth` hashes exactly match `reviewed-pth-inventory.json`. Its after-inventory preserves all seven bytes/hashes and renames only the two reviewed editable hooks to `.pth.disabled-uat`. Exact copied files were hash-checked; both original active editable-hook names are absent. No directory tree or runtime-generated file was walked. All four frontend copy receipts consistently record the three copied dependency trees, local `.` alias, removed copied caches and no copied Next build. Installed dependencies are reused; this does not certify a clean installation or a full dependency-file inventory.

## Boundaries and limits

- Original archive manifests, captured before runtime/profile initialization, are the tracked-source boundary proof. Their creation precedes `complete.json`; the completion receipt's no-runtime/profile flag is historical, not a claim about current running state.
- This audit did not walk or re-hash mutable cell source trees after startup, inspect runtime files, read private credentials/configuration, create profiles or databases, start processes/browser sessions, or copy/extract any archive.
- The original tar hash was independently checked; per-file equality is between the four retained original manifests, with each preflight's nine selected hashes also checked against its manifest. This is not a fresh per-file re-extraction verification of the tar.
- Shared copied Python dependencies and reused browser/model services remain explicit isolation limits. No account creation, bootstrap success, API health, native workflow, inference, tenant isolation or full matrix acceptance follows from this preparation review.

`independent-review-manifest.json` binds every input and this report. No original evidence file was modified.
