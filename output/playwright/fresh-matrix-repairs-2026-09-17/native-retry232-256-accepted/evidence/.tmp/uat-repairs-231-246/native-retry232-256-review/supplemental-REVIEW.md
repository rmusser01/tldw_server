# UAT232/256 supplemental native review

## Verdict

**Pass.** The retained synthetic UI evidence contains the expected unavailable
selected-model guidance and retry action before the later retry. The canonical
reload contains the expected success markers and has no active unavailable
guidance or retry control. The canonical record grows from five to seven rows;
all five original rows remain byte-identical, with only hashes emitted.

For each immutable PostgreSQL upgrade copy, the live-capture receipt is matched
to the actual binding file. The binding's profile, initialization, and holder
hashes match the corresponding retained records. Each copy's
`chat_service.py` bytes match the revision-bound source manifest, and the two
immutable copies have identical service hashes.

## Audit

Run from the repository root:

```sh
node .tmp/uat-repairs-231-246/native-retry232-256-review/supplemental-audit.mjs
```

[supplemental-audit.json](supplemental-audit.json) records 14 passing checks,
safe counts, and SHA-256 digests for 21 retained inputs. It does not contain
raw UI, provider, profile, initialization, holder, binding, or source-manifest
contents.

## Limits

- This is read-only evidence review. It did not invoke Git, inspect a live
  process, or alter source, a profile, database, runtime, or browser state.
- Revision provenance is established from immutable binding and source-manifest
  records; this review does not independently ask Git to resolve the revision.
- It covers the retained retry sequence and immutable-copy parity only.
