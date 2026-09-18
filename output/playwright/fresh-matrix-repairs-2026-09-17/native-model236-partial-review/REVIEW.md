# UAT236 AC3 retention correction

**Scope:** additive to the retained native UAT236 review only. This document does not modify the earlier review, its audit, or the old retained packet.

## Criterion and decision

Task AC3 requires a **native fresh setup** that reaches the TestBot library Chat and produces the intended real completion **without unnecessary reselection**.  The preserved picker evidence is useful boundary evidence for the provider/model availability repair, but it does not satisfy that end-to-end criterion.

**Decision: UAT236 AC3 remains pending.** No accepted task finding is retained for UAT236. UAT261's output/reasoning result remains a separate outcome and cannot be used to weaken or close AC3.

## Why the earlier packet cannot close AC3

- The timestamp-bound picker sequence contains a picker reselection. Its canonical projection also does not establish the required exact completion predicate.
- The retained PostgreSQL library captures establish library-page state only. They do not jointly prove fresh setup, a subsequent real completion, and absence of reselection.
- The retained SQLite library capture records a setup-blocked model-selection path, not a successful current native completion.
- The retained fresh TestBot first-turn records and the earlier MCP-profile wizard evidence do not supply all three AC3 facts in one post-fix native sequence. In particular, the available wizard evidence does not establish a native Character completion.

No retained artifact found during this review proves all AC3 elements together. The evidence was inspected by path, hash, timestamps, and safe structural predicates only; provider content was not copied into this report.

## Corrected retained packet

`output/playwright/fresh-matrix-repairs-2026-09-17/native-model236-partial/manifest.json` has SHA-256:

`e704172149b03af0353ad2b18f1e70b9e3fbc5ba3764d43e8dc51b43fc0f98c5`

Independent retention checks found:

- `acceptedFindings` is empty; `partialBoundaryEvidence` and `nativeAcceptancePending` each contain `236`; `fullMatrixAccepted` is false.
- All four retained evidence payloads match their declared checksums and their source bytes. All six checkpoint entries match.
- The manifest's ten omitted provenance records equal the timestamp-provenance input records after removing the packet-only nonempty `reason` field.
- The manifest records zero secret and JWT matches under its 122-variant scan.
- Retainer SHA-256: `175b77f69470a119d6f23f1b75bf59dba2ef87d31f66b5df032f701eef46d417`.

The old `native-model236-accepted` packet remains a valid byte-preservation record, but it is superseded for task-closure purposes by this partial packet. Its previous integrity result must not be read as AC3 acceptance.

## Verification performed

```text
python3 retention/provenance projection check
exit 0
```

The check recomputed the corrected manifest and payload SHA-256 values, compared packaged payloads to their mapped sources, verified every checkpoint entry, projected the packet-only omission reason out of the ten provenance records, and inspected the manifest acceptance flags. No product, runtime, browser, provider, Git, or task files were changed.
