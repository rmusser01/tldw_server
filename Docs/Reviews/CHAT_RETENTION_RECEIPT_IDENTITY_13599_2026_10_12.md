# Original typed retention receipt identity — TASK-13599

The private operation store now refuses malformed committed retention records instead of returning committed with receipt=None. It parses the existing strict frozen request/result, recomputes the original semantic digest and binds owner, scope, operation, projection, source/target and exact reference-map keys. Original binding validates before a substituted missing target can erase the recovery record through lifecycle handling.

ADR assessment: no new ADR; ADR049/050, proposed070 and approved H2 §§3/8/9/Task2.3 govern this original-key structural result boundary. Scope reaches existing schema, projection digest module, operation store and one new real-DB/property fixture. No public route, capability, schema shape, client/UI, quota, byte adapter or source publication changes.

## Minimal shared correction

Retention request references must be unique. The canonical digest uses the existing exact-Unicode compact UTF-8 SHA-256 machinery over this fixed tuple, with no normalization or object-order dependency:

```text
["tldw-native-asset-retention", 1, operation_id, owner_key,
 [scope_kind, workspace_id_or_null], source_conversation_id,
 expected_asset_manifest_revision,
 [[reference_id, reference_revision, asset_id, revision, representation,
   role, context_enabled, hash_or_null, fidelity_disposition], ...]]
```

Supplied request_digest is excluded from itself. All original ordered reference fields and selected opaque manifest revision remain bound. Stored source and target must both equal the original source. Output manifest revisions remain opaque; no +1 inference or current source recapture repairs the result.

The existing result dataclass admits immutable fork or retention receipts. Retention asset_map keys exactly cover unique original references. Multiple references may share one output asset/revision only with compatible frozen hash, representation and fidelity; role and context-enabled flags may differ. Target/workspace authorization and gone tombstones remain. The retention target's creation operation is not repinned to this retention operation. Existing fork-kind commit behavior stays unchanged; the asset reader continues to refuse retention-kind claims.

## Causal and connected evidence

Corrected causal RED37 failed/1 passed/22 official unavailable PostgreSQL skips precedes production edits. This includes fifteen missing digest cases, duplicate request acceptance, nineteen unrefused corrupt/alias cases and untyped seeded results. Two initial test-only diagnostics are excluded: workspace setup omitted house scope_type and a house query cursor was indexed as a list. Both are corrected through existing conventions while production remains unchanged.

Affected green38 passed/22 officialPG skips/12 warnings,11.04s. Final formatted eight-file connected run323 passed/205 skipped/11 warnings,137.04s. Collection metadata attributes skips to201 unavailable official PostgreSQL cases (197 parameterized plus four direct fixture cases) and four SQLite controls that require PostgreSQL driver transaction behavior. Passing tests were not rerun to attribute skips. The first metadata parsers assumed node-ID/full-line formats; actual collection emits a function/coroutine tree, whose source-backed counts resolve attribution.

Tests exercise real store transactions, missing/corrupt/wrong-kind records, every original identity field, exact map keys, duplicate references, missing-target substitution preserving committed state, legal shared aliases, contradictory hash/representation/fidelity aliases, nested immutability, owner/digest denial and deletion tombstones. Pure controls cover all selected fields/order/Unicode and property-based exact wire roundtrip. Typed committed rows are explicitly fixture-seeded: they do not execute retention publication, install claims/roles, charge quota, read bytes or establish canonical readiness.

Four-file Black formatting preserves exact AST. New-test import ordering and constant-getattr lint diagnostics are repaired using normal imports/direct attribute access after the production helper exists. Matched-filename Ruff HEAD/current0/0, no introduced findings. Three-production-file Bandit baseline/current0 findings/0 errors with coverage asserted; supported3.12 four-file compilation exits0. Both independent source reviewers CLEAR.

Normal full canonical API export, fingerprint and path union match at2129 paths/3301 schemas, SHA3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691. The initial anchored diagnostic grep missed main's logger-prefixed exporter messages and returned1 after the exporter succeeded; actual artifacts and fingerprint/path success markers are verified. No contract file regeneration or source workaround occurs. Startup diagnostics and canonical request bodies remain private; finite evidence lives under `/private/tmp/chat-retention-receipt-13599-*`.

## Bounded closure

Nine initial exact-owned normal repository hooks exit0 without bypass. Manual matched Ruff/Black/Bandit checks qualify paths filtered out by normal hooks. Final eight-file metadata hooks, canonical task recording and local incremental commit close this unit. Only its own completed plan retires privately. No server/browser/profile/key/authentication or immutable original conversation baseline changes. Latest fetched dev a744577 remains included.

Actual retention source-fence admission/publication, canonical role/context manifest, live claims/bytes/readiness, rich fork/read/send/frozen character/generated revision/client/Sync, public capability, physical PostgreSQL/AT/installed extension and whole-ten/study acceptance remain open. An internally consistent stored result is not proof of those later adapters.
