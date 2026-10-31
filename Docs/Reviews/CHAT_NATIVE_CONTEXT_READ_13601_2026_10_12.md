# Private canonical native attachment context — TASK-13601

The accepted H2 context interface requires explicit ordered reference roles, exact asset/reference revisions and a persisted opaque manifest fence. Role storage was added in TASK13600; no canonical reader existed. This private prerequisite now reads that metadata without mutation or byte access. It does not expose routes, readiness flags or new capabilities.

## Change and boundary

NativeAssetContextV1 is strict, frozen, rejects unknown members/numeric revision coercion and keeps the ordered manifest as immutable entries. Distinct references may share exact bytes with different roles/context flags; duplicate reference IDs and conflicting aliased byte descriptors refuse.

NativeChatAssetStore.read_native_asset_context requires the actual owning transaction, exact client/scope and active protected conversation. Workspace locks precede conversation locks. Bundle version/state/projection/settings-revision shape must be valid and its original nonempty bounded string asset_manifest_revision is preserved exactly. No implicit empty/default bundle, new column or derived fake revision is introduced.

Original operation provenance is read without locking it after the conversation; neither _row nor lifecycle-mutating resolve_operation is called. Exact scope/target/projection must match. An active restored conversation may retain a creation key in permanently gone state; the reader never revives it. Preparing/rejected/expired descriptors refuse. The existing byte reader retains its narrower committed-only admission and filesystem checks.

Every reference in the owned conversation is validated, including disabled ones. Unknown NULL role, incompatible role/representation, wrong owner, duplicate or negative positions, invalid revisions, dangling/foreign/released claims and ambiguous asset identities refuse. Lawful stored position gaps remain in order. Candidates lock in sorted ID order before sorted claims, matching existing reclamation. Fractional SQLite asset revisions refuse before string conversion.

A shared DB-only claim helper preserves the original byte-reader descriptor checks: exact original generation/operation, adopted candidate, live matching owned claim, confirmed reserve intent, no release intent and exclusive canonical namespace/upload/storage key. AST comparison confirms the extracted validation tail is unchanged except the already-validated original operation ID source. No byte I/O/network occurs in the metadata reader.

Scope is fork-backed retained claims and known empty protected native context. Retention-backed claims, unavailable-marker storage, ordinary pre-retention source fences, context CAS/replacement/restoration mutations, publisher, public routes/capabilities and all-entry send admission remain pending. Ready bundle metadata in the nonempty fixtures is explicitly seeded over genuinely prepared/adopted bytes; those cases do not qualify a rich publisher. Restored context metadata does not qualify the separately narrower restored byte reader.

ADR required: no. ADR-049/050 and accepted H2 design/Task2.3 govern this private interface; no new durable architectural rule or migration.

## Verification evidence

Supported Python3.12.13 with the project environment activated first. Official PostgreSQL fixture with TLDW_TEST_NO_DOCKER=1; no alternate database provisioning.

- Initial causal red: 27 failed/27 official PostgreSQL skips/11 warnings in12.09s at missing reader. Initial green with the connected byte-reader controls:59 passed/59 skipped/11 warnings in28.16s.
- Additional causal fractional-revision control:1 failed/35 passed/31 skipped/12 warnings in12.93s. SQLite accepted a fractional stored value, and the old reader did not refuse it. Positive-integer validation addresses that exact hole.
- Both source reviewers identified original-operation projection mismatch. Actual-row red:1 failed/1 officialPGskip/65 deselected/11 warnings in1.45s; exact operation/child projection validation addresses it.
- Ten-file connected run:361 passed/238 skipped/12 warnings and2 test setup errors in134.76s. Automatic import cleanup had removed an aliased real-fork fixture. This is explicitly not an overall passing run. Two actual empty/restore controls moved into the existing fork-commit module with its existing registered fixture, preserving prepare/commit/delete/restore assertions. No redundant fixture wrapper, lint suppression or disabled test.
- Final affected reader plus fork-commit suites:88 passed/81 skipped/12 warnings in48.46s, exit0. They exercise actual empty commits, original manifest revision1, actual deletion→restore with unchanged gone keys, shared/disabled references and lawful gaps, malformed metadata, foreign/scope/closed workspace denial, ambiguous claims, descriptor/quota failures, strict immutable wire identity and all existing fork-commit controls. Previously passing unaffected suites were not rerun without another source change.
- Final additional actual shared-reference soft-delete/restore:1 passed/1 officialPGskip/65 deselected/12 warnings in1.38s. Refuses while deleted; after restore complete original manifest/revision, claims and permanently gone operation remain exact. Nonempty publication metadata remains explicitly seeded.
- Final four-file direct Ruff passes; matched-filename baseline/current0/0. New model and new methods formatted with AST-preserving checks; new test Black check passes. No broad legacy formatting. Supported Python compile all4 files and diff-check pass.
- Actual two-production-file Bandit baseline/current0/0 findings/0 errors; coverage verified. Extracted shared byte-claim validation AST comparison passes.
- Canonical full OpenAPI export and checked-in path union match:2129 paths/3301 schemas; SHA256 3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691. New private model does not widen public schema/routes.
- Both independent source reviewers return final bounded CLEAR, including moved actual lifecycle controls and final shared-reference restoration control. Reviewers ran no tests.

Physical PostgreSQL execution and races remain unqualified; skips use the official unavailable fixture and existing PostgreSQL-driver-specific SQLite controls. Raw startup logs/configuration/headers/transcripts are private under /private/tmp/chat-native-context-13601-* and are not published.

## Runtime and remaining work

No browser/profile/global config/auth/key/card/runtime database edits, new sends, windows or tabs were needed for this private metadata-only unit. The prior owned API/browser services and immutable original preservation baseline were left untouched. No installed-extension, real-model or accessibility study claim follows from these checks.

Initial normal exact-owned nine-file hooks pass (exit0); house Ruff/Black hook filters skip these paths and meaningful manual checks above pass. All three bounded stages complete; only the own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_native_context_read_13601_completed.md. Final normal exact-owned eight-file hooks pass (exit0), no bypass; local commit follows. Fresh dev remains a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero missing commits. Master TASK13458 and H2 parent TASK13261.4 remain In Progress.
