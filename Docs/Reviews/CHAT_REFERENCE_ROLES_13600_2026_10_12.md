# Explicit native attachment reference roles — TASK-13600

The reference table previously stored a claim, order, revision and context flag but omitted the role required by the accepted H2 manifest. Embedded-image representation cannot distinguish a message image from a reference image. This bounded storage prerequisite adds an explicit role without interpreting legacy records or granting canonical context/readiness authority.

## Result and scope

New registered SQLite 79→80 and PostgreSQL 83→84 migrations add nullable TEXT role with the four existing values: message_image, document_context, generated_image and reference_image. SQL NULL preserves unknown legacy roles. Explicit unknown/blank/case-altered values fail the constraint. Published native 72/76 DDL, old migrations, historical catalog recognition, existing readers and forced PostgreSQL owner RLS remain unchanged.

The SQLite linear registry and both legacy initialization ladders and both PostgreSQL ladders are wired. The PostgreSQL completed-schema short path also includes 82/83, avoiding replay of broad legacy reconciliation on a current installation. DDL and guarded schema-version updates remain inside the owning migration transaction. Existing reference inserts name columns explicitly and continue to work with role omitted.

ADR assessment: ADR-049 and ADR-050 and the approved H2 design govern this additive storage detail. No new durable architecture rule or ADR is needed. Canonical ordered manifest validation, unknown-role refusal by its future reader, replacement compatibility, source/attachment fences, context CAS, retention publication and public routes/capabilities remain pending. Persisted role alone proves none of those gates.

## Verification

- Supported Python 3.12.13; project virtual environment activated first. Official PostgreSQL fixture with TLDW_TEST_NO_DOCKER=1, no alternate provisioning.
- Causal red: 3 SQLite failures at missing role/registered method and 3 official unavailable PostgreSQL skips; production was unchanged during that run.
- Migration green: 9 passed, 9 skipped, 12 warnings in 6.31s. The genuine prior79 SQLite constructor, fresh/repeated reopen, four accepted roles, invalid-role rollback, and an interruption after actual DDL/version mutation are covered. Full seeded conversations/messages/operations/candidates/claims/references/quota records preserve every old field; only role=NULL is added. Opaque original bundle JSON, manifest token, reference/asset revisions, order and both shared-reference context flags remain exact.
- Final nine-file connected run: 326 passed, 208 skipped, 12 warnings in 107.57s. Covers native role and published/colliding migration lineages, retention results, operations, fork commit/projection, asset admission/read and workspace lifecycle. Skips retain official unavailable PostgreSQL and existing SQLite controls that require the PostgreSQL driver's locking behavior. PostgreSQL counterparts retain the genuine prior83 constructor and official fixture, but physical PostgreSQL execution is unqualified.
- Matched-filename Ruff baseline/current: 0/0 for all three changed Python files. Final direct scoped Ruff passes. New test Black check passes; only new production methods and the affected old version assertion were formatted, preserving their ASTs. No broad legacy formatting.
- Production-scope Bandit: one actual ChaChaNotes_DB.py file verified covered, baseline/current 0/0 findings, 0 errors. Supported Python compile of all three touched Python files passes.
- Both independent reviewers return bounded source CLEAR. The minor latest-version fixture note is addressed with current constants while retaining genuine fixed prior 79/83 setup and rollback expectations.

One exploratory connected invocation stopped before collecting tests because of a wrong test path; it is excluded from passing evidence. The corrected final invocation above passed. No test was disabled or suppressed.

## Current owned runtime preservation

Fresh dev reference a744577356ecd4db2d101689b5564a7a7c3c3f5a is included in the remediation branch. The verified original API PID27579/listener19767/supported Python/worktree exited after SIGTERM; no forced kill. The latest supported-Python API PID59482 uses the same sealed private profile/key/auth configuration. No auth reinitialization or profile reset.

Before and after startup, the original nine conversations and 28 metadata-bearing canonical records deep-equal the immutable original preservation baseline; it was never rewritten. The distinct d9d228ea-19e2-4eac-8e43-af4d78bcbbc9 post-greeting-fix case retains the exact three original canonical records. No new send occurred. One existing owned headless tab at19769 shows three articles, Character setup: Selected and one empty textarea after cold reload. No new browser/tab/window; ambient19764 is untouched.

The initial navigation completed but its network-idle wait timed out. The same already-reloaded tab was checked through actual visible controls; no second navigation or send was needed. This harness diagnostic is not evidence of a product failure. Runtime checks do not qualify physical PostgreSQL, new attachment publication or an installed extension.

Finite private evidence uses /private/tmp/chat-reference-roles-13600-*; raw logs/configuration/headers/transcripts are not published. Review records remain source-only for the migration, with root-owned runtime checks separately bounded above.

Initial normal exact-owned eight-file hooks pass (exit0). House hook filters skip Ruff/Black for these paths; the meaningful manual scoped checks above pass. All three bounded stages are complete and only the own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_reference_roles_13600_completed.md. Final normal exact-owned seven-file hooks pass (exit0), with no bypass. The local commit follows; broad TASK-13458 and TASK-13261.4 remain In Progress. Fresh fetch still resolves dev to a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero commits missing from this branch.
