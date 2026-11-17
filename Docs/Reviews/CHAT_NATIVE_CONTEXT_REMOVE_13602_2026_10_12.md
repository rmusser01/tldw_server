# Reviewed native context removal — TASK-13602

The accepted H2 context update requires exact reviewed manifest, reference and asset revisions. This bounded private store increment implements remove_from_context only. It preserves displayed references/bytes and shared claims, without source-job cancellation or public admission/readiness claims.

## Behavior

NativeChatAssetStore.update_native_asset_context requires the actual owning transaction and revalidates the existing frozen NativeAssetContextUpdateV1 through model serialization/validation, refusing raw or unchecked invalid models. Replace/restore explicitly refuse before mutation. Reference storage lacks immutable original-hash mapping and replacement still needs physically verified byte authority; a supplied expected_original_hash or current claim hash cannot establish missing provenance.

Inside a fixed local savepoint, the existing canonical reader obtains workspace→conversation→reference/candidate/claim locks without reversing operation locks. Compare exact opaque manifest token, target reference ID, reference revision and asset revision before no-op or writes. A current already-disabled reference returns unchanged only after all comparisons match; stale lost-response retry still conflicts.

An enabled target changes only context_enabled, reference revision and update timestamp. Its ID, role, order, claim and byte identity remain. The original bundle retains every other field and gains a new opaque manifest token from the existing UUID generator. The existing history-version writer advances the context fence in that same transaction. A canonical reread returns the saved metadata. No new schema/version column, inferred original hash, byte release, quota update or receipt replay.

The local savepoint rolls back the complete reference/bundle/history mutation on any tail error, even if a caller catches the error and commits unrelated parent work. Successful local work still belongs to the outer caller transaction and rolls back with it. Actual restored active chats with permanently gone creation keys retain that key unchanged.

ADR required: no; ADR-049/050 and accepted H2 Task2.3 govern this atomic reference mutation. Canonical retention-backed/unavailable references, source attachment fence/publisher, physically verified replace, original-hash Restore authority, public routes/shared client and actual send-context admission remain pending. The real prepared/adopted fixture explicitly seeds nonempty publication metadata; it does not qualify rich publication.

## Verification

Supported Python3.12.13 with project environment activated first. Official PostgreSQL fixture with TLDW_TEST_NO_DOCKER=1; no alternate database provisioning.

- Causal red13 failed/13 officialPGskipped/11 warnings6.98s at missing update method; original production unchanged during run.
- Initial remove/read green49 passed/44 skipped/12 warnings19.91s.
- Final five-suite connected run177 passed/177 skipped/13 warnings86.41s, exit0: removal, canonical read, byte read, actual fork commit and workspace lifecycle. Covers exact three preconditions/target/owner, unsupported actions and raw/unchecked model refusal, shared peer participation, exact canonical reload/lost-response stale refusal, reference/bundle/history/final-read rollback, outer caller rollback, preserved bytes via actual owned handle, and actual delete→restore with permanently gone keys.
- A reviewer identified that an empty parent transaction did not prove unrelated caller work survives. Tests add an actual caller conversation before the savepoint and compare the original full CAS graph after catching each tail failure and committing. The first reference-boundary variant failed during caller UUID setup (1 failed/2 passed/3 skipped/28 deselected11 warnings2.59s), not CAS. An explicit declared caller ID bypasses only setup generation and preserves the CAS failpoint. Final changed three cases3 passed/3 officialPGskipped/28 deselected/11 warnings1.93s. This is additional evidence, not a fabricated aggregate full-run count. Passing unrelated checks were not repeated without production changes.
- Final two-file direct Ruff passes and matched-filename baseline/current0/0. New method Black formatting preserves its AST; new test Black check passes. Supported Python compile bothfiles and diff-check pass.
- Actual one-production-file Bandit baseline/current0/0 findings/0 errors, actual scanned file verified. No public schema/route change; TASK13601's unchanged full canonical contract remains the governing contract evidence.
- Both independent reviewers give final bounded source CLEAR, including explicit-ID caller preservation. Reviewers ran no tests.

Physical PostgreSQL execution/concurrency remains unqualified; official unavailable fixture and existing PostgreSQL-driver-specific SQLite controls account for skips. Fault-model serializer diagnostics are retained in warning totals; no warning/test suppression. Private evidence is /private/tmp/chat-native-remove-13602-*; raw runtime logs/configuration/headers/transcripts are not published.

No browser/runtime profile/auth/key/card/DB changes, new sends or new windows/tabs were needed. Original baseline and existing owned services remain untouched. Full all-ten NN/G, installed-extension/Compact/public H2, physical AT/PostgreSQL and real-model/study acceptance remain open.

Initial normal exact-owned seven-file hooks pass (exit0). House Ruff/Black hooks skip these native paths; meaningful manual checks above pass. All three bounded stages complete; only the own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_native_context_remove_13602_completed.md. Final normal exact-owned six-file hooks pass (exit0), no bypass. Fresh dev remains a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero missing commits. Local commit follows. Master TASK13458 and H2 parent TASK13261.4 remain In Progress.
