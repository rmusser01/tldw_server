# Exact native bytes after chat restore — TASK-13603

Actual soft deletion retains native claims/bytes but permanently burns the original creation key. Restore reactivates the protected chat without reviving replay. The private byte reader incorrectly required committed replay state, so restored exact owned bytes remained unreadable. This bounded root repair separates current asset ownership from creation replayability.

## Change and preserved authority

NativeChatAssetStore.read_committed_asset now permits committed provenance or the existing gone/child_deleted tombstone, still requiring the exact original target. The protected child and original operation projection must match. Active child, owner/scope, open workspace, original creation identity, candidate operation/generation, adopted candidate/live claim, exact immutable byte descriptor, confirmed unreleased quota and exclusive canonical namespace/storage guards remain.

Only those two reviewed conditions and the method docstring change. AST comparison against HEAD verifies every other source statement is unchanged; NativeChatAssetIO.read_owned_asset is byte-for-byte unchanged. Operation→workspace→conversation→candidate→claim lock order is preserved. No lifecycle resolver/receipt mutation, creation replay, key resurrection, source fallback, new column, public route/header/capability or retention-kind admission is introduced.

The physical reader retains its repeated current descriptor checks, stable anchored guard, bounded trusted consumption, directory/symlink constraints and exact hash/size/inode validation before yield and after consumption. Restored ownership never authorizes still-deleted, purged, foreign, expired/rejected/preparing, unknown gone-reason, generation/claim/quota/projection mismatch or corrupt file access.

ADR required: no; ADR-049/050 and accepted H2 explicitly separate restored assets from a burned creation operation. This repairs that governing lifecycle contract. Original-hash provenance and verified replace/Restore publication remain pending; no hash is backfilled from a current claim.

## Verification

Supported Python3.12.13 after activating the project environment. Official PostgreSQL fixture with TLDW_TEST_NO_DOCKER=1, no alternate provisioning.

- Causal red2 failed/11 passed/13 officialPGskipped/11 warnings6.19s: exact bytes after actual restore refused at committed-only gate; unsupported original projection still yielded bytes. Original production unchanged during that run.
- Initial restored/read green51 passed/51 skipped/12 warnings20.62s.
- Final seven-suite connected run218 passed/218 skipped/12 warnings97.57s, exit0: restored bytes, existing byte guard/read, prepared claim adoption, canonical context read/remove, actual fork commit and workspace lifecycle. No test suppression or setup errors.
- Real upload validation/preparation and claim adoption precede actual soft-delete, denied read, actual cleanup refusal while live claims remain, actual restore and exact UTF-8 byte read. Handle closes; no DB transaction is held during byte consumption. Complete operation/candidate/claim/quota records remain exact. Original operation is gone with no result; actual same-key resolution remains gone and never recreates a child.
- Actual deletion while the restored reader waits at the existing owned guard completes outside its descriptor transaction; subsequent admission refuses before yielding any handle. Missing/tampered/truncated restored bytes also refuse without durable authority writes. Existing symlink/root/identity/held descriptor checks remain in the connected suite.
- Actual workspace deletion with an injected cascade failure leaves durable closure; restored byte read returns workspace_native_unavailable with no handle or asset authority change. Scope metadata is explicitly seeded for this negative and does not qualify a workspace publisher.
- Final two-file Ruff and matched-filename baseline/current0/0; new gate formatted with AST preservation, new test Black check and supported Python compile bothfiles pass. Diff-check passes. Actual one-production-file Bandit baseline/current0/0 findings/0 errors; scanned file verified. No public schema change; unchanged canonical contract evidence from TASK13601 remains applicable without rerunning its exporter.
- Both independent final source reviews are bounded CLEAR, including held deletion and actual workspace closure. Reviewers ran no tests.

The committed headers in the nonempty claim fixture are explicitly seeded over actually prepared/adopted bytes; they do not qualify a rich fork/retention publisher. Physical PostgreSQL execution/concurrency, public streaming and full public H2 remain unqualified; skipped tests use the official unavailable fixture and existing PostgreSQL-driver-specific SQLite controls.

No runtime profile/auth/key/card/database changes, new browser/window/tab or send were needed. Existing owned services and immutable original preservation baseline remain untouched. Finite private evidence is /private/tmp/chat-restored-assets-13603-*; raw logs/configuration/headers/transcripts are not published.

Initial normal exact-owned seven-file hooks pass (exit0). House Ruff/Black hook filters skip these native paths; meaningful manual scoped checks above pass. All three bounded stages complete; only the own plan retires to /private/tmp/IMPLEMENTATION_PLAN_chat_restored_asset_reads_13603_completed.md. Final normal exact-owned six-file hooks pass (exit0), no bypass. Fresh dev remains a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero missing commits. Local commit follows. Master TASK13458 and H2 parent TASK13261.4 remain In Progress; original-hash mapping, retention/source fences/publication, verified replace/Restore, actual UI/send context and broader NN/G/installed-extension/Compact/physical AT/real-model/study acceptance remain open.
