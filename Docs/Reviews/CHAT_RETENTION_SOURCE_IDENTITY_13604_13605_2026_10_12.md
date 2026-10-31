# Original ordinary retention source — TASK-13604 / TASK-13605

Ordinary pre-retention chats have no native manifest. This private prerequisite reads their coherent source state and binds an opaque source revision to the original server-issued row incarnation. It does not create claims, publish retention, authorize a selected view or expose readiness.

## Root and final behavior

Resettable counters and content hashes did not distinguish actual hard-delete followed by same-ID/same-clock recreation. Both reviewers identified the gap; the actual fixed-clock regression reproduced it. Creation timestamps and latest remaining Sync journal records were rejected: PostgreSQL lacks the equivalent creation trigger, sequence repair and selective pruning can reuse apparent journal identity. A missing explicit owner key also manufactured an internal fallback namespace and now refuses.

New registered SQLite80→81 and PostgreSQL84→85 migrations issue source_incarnation_id once for existing rows without rewriting original fields. Issuance is at the DB insertion boundary, covering ordinary and Sync writers without accepting copied authority. SQLite reserved-input/creation/immutable-update triggers reject supplied SQL identity and issue random row nonce on actual insert. PostgreSQL volatile UUID default backfills existing rows, before-insert guard replaces supplied identity, and update guard prevents replacement. UUID generation already exists in house PostgreSQL code/documented baseline. Updates/restore retain identity; hard-delete/reinsert mints another. All five migration paths and Pg84 short admission are wired. Published native DDL/old migrations and existing RLS remain unchanged.

Current DTOs, Chatbook export/import and Sync materialization enumerate fields; they neither expose nor reuse the nonce. No public schema/route or new dependency is added. Proposed ADR071 records the durable decision; existing accepted ADR049/050 rationale is unchanged. Broader H1/backup integration and public rollout remain pending.

NativeForkStore.read_retention_source_revision requires the original active owner transaction and explicit valid namespace. Workspace locks precede conversation locks. Exact live owner/scope, ordinary-only native marker absence and no native claim/reference carrier are required. Existing one-statement history snapshot selects no content/attachment bytes. Header recheck detects capture-time changes; carrier state rechecks after capture.

Fixed tuple: [tldw-ordinary-retention-source,1,client_id,owner_key,[scope_kind,workspace_id_or_null],conversation_id,source_incarnation_id,[conversation/history/settings fences],source_digest,storage_context_digest]. Existing canonical digest returns opaque sha256 token. It is distinct from native manifest/readiness and H1 selected-view authority. Later publication must resolve existing keys first, recheck active Sync policy, lock operation before this helper and compare the frozen request; none of that is claimed here.

## Verification

SupportedPython3.12.13 after project environment activation; official PostgreSQL fixture TLDW_TEST_NO_DOCKER=1, no alternate provisioning.

- Source causal14failed/14officialPGskip/12warnings10.83s, then14pass/14skip16.93s. Initial connected188pass/170skip97.06s was not original-record qualification.
- Actual fixed-clock record recreation and missing owner-key causal2fail/2PGskip/34deselected12warnings1.88s. Incarnation creation causal2fail/2PGskip12warnings3.17s at missing column. No failing regression was suppressed.
- Combined source/incarnation green21pass/21skip12warnings11.79s closes those actual failures. Includes real message/settings/metadata/image changes, same-capture metadata/carrier interleaving and real history ABA, stable unchanged source, no selected bytes, exact owner/scope and actual workspace closure. Image change now uses equal-length bytes.
- Initial connected137pass/133skip with1 old preservation-fixture failure74.05s reflected the new additive metadata field. The fixture now validates nonempty stable incarnation while still comparing every original field exactly; it does not discard arbitrary changes.
- Final six-suite connected138pass/133skip/12warnings72.19s, exit0: incarnation, source capture, published/colliding native and role migration lineages, actual fork commit and coherent history transactions. Original conversation/message data and nonce reopen/restore/ordinary+Sync behavior are covered.
- Additional direct SQL copied-incarnation and interrupted migration after actual DDL/version work:2pass/2officialPGskip/6deselected12warnings2.08s. SQLite reserved input refuses; PostgreSQL requires server replacement. Interruption preserves original row and prior schema and removes the new triggers/column. Source reviewers clear both controls.
- Five-file matched/direct Ruff0/0, new methods formatted with AST preservation, new test Black checks and supported Python compile all5 pass. No broad legacy formatting. Diff-check passes.
- Actual2-production-file Bandit baseline/current0/0 findings/0 errors; actual coverage verified. Fixed lock-suffix SQL uses bound identities; false-positive annotation is on the reported literal, not a hidden security issue. No runtime logs/secrets are published.
- Both independent final source reviewers bounded CLEAR. Reviewers ran no tests.

Physical PostgreSQL execution/concurrency remains unqualified; official fixture and existing backend-specific controls account for skipped checks. Private finite evidence: /private/tmp/chat-retention-source-13604-*, /private/tmp/chat-incarnation-13605-* and /private/tmp/chat-source-identity-13604-13605-*.

No runtime/profile/auth/key/card database changes, new browser/window/tab/send or baseline rewrite. Existing owned services remain untouched. Actual retention reservation/publication/committed-first recovery, active Sync admission, native-source bridge, original reference-hash mapping, replace/Restore, UI/send integration and broader NN/G/installed-extension/Compact/AT/real-model/study gates remain open.

Initial normal exact-owned fourteen-file hooks pass (exit0). House Ruff/Black hooks skip these native paths; meaningful manual checks above pass. Both bounded three-stage plans complete; only own plans retire to /private/tmp/IMPLEMENTATION_PLAN_chat_retention_source_revision_13604_completed.md and /private/tmp/IMPLEMENTATION_PLAN_chat_source_incarnation_13605_completed.md. Final normal exact-owned twelve-file hooks pass (exit0), no bypass. Fresh dev remains a744577356ecd4db2d101689b5564a7a7c3c3f5a with zero missing commits; local commit follows. Master TASK13458/H2 parent13261.4 remain In Progress.
