# UAT197 / TASK13260.135 — Study Pack named PostgreSQL count rows

The actual StudyPack generation service reaches add_study_pack_cards after fake-model generation and real card persistence, then fails at before_row[0]. The adjacent after_row[0] consumes the same COUNT query. PostgreSQL returns mapping key `count`; SQLite's unchanged unaliased aggregate supports positional index 0. This defect is separate from UAT181 job checkout ownership. No native StudyPack failure or model-quality claim.

Approved minimal implementation: use `row["count"]` for PostgreSQL and preserve `row[0]` for SQLite at exactly the two existing aggregate reads. Keep SQL, parameters, transaction scopes, null-row fallbacks, duplicate handling and delta return unchanged. Shared schema 193/194 regions belong to Sidebar and are outside this diff.

Stage 1 — RED: official required PostgreSQL and SQLite empty, append/duplicate/pack scope, outer rollback and failed-batch rollback controls. Status: complete; 3PG RED/5controls, then isolated after-count RED1.

Stage 2 — GREEN: two conditional mapping accesses only, then rerun unchanged tests plus real StudyPack worker/service persistence. Status: complete; author GREEN/static, frozen for independent review.

Stage 3 — freeze: scoped lint/Bandit, AST and query equivalence, separate patch/snapshots/hashes for independent review. Status: complete; author GREEN/static, frozen for independent review. Root owns tracker, commits and native/runtime work.
