# Independent UAT197 count-row review

TASK13260.135. **Clear**, separately from worker181. Fresh AST inspection verifies that undoing precisely two `int(...)` argument conditionals makes `add_study_pack_cards` identical to the saved original function AST. PostgreSQL takes the unaliased COUNT column `count`; SQLite retains index0. SQL, row fallbacks, transaction scope, parameters and duplicate handling remain unchanged. Full shared-file changes outside this function are not attributed to197.

The new eight cases ran independently in the 29-case requiredPG command recorded in REVIEW181-study-pack.md: all pass with zero skips. These use actual official PostgreSQL or SQLite rows. Cases cover empty input, first/all-duplicate/mixed append, separate packs, caller rollback, and failing foreign-key batch rollback preserving prior members. Test hash matches the author's frozen value. The author's separate first-count and second-count RED receipts show both original positional reads fail; independently inspected, not rerun here.

No remaining source finding. Native/model acceptance remains outside this test claim. See count197-independent-scope.json for fresh exact-function and test identity proof.
