---
id: TASK-12092
title: Implement visual identity asset storage validation
status: Done
references:
- Docs/superpowers/plans/2026-07-01-visual-identity-expression-packs-implementation-plan.md
- Docs/superpowers/specs/2026-07-01-visual-identity-expression-packs-design.md
modified_files:
- tldw_Server_API/app/core/Visual_Identities/storage.py
- tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py
- tldw_Server_API/app/core/DB_Management/db_path_utils.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement Stage 3 from the Visual Identity Expression Packs plan: visual identity asset storage directory helper, image validation/storage helpers, animated metadata extraction, AVIF capability gating, generated-file copy helper, and focused storage tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Stage 3 storage slice implemented with TDD, then review-fix and re-review fix passes completed.

Original Stage 3 changed:
- Added `DatabasePaths.get_user_visual_identities_dir(user_id)` for `Databases/user_databases/{user_id}/visual_identities`.
- Added `Visual_Identities.storage` with content-hash original storage, MIME/header/Pillow validation, dimension/frame limits, animated metadata, first-frame previews, safe relpath resolution, and lower-level generated-file record copy helper.
- Added focused Visual Identity storage tests.

Review-fix changed:
- Removed public `source_path` override from generated-file copy helper so records must resolve through contained `storage_path` under the user outputs directory.
- Made generated-file `source_feature` validation strict when a caller supplies an expected feature.
- Rechecked source byte size after reading content.
- Replaced silent existing-target dedupe with same-directory temp writes plus existing-target regular-file, size, and SHA-256 verification; corrupt targets now raise `stored_asset_hash_mismatch`.
- Added regression coverage for generated record negative metadata, public path override rejection, duplicate dedupe/corruption, extension mismatch/unsupported extension, backslash traversal, frame-count limit, preview failure tolerance, and post-read size recheck.

Re-review fix changed:
- Required non-empty expected `source_feature` for generated-file record copies by default; omitted, `None`, blank, missing, blank-record, and wrong-feature cases now fail closed.
- Removed the `_write_once()` `replace()` fallback so hardlink publish failures fail closed when no target already exists, instead of overwriting a concurrently created target.
- Rendered first-frame previews to memory and published them through the hash-verified write helper; corrupt existing previews now raise `stored_asset_hash_mismatch`.
- Added explicit generated-file `storage_path` rejection coverage for `../`, absolute paths, and backslashes.

Red evidence:
- First review-fix red run after regression tests: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` failed with intended production behavior failures for post-read size recheck, corrupt existing hash target, public `source_path` override, and missing generated-file `source_feature` strictness. Test fixture setup issues were corrected and rerun before implementation.
- Re-review red run after adding regressions: same storage test command failed as expected with 5 failures and 25 passes for hardlink publish fallback, corrupt preview trust, and `source_feature=None`/blank acceptance.

Green evidence:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` passed: 30 passed, 3 warnings.
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Visual_Identities/test_expression_slots.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_capabilities.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_db.py tldw_Server_API/tests/Visual_Identities/test_visual_identity_storage.py` passed: 57 passed, 3 warnings.

Bandit:
- `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Visual_Identities tldw_Server_API/app/core/DB_Management/db_path_utils.py -f json -o /tmp/bandit_visual_identity_stage3.json` exited 0; report totals show 0 findings.

Diff check:
- `git diff --check 65365986f3ae04a2477566af43d073bc58192cb3..HEAD` exited 0.

Concern:
- The helper remains lower-level (`copy_generated_file_record_to_expression_asset`) for Stage 11 service/API reuse; AuthnzGeneratedFilesRepo lookup wiring remains intentionally out of Stage 3 scope.
- Symlink escape coverage was not added in this re-review pass; existing resolver uses resolved containment, and explicit traversal forms are now covered.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
