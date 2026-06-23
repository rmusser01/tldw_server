---
id: TASK-2348
title: Implement Audio Studio backend foundation
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-23 15:28
labels:
- audio
- backend
dependencies: []
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
priority: high
modified_files:
- tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/core/DB_Management/Collections_DB.py
- tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py
- tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py
- tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py
- tldw_Server_API/tests/Services/test_router_groups_contract.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the backend foundation for Audio Studio: Pydantic schemas, Collections DB persistence/revision/idempotency primitives, and the initial /api/v1/audio-studio project/section/track/clip endpoints from the accepted Audio Studio MVP plan.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Audio Studio schemas exist with validation for workflows, resources, base revisions, idempotency keys, and secret-free provider payloads.
- [x] #2 Collections DB supports Audio Studio projects, revisions, sections, tracks, clips, artifacts, generation-job links, and idempotency records with owner isolation.
- [x] #3 Initial /api/v1/audio-studio project and resource endpoints are registered and covered by integration tests.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Stage 1 tasks 1.1 through 1.3 in Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md. Use TDD, preserve existing /api/v1/audiobooks behavior, and run the listed pytest commands before finalizing this slice.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Follow-up spec compliance fixes:
- Strengthened Audio Studio client payload validation to reject common credential key variants including access_token, refresh-token, private_key, credentials/clientCredential, while allowing harmless keys such as tokenizer.
- Added AudioStudioProjectArchiveRequest and required base_revision_id on DELETE /api/v1/audio-studio/projects/{project_id}; stale revisions now return 409 and cross-user lookups remain 404.

Follow-up code-quality fixes:
- Moved Audio Studio project/create/update/archive and section/track/clip upsert mutation flows into repository-level transaction methods using CollectionsDatabase.transaction().
- Repository mutations now perform atomic base_revision_id consumption with conditional project current_revision_id updates, insert revision rows in the same transaction, and roll back project/resource changes if revision insertion fails.
- Archive now creates a project.archive revision and returns the archived project revision info while preserving endpoint 404 owner isolation.
- Clip upsert now rejects dangling track_id, section_id, and artifact_id references that are missing, deleted, or archived.
- Added conservative max length/pattern validation for Audio Studio path IDs and fixed project description clearing for explicit empty string/null values.

Follow-up spec re-review fixes:
- Changed the new Audio Studio revision table DDL from globally unique revision_id primary keys to numeric primary keys plus owner-scoped unique indexes on (user_id, revision_id) for both SQLite and PostgreSQL branches.
- Added a shared-DB regression proving two users can independently create and retrieve the same revision_id without cross-user conflict.
- Set Audio Studio Pydantic models to forbid top-level extra fields so external_url, api_key/secret-like fields, and unknown client fields are rejected instead of silently ignored.

Follow-up verification:
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py -v: 26 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py -v: 8 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v: 8 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -v: 4 passed
- .venv/bin/python -m bandit -r touched backend files -f json -o /tmp/bandit_audio_studio_backend_foundation.json: exit 0, no findings
- git diff --check scoped to Audio Studio touched files: pass; global dirty worktree still contains unrelated changes outside this task scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Follow-up code-quality review fixes:
- Rejected nested URL-bearing keys and http/https URL string values in Audio Studio client-controlled provider/options/settings/metadata payloads.
- Made audio-studio router registration stable/default-enabled and added router group contract coverage.
- Made canonical public project/resource mutation repository methods revision-aware transactional operations while keeping private row helpers for low-level setup.
- Changed idempotency records to first-insert-wins with audio_studio_idempotency_conflict on mismatched request hashes.
- Cleared deleted_at when section/track/clip upserts resurrect resources.

Follow-up verification:
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py -v: 32 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py -v: 9 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v: 8 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py::test_iter_content_router_specs_registers_audio_studio_as_stable_route -v: 1 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -v: 4 passed
- .venv/bin/python -m bandit -r touched backend files -f json -o /tmp/bandit_audio_studio_backend_foundation.json: exit 0, zero findings
- Scoped git diff --check for Audio Studio touched files: pass
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audio Studio backend foundation added for Stage 1.1-1.3. Created schema contracts with string enums, revision/idempotency validation, strict secret/external URL payload guards, and top-level extra-field rejection; extended Collections DB with Audio Studio project/revision/resource/artifact/job/idempotency tables and repository methods with owner isolation; changed revision table DDL to owner-scoped revision_id uniqueness; added transactional repository mutation methods that atomically validate base revisions, mutate project/resources, insert revision rows, advance current_revision_id, and roll back on failures; added clip reference validation and path ID validation; fixed explicit project description clearing; added /api/v1/audio-studio workflow, project CRUD, and section/track/clip upsert endpoints; registered the route key audio-studio; required base_revision_id for project archive; and added unit/integration coverage plus Bandit verification.

Follow-up hardening completed: Audio Studio client payloads now reject nested external URLs, non-http network URL schemes, protocol-relative URLs, and data URI values; project status is constrained to draft/active/archived/error in schemas and repository write paths; standalone revision creation now requires the parent revision to match the current project revision; idempotency records use an insert-first, first-writer-wins transaction with conflict detection by request hash; /audio-studio remains registered as a stable/default-enabled content route with contract coverage; canonical public repository mutation methods are revision-aware transactional paths; and resource resurrection clears deleted_at.

Follow-up verification on 2026-06-23:
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py -v: 39 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py -v: 12 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v: 8 passed
- .venv/bin/python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k audio_studio -v: 1 passed, 173 deselected
- .venv/bin/python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -v: 4 passed
- .venv/bin/python -m bandit -r touched backend files -f json -o /tmp/bandit_audio_studio_backend_foundation.json: exit 0, results []
- Scoped git diff --check for Audio Studio touched files: pass
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
