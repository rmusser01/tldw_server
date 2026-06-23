---
id: TASK-2348
title: Implement Audio Studio backend foundation
status: Done
labels:
- audio
- backend
priority: high
documentation:
- Docs/superpowers/plans/2026-06-23-audio-studio-mvp-implementation-plan.md
- Docs/superpowers/specs/2026-06-23-audio-studio-design.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/audio_studio_schemas.py
- tldw_Server_API/app/api/v1/endpoints/audio/audio_studio.py
- tldw_Server_API/app/api/v1/router_groups/content.py
- tldw_Server_API/app/core/DB_Management/Collections_DB.py
- tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py
- tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py
- tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py
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

Follow-up verification:
- python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_schemas.py -v: 22 passed
- python -m pytest tldw_Server_API/tests/Audio_Studio/unit/test_audio_studio_collections_db.py -v: 7 passed
- python -m pytest tldw_Server_API/tests/Audio_Studio/integration/test_audio_studio_projects_api.py -v: 8 passed
- python -m pytest tldw_Server_API/tests/Audiobooks/integration/test_audiobook_jobs_endpoints.py -v: 4 passed
- python -m bandit -r touched backend files -f json -o /tmp/bandit_audio_studio_backend_foundation.json: exit 0, no findings
- git diff --check scoped to Audio Studio touched files: pass; global git diff --check still reports unrelated pre-existing whitespace in Docs/Design files outside this task scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Audio Studio backend foundation added for Stage 1.1-1.3. Created schema contracts with string enums, revision/idempotency validation, and strict secret/external URL payload guards; extended Collections DB with Audio Studio project/revision/resource/artifact/job/idempotency tables and repository methods with owner isolation; added transactional repository mutation methods that atomically validate base revisions, mutate project/resources, insert revision rows, advance current_revision_id, and roll back on failures; added clip reference validation and path ID validation; fixed explicit project description clearing; added /api/v1/audio-studio workflow, project CRUD, and section/track/clip upsert endpoints; registered the route key audio-studio; required base_revision_id for project archive; and added unit/integration coverage plus Bandit verification.
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
