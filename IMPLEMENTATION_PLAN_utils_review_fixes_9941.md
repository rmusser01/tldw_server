# Implementation Plan: Utils Module Review Findings

## Stage 1: Image Validation Hardening

**Goal**: Ensure image data URI helpers accept only valid raster image payloads and enforce original pixel bounds before resizing.
**Success Criteria**: Small image validation rejects non-image bytes; MIME mismatches are rejected; chunked processing rejects images above the configured original pixel cap before resize.
**Tests**: Focused tests in `tldw_Server_API/tests/Utils/test_image_validation.py` and `tldw_Server_API/tests/Utils/test_chunked_image_processor.py`.
**Status**: Complete

## Stage 2: CPU Batcher Correctness

**Goal**: Prevent `CPUBoundBatcher` from leaving futures pending when operations arrive while a prior batch is draining.
**Success Criteria**: Concurrent/staggered additions all complete without timeout and no pending work is stranded.
**Tests**: Focused async test in `tldw_Server_API/tests/Utils/test_cpu_bound_handler.py`.
**Status**: Complete

## Stage 3: Safe Metadata Index Reliability

**Goal**: Fail metadata updates when identifier index writes fail unexpectedly, while preserving compatibility for known missing-table migrations.
**Success Criteria**: Missing identifier table remains non-fatal; unexpected database/index failures propagate to the caller transaction.
**Tests**: Unit tests in `tldw_Server_API/tests/MediaDB2/test_safe_metadata_utils.py`.
**Status**: Complete

## Stage 4: Sensitive Logging and Legacy Cleanup

**Goal**: Remove raw sensitive payload logging, avoid unauthenticated ffmpeg executable download behavior, and delete placeholder/no-op legacy code.
**Success Criteria**: Segment extraction and rejected external image URL logging avoid raw payloads; ffmpeg helper no longer downloads executable archives without verification; `get_user_database_path` and the no-op temp path line are removed.
**Tests**: Focused tests in `tldw_Server_API/tests/Utils/test_utils_general.py` and existing Utils tests.
**Status**: Complete

## Stage 5: Verification and Task Finalization

**Goal**: Run focused tests and security checks, update Backlog task, and report remaining risk.
**Success Criteria**: Targeted pytest checks pass; touched Python files compile; Bandit reports no new findings for touched Utils scope; Backlog task records touched files and verification.
**Tests**: `python -m pytest` for touched Utils/MediaDB2 tests, `python -m py_compile` or compileall for touched modules, and Bandit on touched source files.
**Status**: Complete
