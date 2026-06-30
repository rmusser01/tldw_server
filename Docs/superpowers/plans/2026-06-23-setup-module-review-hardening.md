# Setup Module Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the current `tldw_Server_API/app/core/Setup` module against the validated review findings without broad setup refactoring.

**Architecture:** Keep each fix on the existing setup boundaries: install-plan validation stays in `install_schema.py`, preview-plan construction stays in `readiness_service.py`, config writes stay in `setup_manager.py`, and persistence/install hardening stays inside the Setup stores and install manager. Add focused regression tests beside the existing setup tests and preserve existing API payload shapes except for the new explicit custom-embedding acknowledgement field.

**Tech Stack:** Python, FastAPI setup core, Pydantic v2 models, pytest, Bandit.

---

### Stage 1: Regression Tests

**Goal:** Reproduce each validated review finding with focused tests before production changes.

**Files:**
- Modify: `tldw_Server_API/tests/Setup/test_install_manager_dependencies.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_manager_masking.py`
- Modify: `tldw_Server_API/tests/Setup/test_audio_readiness_store.py`
- Modify: `tldw_Server_API/tests/Setup/test_setup_readiness_preview.py`

**Success Criteria:** New focused tests fail for the current vulnerable behavior and fail for the expected reason.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Setup/test_install_manager_dependencies.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py tldw_Server_API/tests/Setup/test_audio_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q`

**Status:** Complete

### Stage 2: Installer Execution Hardening

**Goal:** Prevent installer logs from exposing index credentials, bound subprocess execution, and block unpinned VCS requirements by default.

**Files:**
- Modify: `tldw_Server_API/app/core/Setup/install_manager.py`
- Test: `tldw_Server_API/tests/Setup/test_install_manager_dependencies.py`

**Success Criteria:** Installer command logging and errors are redacted, subprocess execution uses a configurable timeout, and unpinned VCS dependencies are skipped unless explicitly allowed.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Setup/test_install_manager_dependencies.py -q`

**Status:** Complete

### Stage 3: Persistence Hardening

**Goal:** Make setup config, setup readiness, audio readiness, and install status persistence resistant to truncation and lost updates.

**Files:**
- Modify: `tldw_Server_API/app/core/Setup/setup_manager.py`
- Modify: `tldw_Server_API/app/core/Setup/readiness_store.py`
- Modify: `tldw_Server_API/app/core/Setup/audio_readiness_store.py`
- Modify: `tldw_Server_API/app/core/Setup/install_manager.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_manager_masking.py`
- Test: `tldw_Server_API/tests/Setup/test_audio_readiness_store.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_store.py`

**Success Criteria:** Atomic replace failures preserve existing files and concurrent in-process updates preserve independent fields; file lock scopes cover load/merge/save writes where the store owns the persistence path.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Setup/test_setup_manager_masking.py tldw_Server_API/tests/Setup/test_audio_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py -q`

**Status:** Complete

### Stage 4: Custom Embedding Trust Boundary

**Goal:** Require explicit custom embedding trust acknowledgement in the install plan schema, not only the preview path.

**Files:**
- Modify: `tldw_Server_API/app/core/Setup/install_schema.py`
- Modify: `tldw_Server_API/app/core/Setup/readiness_service.py`
- Test: `tldw_Server_API/tests/Setup/test_install_manager_dependencies.py`
- Test: `tldw_Server_API/tests/Setup/test_setup_readiness_preview.py`

**Success Criteria:** Direct install plans containing `embeddings.custom` without acknowledgement are rejected, acknowledged preview plans include the acknowledgement marker, and ordinary curated Hugging Face/ONNX install plans remain accepted.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Setup/test_install_manager_dependencies.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q`

**Status:** Complete

### Stage 5: Verification and Closeout

**Goal:** Run focused setup verification, security scanning, and record final task evidence.

**Files:**
- Modify: `backlog/tasks/task-12010 - Harden-Setup-module-review-findings.md`
- Modify: `Docs/superpowers/plans/2026-06-23-setup-module-review-hardening.md`

**Success Criteria:** Focused tests pass, Bandit runs on touched Setup production files, diff whitespace check passes, and Backlog records verification results and any skips.

**Tests:**
- `python -m pytest tldw_Server_API/tests/Setup/test_install_manager_dependencies.py tldw_Server_API/tests/Setup/test_setup_manager_masking.py tldw_Server_API/tests/Setup/test_audio_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_store.py tldw_Server_API/tests/Setup/test_setup_readiness_preview.py -q`
- `python -m bandit -r tldw_Server_API/app/core/Setup/install_manager.py tldw_Server_API/app/core/Setup/install_schema.py tldw_Server_API/app/core/Setup/setup_manager.py tldw_Server_API/app/core/Setup/readiness_service.py tldw_Server_API/app/core/Setup/readiness_store.py tldw_Server_API/app/core/Setup/audio_readiness_store.py -f json -o /tmp/bandit_setup_module_review_hardening.json`
- `git diff --check`

**Status:** Complete
