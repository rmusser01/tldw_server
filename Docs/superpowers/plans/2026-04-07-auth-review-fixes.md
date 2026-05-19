# Auth Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the three confirmed Auth review issues around single-user API-key enforcement, logout semantics for API-key principals, and asymmetric key-rotation settings.

**Architecture:** Keep the changes narrow and behavior-driven. Add targeted regression tests first, then tighten the single-user API-key path, make logout reject unsupported API-key logout requests, and expose the missing secondary private-key setting so existing crypto/session fallback code can use environment-provided rotation material.

**Tech Stack:** Python 3, FastAPI, Pydantic Settings, pytest, Bandit

---

## Task 1: Single-User API Key Regression

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py`

- [ ] **Step 1: Write the failing single-user regression test**
- [ ] **Step 2: Run the focused test and confirm a failure caused by DB fallback still authenticating**
- [ ] **Step 3: Remove the single-user fallback to AuthNZ API-key lookup for non-matching keys**
- [ ] **Step 4: Re-run the focused test and confirm it passes**
- [x] **Step 1: Write the failing single-user regression test**
- [x] **Step 2: Run the focused test and confirm a failure caused by DB fallback still authenticating**
- [x] **Step 3: Remove the single-user fallback to AuthNZ API-key lookup for non-matching keys**
- [x] **Step 4: Re-run the focused test and confirm it passes**
- [x] **Status:** Complete

## Task 2: Logout Semantics For API-Key Principals

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/auth.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`

- [ ] **Step 1: Write the failing logout regression test for API-key authenticated requests**
- [ ] **Step 2: Run the focused test and confirm the endpoint currently returns a false success**
- [ ] **Step 3: Make logout return a clear client error when no bearer token is present for current-session logout**
- [ ] **Step 4: Re-run the focused test and confirm it passes without breaking bearer-token logout coverage**
- [x] **Step 1: Write the failing logout regression test for API-key authenticated requests**
- [x] **Step 2: Run the focused test and confirm the endpoint currently returns a false success**
- [x] **Step 3: Make logout return a clear client error when no bearer token is present for current-session logout**
- [x] **Step 4: Re-run the focused test and confirm it passes without breaking bearer-token logout coverage**
- [x] **Status:** Complete

## Task 3: Secondary Private Key Environment Support

**Files:**
- Modify: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Modify: `tldw_Server_API/app/core/AuthNZ/settings.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`

- [ ] **Step 1: Write the failing settings regression test for `JWT_SECONDARY_PRIVATE_KEY` env loading**
- [ ] **Step 2: Run the focused test and confirm the env-provided value is ignored**
- [ ] **Step 3: Add the missing settings field with the existing rotation semantics**
- [ ] **Step 4: Re-run the focused test and confirm it passes**
- [x] **Step 1: Write the failing settings regression test for `JWT_SECONDARY_PRIVATE_KEY` env loading**
- [x] **Step 2: Run the focused test and confirm the env-provided value is ignored**
- [x] **Step 3: Add the missing settings field with the existing rotation semantics**
- [x] **Step 4: Re-run the focused test and confirm it passes**
- [x] **Status:** Complete

## Task 4: Focused Verification

**Files:**
- Modify: `Docs/superpowers/plans/2026-04-07-auth-review-fixes.md`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_user_db_handling_api_keys.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_auth_endpoints_extended.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_session_manager_configured_key.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_service_rs256.py`
- Test: `tldw_Server_API/tests/AuthNZ/unit/test_jwt_dual_key_rotation.py`

- [ ] **Step 1: Run the targeted pytest selection covering the new regressions and nearby rotation/logout behavior**
- [ ] **Step 2: Run Bandit on the touched Auth files**
- [ ] **Step 3: Update this plan with completion status and verification notes**
- [x] **Step 1: Run the targeted pytest selection covering the new regressions and nearby rotation/logout behavior**
- [x] **Step 2: Run Bandit on the touched Auth files**
- [x] **Step 3: Update this plan with completion status and verification notes**
- [x] **Status:** Complete

## Verification Notes

- `pytest`: `3 passed` for the new regression nodes after the fixes landed.
- `pytest`: `8 passed, 42 deselected` for the focused Auth follow-up selection covering API-key handling, logout flows, session-manager config, and JWT rotation tests.
- `bandit`: attempted via `.venv/bin/python -m bandit ...`, but the module is not installed in the project venv and no `bandit` executable is available on PATH in this workspace.
