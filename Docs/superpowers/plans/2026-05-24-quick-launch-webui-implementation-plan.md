# Consolidated Quick-Launch WebUI Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the repo-root quick-launch scripts the canonical no-Docker API/WebUI launcher and reduce installer run scripts to compatibility wrappers.

**Architecture:** Root launchers own setup and launch behavior: local-single API setup through the existing wizard, WebUI launch from `apps/tldw-frontend`, and `api`/`webui`/`all` modes. Installer run scripts only resolve their installed checkout and delegate to the root launcher with legacy `venv` defaults.

**Tech Stack:** Bash, PowerShell, Windows batch compatibility wrappers, FastAPI/Uvicorn, Next.js WebUI via Bun, pytest contract tests, Bandit.

---

### Task 1: Canonical Launcher Contract Tests

**Files:**
- Modify: `tldw_Server_API/tests/Utils/test_quick_launch_scripts.py`
- Delete: `Helper_Scripts/Installer_Scripts/Tests/test_run_tldw_launchers.py`

- [x] Write tests requiring root `quick-launch.sh` and `quick-launch.ps1` to expose `api`, `webui`, and `all` modes with default `all`.
- [x] Assert root launchers keep the local-single setup contract, launch FastAPI through `tldw_Server_API.app.main:app`, and launch the WebUI from `apps/tldw-frontend` with Bun.
- [x] Assert installer run scripts delegate to root quick-launch scripts instead of duplicating launch logic.
- [x] Run `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_quick_launch_scripts.py -v` and confirm the new tests fail before implementation.

### Task 2: Root Bash Launcher

**Files:**
- Modify: `quick-launch.sh`
- Modify: `quick-launch.command`

- [x] Add mode parsing for `api`, `webui`, `all`, and help. Default to `all`.
- [x] Preserve venv creation, editable install, and local-single wizard init for API/all modes.
- [x] Add WebUI launch from `apps/tldw-frontend`, Bun checks, `TLDW_WEBUI_PORT`, and `NEXT_PUBLIC_API_URL`.
- [x] For `all`, start API in the background, run WebUI in the foreground, and clean up the API process on exit.
- [x] Keep `quick-launch.command` as a Finder-friendly wrapper around `quick-launch.sh`.

### Task 3: Root PowerShell Launcher

**Files:**
- Modify: `quick-launch.ps1`

- [x] Add a positional `Mode` parameter for `api`, `webui`, and `all`, defaulting to `all`.
- [x] Preserve Windows venv creation, editable install, and local-single wizard init for API/all modes.
- [x] Add WebUI launch from `apps/tldw-frontend`, Bun checks, `TLDW_WEBUI_PORT`, and `NEXT_PUBLIC_API_URL`.
- [x] For `all`, start the API in a new PowerShell process and run WebUI in the current console.

### Task 4: Installer Compatibility Wrappers

**Files:**
- Modify: `Helper_Scripts/Installer_Scripts/MacOS_Run_tldw.sh`
- Modify: `Helper_Scripts/Installer_Scripts/Linux_Run_tldw.sh`
- Modify: `Helper_Scripts/Installer_Scripts/Windows_Run_tldw.bat`
- Modify: `Helper_Scripts/Installer_Scripts/MacOS_Install_Update.sh`
- Modify: `Helper_Scripts/Installer_Scripts/Linux_Install_Update.sh`
- Modify: `Helper_Scripts/Installer_Scripts/Windows_Install_Update.bat`

- [x] Replace duplicated installer run logic with wrappers that resolve the installed checkout and delegate to root quick-launch scripts.
- [x] Default installer wrappers to `TLDW_VENV_DIR=venv` and `TLDW_SKIP_INSTALL=1` so they reuse legacy installer environments.
- [x] Keep installer completion output pointing to `all`, `api`, and `webui` modes.

### Task 5: Documentation, Backlog, And Verification

**Files:**
- Modify: `README.md`
- Modify: `Docs/Getting_Started/Profile_Local_Single_User.md`
- Modify: `backlog/tasks/task-500 - Add-WebUI-support-to-OS-quick-launch-scripts.md`
- Modify: `Docs/superpowers/plans/2026-05-24-quick-launch-webui-implementation-plan.md`

- [x] Update local quick-launch docs so root scripts are described as API + WebUI launchers by default, with API-only and WebUI-only modes.
- [x] Update Backlog task `TASK-500` with consolidated scope, touched files, and final verification.
- [x] Run `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Utils/test_quick_launch_scripts.py -v`.
- [x] Run Bash syntax checks for touched shell scripts.
- [x] Run Bandit on the touched Python test file.
- [x] Run `git diff --check`.
