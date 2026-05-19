# Quick-Launch Shortcut Scripts Design

Task: `TASK-419`

## Goal

Add OS-native no-Docker shortcut scripts for self-hosters who want the local single-user API path without needing `make`.

## Design

The scripts are thin wrappers around the existing local single-user setup contract. They do not create a parallel installer or duplicate the setup wizard logic. Each launcher creates `.venv` if needed, installs the editable package into that venv, runs `tldw_Server_API.cli.wizard.cli init --profile local-single`, and starts `uvicorn` on `127.0.0.1:8000`.

## Files

- `quick-launch.sh`: Linux and shell-first macOS launcher.
- `quick-launch.command`: macOS Finder-friendly wrapper that delegates to `quick-launch.sh`.
- `quick-launch.ps1`: Windows PowerShell launcher.
- `tldw_Server_API/tests/Utils/test_quick_launch_scripts.py`: contract tests for launchers.
- `README.md` and `Docs/Getting_Started/Profile_Local_Single_User.md`: onboarding references.

## Constraints

- Do not call Docker or require Make.
- Do not reference deprecated Gradio-era entrypoints such as `summarize.py -gui`.
- Do not print `SINGLE_USER_API_KEY` by default; point users to the env file or existing explicit secret paths.
- Keep custom host/port support limited to environment variables so the default remains boring and local.

## Verification

Focused verification should include the new contract tests, existing onboarding contract tests, and Bandit over the touched scripts/tests/docs scope.
