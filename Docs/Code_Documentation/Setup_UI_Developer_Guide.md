# Setup UI Developer Guide

## Overview

The first-run setup experience lives in `tldw_Server_API/app/Setup_UI/` with assets under `tldw_Server_API/app/static/setup/`, and is backed by the `app/api/v1/setup.py` API. It provides:

- A guided wizard that selects relevant configuration sections.
- A configuration explorer that renders every entry in `Config_Files/config.txt` with inline context.
- A floating “Setup Assistant” bubble that answers questions by mining local configuration metadata.

This document explains how the setup stack is structured, how data flows from the backend to the UI, and the key extension points for developers.

## Native WebUI Readiness Flow

The WebUI also exposes a native setup readiness screen at `/setup` for users who have already configured a backend URL in the WebUI. This screen is separate from the static backend fallback UI, but both use the same setup contracts.

- Before first-run setup is completed, `/setup` uses the first-run readiness endpoints under `/api/v1/setup/readiness/*`. These endpoints keep the same local-only unauthenticated guard as the backend setup UI.
- After WebUI first-run setup is completed, the same screen switches to admin mode and calls `/api/v1/setup/admin/readiness/*`. Those endpoints require the admin setup dependency used by the server-side audio installer.
- If the WebUI has no configured backend URL, or connection setup still needs attention, `/setup` keeps rendering the connection onboarding wizard.
- The backend `/setup` page remains the recovery fallback and is always linked from the native readiness screen.
- Profile selection and preview are non-mutating. Config writes and model/download provisioning only happen through the explicit `Provision now` action.

The native screen is implemented in `apps/packages/ui/src/components/Option/Setup/ReadinessSetupScreen.tsx` and uses `useSetupReadiness()` from `apps/packages/ui/src/components/Option/Setup/hooks/useSetupReadiness.ts`. The hook is backed by the shared client in `apps/packages/ui/src/services/tldw/setup-readiness.ts`, which selects first-run or admin endpoint paths based on mode.

## Architecture Diagram (Textual)

```
config.txt (with comments)
        ↓ (parsed by ConfigParser + comment index)
setup_manager.get_config_snapshot()
        ↓ (JSON)
GET /api/v1/setup/config
        ↓
setup.js → render sections, wizard, hints
        ↓
User edits values → POST /api/v1/setup/config → setup_manager.update_config()
```

The assistant flow taps the same snapshot and searches its metadata:

```
User question → POST /api/v1/setup/assistant
        ↓
setup_manager.answer_setup_question() → fuzzy match over sections/fields → result JSON
        ↓
setup.js assistant UI → show response + deep-link buttons
```

## Backend Components

### `setup_manager.py`

- **Comment Preservation**: `_build_comment_index()` reads `config.txt` verbatim and associates comment blocks with sections/fields. `update_config()` rewrites only the touched keys while preserving the original comment/text layout. Never bypass this function when writing setup changes.
- **Snapshot Generation**: `get_config_snapshot()` merges ConfigParser data, `SECTION_LABELS/DESCRIPTIONS`, and comment-derived hints. Every field now has a non-empty `hint` to guarantee UI context.
- **Assistant Search**: `answer_setup_question()` tokenises the user query and scores each section/field hint using `SequenceMatcher`. The method returns the top matches (with scores) and a short prose answer assembled from description + hint text.

### `setup.py` (API layer)

- `/status`: gatekeeper for first-run logic (used by the guard script and the wizard).
- `/config` (GET/POST): fetch and persist the configuration snapshot. POST writes through `update_config()` so comments stay intact.
- `/assistant` (POST): lightweight Q&A endpoint; no external LLM. Wraps `answer_setup_question()` and translates validation errors into `HTTP_400`.
- `/install-status` (GET): exposes progress from the background installer (see below). The Setup UI polls this endpoint while dependencies/models are being provisioned.
- `/readiness/profiles`, `/readiness/status`, `/readiness/preview`, `/readiness/provision`, `/readiness/verify`: first-run readiness endpoints for chat defaults, embeddings/RAG, and speech setup. These are local-only and available only while setup is still required.
- `/admin/readiness/profiles`, `/admin/readiness/status`, `/admin/readiness/preview`, `/admin/readiness/provision`, `/admin/readiness/verify`: post-setup admin readiness endpoints with the same behavior, gated by admin setup access.

### `install_manager.py`

- **Dependency bootstrap**: Before downloading models the installer aggregates all required Python packages (per backend) and runs `pip install` with `sys.executable`. A new env flag, `TLDW_SETUP_SKIP_PIP`, skips this phase - steps are marked as `skipped` and model downloads are suppressed to avoid half-configured states. Optional `TLDW_SETUP_PIP_INDEX_URL` rewires pip to a custom/simple index for air-gapped environments.
- **Model downloads**: Existing logic remains in place. When `TLDW_SETUP_SKIP_DOWNLOADS` is set the steps are recorded as skipped and no network calls occur.
- **Status reporting**: Every dependency/model action is logged to `Config_Files/setup_install_status.json` (or the override specified by `TLDW_INSTALL_STATE_DIR`). The Setup UI renders these steps verbatim in the “Installer Progress” panel.

## Frontend Components (Setup UI)

### Native WebUI Readiness Components

- `ReadinessSetupScreen.tsx`: compact readiness dashboard used by the WebUI `/setup` route. It renders the profile picker, Chat/Embeddings/RAG/Speech lane summaries, secondary TTS details, preview summary, verification result, and the explicit `Provision now` action.
- `useSetupReadiness.ts`: React hook that loads profiles and status in parallel, maps first-run/admin guard failures into display states, and polls status while provisioning is active.
- `setup-readiness.ts`: typed client for the first-run and admin setup readiness endpoints. Keep this client aligned with `openapi-guard.ts` whenever endpoint paths change.

### Entry Point

- `setup.html` loads `setup.js`, `setup-guard.js`, and `setup.css`. The new assistant markup is appended at the bottom of the body.

### Styles (`css/setup.css`)

- `section-content` now stacks `section-info` above the fields for readability.
- `.assistant-*` classes provide layout/animation for the floating chat bubble. The bubble hides the toggle button while open (`.assistant-root.assistant-open`).

### Logic (`js/setup.js`)

- **Wizard Constants**: `FEATURE_OPTIONS` drives recommendation mapping; extend this array to expose new modules.
- **Hints**: every form field uses `humaniseKey()` plus the backend hint to display contextual help.
- **Assistant**:
  - `initAssistant()` wires toggle, close, submit, and Escape handlers.
  - `sendAssistantQuestion()` posts questions to the API and handles loader state.
  - `addAssistantMessage()` renders conversation bubbles and suggestion buttons. Clicking a suggestion opens the relevant accordion and focuses the input via `focusSectionFromAssistant()`.
- **Installer Progress**: once setup completes with an install plan, `beginInstallStatusMonitoring()` kicks off polling and updates the new progress card. The summary message now reminds users we are installing Python dependencies *and* model files.

### Comment Awareness

Because `update_config()` now preserves comments, developers should continue to keep documentation inline in `config.txt`. When adding new keys, include descriptive comments immediately above the entry; they will automatically surface as hints in the UI and via the assistant.

## Extending the Setup Flow

1. **Add a new configuration section**: update `SECTION_LABELS`/`SECTION_DESCRIPTIONS`, teach the wizard how to surface it (`FEATURE_OPTIONS`), and ensure the config template includes comments.
2. **New wizard step**: extend `WIZARD_STEPS` in `setup.js` and adjust `TOTAL_GUIDED_STEPS`. For multi-select behaviour, reuse `toggleWizardSelection()`.
3. **Assistant special cases**: if a question should prioritise a specific field, give that field a distinctive hint/comment; the scoring function rewards token matches.

## Testing & Verification

- **Manual**: run through the wizard, save the config, and diff `Config_Files/config.txt` to confirm comments remain unchanged.
- **Assistant**: ask for a known field (e.g. “single_user_api_key”) and verify the response contains a deep link that scrolls to the proper input.
- **Accessibility**: ensure the assistant opens with focus on the textarea, responds to Escape, and the wizard retains keyboard navigation.
- **Installer smoke test**: run the wizard, select one STT and one TTS backend, and watch the progress panel for dependency → model steps. Confirm the status file lands in `Config_Files/setup_install_status.json` (or the overridden cache directory).

## Future Ideas

- Swap the fuzzy matcher for a vector-store when a lightweight embedding pipeline is available.
- Add automated UI tests (Playwright) to cover wizard validation and assistant interactions once test infrastructure is in place.

## Security Model

- Setup API is unauthenticated during first-run but restricted to local requests by default.
- Mutating endpoints (POST `/api/v1/setup/config`, `/api/v1/setup/complete`, `/api/v1/setup/assistant`) are local-only.
- Read endpoint GET `/api/v1/setup/config` is also local-only to reduce configuration surface exposure.
- Environment variables:
  - `TLDW_SETUP_ALLOW_REMOTE=1` - temporarily allows remote access to setup endpoints on trusted networks.
  - `TLDW_SETUP_TRUST_PROXY=1` - when set, honors `X-Forwarded-For` to determine client origin; otherwise the header is ignored.
- Remote setup access requires authentication as an admin (admin role + `system.configure` permission), or the single-user principal when running in single-user mode.
- Secret values never leave the server; snapshot marks them with `is_secret: true`, returns an empty `value`, and includes an `is_set` flag.

## Update Validation

Setup writes are validated against the existing configuration:

- Sections and keys must already exist; unknown sections/keys are rejected.
- Values must conform to the inferred type of the current value for boolean/integer/number fields (strings are accepted as-is).
