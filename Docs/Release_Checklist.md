## tldw_server Release Checklist

This document is the broad readiness checklist for preparing a full tldw_server release. The authoritative operator path lives in [Docs/Development/Release_Process.md](Development/Release_Process.md); use that document for the actual release command flow, then use this checklist to confirm scope, readiness, and follow-through. Keep this checklist up to date as the project evolves and adapt the scope (for example, use a subset for small bugfix releases).

---

## 1. Versioning & Metadata

Ensure version numbers and release metadata are consistent and discoverable.

- [ ] Bump the version in `pyproject.toml` and verify it matches all visible references (for example `README.md`, `Docs/`, Docker tags).
- [ ] Update release notes (for example `CHANGELOG.md` or a dedicated release notes file) with all user-visible changes, including any notable performance or behavior changes.
- [ ] For WebUI route-boundary/recovery changes, use and adapt `Docs/Product/Completed/WebUI-related/M2_Release_Note_Template_Route_Recoverability_2026_02.md` so recoverability claims and troubleshooting links stay consistent.
- [ ] For WebUI releases, include UX regression status using `Docs/Product/Completed/WebUI-related/M5_3_Release_Note_UX_Regression_Reporting_Template_2026_02.md` (gate outcomes, highest severity, and any conditional-ship exceptions).
- [ ] Verify API docs that mention a version or base URL are current.
- [ ] Document any new or changed environment variables in `README.md` and the relevant docs under `Docs/`.
- [ ] Verify that the chosen version bump (patch, minor, major) matches the project’s versioning guidelines (for example `Versioning_Scheme.jpg`).
- [ ] If the Web UI shows release notes or a “What’s new” section, ensure it reflects the same changes as the backend release notes.

---

## 2. Documentation Health

Confirm that all user and developer documentation matches the current behavior of the system.

- [ ] Walk through the Quickstart in `README.md` from a clean environment (install → run server → open `/docs`) and update any out-of-date steps.
- [ ] Verify AuthNZ setup docs (`.env.example`, AuthNZ docs in `Docs/AuthNZ/`) match the actual initialization flow.
- [ ] Update media ingestion documentation for any new formats, providers, or configuration options.
- [ ] Update RAG, embeddings, and evaluation docs for new models, providers, or configuration changes.
- [ ] Update MCP Unified docs for any endpoint, schema, or behavior changes.
- [ ] Update Web UI / Next.js documentation for any breaking API or route changes.
- [ ] Build the docs site (for example `mkdocs build`) and fix any build errors or broken links.
- [ ] Ensure `Env_Vars.md`, `Docs/Documentation.md`, and any other configuration overviews reflect new environment variables or configuration options.
- [ ] Spot-check for broken internal links and obvious spelling or grammar issues in `Docs/` and `README.md`.

---

## 3. Code Review & Hygiene

Review the diff relative to the previous release and ensure the codebase is clean and consistent.

- [ ] Review changes for stray debug code, temporary hacks, and TODOs without associated issues.
- [ ] Verify new code follows existing style and conventions (PEP 8, type hints, `loguru` logging, error handling patterns).
- [ ] Confirm names of modules, classes, functions, and variables reflect their current responsibilities.
- [ ] Ensure comments and docstrings have been updated to match the latest behavior.
- [ ] Run static checks (for example `ruff`, `flake8`, or the project’s linter) and confirm there are no new warnings.
- [ ] Run type checks (for example `mypy`) on `tldw_Server_API/` and fix any new errors.
- [ ] Confirm changes align with `AGENTS.md`, `Project_Guidelines.md`, and any relevant design documents under `Docs/Design/`.
- [ ] Ensure new modules or components under `tldw_Server_API/app/` have corresponding tests under `tldw_Server_API/tests/` following existing patterns.

---

## 4. Core Build & Install

Verify that a new user can cleanly install and start the backend using the documented instructions.

- [ ] In a fresh virtual environment, run `pip install -e .[dev]` and confirm it completes successfully.
- [ ] Start the server (for example `python -m uvicorn tldw_Server_API.app.main:app --reload`) and verify there are no import or configuration errors.
- [ ] Follow the minimal configuration flow: copy `.env.example` to `.env`, run the AuthNZ initialization script, and confirm it completes successfully.
- [ ] Verify the server handles missing but optional configuration files gracefully (clear error messages, no crashes).
- [ ] Build and run the main Docker image, confirming the container serves `/docs` and `/api/v1/config/quickstart` as expected.
- [ ] Verify that `ffmpeg` is installed and available on `PATH` so that audio/video ingestion and STT features work as documented.
- [ ] After startup, call a simple health or status endpoint (for example `GET /docs` or a dedicated health check) and confirm it responds successfully.

---

## 5. Test Matrix (Python / Backend)

Run the test suite and confirm that core backend behavior is stable and covered.

- [ ] From the repo root, run `python -m pytest -m "unit" -v` and confirm all unit tests pass.
- [ ] Run `python -m pytest -m "integration" -v` and confirm integration tests pass, or that any skips are intentional and documented.
- [ ] Confirm tests involving the per-user content DB, notes/chats databases, and `users.db` pass with the default SQLite configuration.
- [ ] Run AuthNZ PostgreSQL fixture tests (for example `tldw_Server_API/tests/AuthNZ/`) when Postgres is available and confirm they pass or skip cleanly based on the fixture (no hangs or unexpected failures).
- [ ] Verify embeddings and RAG tests pass with at least one configured embedding provider (mocked or real).
- [ ] Verify OpenAI-compatible `/chat/completions` tests pass with mocked providers and that provider routing and error handling tests succeed.
- [ ] Verify audio STT and TTS tests pass for `/audio/transcriptions` and `/audio/speech` endpoints (using real engines or mocks as appropriate).
- [ ] Confirm MCP Unified status and tool execution tests pass, including any WebSocket tests where available.
- [ ] Run coverage (for example `python -m pytest --cov=tldw_Server_API --cov-report=term-missing`) and confirm coverage meets the project target.

---

## 6. Test Matrix (Modes & Configurations)

Exercise the system under different authentication modes and provider configurations.

- [ ] Test `AUTH_MODE=single_user` with `SINGLE_USER_API_KEY` by making authenticated API calls and verifying expected behavior.
- [ ] Test `AUTH_MODE=multi_user` with JWT: user registration, login, and authenticated requests should all work.
- [ ] Verify rate limiting for embeddings, chat, and evaluations behaves as documented (throttling, error messages, and headers).
- [ ] Start the server with major external providers disabled (no API keys set) and confirm it runs and handles missing providers gracefully.
- [ ] Configure at least one commercial provider and confirm basic flows (chat, embeddings, RAG) work end-to-end.
- [ ] Configure at least one local LLM backend (for example Llama.cpp, vLLM, or Ollama) and confirm basic flows work.
- [ ] Install optional extras (for example `.[multiplayer]`) and run any associated tests, verifying they pass.
- [ ] When no external LLM providers are configured, verify that supported non-LLM functionality (for example basic media ingestion and search) still works as documented.
- [ ] Where possible, run at least a basic subset of unit and integration tests on both Linux and macOS (and Windows if supported) using the same commit.

---

## 7. Web UI / Frontend Integration

Confirm that the Web UI integrates correctly with the backend and that core user flows work.

- [ ] Run the `apps/tldw-frontend` dev build and verify it can connect to the backend in at least single-user mode (and multi-user, if supported).
- [ ] Run the production build for `apps/tldw-frontend` (for example `npm run build`) and ensure it completes without errors.
- [ ] From the Web UI, upload and process media (video, audio, or documents) and confirm processing completes successfully.
- [ ] Use the Web UI to search ingested content and verify results match expectations.
- [ ] Start a chat session from the Web UI and confirm messages stream or complete as expected.
- [ ] Exercise at least one RAG-backed chat or research flow through the Web UI and verify it behaves correctly.
- [ ] Check browser dev tools for console errors during common flows and fix any issues.
- [ ] Verify frontend environment variables and base URL configuration match the documented setup.
- [ ] Open the main views (for example Home, Media Library, Chat, Settings) and confirm they load without 500 errors or infinite loading spinners.

---

## 8. Database & Migration Safety

Ensure database schemas, migrations, and upgrade paths are safe and well understood.

- [ ] For each database (`Databases/users.db`, `Databases/evaluations.db`, the per-user content DB, per-user `ChaChaNotes.db`), verify that any schema changes are reflected in migration scripts or initialization logic.
- [ ] Ensure any schema changes are described and scripted according to `Docs/Database_Migrations.md`.
- [ ] Test an upgrade flow by starting from a database created by the previous release and launching the new server, confirming that it starts and runs without manual intervention or with clearly documented migration steps.
- [ ] Verify that data created by the current version remains readable and usable by the previous minor version where compatibility is expected, or document incompatibilities clearly.
- [ ] Exercise soft delete and restore flows (for example on notes or media) to confirm data integrity.
- [ ] Test chatbook export and import on a sample dataset to ensure backups and restores function correctly.
- [ ] Run a small batch of concurrent ingest and query operations and verify there are no persistent database lock errors or integrity issues in logs.

---

## 9. Performance & Resource Checks

Look for major performance regressions or resource issues compared to the previous release.

- [ ] Ingest a medium-sized video or large document, including transcription and chunking, and compare processing time with the previous release.
- [ ] Run a small, fixed RAG benchmark (for example a set of known queries) and compare latency and result quality to prior versions.
- [ ] Perform a basic long-running smoke test (for example ingest + search + chat cycles for around 30 minutes) and monitor for memory growth or instability.
- [ ] If profiling tools are available, run a focused profile on a representative workload and investigate any significant regressions.
- [ ] Optionally run the LLM gateway benchmarks (for example `make bench-sweep` or `make bench-full`) and compare results against previous release baselines.
- [ ] Record key latency and throughput numbers for representative workloads (for example under `Docs/Monitoring/` or `Docs/Maintenance/`) to help track trends over time.

---

## 10. Security & Hardening

Validate basic security properties and ensure new changes do not weaken protections.

- [ ] Confirm file upload validation is enforced (type, size, and content checks) and that invalid uploads produce clear, non-crashing errors.
- [ ] Verify that URLs used for yt-dlp or web scraping are validated or restricted according to project policies.
- [ ] Audit authentication and authorization: ensure no new endpoints that should be protected are left unauthenticated, and that role-based behavior (where applicable) is correct.
- [ ] Review logs for any leakage of secrets (API keys, passwords, JWTs) and remove or redact as needed.
- [ ] Verify CORS configuration matches the intended deployment environments (local development and production).
- [ ] Review dependency updates using your preferred tools and address any critical or high severity security advisories, or document exceptions.
- [ ] Run a dependency or container scan (for example pip-audit, Trivy, or an equivalent tool) and address or document any critical findings.
- [ ] Verify rate limiting and abuse protections for public-facing endpoints prevent unbounded requests and behave as documented.

---

## 11. Packaging & Distribution

Confirm that you can build, install, and run the project from release artifacts.

- [ ] Build source and wheel distributions (for example with `python -m build`) and confirm the build completes.
- [ ] In a fresh virtual environment, install from the built artifacts and verify that the server starts and passes a small smoke test (basic API calls).
- [ ] Build the main Docker image with the new version tag and confirm that it starts correctly and reports the expected version (via logs or an explicit version endpoint).
- [ ] If you publish images or packages to a registry, verify that tags and metadata match the release version and that consumers can pull and run them.
- [ ] Confirm the GitHub Actions publish workflow ran for the release, and that GHCR + Docker Hub have updated tags for `tldw-server`, `tldw-server-worker`, and `tldw-server-audio-worker`.
- [ ] Verify release publish prerequisites are set: `DOCKERHUB_IMAGE` repo variable and `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets.
- [ ] Verify that release-published tags are the source of truth for artifact verification and are not replaced by main snapshots (note: WebUI/Admin UI currently have snapshot tags only).
- [ ] Validate GHCR attestations exist for each published image version (app/worker/audio-worker).
- [ ] From a clean host environment, run the released Docker image with minimal configuration and exercise a short ingest → search → chat flow end-to-end.
- [ ] Verify that contributor-friendly dev entrypoints (for example `make server-up-dev`) still work as documented.

---

## 12. Issue & Feature Hygiene

Align the release with the project’s issue tracker and feature planning.

- [ ] For all issues targeted at this release, verify that the fixes are present, tested, and reflected in the release notes.
- [ ] Confirm new features have tests, documentation, and any necessary configuration updates (for example templates or sample configs).
- [ ] Ensure there are no remaining P0/P1 bugs for this version, or that they are explicitly called out in a “Known Issues” section.
- [ ] Close or update relevant issues in the tracker to reflect the final state of the release.
- [ ] Ensure any `IMPLEMENTATION_PLAN.md` entries related to this release are updated to “Complete” or archived.
- [ ] If there are known limitations or partial features, document them in `Docs/Issues/` and/or the release notes so users can discover them easily.

---

### Using This Checklist

- Treat this document as a guide, not a rigid contract. Adjust items to match the scope and risk level of each release (for example, use a focused subset for small bugfix releases).
- When in doubt, prefer smaller, well-tested changes over large, risky ones.
- Keep this checklist updated as new subsystems are added or old ones are retired.
