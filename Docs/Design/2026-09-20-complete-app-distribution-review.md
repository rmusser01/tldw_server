# Complete application distribution: design review

Date: 2026-09-20 (America/Los_Angeles)
Task: TASK-13265
Review baseline: working tree based on `dfe36e91e7`; concurrent unrelated edits exist.
Status: review complete; corrections below are recommendations for the continuing design, not implemented behavior.

## Assessment

Retain the approved small PyPI launcher plus managed release bundles and the paired prebuilt Docker distribution. The architecture can support the requested experience, but packaging the existing server and WebUI is insufficient. The principal gaps concern native dependencies, runtime routing, upgrade ownership, and complete recovery.

The user has approved the distribution architecture, first-run/background lifecycle, and data/update/optional-engine policy in principle. They requested this review before continuing. Exact platform support, detailed release tests, and the final written specification remain open. No installation, runtime implementation, or publication was performed for this review.

## Findings and proposed corrections

### R1 — High: a lightweight launcher does not produce a lightweight backend

The default dependencies in `pyproject.toml:108` through `:150` include YARA, PyAudio, Docling, PyTorch, Transformers, and local inference libraries. Installing the existing backend with its normal dependencies would preserve much of the installation friction. Merely removing dependencies from metadata would also be unsafe if core startup imports them unconditionally.

Define an executable core profile and verify its startup/import boundary with heavy packages absent. Lock its complete dependency set for each supported platform; install tested wheels or managed binaries without compiling on the user's machine. Keep the API-only package compatible through an explicit packaging decision, rather than silently changing its dependency contract. Do not create a second divergent copy of the backend source.

The native profile also needs an explicit external-service policy. Docker supplies Redis (`Dockerfiles/docker-compose.single-user.yml:23`), while native AuthNZ sessions have a database fallback (`tldw_Server_API/app/core/AuthNZ/session_manager.py:149`). This is evidence for that session path, not proof every promised core feature works without Redis. Verify the complete core feature list without Redis/PostgreSQL. If baseline media operations require FFmpeg, provide a managed executable; otherwise acquire it through the audio component flow. One existing Windows path assumes `./Bin/ffmpeg.exe` (`tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Files.py:1620`).

### R2 — High: the reusable WebUI needs a real runtime routing contract

`apps/tldw-frontend/next.config.mjs:147` constructs API rewrites from configuration evaluated during the build. `Dockerfiles/Dockerfile.webui:33` defaults the build's internal origin to `http://app:8000`. Session bootstrap separately reads an internal origin at runtime (`apps/tldw-frontend/pages/api/_tldw-webui/runtime-auth-policy.ts`). A native launcher changing only environment variables could therefore configure session bootstrap and ordinary API calls differently.

Use same-origin browser requests and make the private upstream runtime-configurable through an explicitly designed server-side routing path. Preserve cookies, CSRF checks, redirects, uploads, streaming, cancellation, and WebSockets. Do not rely on editing generated bundles. Fixed per-distribution builds are an alternative only if the design explicitly accepts fixed private addresses and rejects conflicting ports.

The public browser origin must remain stable after first installation. Persist the selected origin; do not silently change it when a port is occupied. Identify the existing managed instance before reusing it, and provide recovery for unrelated port occupants. Browser storage and cookies cannot be treated as equally isolated: cookies are not scoped by port, so simultaneous managed instances also need deliberate cookie naming and identity rules.

Next.js documents that `NEXT_PUBLIC_*` values are fixed during build: [environment-variable documentation](https://nextjs.org/docs/pages/guides/environment-variables). The installed framework's build code serializes standalone configuration and generates routing manifests. An existing local standalone artifact was inspected, but had no rewrites; it is not evidence that the proposed native quickstart works.

### R3 — High: current backups do not cover whole-application recovery

`tldw_Server_API/app/services/admin_bundle_service.py:51` lists six database datasets, and `:420` explicitly rejects vector-store export. Jobs owns another database (`tldw_Server_API/app/core/Jobs/manager.py:657`); Workflows and its scheduler have further paths (`tldw_Server_API/app/core/DB_Management/db_path_utils.py:684`). Uploads, generated assets, configuration, and decryption keys must also be considered. The existing bundle is not a complete upgrade recovery point.

First define an inventory of authoritative stores, files, and secrets versus rebuildable caches and downloadable models. For launcher-managed local storage, stop every writer and produce a verified, coherent snapshot plus a manifest of its coverage. Preserve keys needed to read restored data. Avoid copying unchanged model weights for every update, and preflight storage for downloads, extraction, the candidate environment, and recovery data.

Automatic restoration should initially cover only data the launcher owns and can quiesce. External databases, external workers, and custom storage need their own explicit recovery contracts. Validate restoration into disposable storage before claiming a backup is usable.

### R4 — High: browser data and browser migrations need their own recovery policy

The WebUI maintains chat/message tables, settings, drafts, files, recordings, and generated audio in IndexedDB (`apps/packages/ui/src/db/dexie/schema.ts:28`). Composer drafts also use local storage (`apps/packages/ui/src/hooks/useDraftPersistence.tsx:30`, `apps/packages/ui/src/services/settings/local-bucket.ts:23`). A server backup cannot capture these stores automatically.

Preserve the browser origin and explicitly list browser-only data in backup coverage. Provide export/synchronization for authoritative browser data, or accurately scope recovery to exclude it. Do not promise that all data survives a browser reset or a change of browser. Upgrade/downgrade checks must include IndexedDB schema compatibility and stale open tabs; restoring an older backend and WebUI does not reverse an already-applied browser migration. Flush active drafts and prevent incompatible writes during activation.

### R5 — High: a candidate startup can mutate data before activation

Normal startup ensures AuthNZ schema and seed state (`tldw_Server_API/app/services/startup_auth.py:55`) and invokes worker startup (`tldw_Server_API/app/services/lifespan_startup_sequence.py:88`). Starting a candidate to check readiness while pointing it at live data is not a harmless probe.

Validate downloads and package contents while the old release runs. Any trial backend boot must use disposable data/configuration roots with worker execution and external effects disabled. Only after quiescence and a verified recovery snapshot may the candidate migrate live data. Keep ingress and background work gated until activation checks pass.

Ordinary shutdown is insufficient evidence of quiescence: Jobs acquisition gating is process-wide (`tldw_Server_API/app/core/Jobs/manager.py:685`), and lease waiting defaults to zero (`tldw_Server_API/app/services/shutdown_job_poller_handoff.py:115`). Updates need a bounded maintenance sequence across all managed writers. If work cannot safely drain, postpone the update and identify it to the user. Persist the upgrade phase so a crash or reboot can be recovered without guessing which schema is active.

### R6 — High: optional installation must isolate configuration as well as packages

The current setup installer runs `sys.executable -m pip install --upgrade`; its uv fallback does not specify a target interpreter (`tldw_Server_API/app/core/Setup/install_manager.py:254`). Verification writes readiness state (`:1066`), and some installs patch configuration (`:1439`). Calling these operations from the active server would violate the approved promise that a failed optional install leaves the working app unchanged.

Target a candidate interpreter explicitly and give installation and verification private configuration/readiness paths. Publish the tested environment and its configuration together under the same update lock. Use a small curated set of supported component combinations, rather than allowing arbitrary packages or promising every engine on every machine. Driver requirements and unsupported hardware should be detected before large downloads.

### R7 — High: lifecycle and update authority must survive application replacement

One launcher-owned supervisor outside replaceable release directories should own native start/stop/update operations, the exclusive installation/data-root lock, verified process identity, descendant cleanup, and the durable update record. Route CLI and authenticated WebUI requests to that same owner. Do not infer ownership from a PID or occupied port alone. The existing ephemeral wizard helpers (`tldw_Server_API/cli/wizard/cli.py:433`) are not a persistent application supervisor.

Expose only narrowly defined managed operations and curated release/component identifiers. Preserve the existing same-origin/session protections; do not introduce a generic shell or package-install endpoint. Login startup must invoke the same owner rather than launching duplicate services.

Docker updates require host authority. The app container should not receive the Docker socket to implement an Update button. Use the release bundle's host-side helper, or separately design a restricted host controller if full WebUI-driven Docker updates are required. That boundary is a necessary clarification of the earlier shared update experience. Docker describes the authority associated with daemon access in its [security documentation](https://docs.docker.com/engine/security/#docker-daemon-attack-surface).

### R8 — Medium: installation still depends on path and bootstrap assumptions

The setup helper defaults `.env` to the current directory outside a repository (`tldw_Server_API/cli/wizard/profiles.py:79`). Configuration already has useful platform-aware helpers (`tldw_Server_API/app/core/config_paths.py:57`), but user database defaults remain tied to the project root (`tldw_Server_API/app/core/config.py:1079`). A new launcher should reuse those helpers while explicitly supplying every authoritative path and controlling inherited configuration.

Test launching from arbitrary directories, paths with spaces/non-ASCII characters, and read-only installed packages. Reinstall must discover existing managed data without silently adopting a developer checkout or unrelated `.env`.

The two-command example assumes uv and the installed command are on PATH. uv documents that shell setup can be required: [tool installation](https://docs.astral.sh/uv/guides/tools/#installing-tools). The first-install instructions need to handle that explicitly and prove first launch on a fresh Windows/macOS/Linux account. Define launcher upgrades separately from application upgrades; updating the small PyPI tool alone must not silently migrate application data.

### R9 — Medium: a product release must identify all artifacts and runtimes

Version labels alone do not establish an installable, reproducible release. Publish a release manifest that identifies exact backend, WebUI, Python/Node runtime, dependency locks, supported platform tuples, required launcher version, data/browser compatibility, and artifact digests. Authenticate the manifest through a defined trusted release mechanism. Publish the installable release only after every required artifact and acceptance check succeeds. Interrupted or corrupt downloads must never become active; ordinary later launches must work from the installed release without the distribution service being available.

Validate archive paths, symlinks, extraction bounds, and contents before activation. Next standalone output needs public/static assets and can include incorrect traces, particularly in a monorepo: [Next.js output documentation](https://nextjs.org/docs/app/api-reference/config/next-config-js/output). Build from clean source in CI and enforce a release-content allowlist so development databases, secrets, caches, and model directories cannot enter bundles.

The current WebUI Docker runtime is Node 20 (`Dockerfiles/Dockerfile.webui:48`), which upstream now lists as EOL. Select and test a supported LTS runtime and define how runtime security fixes reach explicit application updates: [Node.js release status](https://nodejs.org/en/about/previous-releases). No particular replacement version is certified by this review.

Define support as OS version, architecture, and relevant ABI/libc, not just three OS names. Test native dependency wheels on each tuple. Keep optional engine compatibility separate from core application compatibility. The existing Docker publish matrix omits the WebUI (`.github/workflows/publish-docker.yml:28`).

### R10 — Release constraint: existing publication policy must be reconciled

`Docs/Development/Packaging_and_Distribution_Strategy.md:8` currently records a frontend publication freeze and a later release-specific gate. `Docs/ADR/029-tldw-frontend-static-pypi-bundle.md` is an older Proposed alternative. The new specification should explicitly relate to both; design approval does not establish that external distribution requirements have been completed.

Continue design and local build preparation. Before publishing, resolve the documented release gate, confirm package/image ownership, and verify the chosen name. A PyPI pending Trusted Publisher does not reserve a name: [PyPI documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/). No publication or registry configuration was attempted in this review.

## Acceptance checks to add to the design

| Scenario | Required evidence |
| --- | --- |
| Fresh native account on every supported tuple | Complete released-artifact installation, core startup, authenticated WebUI, provider-config flow, one chat with a mock provider, document ingest/search, no frontend compilation or undeclared native dependency. |
| Fresh Docker host | Bundle starts matching prebuilt images with generated credentials and persistent storage; no repository, Make, host Python, or manual credential copying. |
| Core dependencies only | Promised core workflows run with heavy optional libraries and external services absent. |
| Different working directory and occupied ports | Existing data/identity retained; no silent origin change or adoption/termination of unrelated processes. |
| Upstream changes at runtime | Same released WebUI reaches the selected backend for session bootstrap, ordinary API, uploads, streaming, and WebSockets. |
| Interrupted update at each phase | Next invocation recovers deterministically; no incompatible database opens, lost authoritative data, or duplicate job execution. |
| Long-running work and candidate probes | No snapshot while managed writers remain; candidate validation does not touch live stores or cause external effects. |
| Backup restore and downgrade | Restore proves complete declared coverage, key usability, cross-store consistency, and browser schema compatibility; destructive rollback requires informed user choice. |
| Failed optional component installation | Active packages, configuration, readiness state, and existing workflows remain usable. |
| Corrupt/unavailable download and low disk | Installation fails before activation, preserves recovery material, supports retry, and does not block later offline startup of an installed app. |
| Reinstall and uninstall | Existing managed data rediscovered; deletion requires a separate explicit action; external storage is not removed. |

## Review method and limits

Source inspection plus an independent backend lifecycle/recovery review; primary findings were checked against the cited source. Upstream Next.js, uv, Docker, Node.js, and PyPI documentation was consulted. No fresh machine installation, candidate update, restore, platform certification, or runtime smoke test was executed. This is a design review, not a claim that these checks pass.

Only this review document and TASK-13265 were edited. Bandit and application tests are not applicable to these documentation/task-only edits; whitespace and report structure/link checks are recorded with the task.
