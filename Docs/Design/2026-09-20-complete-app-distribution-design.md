# Complete WebUI and server distribution

Date: 2026-09-20
Task: TASK-13265
Status: written specification approved in conversation on 2026-09-22; implementation planning in progress.
Review: [2026-09-20-complete-app-distribution-review.md](2026-09-20-complete-app-distribution-review.md)

## 1. Product contract

A new user installs the complete application, opens the WebUI, configures a model provider, and starts using their own documents. They do not clone the repository, build the frontend, coordinate component versions, or copy credentials between services. Native installation works without Docker or a separately installed Node runtime. Windows, macOS, and Linux are first-release targets.

The two distribution paths are:

| Path | Installation and operation |
| --- | --- |
| Native | Install uv, then `uv tool install tldw-app` and `tldw start`. The launcher installs the matched application, starts it in the background, and opens the browser after authenticated readiness succeeds. |
| Docker | Download and extract a versioned release bundle, then run its `start.sh` or `start.ps1` helper. Docker with Compose is the only runtime prerequisite. The helper pulls prebuilt images, initializes private configuration, starts the stack, and reports the browser URL. |

`tldw-app`, the `tldw` command, and image names are implementation names for this specification, not claims of available packages or reserved registry names. Before publishing installation instructions, verify ownership and substitute any necessary final name consistently.

The existing `tldw-server` API-only distribution remains available. The browser extension and admin-ui are separate products; this project does not package them into the application. Existing developer/source workflows remain supported.

## 2. Core features and dependency boundary — R1

The native core profile provides the full primary WebUI, hosted-provider and existing local-provider chat, server-saved chat history, notes, prompts, plain-text/Markdown/HTML/DOCX/EPUB/text-PDF ingestion, and lexical document search. Document processing that requires OCR clearly offers its optional component. Remote embedding/provider connections are included where they do not require local inference libraries. Local vector inference, reranking, OCR, transcription, speech synthesis, browser-based scraping engines, and GPU runtimes are guided components with their own compatibility records.

Local model-server connections must pass the actual backend connectivity/egress check. Native loopback and Docker host connections have different addresses; the wizard must explain and validate the detected path. It must not weaken outbound restrictions globally to make a provider test pass.

The core runs with SQLite-backed managed storage and without an external Redis or PostgreSQL service. Optional packages must be absent during core acceptance tests. Core imports and startup must not load them or emit misleading startup failures. Unavailable features expose structured capability/setup state without disabling unrelated workflows. Retain mandatory validation, upload checks, authentication, quotas, and rate limits when selecting lighter dependencies.

### Backend packaging choice

Preserve the public `tldw-server` package's dependency contract in this work. CI produces an application-only `tldw-server-runtime` wheel from the same backend sources and package-data inventory, with independently declared core dependencies and supported component extras. Build staging may copy source into a temporary directory; there is no second maintained source tree. The managed runtime wheel is a release asset, not an additional user-facing PyPI installation path.

The runtime wheel and API-only wheel own the same Python imports and must never coexist in one environment. The launcher always uses a dedicated environment and rejects this combination. Metadata must accurately describe installed dependencies; installing a full wheel with unsatisfied requirements and declaring it healthy is prohibited. Runtime and API package builds share source-content checks and explicitly test their distinct dependency contracts.

Resolve and lock transitive dependencies for every supported platform/component combination in CI. Installation uses the released wheels, hashes, and exact dependency lock. No source distribution fallback or compiler installation on the user's machine is permitted for a supported core target. Missing wheels fail the release gate, rather than becoming a customer troubleshooting step. Use uv's dependency tooling rather than implementing a resolver ([uv documentation](https://docs.astral.sh/uv/pip/compile/)).

FFmpeg/ffprobe are acquired with the media/audio component that needs them; text-document core operation must not require them. Their absolute executable paths come from managed component configuration, including on Windows. Dependencies needed by promised core document formats, such as native PDF libraries, must have tested distributable wheels/binaries on every core target.

## 3. Platform and runtime targets — R9

The initial qualification targets are explicit defaults for this written-spec review:

| Distribution | OS qualification baseline | Architecture |
| --- | --- | --- |
| Native | Windows 11, vendor-supported servicing release | x86-64 |
| Native | macOS 14 or later while vendor-supported; test the oldest advertised version and current supported version | Apple Silicon arm64 |
| Native | macOS 14 or later while vendor-supported; test the oldest advertised version and newest supported Intel version | Intel x86-64 |
| Native | Ubuntu 24.04 LTS | x86-64 and arm64, separately qualified |
| Docker | Linux containers on supported Docker/Compose hosts | linux/amd64 and linux/arm64 |

Native Windows ARM, 32-bit systems, musl/Alpine, and other Linux distributions are not initially advertised as qualified. Compatible-but-untested systems may receive a clear unsupported-target message; they must not silently receive a wrong-architecture bundle. Adding a target requires the same tests as the original matrix. A broken required target blocks the full native release unless the user explicitly changes its advertised scope.

Use managed CPython 3.12 and a supported Node 24 LTS patch as the initial runtime families. CI selects exact supported patch versions, checks upstream security/support status, and freezes their versions and hashes in each release. Runtime-family selection is a design target, not certification of the current application. Node's platform matrix includes OS, architecture, and libc requirements; application dependencies may impose stricter requirements ([Node 24 platform documentation](https://github.com/nodejs/node/blob/v24.x/BUILDING.md)).

Release metadata records actual tested OS versions, architecture, libc/ABI constraints, minimum resources, measured installed/download size, and core/component qualification results. Do not invent size or startup-speed claims before measurement. Optional engines advertise their own subset of platforms and hardware; supporting the core on a machine does not promise every engine will run there.

## 4. Native installation and ownership — R7, R8

The PyPI launcher has only control/verification dependencies and never imports the application to perform lifecycle operations. It installs a managed application interpreter separately from its own uv tool environment. Its long-running supervisor lives outside replaceable application release directories.

One supervisor owns the default managed installation for an OS account. It holds an exclusive OS-backed installation/data-root lock, launches and tracks the gateway/backend/WebUI and managed descendants, and records process identities with creation information. A PID or occupied port alone is not evidence of ownership. CLI actions, optional login startup, and native WebUI lifecycle requests reach this same owner.

| Command | Contract |
| --- | --- |
| `tldw start` | Recover an interrupted transaction if necessary; install on first use; reuse a verified existing instance; open the browser only after readiness; run in the background. |
| `tldw stop` | Stop owned services and descendants cleanly; preserve data; report active work when shutdown cannot safely complete. |
| `tldw status` | Report installed/active release, actual service state, pending recovery, and browser URL without depending on the main API. |
| `tldw logs` | Read bounded, rotated managed logs with credential redaction. |
| `tldw update` | Perform the explicit product update transaction in section 8. |
| `tldw backup` | Create a verified managed-server snapshot and report precise browser-copy coverage from section 7. |
| `tldw restore <snapshot>` | Validate coverage/compatibility, preview affected data, obtain explicit confirmation, and restore under the maintenance/recovery contract. |
| `tldw uninstall` | Stop owned services, remove login startup and replaceable application files, and preserve data/configuration/backups/models. |

The supervisor is reachable only through a private authenticated local control channel. Its secret belongs to the OS user and is never sent to the browser. The WebUI submits a bounded operation through its authenticated same-origin session; the gateway validates it and delegates to the supervisor. Requests carry an operation identifier to avoid duplicated update/install actions after retries. Accepted inputs are release/component identifiers from the verified catalog, not shell commands, arbitrary executable paths, or installation URLs. A repeated conflicting operation receives a useful busy/current-operation response.

Closing the terminal or browser does not terminate the native app. Logging out, shutting down the computer, or an OS termination may stop it; recovery is checked at the next start. Login startup is off by default and uses this same entry point when enabled. A crash must not trigger an unlimited restart loop.

First-use instructions include uv installation, PATH setup, and a fallback invocation that works before reopening the shell. Test the published instructions literally on a new account ([uv tool installation](https://docs.astral.sh/uv/guides/tools/#installing-tools)). Launcher upgrades are separate from application-data migration: `uv tool upgrade tldw-app` must not change application data. If a product update requires a newer launcher, report that before maintenance and provide this one-command prerequisite. Ordinary app starts reuse the installed version without a network availability check.

## 5. One stable browser entry point — R2

Ship a small Node gateway beside the unmodified Next.js standalone server. Both use the private Node runtime. The gateway owns the public browser port and selects private upstreams from supervisor-provided runtime configuration. This extra process removes dependency on Next's build-time rewrites and can display a minimal progress/recovery page while application services restart.

Use Node's HTTP server with a declared, release-pinned `http-proxy-middleware` runtime dependency for HTTP and WebSocket forwarding. Do not implement a new proxy protocol stack or depend on Next's private bundled modules. The gateway is a separate proxy, not a custom Next server; Next explicitly distinguishes custom-server deployment from standalone output ([Next documentation](https://nextjs.org/docs/pages/guides/custom-server)). The proxy library documents runtime targets and explicit WebSocket upgrade handling ([proxy documentation](https://github.com/chimurai/http-proxy-middleware)).

Gateway routing order is explicit:

| Path | Destination |
| --- | --- |
| `/_tldw/*` | Minimal managed status/maintenance UI and authenticated control operations. |
| `/api/_tldw-webui/*`, `/api/documentation/*`, `/api/hello`, `/setup`, `/setup/*` | Next API routes and application setup wizard. Runtime session bootstrap uses the same private backend origin as the gateway. |
| `/api/v1`, `/api/v1/*`, `/health`, `/internal/ready`, `/openapi.json`, `/docs`, `/docs/*`, `/docs-static`, `/docs-static/*`, `/static/favicon.ico`, `/redoc` | Managed backend, subject to its existing authorization and setup gates. Backend setup APIs remain under `/api/v1`; its legacy HTML `/setup` does not replace the application wizard. |
| Other paths | Next standalone WebUI and static assets. |

The managed WebUI build uses a dedicated managed mode with relative browser API URLs and no build-time private API rewrites. Configure session bootstrap from runtime-only private configuration. This requires adapting the current mode validator/runtime policy and shared browser-networking code, not merely changing environment variables. The existing quickstart/developer modes retain their supported behavior. Never embed master API keys in public configuration or client bundles.

Default public origin is `http://127.0.0.1:8080`. If unavailable on first installation, offer another available local port and persist the chosen origin before browser data is created. Later starts never silently change the origin. Private backend/Next ports may change; readiness proves that all gateway paths reach the selected services. Reuse an existing process only after validating managed identity. A conflict with an unrelated process results in a recovery action, not termination or adoption of that process.

Native listeners bind to loopback. Docker exposes only the gateway on the host's loopback; the backend and Next ports are private to the Compose network. The gateway rejects unexpected Host/Origin values, discards untrusted forwarding headers, and establishes the trusted-hop contract explicitly. Preserve streaming, cancellation, multipart bodies, redirects, WebSocket upgrades, cookies, and CSRF behavior. Never inject the master API key into arbitrary browser-proxied requests; exchange it through the bounded local session bootstrap path. Use instance-specific session and CSRF cookie names supplied to both backend and client, since cookies are not isolated by port.

Readiness includes process identity/release checks, backend readiness, WebUI assets, and an ephemeral authenticated cookie-session exchange through the public gateway. Readiness sessions are revoked after probing. The browser opens only after these succeed; provider setup is the next user step. Readiness must not download models or call paid model providers.

## 6. Data and configuration layout — R3, R8

Use the existing platform-aware configuration helpers and explicit paths throughout the managed profile. Separate configuration/secrets, durable data, replaceable releases, caches, logs, and the control transaction record. Defaults use the OS user's application-data locations; large caches/models live in appropriate local-data/cache locations rather than roaming configuration. Preserve supported explicit data-location overrides through the same inventory.

All backend databases, jobs/schedulers, uploads, generated assets, component configuration, and keys required for decryption must resolve through the managed storage inventory. Set `DATABASE_URL`, `USER_DB_BASE_DIR`, `JOBS_DB_PATH`, module configuration roots, and other discovered store paths explicitly. Audit the full enabled-core call path before freezing the inventory. New enabled stores must register inventory and snapshot behavior; an unknown authoritative path fails snapshot preflight rather than being omitted.

Do not discover configuration from the caller's current directory or modify installed package files. Reinstall discovers the managed installation registry and reuses data/credentials. Existing developer installations require an explicit import/adoption workflow; no automatic adoption of a nearby `.env` or repository database. Unix permissions and Windows user ACLs protect private files; secrets are not printed, placed in URLs, or exported in logs.

Uninstall stops owned services, removes optional login startup, and removes replaceable application files. It preserves server data, configuration, backups, and downloaded models by default. Deleting data is a separate explicit operation that previews the managed paths. External databases/files and browser profiles are never deleted by launcher uninstall.

Run `tldw uninstall` before optionally removing the launcher with `uv tool uninstall tldw-app`. Removing the uv tool alone does not stop an already running supervisor or remove the separately managed application. Published uninstall instructions must make that distinction clear; reinstalling the launcher must recover the existing installation registry.

## 7. Backup scope and browser compatibility — R3, R4

The existing six-dataset admin bundle is not the implementation of whole-app recovery. Reuse its DB abstractions where appropriate, but drive recovery from the complete managed storage inventory.

| Data class | Recovery treatment |
| --- | --- |
| Authoritative SQLite databases, queued-work state, uploads, generated assets, configuration, encryption keys | Coherent server snapshot after all managed writers are stopped. Include identities, schema versions, checksums, and coverage manifest. |
| Vectors/indexes | Snapshot when authoritative. Exclude only when the manifest proves they are reconstructible from retained content/model/settings and declares the rebuild behavior. |
| Downloaded immutable runtimes/model weights | Record identity/hash and reuse verified objects; avoid duplicating them in each snapshot. User-provided or modified models are authoritative files, not disposable downloads. |
| Logs, temporary files, fully reconstructible caches | Excluded explicitly. Retain bounded logs for diagnosis without treating them as recovery data. |
| Browser IndexedDB/local storage/session drafts | Separate browser export/compatibility contract below; never claim a server snapshot captures them. |
| External databases, workers, and storage | Outside initial automatic recovery. Block an automatic migration that depends on them until a separately supported recovery procedure is selected. |

Check available disk for retained old release, downloads, extraction, candidate environment, snapshots, and a recovery reserve. Refuse maintenance if space is insufficient. Write snapshots into staging, validate the manifest and store integrity, then mark the snapshot complete. Verify restore into disposable roots; keep the last usable recovery snapshot until a later verified one exists. A failed restore preserves original snapshot and failed-state evidence and stops safely.

### Browser policy

Keep the same browser origin and retain browser data during every launcher operation. The WebUI offers a scoped export of authoritative browser records/blobs and useful settings to managed server storage, with browser-profile identity, data schema, timestamp, and coverage. Exclude session/auth tokens. Show whether a backup contains a current browser copy, an older copy, or server data only. A CLI backup or update cannot access closed browsers and must report that limitation accurately.

Automatic updates require browser-store compatibility with the previous release in the rollback window. For the first distribution series, do not increment the IndexedDB database version inside that window; code/record changes must remain readable by both releases. A release that needs an incompatible browser migration is blocked from this automatic update path until a separate tested browser export/migration/restore procedure is delivered. An additive IndexedDB version bump alone is not proof that older Dexie clients can open it.

Before native WebUI-initiated maintenance, flush active drafts and export current browser data. Coordinate active tabs, reject incompatible stale writes, and show a reconnect/reload state after activation. CLI-initiated updates may proceed with browsers closed only under the compatibility rule. Server snapshots do not promise recovery of newer browser-only drafts, browser resets, or another browser profile.

Immediate automatic rollback occurs before the new release is exposed to user traffic, so browser migration/new user work has not begun. Later user-requested restoration previews server changes that would be lost, preserves a snapshot of the current state, checks browser compatibility, and requires explicit confirmation. It does not silently rewind browser data or repeat external job side effects.

## 8. Application update transaction — R5, R7

The independent native supervisor, or the Docker host helper, owns one serialized transaction. Persist phase transitions durably outside replaceable releases. Every phase is restartable or has a defined recovery branch; do not infer success from a renamed directory or last-known PID.

| Phase | Action and failure behavior |
| --- | --- |
| Acquire and preflight | Lock installation, identify current release/data schemas, authenticate target manifest, check launcher/platform/component/browser compatibility and free space. Failure leaves the running application unchanged. |
| Stage | Download/resume verified artifacts and construct the candidate environment/configuration. Validate metadata/imports and perform any trial boot against disposable roots with workers and external effects disabled. No access to live mutable stores or real provider credentials. |
| Enter maintenance | Notify connected clients and allow a bounded draft-flush/browser-export handshake before the write barrier. Then block new mutations/submissions at the gateway, stop new job acquisition in every managed process, and drain existing work. A failed WebUI-initiated export postpones maintenance; unavailable browsers use the declared compatibility/coverage policy. Never use the ordinary zero-second lease wait as proof of safety. |
| Quiesce and snapshot | Confirm every managed writer/descendant has exited. If bounded draining cannot finish, postpone, restore ordinary service, and identify active work. Produce and verify the complete server snapshot. No forced update shortcut that silently discards work. |
| Migrate and verify | Record migration-in-progress, boot/migrate candidate with public ingress and background jobs still gated, then run authenticated and store-integrity checks. Failure restores the paired prior app/configuration/server snapshot and verifies it before serving again. |
| Activate | Record the new active generation, expose it through the gateway, enable workers, and reconnect clients. Retain the prior release/snapshot. This is the boundary after which automatic data rollback is prohibited. |
| Cleanup | Remove only unreferenced incomplete staging and caches within configured retention. Cleanup failure cannot erase the current or required recovery generation. |

Use a published maintenance/drain timeout with an actionable postpone result. Long-running transcription or external workers must not be assumed idle after a timeout. After a crash/reboot during migration, recovery checks the persisted phase and restores/finishes the recorded transaction with ingress blocked. If it cannot verify a safe state, it reports the recovery artifact locations and stops.

The gateway is versioned application code and may itself require replacement. Its supervisor remains available through the private control channel. Allow a brief browser reconnect while transferring the same public origin; replacement gateways start with ingress blocked until the supervisor supplies the verified generation and phase. A gateway restart must not bypass maintenance or reopen the prior release against migrated data.

A health probe is not sufficient proof of job consistency. Preserve queued-work state, reconcile leases after controlled restart, and test that interrupted local operations do not create duplicate committed results. Do not promise exactly-once execution for arbitrary external side effects; candidate probes must never generate those effects.

## 9. Optional processing components — R6

Use a curated catalog of tested component combinations: local transcription/TTS, OCR/document processing, and local embeddings/reranking. A catalog entry supplies the supported platform/hardware, dependency lock, binaries/models, download and installed size, license/setup notices where required, restart behavior, and validation actions. Unsupported combinations explain why before download. GPU driver installation remains an OS/user operation; the app detects requirements and gives a concrete supported path.

Installation runs in an explicitly selected candidate interpreter with private configuration and readiness storage. Reuse existing installer logic only after it accepts these target boundaries; the active server must not execute an unqualified pip/uv upgrade. Downloaded models can be staged as immutable shared objects but are not marked ready until checks pass. Activation of dependencies, configuration, and readiness occurs together under the lifecycle lock, using the same relevant maintenance/recovery steps as an app update.

A failed install leaves active packages, settings, and readiness unchanged and preserves reusable downloads. Removal detaches the component safely; deleting model files is a separate explicit choice. Core workflows remain usable. This is managed application setup, not an arbitrary package/plugin manager.

In Docker, select a prebuilt compatible component image/profile through the host helper; never install packages into a running immutable app container. Model downloads persist in managed storage. The WebUI may configure/inspect components and initiate model acquisition through existing authorized jobs; changes requiring container replacement are performed by the host helper.

## 10. Docker bundle and update boundary — R7

The versioned bundle contains a Compose file pinned to image digests, platform-specific start/update/stop/status scripts, the release manifest/signature, and a concise README. The core stack includes gateway, standalone WebUI, backend, and any explicitly documented bundled service required by that Docker profile. Prefer the same Redis-free core behavior as native; do not inherit unrelated production services by accident. Optional service overlays remain separate.

Run initialization with a pinned one-shot control entry point inside a released image to generate credentials/configuration in persistent managed storage. Host users need neither Python nor crypto/toolchain packages. Initialization is idempotent: it never replaces existing valid credentials or resets data. Use a persisted Compose project identity and bind only the gateway to host loopback. The helper handles browser-open where available and always prints the usable URL.

Updates and backup/restore run from the host scripts, using Docker/Compose and one-shot control containers with precisely scoped volumes. All service containers that can write the snapshot scope must be stopped before snapshot work. Container restart policies must not resurrect those writers during maintenance. Shared/external workers are outside this managed profile.

The first Docker release's WebUI Update action shows the available release and exact host-helper action; it does not directly replace containers. This is the chosen clarification of the review finding. Full Docker updates initiated entirely in the browser require a separately designed restricted host controller. Do not mount the Docker socket into the application to achieve them. Native WebUI updates remain fully delegated to the native supervisor.

Compose `down` and helper stop retain volumes. Destructive reset is a separately confirmed operation that enumerates the named managed volumes. Neither command guesses project identity from the current folder name.

## 11. Release integrity and publishing — R9, R10

Build from a clean tagged commit in CI. Produce the API package, application runtime wheel/dependency artifacts, per-platform WebUI/gateway/runtime bundles, Docker images, and helpers from that release source. Every installable product release has one signed manifest with these required fields:

| Field group | Required contents |
| --- | --- |
| Identity | Manifest schema, product version, source commit, creation time, release channel, signer/key identifier. |
| Compatibility | Minimum launcher, supported platform tuples, exact Python/Node versions, backend and browser compatibility generations, allowed upgrade sources, component combinations. |
| Artifacts | Stable identifier, trusted download location, byte size, extraction/installed size bounds, SHA-256 digest, platform, role; Docker image digest where applicable. |
| Dependencies and data | Dependency-lock digests, required inventory/schema versions, migration/rollback eligibility, component catalog digest. |
| Qualification | Required gate results and evidence links, including fresh install and update/restore tests. |

Use detached Ed25519 signatures over the exact manifest bytes and a trusted public key set shipped with the launcher/control bootstrap. Use a maintained cryptographic implementation, not custom cryptography. Protect the release signing operation in CI and authenticate release provenance. Key rotation requires a previously trusted signature or an explicit launcher/bootstrap update; a downloaded key is not trusted merely because it accompanies a manifest.

For Docker's initial bootstrap, the official downloaded helper contains a fixed control-image digest and verification key identity. The control entry point verifies manifests/artifacts before mutation. Trust in that initial helper is the documented official-distribution bootstrap boundary; subsequent manifests cannot substitute arbitrary control images or signing keys without verification.

Reject unsafe archive paths, escaping symlinks, device files, and extraction/size limit violations. Enforce artifact-content rules excluding development databases, secrets, source caches, and unrelated models. Standalone WebUI bundles include the required public/static assets, gateway dependency closure, and platform-correct traced files. Do not bundle a developer's existing `.next` tree.

Promote a release to the install/update catalog only after every required artifact exists and the matrix passes. A tag, GitHub draft release, or partially published image set is not installable product status. Use immutable digests for installation; a mutable latest tag is only discovery metadata. Retain supported known-good releases for recovery. Explicit downgrades require compatibility checks rather than accepting a lower manifest version automatically.

The launcher is published separately through a protected PyPI Trusted Publisher workflow, first exercised on TestPyPI. Verify package ownership, metadata, entry point, isolated install, and name consistency before directing users to it. Runtime bundles remain release assets; npm is not a separate end-user installation requirement. PyPI's pending-publisher setup does not reserve names ([PyPI documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)).

The existing frontend publication freeze remains a release constraint in `Docs/Development/Packaging_and_Distribution_Strategy.md`. This specification supersedes the older static-PyPI proposal as the technical direction; it does not certify that the separate release gate is satisfied. Local design/build work can proceed. Publishing protected artifacts requires the existing release-specific gate to be recorded as satisfied and the relevant publication authorization. Do not weaken package-content guards to work around that gate.

## 12. Required acceptance evidence

Tests use the downloadable candidate artifacts and published instructions, outside a checkout. Unit tests support these gates but do not substitute for them. Use deterministic mock providers for automated chat/STT/provider connectivity; record any additional real-service/manual qualification separately.

| Gate | Required proof |
| --- | --- |
| G1 Native first install | Every target installs from a fresh OS account with uv and no Git/Node/compiler/Docker; authenticated browser setup, chat, notes, every core document format, and lexical search work. Record exact target and dependency inventory. |
| G2 Docker first install | Fresh Docker/Compose host runs the extracted bundle without host Python/Make or manual credentials; matching images become ready; data survives stop/start and changing shell directory. |
| G3 Dependency boundary | Core tests run with heavy extras and external databases/services absent. No accidental fallback to globally installed tools, cached development dependencies, or repository files. |
| G4 Networking and auth | Reuse a single platform WebUI artifact with different private upstream ports/hosts; test cookies, CSRF, docs API routes, redirects, uploads, streaming cancellation, WebSockets, hostile Host/forwarding input, and session/CSRF isolation. |
| G5 Lifecycle and paths | Repeated simultaneous starts, unrelated port occupants, stale/reused PIDs, child crashes, terminal closure, login startup, spaces/non-ASCII paths, read-only installed packages, and unrelated `.env` files behave predictably. |
| G6 Update failures | Interrupt every transaction phase with process termination/reboot simulation, failed downloads, disk exhaustion, incompatible schemas, and migration/probe failures. Verify paired recovery, preserved data, deterministic next-start behavior, and no candidate external effects. |
| G7 Work draining | Run long jobs and multiple managed writers; update postpones until safely quiescent. Queued state survives, committed local results are not duplicated, and unsupported external workers cause preflight refusal. |
| G8 Restore and browser | Restore the full declared inventory into disposable storage; verify data relationships, assets, keys, jobs, indexes or rebuild evidence. Exercise browser export timestamps, closed browsers, stale tabs, draft flushing, and both directions of the allowed rollback window. |
| G9 Optional components | Install, conflict/fail, retry, activate, remove, and preserve models on each advertised component target. Failed operations leave the active environment/configuration/readiness and core workflows unchanged. |
| G10 Artifact trust | Tampered manifest/artifact, untrusted key, escaping archive entry, missing file/platform binary, incomplete release, and wrong interpreter fail before activation. Exact dependency metadata remains satisfied. |
| G11 Retention and uninstall | Reinstall reuses managed credentials/data; stop/uninstall retain data; explicit deletion previews only owned paths. Offline later starts work when the release service is unavailable. |
| G12 Release policy | Supported runtimes and platform statuses verified; required matrix evidence present; protected publisher configuration and existing frontend release gate satisfied before publication. |

## 13. Delivery decomposition

This is an umbrella product contract. Create separate executable plans for these reviewable work packages; do not turn the entire system into one oversized implementation task.

| Work package | Deliverable and dependencies | Completion evidence |
| --- | --- | --- |
| WP1 Paired Docker distribution and runtime gateway | Release-manifest/artifact layout, gateway + managed WebUI mode, paired prebuilt Docker bundle and idempotent initialization. Local/CI artifacts first; publication remains gated. Establish shared contracts used by later packages. | G2, G4, relevant G10/G12; fresh paired-stack proof. |
| WP2 Lean backend and managed paths | Application runtime wheel from shared sources, core imports/capabilities, complete path inventory, explicit native prerequisites, per-target core locks. Can proceed independently of gateway implementation against the shared contract. | G3, core format coverage, package metadata checks, relevant G5. |
| WP3 Native launcher and supervisor | PyPI launcher, managed runtime acquisition, authenticated single owner, first-start readiness, background lifecycle, status/logs/stop, reinstall and optional login startup. Uses WP1/WP2 artifacts; unsupported update actions remain explicitly unavailable until WP4. | G1/G5/G10/G11 for start/stop/install, with no advertised update promise before WP4. |
| WP4 Safe updates and recovery | Complete snapshots, browser exports/compatibility enforcement, candidate probe mode, maintenance gates, transaction journal, native WebUI updates and Docker host-helper updates. | G6/G7/G8 and update portions of G10/G11. |
| WP5 Guided components and release qualification | Candidate component installs/configuration, curated target matrix, Docker component profiles, exact novice instructions, protected publishers, full release evidence. | G9 and full G1–G12 for the advertised product release. |

Intermediate artifacts are explicitly labeled development/preview and list unavailable operations. The complete native product is advertised only after its approved lifecycle/recovery and guided-component promises are qualified. Docker can ship earlier once its own advertised feature set and applicable gates pass. Each package uses TDD for behavioral changes, scoped lint/tests/Bandit, and review before completion; the implementation plans identify exact files and checks.

## 14. Review correction coverage

| Review item | Design sections | Gates |
| --- | --- | --- |
| R1 Core dependencies and native services | 2, 3, 6 | G1, G3, G9 |
| R2 Runtime routing and browser origin | 5 | G4, G5 |
| R3 Complete coherent snapshots | 6, 7, 8 | G6, G7, G8 |
| R4 Browser data and migration compatibility | 5, 7, 8 | G4, G8 |
| R5 Probe side effects and update quiescence | 8 | G6, G7 |
| R6 Isolated component installation | 9 | G9 |
| R7 Lifecycle owner and Docker authority | 4, 8, 10 | G2, G5, G6 |
| R8 Paths, bootstrap, launcher lifecycle | 4, 6 | G1, G5, G11 |
| R9 Matched artifacts, trust, runtime/platform support | 3, 11 | G1, G2, G10, G12 |
| R10 Publication policy and earlier proposal | 11, policy/ADR cross-references | G12 |

This document specifies required behavior. It does not report completed implementation, certified platforms, successful install/restore trials, reserved package names, or publication. The companion review contains the inspected source locations underlying these corrections.
