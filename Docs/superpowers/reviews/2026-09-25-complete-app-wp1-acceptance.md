# WP1 paired Docker candidate: acceptance record (2026-09-25)

This is a development acceptance record for TASK-13343, not a product-release
approval. The worktree branch is `codex/complete-app-wp1`. The existing
frontend publication freeze remains in force; no image, installer, or catalog
entry has been published.

| Contract | Current evidence | Status |
| --- | --- | --- |
| Section 5: one browser gateway, same-origin managed WebUI, private backend/Next ports | Gateway real-socket tests cover routing, Host/Origin rejection, forwarded-header stripping, multipart body forwarding, redirects, cookies, SSE cancellation, and WebSocket upgrade. Managed standalone Next and runtime session tests cover the private hop. Compose publishes only `127.0.0.1:${TLDW_PUBLIC_PORT}:8080`; `docker compose config` parsed on both platform settings. | Implemented locally; exact extracted-container and two-instance browser proof pending. |
| Section 10: Docker-only extracted first start, stable identity, stop/status, retained data | Control tests cover signature-before-init, mode-0600 credentials, repeat identity, conflicting release, and tampering. Fake-Docker helper tests cover ordering, unavailable daemon/Compose, occupied port, and stop/status from another directory. The backend DB and generated config use named volumes. | Live G2 first/repeat/stop/restart and browser setup pending. |
| Section 11: signed manifest and immutable artifacts | Ed25519 exact-byte manifest verification, per-file SHA-256, platform selection, path/symlink checks, fixed control-image/key bootstrap, and candidate-gate tests pass. Candidate CI uses a job-local registry and checks image revision labels, captured digests, runtime versions, and selected image contents. | Candidate build/smoke not yet run; one published multi-platform manifest is not produced. |
| G2 Docker first install | Source tests and manual CI smoke script exist. | Open: no extracted bundle has run on this host or CI, and the guided browser setup has not been exercised. |
| G4 networking/auth | Focused gateway, backend AuthNZ, Next runtime, and browser-networking tests pass; real standalone Next session exchange through the gateway passed before image work. | Open: exact two-instance browser/auth and real-container upload/stream/WebSocket cases are not yet qualified. |
| G10 artifact trust | Manifest/control/candidate test suites pass, including altered signature, wrong platform, unsafe path, changed file, and missing image. | Partially evidenced; runtime image-content guards and exact extracted tamper smoke need Docker. |
| G12 release policy | Docker targets use Python 3.12 and Node 24 families; candidate code records exact runtime patches and refuses unsupported families/mismatches. | Open: actual patches, upstream support/security status, both platform results, and protected publication gate are unverified. |

Local verification before this WP1 review: 46 lean Release tests pass;
Black checks, shell syntax, Compose config for amd64 and arm64 settings,
gateway dependency lock installation, `git diff --check`, and scoped Bandit
(zero findings) pass. Prior slices recorded 64 managed-WebUI Vitest tests,
32 AuthNZ unit tests, 6 focused HTTP/logout tests, 12 gateway socket tests,
and the standalone Next through-gateway session test. Whole-frontend
typechecking still reports 93 diagnostics in untouched files; that baseline
is not counted as a passing gate.

Local Docker execution is blocked: the installed Docker Desktop socket returns
“Docker Desktop is unable to start” after CLI and direct-app start attempts.
`pwsh` is absent locally, so only shell helpers ran here; the manual CI lane
parses PowerShell syntax on Windows. No actual image size, download size,
startup duration, or platform patch measurement is available yet. The start
helpers now wait for Compose health before reporting/opening the URL and stop
partial services after failed readiness while retaining the instance state.
Do not infer or advertise values from Dockerfile tags.

The manual `verify-app-bundle.yml` lane builds separate job-local candidates
for linux/amd64 and linux/arm64 and leaves G2/G4/G12 false. Its required-both
status job confirms both provisional runs were exercised; it does not
promote them. The final signed public bundle and release-catalog boundary
remain WP5 work after WP1's applicable gates pass. Native runtime slimming,
full storage inventory, native supervision, update/restore, and optional
component management remain the explicit WP2–WP5 boundaries in the design.

TASK-13343 stays **In Progress**. To close it, run the manual candidate lane,
repair failures on both platforms, add the missing browser/two-instance and
runtime-support evidence, produce a single qualified multi-platform signed
candidate, and record its immutable manifest plus measured sizes and startup
times. Public promotion remains a separate authorized release action.
