# WP1 paired Docker candidate: acceptance record (2026-09-25)

This is a development acceptance record for TASK-13343, not a product-release
approval. The worktree branch is `codex/complete-app-wp1`. The existing
frontend publication freeze remains in force; no image, installer, or catalog
entry has been published.

| Contract | Current evidence | Status |
| --- | --- | --- |
| Section 5: one browser gateway, same-origin managed WebUI, private backend/Next ports | Gateway real-socket tests cover routing, Host/Origin rejection, forwarded-header stripping, multipart body forwarding, redirects, cookies, SSE cancellation, and WebSocket upgrade. Managed standalone Next and runtime session tests cover the private hop. Compose publishes only `127.0.0.1:${TLDW_PUBLIC_PORT}:8080`; `docker compose config` parsed on both platform settings. | Implemented locally; exact extracted-container and two-instance browser proof pending. |
| Section 10: Docker-only extracted first start, stable identity, stop/status, retained data | Control tests cover signature-before-init, mode-0600 credentials, repeat identity, conflicting release, and tampering. Fake-Docker helper tests cover ordering, unavailable daemon/Compose, occupied port, and stop/status from another directory. The backend DB and generated config use named volumes. | Live G2 first/repeat/stop/restart and browser setup pending. |
| Section 11: signed manifest and immutable artifacts | Ed25519 exact-byte manifest verification, per-file SHA-256, platform selection, path/symlink checks, fixed control-image/key bootstrap, and candidate-gate tests pass. Native amd64 and arm64 CI built all four images; signed extracted-bundle verification and initialization passed. | Both CI smoke checks stopped after service readiness; one qualified multi-platform manifest is not produced. |
| G2 Docker first install | Native amd64 and arm64 CI signed bundles ran outside the checkout and reached healthy backend, WebUI, and gateway services. | Open: an HTTP smoke request failed before repeat start and guided setup. A missing WebUI auth-mode setting is fixed and awaiting the corrected CI smoke. |
| G4 networking/auth | Focused gateway, backend AuthNZ, Next runtime, and browser-networking tests pass; real standalone Next session exchange through the gateway passed before image work. | Open: exact two-instance browser/auth and real-container upload/stream/WebSocket cases are not yet qualified. |
| G10 artifact trust | Manifest/control/candidate test suites pass, including altered signature, wrong platform, unsafe path, changed file, and missing image. Local image-content guards and extracted control verification passed. | Partially evidenced; the extracted tamper check was after the failed readiness step and has not run. |
| G12 release policy | The local arm64 images reported Python 3.12.14 and Node 24.21.0. The candidate code records exact patches and refuses unsupported families/mismatches. | Open: upstream support/security status, both-platform results, download size, and protected publication gate are unverified. |

Local verification after the CI follow-up: 47 lean Release tests pass;
Black checks, shell syntax, Compose config for amd64 and arm64 settings,
gateway dependency lock installation, `git diff --check`, and scoped Bandit
(zero findings) pass. Prior slices recorded 64 managed-WebUI Vitest tests,
32 AuthNZ unit tests, 6 focused HTTP/logout tests, 12 gateway socket tests,
and the standalone Next through-gateway session test. Whole-frontend
typechecking still reports 93 diagnostics in untouched files; that baseline
is not counted as a passing gate.

Docker Desktop recovered on September 25. A local arm64 candidate
built all four images and passed runtime-version/content guards. Its signed
bundle verified and initialized from a temporary directory, but Compose
stopped when the backend became unhealthy. A separate fresh backend container
reproduced `ModuleNotFoundError: No module named 'tldw_profile_core'`. The
backend Dockerfile omitted the local package expected by root `pyproject.toml`.
After copying the package into the builder, the corrected image imported it,
contained 2 schemas and 45 fixtures, and reached `/internal/ready` health in
30.29 seconds with new isolated volumes. The candidate script now checks that
import before packaging. This corrected image has **not** passed the complete
signed-bundle smoke; its isolated health result does not close G2 or G4.

The local arm64 inventory reported Docker `image inspect .Size` values of
4,302,324,090 bytes for backend, 153,274,684 for WebUI, 80,833,945 for
gateway, and 50,144,556 for control. The corrected backend reported
4,302,321,900 bytes. These are local image metadata values, not download or
installed-footprint measurements. No end-to-end setup duration, arm64 browser
result or PowerShell runtime result is available. `pwsh` is
absent locally; CI PowerShell parsing passes. Docker build data reduced host
free space to about 15 GiB, so a third four-image rebuild was not attempted.
The start helpers wait for Compose health before reporting/opening the URL and
stop partial services after failed readiness while retaining instance state.

The user authorized branch push and native CI on September 25. Manual
dispatch was unavailable because the new workflow was absent from the default
branch, so a push trigger restricted to this branch and workflow file bootstraps
the lane. The first run passed 46 release tests but stopped on formatter drift;
Black is now pinned to the locally verified 25.1.0 version. The
[second run](https://github.com/rmusser01/tldw_server/actions/runs/36206663211)
at `754c9dd521` passed the focused Python/browser/gateway checks and Windows
helper parsing, built all four images on both native platforms, verified and
initialized the signed bundles, and reached healthy services. Each platform
then exited 22 on a silent HTTP smoke request. The WebUI service omitted
`AUTH_MODE=single_user`, which its session policy requires. A failing Compose
regression and a real standalone Next through-gateway reproduction confirm
503 with the setting omitted versus 204 with two cookies when supplied. The
configuration fix passes 47 release tests and awaits the next container run.
HTTP smoke errors now report their line/status, and failed CI runs retain the
provisional bundle/inventory/evidence without the private signing key.

The production release Python scope still has zero Bandit findings. The helper
test file reports its existing subprocess warnings and test assertions; the
new finding is the regression assertion, not a production security change.
The arm64 runner reported 104 GiB free before build and completed all image
builds, so the standard runner disk concern did not block this attempt. Local
Docker image inspection now reports a content-store I/O error and the host has
about 4.5 GiB free; further local container builds are held.

Further routing review found that `/setup` was sent to the backend's legacy
credential setup page instead of Next's application wizard, while backend
`/docs-static` and `/static/favicon.ico` were sent to Next. A real-socket
regression failed on the setup collision. The corrected route table passes 13
gateway tests and lint; the existing standalone Next build serves `/setup`
through the gateway with HTTP 200 and Next static-asset references. The smoke
now checks that setup response and real backend documentation assets. This
follow-up still needs its own exact container candidate run.

Upstream runtime support was checked on September 25. Python's
[version status](https://devguide.python.org/versions/) keeps 3.12 in security
support until October 2028, and
[3.12.14](https://www.python.org/downloads/release/python-31214/) is the current
3.12 security release. Node's
[release schedule](https://github.com/nodejs/Release) keeps 24 in LTS until
April 2028; [24.21.0](https://nodejs.org/en/blog/release/v24.21.0) is its current
LTS patch, newer than the 24.18.1 fixes in the latest listed
[security release](https://nodejs.org/en/blog/vulnerability/july-2026-security-releases).
The [24.21.0 platform table](https://github.com/nodejs/node/blob/v24.21.0/BUILDING.md)
lists GNU/Linux x64 and arm64 as Tier 1 with kernel >=4.18 and glibc >=2.28.
This establishes upstream eligibility of the observed runtime families/patches;
it does not replace image dependency/security review or the complete G12 gate.

The manual `verify-app-bundle.yml` lane builds separate job-local candidates
for linux/amd64 and linux/arm64 and leaves G2/G4/G12 false. Its required-both
status job confirms both provisional runs were exercised; it does not
promote them. The final signed public bundle and release-catalog boundary
remain WP5 work after WP1's applicable gates pass. Native runtime slimming,
full storage inventory, native supervision, update/restore, and optional
component management remain the explicit WP2–WP5 boundaries in the design.

TASK-13343 stays **In Progress**. To close it, rebuild the corrected exact
candidate and run the manual candidate lane on both platforms, add the missing
browser/two-instance and runtime-support evidence, produce a single qualified
multi-platform signed candidate, and record its immutable manifest plus
measured sizes and startup times. Public promotion remains a separate
authorized release action.
