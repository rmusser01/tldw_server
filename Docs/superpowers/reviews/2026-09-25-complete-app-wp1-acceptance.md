# WP1 paired Docker candidate: acceptance record (2026-09-25)

This is a development acceptance record for TASK-13343, not a product-release
approval. The worktree branch is `codex/complete-app-wp1`. The existing
frontend publication freeze remains in force; no image, installer, or catalog
entry has been published.

| Contract | Current evidence | Status |
| --- | --- | --- |
| Section 5: one browser gateway, same-origin managed WebUI, private backend/Next ports | Gateway real-socket tests cover routing, Host/Origin rejection, forwarded-header stripping, multipart body forwarding, redirects, cookies, SSE cancellation, and WebSocket upgrade. Run 36210634045 passes real-container setup/docs routing, named session cookies, cookie-only profile authentication, hostile Host rejection, and private-port checks on both native platforms. | One-instance extracted-container proof passes; two-instance browser/network qualification remains open. |
| Section 10: Docker-only extracted first start, stable identity, stop/status, retained data | Control tests cover signature-before-init, mode-0600 credentials, repeat identity, conflicting release, and tampering. Fake-Docker helper tests cover ordering, unavailable daemon/Compose, occupied port, and stop/status from another directory. Both native jobs in run 36210634045 passed first/repeat/stop/restart and persistence with the corrected configuration. | Container lifecycle proof passes; interactive browser setup and Windows helper runtime qualification remain open. |
| Section 11: signed manifest and immutable artifacts | Ed25519 exact-byte manifest verification, per-file SHA-256, platform selection, path/symlink checks, fixed control-image/key bootstrap, and candidate-gate tests pass. Both corrected native candidates pass extracted initialization/tampering; independently downloaded signatures and all eight helper hashes verify for each platform. | Per-platform provisional artifacts verified; one qualified multi-platform manifest is not produced. |
| G2 Docker first install | Both corrected native signed bundles ran outside the checkout, reached healthy services, served managed setup, authenticated the profile with cookies only, and retained credentials/data across stop/start from another directory. | Remains false in provisional evidence until its planned interactive browser setup qualification is implemented. |
| G4 networking/auth | Focused gateway, backend AuthNZ, Next runtime, and browser-networking tests pass; real standalone Next session exchange through the gateway passed before image work. | Open: exact two-instance browser/auth and real-container upload/stream/WebSocket cases are not yet qualified. |
| G10 artifact trust | Manifest/control/candidate test suites pass, including altered signature, wrong platform, unsafe path, changed file, and missing image. Both corrected native jobs pass image-content guards, extracted verification, and refusal of a tampered manifest before instance initialization. Downloaded signatures and helper bytes independently verify. | True in both corrected per-platform provisional records; full product qualification is not implied. |
| G12 release policy | Both native jobs recorded Python 3.12.14 and Node 24.21.0. Their upstream support status was checked on September 25. The candidate code records exact patches and refuses unsupported families/mismatches. | Open: full dependency/security review, download/installed-footprint measurements, joined matrix qualification, and protected publication remain outstanding. |

Local verification after the cookie follow-up: 49 lean Release tests and three
focused AuthNZ integration tests pass;
Black checks, shell syntax, Compose config for amd64 and arm64 settings,
gateway dependency lock installation, `git diff --check`, and scoped Bandit
(zero findings) pass. Prior slices recorded 64 managed-WebUI Vitest tests,
32 AuthNZ unit tests, 6 focused HTTP/logout tests, 13 gateway socket tests,
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

The [auth-mode follow-up run](https://github.com/rmusser01/tldw_server/actions/runs/36208320371)
at `dbc1100323` passes the complete arm64 extracted smoke: assets, session
cookies, hostile Host rejection, private ports, persisted credentials/data
across stop/start from another directory, and manifest tampering. Its inventory
records Python 3.12.14 and Node 24.21.0; all eight downloaded bundle file hashes
match manifest `1b7201d4730e230ef59661878e532b5ff87e3ed48a9b39500354ecc054e806db`.
G10 is true in its arm64 evidence; G2/G4/G12 remain false by the provisional
lane's policy. The promotion verifier rejects it as required. The runner's
Docker `.Size` values are 8,515,093,284 backend, 534,186,910 WebUI, 249,104,975
gateway, and 164,181,011 control bytes; these metadata values are not a measured
download/installation footprint and are not compared with Docker Desktop's
earlier values. The amd64 job also passed, and the required-both status passed.
Its manifest hash is `15cbe30fd80b8a3805d4ca7d71cf8c26f0129edeb39a1dbca83e86ef1c6b40ef`;
all eight downloaded file hashes match. The setup/docs source fix is
committed separately at `9755c7eaaf` and is not covered by this candidate. Future
evidence uploads include the ephemeral public verification key, allowing the
downloaded provisional manifest signature to be checked independently; the
private signing key remains excluded.

Subsequent cookie review found that the successful run's generic Set-Cookie
assertion could accept a CSRF cookie without an authenticated session. Compose
passed the instance session-cookie name to Next but omitted it from the backend;
Next therefore filtered out the backend's default-named session cookie. The
backend also defaulted to HTTPS-only cookies despite this bundle's HTTP loopback
endpoint. Two regression tests failed before the correction. Compose now shares
the instance name and explicitly disables Secure cookies for its loopback HTTP
gateway. The smoke requires both named cookies and uses the cookie jar to fetch
the authenticated profile without an API-key header. Run 36209873850 was cancelled
to avoid qualifying the obsolete configuration. All 49 lean release tests and
three existing AuthNZ integration tests pass, as do Black, shell syntax, and
Compose validation. Test-file Bandit reports only assertions and existing
subprocess harness warnings; production Release code still has no findings.
The earlier run is not evidence that the browser session authenticated; the
corrected run below provides that HTTP authentication proof.

The [corrected native run](https://github.com/rmusser01/tldw_server/actions/runs/36210634045)
at `e759322854d3547e239bc97c9dfe1b232b5f6da3` completed successfully, including
both native candidates, Windows parsing, and the required-both status job.
The strengthened extracted smoke on each platform checks the managed setup
page, real backend docs assets, both configured cookie names, and authenticated profile access
with only the cookie jar, followed by Host/private-port, restart persistence,
and manifest-tamper checks. Windows helper parsing also passes. Each Linux job
passes 49 Release tests, 207 frontend tests, 13 gateway tests, scoped lint/format
checks, and production Bandit with no findings. An independent focused code
review of `dbc1100323..e759322854` found no actionable issues and independently
reran the 13 gateway tests.

Downloaded public-key signature verification and all eight helper hashes pass
for each candidate, and both source revisions match the exact run commit.
Both inventories record Python 3.12.14 and Node 24.21.0. The signed provisional
G10 flag is true; G2/G4/G12 remain false, and both promotion checks reject them.

| Platform | Verified manifest SHA-256 | First / repeat Compose health wait |
| --- | --- | --- |
| linux/arm64 | `924401d1667ced52d519ead6e112e0b80a42cb786d5c10d3ccf62b51e2f2da61` | 121.75 s / 91.74 s |
| linux/amd64 | `d95a6bb908abf3ee9cf5bd8baf189808a965981604c052f2a6338166672b0d29` | 122.57 s / 92.19 s |

These timings run from the backend container's Starting event to the gateway's
Healthy event in the CI log. They include healthcheck polling and exclude image
downloads, control/helper initialization, and browser interaction. They are not
end-to-end first-install durations.

| Platform | Backend Docker `.Size` | WebUI | Gateway | Control |
| --- | ---: | ---: | ---: | ---: |
| linux/arm64 | 8,515,099,267 | 534,186,958 | 249,105,023 | 164,181,011 |
| linux/amd64 | 8,803,264,108 | 515,154,934 | 230,428,225 | 139,435,839 |

Sizes are Docker image metadata bytes, not measured network download or
installed storage footprints. The job-local image registries have expired;
uploaded bundles are review evidence and cannot install those images elsewhere.

The manual `verify-app-bundle.yml` lane builds separate job-local candidates
for linux/amd64 and linux/arm64 and leaves G2/G4/G12 false. Its required-both
status job confirms both provisional runs were exercised; it does not
promote them. The final signed public bundle and release-catalog boundary
remain WP5 work after WP1's applicable gates pass. Native runtime slimming,
full storage inventory, native supervision, update/restore, and optional
component management remain the explicit WP2–WP5 boundaries in the design.

TASK-13343 stays **In Progress**. The corrected native smoke now passes on both
platforms. To close it, add the missing interactive browser/two-instance and
full runtime/security-policy evidence, produce a single qualified
multi-platform signed candidate, and record its immutable manifest plus
measured sizes and startup times. Public promotion remains a separate
authorized release action.
