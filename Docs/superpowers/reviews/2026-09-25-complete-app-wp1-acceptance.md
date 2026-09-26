# WP1 paired Docker candidate: acceptance record

This records development qualification for TASK-13343 on `codex/complete-app-wp1`.
Protected frontend publication remains frozen. No public image, installer,
catalog entry, PR or merge is approved by this record.

## Current evidence (2026-09-26)

The final code review fix wave and Tasks19/20/21/22 pass scoped spec/quality review. Exact clean candidate `13d9a0d29cffffc282647ac9b72585af1c15eb8a` native CI [36239373229](https://github.com/rmusser01/tldw_server/actions/runs/36239373229) built all four images on both Linux architectures, passed built-backend MCP/setup and all13 signed lifecycle checks. Both candidates pass25 browser checks including normal wizard/privacy progression on both instances, then fail only hostile_inputs_1. Actual same-product cached diagnosis isolates a foreign-Origin cookie WebSocket upgrade reset instead of the required HTTP refusal; Task22 correction9722148d6f and peer-reset safety fixd0afab1daf are implemented; scoped fix review approves with no remaining findings. Owned cleanup passed; signatures are invalidated and all candidate gates remain false. Local dependency installation stalled twice; bounded diagnostics establish intermittent waiting but no cause/fix, and native builds completed. Full corrected local/native qualification remains pending.

| Contract | Current evidence | Status |
| --- | --- | --- |
| Sections5/10: Docker-only startup, stable identity, gateway readiness and private ports | Reviewed fixes and scoped Docker fixtures cover signed identities, gateway cookie auth/revoke, first-port conflict/retry, authoritative persisted inputs and owned cleanup. | Exact13d9 both native signed lifecycle13 passes; hostile upgrade check fails. Windows runtime unqualified. |
| Section11 / G10: required signed artifacts | Both consumers now require all eight signed paths for the selected platform; omitted/missing/wrong-platform/tampered cases pass. | Failed13d9 signatures invalidated; corrected exact signed bytes pending. |
| G2: credential-free managed connection and initial wizard | Implementation/probe tests exist; authenticated installer readiness is now required before browser success. | Exact13d9 both native managed setup/privacy progression passes; full browser gate remains failed. |
| G4: two-instance networking/auth | Full browser checklist remains strict and uses the same managed WebUI artifact against two private target configurations. | Exact13d9 both native stops at hostile_inputs_1; corrected full evidence pending. |
| G12: release policy | Python3.12 and Node24 upstream eligibility rechecked September26; exact patches will be recorded from built images. | False/open: full matrix, Windows runtime, dependency/security/footprints and protected publication remain separate. |

PowerShell helpers are provisional. Windows parsing does not prove real Docker startup, ACL/private-state handling, port recovery, readiness, or stop/status. Docker image Size metadata does not measure download or installed footprint. The 93 untouched whole-frontend typecheck diagnostics remain a disclosed failing baseline.


## Review fixes and scoped verification

Final whole-WP1 review covered `3c871df717..19f2174383`. Its four Important
findings and one Minor finding were handled in a single combined fix wave,
`19f2174383..1d0bcb9b28`, with one scoped rereview. The fixes make persistent
configuration authoritative over inherited Compose inputs, require all eight
signed bundle files, perform authenticated gateway readiness and temporary
session revocation before startup success, preflight the first port before
persisting an origin, and report failed cleanup accurately.

The final focused Release run passed 158 tests; a final established-origin
regression was then added and its complete amended control file passed 15 tests.
This is not a claim of a full 159-test rerun. Black/Ruff/shell syntax passed;
Bandit on all amended production Python reported 0 findings. A real Docker
fixture caught and fixed an inspection classifier error: container records
also carry Driver. The corrected scoped proof passed actual port reservation,
runtime identity/private network, gateway cookie authentication/revocation,
and healthy-fixture broken-auth refusal. Owned resources were removed.
These scoped results support the code review; exact artifact results above
remain the acceptance evidence.

Task19 repaired the actual gateway/internal-route mismatch with two changed files.
Its focused readiness/control run passed 52 tests with 4 disclosed environment warnings;
Black/Ruff passed and production Bandit reported 0 findings. RealHTTP tests require
operator cookie auth and cover503/not_ready, malformed/non-object/missing-status,
oversized/truncated/stalled responses and exact cleanup. Maintained-source cached
proof returns operator200/ready, DELETE200 and revoked profile/operator401, with
owned cleanup; a controlled actual not_ready injection was not exercised.
The scratch report was briefly committed at `c63cb74bf4` and is removed from the
tracked final tree; its code/test/proof facts are retained here and in task history.

Task20 changes one shared request-core condition and adds18 behavioral cases including
the real setup-method request chain. Expected red6fail/8pass becomes focused14pass.
After correcting test-only runtime-cookie-name cleanup, the affected11-file run
reports291pass/10fail (10passing suites); unchanged-production baseline persistence
suite reproduces10fail/6pass. This is not an overall passing run. Scoped ESLint
has0errors/86identical baseline warnings; whole-file Prettier reproduces baseline
2warnings plus printer failure, while newly added blocks pass formatting. These
baseline limits remain disclosed Minor findings. Bandit is inapplicable to this
TypeScript-only change. Fresh exact compiled/browser artifact evidence is still
mandatory; no backend guard or qualification latch was changed.

Task21 changes only two MJS probe/test files. Expected red reproduces disabled Continue; focused6/6 and complete maintained file58/58 pass. Scoped ESLint/diff clean; Bandit inapplicable. Independent review approves spec compliance and quality without findings. Fixtures require acknowledgement and successful ordered setup POSTs, and retain sticky API/network failure after later GET success. Full fresh local/native artifact proof remains mandatory.

The first69e605 dependency install stayed quiet beyond30minutes. Only its verified
owned build was interrupted; cleanup passed and no manifest/signature remained.
A bounded current-source verbose dependency diagnostic passed uncached in85seconds
and its owned image was removed. The unchanged exact retry's ordinary install
passed in86.5seconds, then compiled successfully. No Bun root cause, version or
verbosity fix is claimed; no production dependency/Dockerfile change was made.

Current compiled wizard diagnosis proves the checkbox starts unchecked and
Continue disabled; normal acknowledgement/click reaches provider in66ms with
setup_path/privacy POST200 and catalog200. Metadata remains local with remote
setup disabled, and strict trackers stay clean. DELETE200 then profile401 used
cleared browser cookies, so it is anonymous refusal, not explicit stale-token
replay. All owned containers/network/backend volumes were removed and unrelated
PostgreSQL preserved. Parent lifecycle/browser revocation evidence remains required.

Exact13d9 local builds were interrupted only at verified owned quiet-install
bounds (972s and924s), and cleanup removed the registry without recovery markers
or signatures. One proc-counter diagnostic timed out302s with idle Bun and a
persistent process named node; a separately scoped identity diagnostic completed
90s and established those successful-run children were Bun, with scripts still
unclassified. Neither diagnoses the stalled script or a root cause. Both native
13d9 builds complete, so no Bun version/verbosity/dependency fix is claimed.
Diagnostic images were absent/removed by verified ownership; no pruning occurred.

Both native13d9 candidates pass all13 lifecycle and25 browser checks, then fail
hostile_inputs_1; the full38-check matrix is not completed. One cached69 runtime
stack proves the fifth subrequest resets in1ms, while the first four return
200/403/403/403 without reflection. Product runtime code is unchanged69..13;
the missing unused cached69 control image required local initializer source for
diagnostic state, so this is not Docker-only/signed-helper qualification.
DELETE200 and explicit captured stale-cookie replay401 pass, all owned volumes/
network/containers removed, unrelated PostgreSQL identities/states preserved.
Task22 now supplies a constant403 with absolute1000ms unreferenced cleanup,
retaining the strict probe and authorization rules. Scoped review found a
peer-reset process-crash risk; actual gateway red reproduced ECONNRESET/exit1,
and the minimal pre-end error handler now proves socket destruction and
subsequent status200/process exit0. Final gateway/routes20/20 and scoped lint
pass without warnings. Scoped fix review approves with no new breakage; exact corrected qualification
remains pending;
Bandit is inapplicable to MJS-only edits. Git housekeeping warnings remain
baseline and no manual GC/prune occurred.

## Rulings I made

- Ruling: Correct the design route table to put `/setup` in Next and backend docs assets in FastAPI — this records the user-approved routing correction already committed/tested at 9755c7eaaf/e759322854, rather than reverting to the legacy backend page. Cost if wrong: route ownership and novice setup need rework; API setup paths remain available.
- Ruling: Use a qualification-only Compose override for the second instance's private hostnames/ports after both signed helper starts — the contract requires one WebUI artifact against varying private targets, and this adds no user-facing override API. Cost if wrong: the fixture may need redesign to reflect a supported supervisor configuration; do not advertise it as an installer feature.
- Ruling: Fix Docker setup locality using a bounded authenticated private gateway hop and exact persisted public origin, rather than blanket proxy trust or remote setup enablement — observed Docker NAT makes loopback-only peer checks reject supported managed ingress; the design requires an explicit trusted-hop contract. Cost if wrong: backend setup security boundary and candidate images require rework, so focused hostile-input tests and independent security review are required before qualification.
- Ruling: Set existing backend ALLOWED_ORIGINS to the exact persisted managed gateway origin — actual cookie WebSocket upgrade fails and runtime trusted origins omit it; reuse existing policy with no wildcard. Cost if wrong: managed cross-origin integrations may require explicit separate configuration; ordinary developer/hosted defaults remain unchanged.
- Ruling: Extend the shared authenticated managed-hop contract to MCP local ingress, preserving effective-loopback allow/block checks and configuring exact MCP origins — actual MCP log rejects Docker bridge peer while same-cookie audio WebSocket succeeds; disabling the IP controller or broad private-IP allowlisting would weaken policy. Cost if wrong: MCP proxy/security integration needs rework; independent security review and actual hostile/blocked-input tests gate qualification.
- Ruling: Ship only existing Docs/Published in the WebUI runtime — actual reader fails without it and works with a read-only mount; nonexistent extension-documentation source already returns an empty list in developer mode. Cost if wrong: additional explicit published sources need packaging, but no unrelated design/private docs are exposed.
- Ruling: Evaluate G2 against its explicit Docker first-install criteria (credential-free managed connection, matching ready images and durable lifecycle), keeping provider/full native setup separate — designG2 does not require paid provider configuration, which belongs to later G1/product onboarding. Cost if wrong: G2 evidence must be tightened and additional novice setup checks added; evidence must still name its initial-wizard scope and never claim full provider setup complete.
- Ruling: Refresh obsolete single-builder packaging assertions to validate both actual paired targets and add only the reviewed local profile package to API source allowlist — baseline relativeTask11 does not justify accepting failing wholeWP1 tests. Cost if wrong: semantic guards could hide a lost quickstart/security contract; negative fixtures and independent review are required.
- Ruling: Allow a validated qualification-only local registry port, default5000 inCI and15000 on this host — macOS ControlCenter alreadyowns5000; changing a user service is unnecessary. Cost if wrong: port/ref validation or testtool cleanup needs rework; no new registryhost/publicpush capability is added.
- Ruling: Explicitly precreate the root-owned public trust directory0755 before copying readonly0444 key files — actualBuildKit gives an automatically created destination directory0444 aswell, blocking caller traversal; publictrust mustbe readablewithout changing helperUID or exposingprivatekeys. Cost if wrong: control packaging modes need rework and exact nonroot signature qualification must be repeated; no caller-write permissions or privatekey relaxations allowed.
- Ruling: Validate the existing session cookie at /api and CSRF cookie at / rather than broadening the production session scope to satisfy the probe — real Chromium cookie-only profile succeeds and production setter explicitly scopes session to /api; the probe root lookup and / assertion are wrong. Cost if wrong: browser qualification path/discovery must be reworked and isolation evidence repeated; no production cookie-policy relaxation.
- Ruling: Run bounded installer readiness in the existing pinned control image on the verified private Compose network, checking running signed identities and the actual gateway listener with the persisted public Host/Origin envelope — fulfills approved authenticated-readiness/revocation contract without host Python/Node or Docker socket mounts. Cost if wrong: probe/network identity design must be reworked and exact readiness qualification repeated; real public host binding remains independently required in lifecycle/browser tests.
- Ruling: Confirm and select the first origin with Docker-owned port preflight before initialization, offering an available alternative for occupied defaults and preserving established origins — current init-before-check makes the documented alternate-port retry unusable. Cost if wrong: preflight ownership/port handoff needs rework and first-install conflict tests repeated; never reset existing credentials/data or stop unrelated listeners.
- Ruling: Preserve the backend loopback-only /internal/ready guard and perform gateway backend readiness via existing permission-protected /api/v1/health/ready after fresh cookie bootstrap/profile, before exact revocation and browser success — actual cached artifacts prove anonymous401/cookie200-ready/revoked401; binding section5 specifies components without requiring their order. Cost if wrong: readiness endpoint/permission/result integration must be reworked and exact local/native readiness qualification repeated, with no master-key or internal-route exemption.
- Ruling: Repair the newly reproduced actual-candidate integration defect as narrow Task19 within Task13 qualification, with its own focused TDD/review and fresh artifact proof — the sole whole-branch fix-wave rereview has no residual findings, while mandatory actual qualification exposed a different cross-component contract mismatch. Cost if wrong: task/review scope must be reorganized and qualification repeated; this does not authorize a second broad review, deferred load-bearing acceptance, or publication.

- Ruling: Keep noAuth as explicit key/bearer omission while adding runtime-instance CSRF for unsafe same-origin cookie transport — real unchanged images prove missing-token first-run POST403 and identical expected-token POST200; cookie-bound mutations remain protected even on local setup routes. Cost if wrong: request transport semantics and security boundary tests need rework and exact local/native browser qualification must repeat; no backend CSRF exemption, cross-origin token leakage or qualification latch relaxation is authorized.

- Ruling: Bound recovery from the unusually quiet dependency install by stopping only the verified owned build after more than30 minutes and retrying the unchanged exact candidate once, after current-source verbose dependency proof85seconds — this preserves the production lockfile/Bun/Dockerfile and avoids claiming an unproven workaround fixes the installer. Cost if wrong: a legitimate long install is interrupted and exact artifact qualification must be repeated; never substitute diagnostic image layers, prune unknown resources or weaken acceptance.
- Ruling: Make the browser probe follow the product's normal required privacy acknowledgement before Continue and model that gate plus real setup writes in its fixture — unchanged compiled UI proves the existing product correctly disables Continue until acknowledgement, then reaches provider setup with200 writes. Cost if wrong: probe flow/scope and fixture semantics must be reworked and exact local/native evidence repeated; no forced click, user acknowledgement bypass, product guard change or success-latch relaxation is authorized.

- Ruling: Run the already-authorized own-branch native CI on the same exact reviewed/tested source13d9a0d29c alongside the recurring quiet local dependency install, rather than waiting serially for unproven local recovery — unchanged Dockerfile/lock/manifests plus prior ordinary86.5s success leave environment/timing unresolved; independent native builds add evidence without relaxing local or candidate acceptance. Cost if wrong: CI compute is spent on a source that may need correction and exact native qualification must repeat; no candidate acceptance, publication, PR or merge is authorized by the push.
- Ruling: Bound the recurring current-source quiet install after fifteen minutes of unchanged resolved256/no completion, interrupt only verified own Buildx process, then take one300s instrumented dependency-only diagnostic before considering one unchanged retry — prior quiet>30min versus successful86.5s and unchanged inputs make another serial30min wait uninformative, while own-RUN CPU/I/O/wait categories can distinguish activity from waiting. Cost if wrong: a legitimate slow install is interrupted and local qualification repeated; diagnostic images cannot substitute for product artifacts, no version/dependency/verbosity fix or pruning is authorized.
- Ruling: Stop the repeated ordinary-install path at the current retry bound and, if it remains quiet, measure all-thread executable identity plus strict known-workspace script identity once — exact Bun image node is a symlink to Bun, so processcomm cannot establish nativeNode or the cause; prior probe omitted thread children and deleted raw lifecycle evidence. Cost if wrong: five minutes of isolated diagnostic compute yields insufficient identity and requires a separately scoped investigation; no fourth blind retry, source/dependency/runtime change or diagnostic image acceptance follows from this ruling.
- Ruling: Diagnose the hostile-input runtime failure using verified cached69 backend/WebUI/gateway and unchanged control source to initialize fresh private diagnostic state when the unused cached69 control image is missing — native13 already independently passed all signed-control/lifecycle checks, and69..13 product runtime source is unchanged. Cost if wrong: the runtime reproduction must be repeated with exact current artifacts; local Python initialization supplies no Docker-only/signed-helper qualification, and no mixed runtime image/source acceptance or image pull/build is authorized.

- Ruling: Return a small bounded constant HTTP403 for gateway Host/Origin-denied upgrades, preserving exact authorization and strict probe, rather than accepting arbitrary connection errors as a security pass — actual foreign-Origin rejection resets while ordinary refusal and synthetic fixture express403. Cost if wrong: upgrade refusal framing/cleanup must be reworked and exact local/native security checks repeated; no auth/private-hop/Origin relaxation or reflected credential material is authorized.

## Historical evidence through September25 and superseded follow-ups

The following chronological notes describe earlier source revisions and
failed/superseded candidates. They are retained as history, not current
qualification or availability claims. The current evidence table above is
controlling, including later failures and the final corrected artifacts.

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
