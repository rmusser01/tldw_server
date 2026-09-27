# TASK-13376.12: WebUI Docker browser-download fix

Approved design: scope `PUPPETEER_SKIP_DOWNLOAD=true` to the shared Docker
dependency install RUN. Keep Bun 1.3.2, the frozen lockfile, other lifecycle
scripts, local developer installs and Playwright acceptance unchanged.
TASK-13376.11 captured Puppeteer as the pending child; its internal wait is
unproven. Removing an unused browser download must pass actual build and
candidate verification before claiming a repair.

## Stage 1: Test and implement the approved command boundary
**Goal**: The Docker install skips Puppeteer browser downloads only.
**Success Criteria**: A failing then passing command-behavior test verifies
the child environment and frozen install arguments; existing contracts pass.
**Tests**: Execute the dependency RUN with an inert Bun fixture, including an
inherited false flag; focused Docker hardening and same-origin suites.
**Status**: Complete

Modify `tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py` first.
Replace its install source-string assertion with a real shell execution against
an inert Bun executable that reports the child flag and arguments. Removal of
the flag or replacing the frozen install with script suppression must fail.
Then apply the approved change to `Dockerfiles/Dockerfile.webui`.

## Stage 2: Validate and review the scoped implementation
**Goal**: Commit a minimal, reviewed fix suitable for exact-source builds.
**Success Criteria**: Focused tests, formatting, Bandit and independent review
pass, with versions/lockfiles untouched.
**Tests**: Project-venv pytest and Bandit on the touched Python test; formatter
check on changed scope; `git diff --check`; read-only code review.
**Status**: Complete

Record red/green evidence, review findings and source scope in TASK-13376.12.
Commit Dockerfile, tests and tracking records together, without bypassing hooks.
Use the existing isolated checkout; preserve unrelated main-checkout changes.

## Stage 3: Build and qualify the committed candidate
**Goal**: Verify the production dependency build and paired candidate.
**Success Criteria**: Exact clean source builds with the ordinary production
Dockerfile, then signed candidate lifecycle/browser evidence is verified, or
any bounded failure is documented without changing acceptance requirements.
**Tests**: Existing `qualify_app_bundle_candidate.sh` on local native arm64,
manifest/source/file verification and current native CI evidence.
**Status**: In Progress

Use a new private candidate directory and private output log. Record baseline
Docker resources, owned process IDs, source SHA and runtime inputs. Preserve all
previous signing/recovery evidence and data. Retain the reviewed 15-minute
no-progress bound for an owned Buildx process; do not kill shared processes.
No publication, promotion or merge. G12 and complete-product gates remain false
unless independently satisfied. New runtime/package/build-strategy decisions
outside the approved flag require review. Remove only this plan after all stages
are complete, preserving its content privately/history and the broader plan.
