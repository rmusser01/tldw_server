# Native frontend candidates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development to implement this plan task-by-task. The user already selected continuous subagent-driven execution.

**Goal:** Qualify vendor-patched frontend runtime candidates on native amd64 without changing production admission.

**Architecture:** Generate temporary recipes from canonical frontend recipes, preserving their builders and runtime contract. Build baseline and candidate exact OCI artifacts, exercise real standalone applications, then compare complete pinned-scanner evidence. Candidate execution and release admission remain separate.

**Tech Stack:** Python standard library, pytest, Docker/Buildx, Node, GitHub Actions, pinned Trivy/Syft/Grype.

**Spec:** `Docs/Design/TASK-13013.7.15-native-frontend-candidates.md`

## Global Constraints

- Work only in the existing `codex/task-13013-7-supply-chain-design` worktree; TASK-13013.7.15 tracks every change.
- No production Dockerfile, release workflow, dependency lock, application-source, exception or custom-ignore edits.
- No FFmpeg source/history work or another task's artifacts. No registry publication or merge.
- Preserve canonical builders byte-for-byte, Node 24.20.0, Bun 1.3.2 and frozen lock installs.
- Candidate runtime is Ubuntu24.04 index `sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517`, zlib1g `1:1.3.dfsg-3.1ubuntu2.2`, libc6 `2.39-0ubuntu8.8`.
- WebUI UID10002/port3000/workdir `/app/apps/tldw-frontend`; Admin UID10003/port3001/workdir `/app/admin-ui`; preserve healthcheck 30s/5s/5 retries.
- Native means host Linux/x86_64 and Docker server linux/amd64 or x86_64, with no QEMU setup. Emulation cannot satisfy qualification.
- Preserve complete findings across severities; a lowered rating or missing match is not a fix. CVE-2026-85091 remains unresolved.
- Activate `/private/tmp/task-13013-7-pyjwt-verification-venv` for local Python/pytest/Bandit; do not mutate shared environments.

## Stage 1: Candidate construction

**Goal:** Fail-closed candidate generation with canonical recipe preservation.
**Success Criteria:** Both canonical inputs render; altered/ambiguous markers fail; tests prove preservation and fixed inputs.
**Tests:** Focused renderer tests and existing Supply_Chain suite.
**Status:** Complete —1f23962f09, independent spec/quality review approved;23 focused and610 Supply_Chain tests pass.

### Task 1: Candidate recipe renderer

**Files:**
- Create `Dockerfiles/candidates/frontend/render.py`.
- Create `Dockerfiles/candidates/frontend/README.md`.
- Create `tldw_Server_API/tests/Supply_Chain/test_frontend_candidate_render.py`.

**Interfaces:**
- `render_candidate(source: str) -> str`: deterministic transformation or ValueError for an unexpected source contract.
- CLI `python Dockerfiles/candidates/frontend/render.py --application {webui,admin-ui} --output PATH`: reads canonical source from this repo, writes candidate only, exits nonzero on rejected input.
- Output preserves builder prefix and runtime remainder exactly around the one known runtime FROM; candidate-only injected block owns the replacement base, copies and backport acquisition.

- [ ] Read the two canonical Dockerfiles, their official Node runtime assumptions and the spec's fixed inputs.
- [ ] Write tests first, using importlib to load the new module as existing Supply_Chain tests do. Include this preservation assertion for each canonical source:

```python
marker = 'FROM node:24.20.0-bookworm-slim@sha256:ba849c60be29959425b8734d57b8b4b7d56f98edd9504c9af091d5281095a71e AS runtime\n'
prefix, remainder = source.split(marker)
output = render_candidate(source)
assert output.startswith(prefix)
assert output.endswith(remainder)
```

- [ ] Test missing/duplicated runtime marker, changed runtime Node version/digest and missing/renamed builder rejection; test real CLI writes only the requested temporary output and leaves canonical hashes unchanged.
- [ ] Run the tests before implementation and retain the expected failure.
- [ ] Implement only a strict known-marker transformation and the explicit CLI, no general Dockerfile parser or runtime migration framework. The core transformation is:

```python
if source.count(EXPECTED_RUNTIME_FROM) != 1:
    raise ValueError('expected exactly one canonical runtime stage')
prefix, remainder = source.split(EXPECTED_RUNTIME_FROM)
if prefix.count(EXPECTED_BUILDER_FROM) != 1:
    raise ValueError('expected canonical Node builder stage')
return prefix + CANDIDATE_RUNTIME_BLOCK + remainder
```

- [ ] The injected runtime uses the spec's Ubuntu pin. Copy `/usr/local/bin/node`, `/usr/local/bin/docker-entrypoint.sh`, and `/usr/local/LICENSE` from builder to identical paths; explicitly restore `ENTRYPOINT ["docker-entrypoint.sh"]`. Verify source paths from the known official Node recipe/artifact, failing visibly if unavailable rather than omitting evidence. The baseline slim recipe purges ca-certificates: verify system-bundle absence and preserve the embedded Node roots via the exact executable rather than adding a missing-path CA COPY. Native controls compare the actual trust-store contract.
- [ ] Install only exact zlib1g through normal signed apt indexes. Retain package download SHA256, signed InRelease identities, installed zlib/libc/libstdc++ versions under `/usr/local/share/tldw-candidate-evidence`. Do not use insecure apt options, upstream1.3.2 or a snapshot claim. Keep application COPY/USER/ENV/WORKDIR/HEALTHCHECK/CMD remainder untouched.
- [ ] Test the injected artifact pins, exact package version, absence of insecure apt flags and preserved application health/CMD/USER text. Document removed non-application package-manager tooling if different from the baseline; do not claim a bit-identical OS.
- [ ] Run focused pytest, Ruff/Black and Bandit on changed Python scope. Run existing Supply_Chain suite once before commit, classify only existing test assertion findings, and commit only this task's files with TASK-13013.7.15 in the message.

## Stage 2: Native application compatibility

**Goal:** Exercise both exact standalone application artifacts with baseline controls.
**Success Criteria:** Correct native identity, non-root application behavior, real GNU sharp transform, configured healthcheck and bounded shutdown.
**Tests:** Probe unit tests for parsing/validation/failure paths, real native job controls.
**Status:** In Progress

### Task 2: Native runtime controls and workflow

**Files:**
- Create `Dockerfiles/candidates/frontend/qualify.py`.
- Create `.github/workflows/frontend-runtime-candidate.yml`.
- Create `tldw_Server_API/tests/Supply_Chain/test_frontend_candidate_qualification.py`.
- Update candidate README with exact local/CI usage and limits.

**Interfaces:**
- CLI `python Dockerfiles/candidates/frontend/qualify.py --application {webui,admin-ui} --baseline SUBJECT --candidate SUBJECT --evidence PATH`.
- Both subjects are locally loaded exact OCI subject SHA256 identities, not mutable tags.
- Produce `qualification.json` with schemaVersion1, application, source/input identities, native host/daemon facts, baseline/candidate observations, passed boolean and explicit scope `native-frontend-compatibility-not-release-admission`.
- Produce raw image/container inspections, command output, version/ownership/sharp/control observations and exit statuses under evidence, including on failure.

- [ ] Write failure-first tests for non-native host/daemon, malformed subject, wrong image identity/platform, wrong Node/UID/GID/workdir, sharp failure, health mismatch and SIGTERM timeout. Use fixed argv subprocess seams for unit tests; mock outcomes are not native qualification evidence.
- [ ] Implement subprocess calls without shell interpolation, bounded timeouts, exact-image validation and retained failures. Reject ambiguous identities; reuse `Helper_Scripts/Supply_Chain/runtime_probe.py config` in workflow to bind OCI layout/config before loading, without changing that helper.
- [ ] Probe each baseline and candidate with real Node controls, including this sharp operation executed inside the image from its default workdir:

```javascript
const assert = require('node:assert/strict');
const sharp = require('sharp');
(async () => {
const input = await sharp({create:{width:2,height:2,channels:3,background:'#123456'}}).png().toBuffer();
const output = await sharp(input).resize(3,3).png().toBuffer();
const metadata = await sharp(output).metadata();
assert.equal(metadata.width,3);
assert.equal(metadata.height,3);
})().catch(error => { console.error(error); process.exitCode = 1; });
```

- [ ] Compare baseline/candidate application configuration, Node version/ABI, UID/GID/home, `tls.rootCertificates` count/hash, optional system CA bundle presence/hash and real sharp results. Application configuration means Cmd/Entrypoint/User/WorkingDir/Healthcheck/ports and ENV keys declared in the canonical runtime remainder; removed inherited Yarn metadata is not an application regression. Validate candidate installed vendor versions separately with dpkg; do not compare glibc version equality across OSes.
- [ ] Start each standalone app privately with a controlled backend container named `backend` on its own task-owned internal Docker network. Backend uses the exact candidate Node image with an overridden Node command serving only `/api/v1/health` on8000. Do not publish host ports or use real credentials.
- [ ] Build Admin with `NEXT_PUBLIC_API_URL=http://backend:8000`, ordinary standalone mode; build WebUI with quickstart and `TLDW_INTERNAL_API_ORIGIN=http://backend:8000`. Keep other canonical arguments as existing container-build-check.
- [ ] Verify WebUI root200/configured healthcheck. Verify Admin `/api/health`200 and `/api/health/ready`200 with backend healthy; stop backend, then require ready503 while liveness stays200 and configured healthcheck fails. Record the controlled-stub limit, not real-backend certification.
- [ ] Use read-only rootfs, dropped capabilities, no-new-privileges, finite PID/memory/CPU limits, non-root image USER, private network, writable tmpfs only for /tmp and application .next/cache owned by expected UID/GID. Retain inspections proving those settings. Clean only exact containers/networks created by this invocation.
- [ ] Require SIGTERM completion within10seconds without SIGKILL (exit0 or143 acceptable); retain elapsed time, process exit and logs. Fail rather than silently continuing after failed controls.
- [ ] Workflow: pinned checkout, setup-docker with containerd snapshotter, buildx and build-push/upload actions matching existing repository pins; native ubuntu-24.04 runner; only contents:read permission; matrixwebui/admin-ui; timeout90minutes; manual dispatch and scoped push trigger on this branch/new candidate paths. No QEMU, registry login, publication or required-check changes.
- [ ] Generate temporary recipe using Task1 CLI; build baseline and candidate with identical source/args to local OCI archives, max provenance/SBOM, no push. Verify/extract/load exact subjects; execute qualifier; always upload evidence and archives with14-day retention, retaining source commit and build metadata.
- [ ] Run focused tests, actionlint, touched-scope Bandit and existing Supply_Chain tests; commit only task files. Document any unavailable native evidence rather than setting passed=true locally.

## Stage 3: Exact-artifact vulnerability comparison

**Goal:** Retain honest native before/after security evidence and run the pipeline.
**Success Criteria:** Same fresh DB per baseline/candidate, complete scan outputs and known fix/status checks; no admission or suppression changes.
**Tests:** Workflow contract tests, comparison validation tests, actual native CI evidence.
**Status:** Not Started

### Task 3: Frozen-database scans and native execution

**Files:**
- Update `.github/workflows/frontend-runtime-candidate.yml`.
- Create `Dockerfiles/candidates/frontend/compare.py` and `tldw_Server_API/tests/Supply_Chain/test_frontend_candidate_comparison.py`.
- Update candidate README and Backlog task evidence links.

**Interfaces:**
- CLI `python Dockerfiles/candidates/frontend/compare.py --baseline-trivy PATH --candidate-trivy PATH --baseline-grype PATH --candidate-grype PATH --baseline-config SHA256 --candidate-config SHA256 --output PATH`.
- Config digests come from the already-validated OCI layouts. Each Trivy ImageID and Grype source imageID must match its corresponding expected config; baseline and candidate are intentionally different subjects.
- Emit complete package/CVE comparisons across all severity levels, original/new matches and explicit `admitted:false`; no vulnerability waiver or fixed verdict from absence alone.

- [ ] Write tests first for below-threshold persistence, package rename/missing-match ambiguity, introduced matches, malformed/missing reports, source identity mismatch and rejection of unproven fixed claims. Example assertion:

```python
assert comparison['admitted'] is False
assert comparison['matches'][0]['classification'] == 'still-reported-severity-changed'
```

- [ ] Implement a compact deterministic report normalizer, keeping Trivy and Grype comparisons separate; match package/CVE keys and retain full versions/severities. Do not create policy overrides or mutate raw reports.
- [ ] Add pinned Trivy/Syft/Grype references from spec. Download each scanner DB once; require valid freshness≤24h (+5min skew), record metadata/checksums, scan both exact OCI subjects/layouts without refresh, then confirm hashes unchanged. Preserve complete Syft binary/package inventories and unfiltered findings.
- [ ] Retain scanner image inspections and executed command transcripts. Validate config/layout/archive binding for scans; distinguish a Docker-archive reader's synthetic manifest digest from the OCI subject if using that format.
- [ ] Require Node24.20.0 inventory and candidate package versions, report glibc fixes via vendor evidence and zlib27171 removal, retain zlib85091 at whichever vendor severity it has. Candidate security findings are evidence, not release admission; malformed or missing evidence still fails.
- [ ] Run comparison and workflow tests, actionlint, touched-scope Bandit and full Supply_Chain suite once. Have fresh task review and final scoped-branch review; fix confirmed problems.
- [ ] Push only this owned PR branch under the existing user authorization and run/observe the native workflow. Address in-scope native failures; after three failed attempts at one issue stop that approach, document cause and choose an evidence-backed alternative. Do not merge or change production recipes on candidate success.
- [ ] Retain run/artifact links and results in Backlog; leave unsuccessful acceptance criteria open. Native compatibility is complete only from actual native run evidence, not local emulation or mock tests.
