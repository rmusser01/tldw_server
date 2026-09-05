# Software Supply-Chain Release Controls Design

**Status:** Approved
**Date:** 2026-08-30
**Backlog:** TASK-13013.7
**Parent:** TASK-13013
**Baseline:** `origin/dev` at `5921014aa9adfbc0cf32232a3d270bdda18c8150`

## Purpose

Close the public release candidate's dependency and artifact supply-chain gaps
with one fail-closed contract spanning Python, both Bun application roots,
production container images, third-party reference runtime images, SBOMs,
vulnerability evidence, and provenance.

The release rule is:

> No production artifact is promoted while it contains an unexcepted Critical
> or High vulnerability, lacks a validated SBOM, lacks immutable identity, or
> lacks the required provenance and scan evidence.

This design makes that rule enforceable and auditable. It does not claim
bit-for-bit reproducible container builds: operating-system repositories and
some upstream build inputs remain time-varying. It instead guarantees an exact
Python resolution, reviewed Bun locks, immutable base identities, exact
artifact digests, recorded build materials, and scan evidence tied to those
digests.

## Current Problems

The audited baseline has several release-blocking gaps:

- the repository has no committed root `uv.lock`, while production Dockerfiles
  resolve Python dependencies with mutable `pip install` commands;
- the current Dependabot configuration tracks root pip declarations and
  GitHub Actions but does not update either `bun.lock` or Docker base images;
- the SBOM workflow looks only for npm `package-lock.json` files and therefore
  skips both checked-in Bun locks;
- Python SBOM generation installs an unpinned CycloneDX package at run time and
  attempts several incompatible fallback interfaces;
- the CycloneDX CLI container tag is resolved dynamically during the workflow,
  so the reviewed workflow does not identify the tool that actually ran;
- SBOM validation is non-blocking through `continue-on-error: true`;
- no release gate scans the exact container digests that will receive public
  tags;
- the Docker release workflow applies semantic and `latest` tags before any
  vulnerability decision;
- production Dockerfile `FROM` references are mutable tags without digests;
- the production reference deployment permits third-party exact tags without
  requiring digests;
- existing GitHub build attestations cover the published backend images, but
  release evidence does not include a durable digest manifest, complete scan
  reports, per-image SBOMs, or scanner database metadata;
- the WebUI lock resolves Next.js 16.1.4 with `@sentry/nextjs` 9.47.1, whose
  declared Next peer range stops at 15;
- the Admin UI lock resolves Next.js 16.2.2 while its Next-specific lint and
  bundle-analysis packages are on mixed 16.1/16.2 lines.

## Goals

1. Move the WebUI and Admin UI to the approved supported Next.js security
   baseline and align their Next-specific companion packages.
2. Make both Bun locks and a committed universal Python lock first-class
   dependency-update, install, SBOM, and CI inputs.
3. Generate separate, validated CycloneDX SBOMs for Python, the applications
   workspace, the Admin UI, their aggregate source graph, and every candidate
   or reference production image.
4. Require immutable `tag@sha256:digest` identities for every production base
   and reference runtime image.
5. Build release candidates once, scan the resulting digests, and promote
   those same digests only after every gate passes.
6. Block on every Critical or High production vulnerability, including
   unfixed findings, unless a narrowly scoped repository exception is valid.
7. Attach durable, digest-bound SBOM, scan, provenance, and manifest evidence
   to the GitHub release in addition to short-lived Actions artifacts.
8. Preserve trusted publishing and make PyPI's PEP 740 attestations explicit.
9. Keep frontend publication disabled while still building and scanning both
   frontend images as release-candidate evidence.

## Non-Goals

- Publishing the WebUI or Admin UI container images.
- Completing TASK-12116's broad frontend strictness, lint-debt, persisted-store
  migration, or cross-workspace dependency-major alignment.
- Claiming provenance for third-party Caddy, PostgreSQL, Redis, Prometheus,
  Alertmanager, or Grafana images.
- Introducing non-public deployment data or downstream environment details
  into the public repository.
- Delivering multi-platform images. This release remains explicitly
  `linux/amd64`.
- Proving bit-for-bit Docker reproducibility.
- Replacing vulnerability remediation with a permanent allowlist.
- Replacing TASK-13013.3's release freeze, TASK-12983's downstream handoff, or
  TASK-13013.9's capacity and soak certification.

## Security and Release Invariants

The implementation must preserve these invariants:

1. A missing lock, SBOM, scan report, scanner database, exception policy,
   digest, provenance record, or expected artifact is a hard failure.
2. Workflow conditions cannot silently skip a required ecosystem or image.
3. Every third-party Action in the changed supply-chain and release paths is
   pinned to a full commit SHA.
4. Every downloaded binary or tool container is pinned by an immutable digest
   or verified checksum and has its human-readable version recorded.
5. Release scans use the exact `name@sha256:digest` returned by the candidate
   build, never a mutable tag. When that digest identifies an OCI index, the
   scan fixes `--platform linux/amd64` and records the selected child manifest
   digest as well as the index digest.
6. `CRITICAL,HIGH` is the blocking severity set and `ignore-unfixed` is false.
7. The complete scanner JSON is retained; a filtered console table is not
   sufficient evidence.
8. The scanner version, vulnerability database identity, database update
   timestamp, scan timestamp, target digest, platform, policy revision, and
   exception revision are recorded together.
9. A release scan requires a successfully updated vulnerability database no
   older than 24 hours. Network or database failure blocks release.
10. Exceptions are empty by default, package-and-component scoped, approved,
    owned, justified, and expiring.
11. No formal or floating public image tag is applied until all owned images
    and reference runtime images in the release matrix pass.
12. Promotion retags the already scanned digest. It never rebuilds.
13. Pull-request workflows remain read-only and receive no registry-publish or
    release-publish authority.
14. Evidence identifies whether an image is project-built or third-party; the
    release must not imply that third-party provenance was produced here.

## Supported Next.js Baseline

The selected release-candidate baseline is Next.js 16.3.3, the current
security release on the supported 16.x Active LTS line at design time.

Both application manifests and locks move together:

- `apps/tldw-frontend`: Next.js 16.3.3, `@sentry/nextjs` 10.46.0, and
  `@next/eslint-plugin-next` 16.3.3;
- `admin-ui`: Next.js 16.3.3, `@next/bundle-analyzer` 16.3.3,
  `eslint-config-next` 16.3.3, and its transitive Next ESLint plugin on the
  matching line.

Direct framework and Next-specific companion versions are exact manifest
pins. Dependabot, rather than permissive ranges, owns later reviewed movement.
The two applications may retain different React and styling stacks where
their existing contracts require that difference; broad major-version
alignment belongs to TASK-12116.

A lock update is accepted only when all of the following pass from clean,
frozen installs:

- WebUI lint, typecheck, unit tests, production build, bundle/token checks,
  critical Playwright journeys, and the existing required UX gates;
- Admin UI lint, typecheck, unit and accessibility tests, production build,
  Playwright smoke coverage, and existing real-backend authentication
  coverage;
- container builds for both application images;
- vulnerability and SBOM gates described below.

If Next 16.3.3 exposes unrelated pre-existing strictness debt, this task makes
only the minimal compatibility repair necessary for the above security
baseline and records broader cleanup under TASK-12116.

## Deterministic Dependency Resolution

### Python

The root project adopts one committed universal `uv.lock`. Target-specific
requirements snapshots are rejected because they create multiple production
truths and complicate update and SBOM review.

The lock lifecycle is:

1. pin one reviewed uv release for developer commands, CI, SBOM export, and
   production container builds;
2. resolve from `pyproject.toml` into the committed universal lock;
3. verify lock freshness without mutation in CI;
4. install each production profile with `uv sync --locked --no-dev
   --no-editable` and only its explicitly declared extras;
5. fail when the lock would change, a requested profile is absent, or uv emits
   an unsupported lock/export result.

The production profile matrix documents at least:

- application: root default production dependencies;
- worker: root production dependencies plus the existing `multiplayer` extra;
- audio worker: the exact existing audio-worker production dependency profile.

Docker builds copy both `pyproject.toml` and `uv.lock` before installing.
Editable installs, unbounded pip bootstrapping, and dependency resolution
during the production image build are removed. The uv binary itself is copied
from a version-and-digest-pinned upstream image or installed from a
checksum-verified release artifact.

uv's CycloneDX 1.5 export is a preview interface. The repository therefore
pins the uv version, checks the output schema and expected root component, and
fails rather than falling back to another generator.

### Bun

`apps/bun.lock` is the lock for the complete applications workspace, including
`tldw-frontend`, the browser extension, and shared workspace packages.
`admin-ui/bun.lock` remains an independent Admin UI lock.

Every CI and Docker install uses `bun install --frozen-lockfile`. A clean
install must leave each lock byte-for-byte unchanged. A manifest/lock mismatch
is a hard failure.

Dependabot adds:

- `package-ecosystem: bun` at `/apps`;
- `package-ecosystem: bun` at `/admin-ui`;
- `package-ecosystem: uv` at the repository root;
- `package-ecosystem: docker` entries covering the production Dockerfile
  roots.

The existing review labels, ownership, scheduling style, and bounded open-PR
limits are retained. Third-party production Compose references stored outside
formats Dependabot understands use a documented manual digest-refresh
procedure with the same build, SBOM, and scan gates.

## SBOM Architecture

All SBOMs use CycloneDX JSON and stable, unambiguous names:

| Artifact | Required contents | Producer |
| --- | --- | --- |
| `sbom-python-root.cdx.json` | universal locked root Python resolution | pinned uv |
| `sbom-apps-workspace.cdx.json` | `apps/bun.lock` workspace graph | digest-pinned cdxgen |
| `sbom-admin-ui.cdx.json` | `admin-ui/bun.lock` graph | digest-pinned cdxgen |
| `sbom-source-aggregate.cdx.json` | merged three-source graph | digest-pinned CycloneDX CLI |
| `sbom-image-<component>-<digest>.cdx.json` | one exact image digest | pinned image-SBOM tooling |

The applications workspace artifact is deliberately named `apps-workspace`,
not `webui`, because the lock also owns the extension and shared packages.
This prevents an incomplete component label from being mistaken for a
WebUI-only inventory.

cdxgen is selected because it reads Bun's text lock format and workspace
layout. It runs at an exact reviewed version from an immutable container
digest, without `latest`, `npx -y`, or a run-time tag-to-digest lookup.

The CycloneDX CLI used for merge and validation is likewise committed as a
readable version plus image digest. Every component SBOM and the aggregate are
validated. Validation has no `continue-on-error` path.

Generation additionally asserts:

- each expected file exists and is nonempty;
- `bomFormat`, spec version, serial number, metadata component, and components
  are present and valid;
- the Python root component and both Bun roots are recognizable;
- package counts are nonzero and within a reviewed sanity range;
- the aggregate contains all three expected source roots;
- no SBOM embeds credentials, raw environment values, or registry tokens.

Source SBOMs describe locked source resolution. Image SBOMs describe actual
artifact contents. They are retained separately rather than treating the
source aggregate as proof of what an image contains.

## Immutable Image Identity

### Project-built production images

The build-and-scan matrix contains:

1. API application from `Dockerfiles/Dockerfile.prod`;
2. worker from `Dockerfiles/Dockerfile.worker`;
3. audio worker from `Dockerfiles/Dockerfile.audio_gpu_worker`;
4. WebUI from `Dockerfiles/Dockerfile.webui`;
5. Admin UI from `Dockerfiles/Dockerfile.admin-ui`.

Every `FROM` in those Dockerfiles uses a readable tag followed by
`@sha256:<manifest-list-digest>` where the upstream publishes a manifest list.
Multi-stage files pin every stage reference, even when stages currently share
the same base.

The build platform is explicitly `linux/amd64`. The digest manifest and
provenance record that platform. The workflow must not claim a multi-platform
release or certify architectures it did not build and scan. BuildKit may wrap
the single-platform image and its attestations in an OCI index; evidence must
distinguish the promoted index/subject digest from the scanned
`linux/amd64` child manifest digest.

The WebUI and Admin UI are built, SBOMed, and scanned but are not pushed under
public release tags. Their candidate evidence proves the repository can
produce candidate artifacts without enabling frontend publication.

### Third-party reference runtime images

The production reference matrix also contains:

- Caddy;
- PostgreSQL;
- Redis;
- Prometheus;
- Alertmanager;
- Grafana.

Production configuration, examples, preflight, and documentation require each
operator value to match `<readable-tag>@sha256:<digest>`. An exact tag without
a digest is no longer accepted.

The release workflow pulls, inventories, and scans the exact configured
digests. Evidence labels them as `third-party-reference` and records their
upstream registry, index digest where applicable, selected `linux/amd64`
manifest digest, and platform. Project provenance is not generated or claimed
for those images. Project-built maximum-level provenance records its own base
materials, including upstream base digests.

## Vulnerability Policy and Exceptions

### Blocking policy

Pinned Trivy is the common enforcement scanner for source SBOMs and exact
image digests. The workflow records complete JSON and may also generate a
human-readable summary.

The release decision blocks when:

- any production component has an unexcepted Critical or High finding;
- a finding has no fix yet;
- the target cannot be scanned;
- the vulnerability database cannot be updated or identified;
- a required SBOM or package identity cannot be parsed;
- the exception document is missing, invalid, expired, or overbroad.

Lower severities remain visible in the complete report and release summary but
do not block TASK-13013.7.

### Repository exception contract

The canonical policy file starts with an empty exception list. Each later
entry requires:

- a unique stable exception ID;
- vulnerability ID;
- exact component class from the documented source/image matrix;
- exact package PURL and affected installed version or bounded version range;
- severity;
- concise technical rationale;
- remediation or compensating-control note;
- accountable owner;
- public approval reference to a repository issue or pull request;
- creation date;
- expiry date.

Critical exceptions expire no later than 7 days after creation. High
exceptions expire no later than 30 days after creation. Shorter upstream or
organizational limits win. Renewals require a new approval reference and
revised rationale; editing only the expiry is invalid.

A small standard-library validator checks schema, enumerations, dates,
durations, PURLs, duplicate IDs, exact component scope, and approval-reference
shape. It emits an ephemeral Trivy ignore policy containing only fields Trivy
understands, including vulnerability, PURL, expiry, and statement. The
ephemeral file is an output, never the reviewed source of truth.

After scanning, a policy evaluator proves that:

- every ignored finding matches exactly one unexpired repository entry;
- every entry matched at least one finding in its declared component;
- no exception suppresses a different package, component, severity, or
  vulnerability;
- the complete unsuppressed and policy-adjusted result sets are both retained.

Stale unmatched exceptions fail the release gate so the policy file cannot
accumulate dead suppressions.

## Source and Pull-Request Gate

The source supply-chain workflow runs on relevant pull requests and protected
branches with read-only permissions. It:

1. verifies pinned tool and Action identities;
2. checks `uv.lock` freshness and all frozen dependency installs;
3. generates and validates the three source SBOMs and aggregate;
4. scans the source SBOMs and preserves complete reports;
5. validates the exception policy;
6. runs framework regression suites when Next or related lock entries change;
7. checks production Dockerfile `FROM` references and production reference
   image examples for required digests;
8. uploads named evidence artifacts with `if-no-files-found: error`.

Required outputs are asserted in a final unconditional gate. A skipped matrix
entry, cancelled producer, missing artifact, or soft-failed validation causes
the gate to fail.

The source gate may report development-only findings separately. The formal
release decision remains based on the packages present in production image
SBOMs, ensuring that a development tool is not mislabeled as shipped while
still keeping its risk visible.

## Candidate Build, Scan, and Promotion

Formal release images follow a staged pipeline:

```text
trusted release ref
        |
        v
build five linux/amd64 candidates once
        |
        v
capture exact digests + max provenance + OCI SBOM
        |
        v
SBOM and scan all five candidate digests
        |
        +---- pull/SBOM/scan six third-party reference digests
        |
        v
validate exceptions and zero Critical/High policy
        |
        v
promote the same app/worker/audio digests
        |
        v
verify tags, attach evidence, publish release
```

Each build receives a run-unique candidate tag only so it can be pushed and
addressed. The workflow immediately captures the registry digest and all
subsequent operations use `name@digest`.

The three publishable images are promoted only in a single downstream job
that depends on every candidate and reference-image gate. Promotion order is:

1. apply immutable full-version tags to all three exact digests;
2. verify every full-version tag resolves to its expected digest;
3. apply major/minor and `latest` floating tags where release policy requires;
4. verify every floating tag;
5. write the final manifest and publish/complete the GitHub release.

No floating tag changes until every immutable full-version tag is verified.
Promotion is idempotent: retrying must resolve the same tag and digest rather
than rebuild. A partial immutable-tag failure leaves only intended immutable
aliases and blocks floating tags and release publication. The operator repairs
or retries the promotion job; automation does not delete registry data.

The existing post-publication `release: published` Docker trigger is not a
safe admission point. The implementation replaces it with a controlled
trusted-ref flow that prepares a draft release and publishes it only after
promotion and evidence upload succeed. Manual runs require an explicit
confirmation input and verify the supplied tag resolves to the intended
protected commit.

## Provenance and Attestation

Project-built images use BuildKit:

- `provenance: mode=max`;
- OCI SBOM attestation enabled;
- exact source commit, Dockerfile, platform, build arguments that are safe to
  disclose, and base material digests;
- existing GitHub artifact build-provenance attestation for each digest.

Secret values must enter only through supported secret mounts and must not
appear in build arguments, image labels, SBOM metadata, provenance parameters,
logs, or release assets.

The PyPI workflow retains OIDC trusted publishing through
`pypa/gh-action-pypi-publish` and sets `attestations: true` explicitly. That
action's PEP 740 attestations are the canonical PyPI provenance; this task does
not add a duplicate attestor. The workflow verifies built distribution hashes,
uses the exact validated distribution artifact for upload, and documents how
consumers inspect the PyPI attestation.

## Durable Release Evidence

Actions artifacts remain useful for pull requests, but formal release evidence
must also be attached to the GitHub release. The release asset set includes:

- three source component SBOMs and the aggregate source SBOM;
- a CycloneDX SBOM for each of the five project-built candidates;
- a CycloneDX SBOM for each of the six third-party reference images;
- complete Trivy JSON for every source component and image;
- policy-adjusted scan decisions without deleting the original results;
- scanner version and database metadata;
- the validated exception policy snapshot;
- a machine-readable release digest manifest;
- SHA-256 checksums for every attached evidence file;
- references needed to verify OCI SBOM and provenance attestations.

The digest manifest has a versioned schema and records:

- repository and source commit;
- release tag/version and workflow run;
- `linux/amd64` platform;
- policy and exception file hashes;
- tool versions and immutable tool identities;
- for each project image: logical name, Dockerfile, candidate reference,
  promoted subject/index digest, selected `linux/amd64` manifest digest,
  promoted tags or `build-and-scan-only` status, base materials, SBOM/report
  filenames and hashes, and attestation references;
- for each third-party image: logical name, upstream `tag@digest`,
  selected `linux/amd64` manifest digest, `third-party-reference` ownership,
  SBOM/report filenames and hashes;
- scanner database identity/update time and scan times;
- final pass/fail decision.

Release publication verifies that every manifest file exists, matches its
recorded hash, and refers to the same target digest. The manifest must not list
an artifact that was rebuilt after scanning.

## Documentation and Operator Contract

The deployment and release documentation will:

- replace exact-tag-or-digest language with mandatory `tag@digest` language;
- provide digest-pinned examples without credentials;
- explain upstream digest refresh, scan, review, and rollback;
- distinguish project provenance from third-party digest evidence;
- document the Critical/High policy and time-bounded exception workflow;
- show how to verify GitHub image attestations, OCI SBOMs, release checksums,
  and PyPI PEP 740 attestations;
- state that the certified image platform is `linux/amd64`;
- state that frontend images are build-and-scan-only;
- avoid any non-public downstream or infrastructure data.

TASK-13013.6's production preflight gains the `@sha256:` requirement but keeps
ownership of deployment topology, secret validation, backup, rollback, and
health behavior.

## Failure Handling and Recovery

- Tool download, checksum, schema, SBOM, database, or registry errors fail
  closed and retain available diagnostics.
- Scanner output is treated as untrusted input and parsed defensively with
  bounded file sizes and explicit schemas.
- A vulnerability database outage does not reuse unidentified or stale cache
  data for a release.
- Candidate images may remain under run-unique tags after failure for
  investigation; they are never presented as released.
- A failed frontend candidate blocks the whole core release gate even though
  the frontend image is not published, because the accepted repository must
  remain buildable at its approved security baseline.
- A failed third-party reference image blocks release of the reference
  deployment set.
- A scan-policy failure is repaired by dependency/base remediation or a valid
  approved exception, followed by a fresh build and scan.
- A scan is never reused for a rebuilt digest.
- Promotion retries are digest-idempotent. If immutable aliases were partially
  created, a retry verifies them before proceeding; floating aliases remain
  unchanged until the complete set passes.
- An already published artifact is never silently overwritten as remediation.
  A corrected release receives a new version or RC identifier.

## Trust Boundaries

- Pull-request code is untrusted and runs without package, attestation, or
  release write tokens.
- Candidate build and promotion run only from a reviewed protected repository
  ref in the trusted workflow context.
- GitHub OIDC is used only by the jobs that need attestation or trusted
  publishing.
- Registry credentials are scoped to candidate push and promotion jobs.
- Scanner and SBOM tools are supply-chain inputs in their own right and are
  immutable and version-recorded.
- Upstream vulnerability data is time-varying; the evidence therefore records
  the database used instead of claiming that a past clean scan proves future
  safety.
- Repository exception approval is visible and reviewable. Workflow inputs
  cannot create an ad hoc suppression.

## Expected Implementation Scope

The implementation plan will confirm exact paths, but the intended scope
includes:

- root `uv.lock` and Python production Docker build changes;
- both Next.js manifests and Bun locks;
- Dependabot ecosystem entries;
- a fail-closed source SBOM workflow;
- supply-chain policy, exception schema/validator, and focused tests;
- digest pinning in the five production Dockerfiles;
- production reference preflight, examples, and documentation updates;
- candidate build, exact-digest scan, staged promotion, evidence manifest, and
  release-asset workflow changes;
- explicit PyPI attestations and verification documentation;
- regression, policy, workflow, Dockerfile, Compose, and documentation
  contract tests.

No non-public deployment repository, downstream configuration, or
infrastructure file is part of this scope.

## Verification Strategy

Implementation follows test-driven development. Required verification covers:

### Locks and installs

- `uv lock --check` or the pinned equivalent from a clean checkout;
- each Python production sync profile with `--locked --no-dev --no-editable`;
- frozen Bun installs for `apps` and `admin-ui` with no lock diff;
- Dependabot configuration validation for uv, Bun, Docker, and Actions.

### Framework regressions

- WebUI lint, typecheck, unit, production build, required UX, and critical E2E
  suites;
- Admin UI lint, typecheck, unit/accessibility, production build, smoke, and
  real-backend auth suites;
- both frontend Docker builds.

### SBOM and policy

- deterministic fixture tests for all three source producers;
- missing/empty/malformed/wrong-root SBOM rejection;
- aggregate membership and validation;
- exception schema, duplicate, scope, expiry, maximum-duration, renewal, and
  unmatched-entry rejection;
- Critical/High fixed and unfixed blocking cases;
- lower-severity reporting without blocking;
- exact package/component suppression and cross-component suppression denial;
- secret-pattern scan over SBOM, provenance, reports, and manifest fixtures.

### Images and release workflow

- every production `FROM` and reference image uses `tag@sha256`;
- preflight rejects tags without digests and malformed digests;
- build outputs include a valid exact digest for `linux/amd64`;
- image SBOM, scan target, provenance subject, promoted tags, subject/index
  digest, and selected `linux/amd64` manifest digest agree;
- missing scanner DB metadata or data older than 24 hours fails;
- promotion is unreachable when any owned or third-party matrix item fails;
- full-version tags precede floating tags;
- frontend images never receive release tags;
- release assets and checksums are complete before publication;
- actionlint and focused workflow policy tests pass.

### Security and repository quality

- focused Bandit over new or changed Python policy tooling;
- shell/static analysis for changed release helpers;
- `git diff --check` and repository secret scanning;
- no `continue-on-error` or conditional-skip escape hatch on required gates;
- no reference to non-public repositories, downstream environments, or
  infrastructure identifiers.

Live network scans and registry promotion require trusted CI. Local tests use
fixtures and mocked metadata without pretending they constitute release
evidence.

## Primary References

- [Next.js support policy](https://nextjs.org/support-policy)
- [Next.js security release announcements](https://nextjs.org/blog)
- [GitHub Dependabot ecosystems](https://docs.github.com/en/enterprise-cloud@latest/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories)
- [uv project locks and sync](https://docs.astral.sh/uv/guides/projects/)
- [uv CycloneDX export](https://docs.astral.sh/uv/concepts/projects/export/)
- [Docker BuildKit attestations](https://docs.docker.com/build/ci/github-actions/attestations/)
- [GitHub artifact attestations](https://docs.github.com/en/actions/concepts/security/artifact-attestations)
- [Trivy filtering and ignore policy](https://trivy.dev/docs/dev/docs/configuration/filtering/)
- [PyPI trusted-publishing action and attestations](https://github.com/pypa/gh-action-pypi-publish)

## Acceptance Criteria Mapping

### AC1: supported Next.js applications and regressions

Satisfied by the exact Next.js 16.3.3 baseline, aligned companions, frozen
locks, complete WebUI/Admin regression matrix, and build-and-scan gates.

### AC2: Bun updates/SBOMs and reproducible Python resolution

Satisfied by Dependabot Bun and uv ownership, both frozen Bun locks, the
committed universal `uv.lock`, locked non-editable production sync profiles,
and separate validated source SBOMs.

### AC3: immutable provenance, vulnerability evidence, and exceptions

Satisfied by digest-pinned bases and reference images, build-once
scan-before-promotion, exact-digest SBOMs and Trivy evidence, maximum-level
project provenance, durable release assets, and the validated expiring
exception contract.

## Delivery Boundary

TASK-13013.7 is one reviewable supply-chain hardening PR. It can make the
minimal frontend compatibility changes required by the security baseline but
does not absorb TASK-12116. It provides evidence consumed by TASK-13013.3 and
TASK-12983, while deployment behavior remains with TASK-13013.6 and
capacity/soak evidence remains with TASK-13013.9.

Implementation does not begin until this written specification is approved and
the task-specific implementation plan is reviewed.
