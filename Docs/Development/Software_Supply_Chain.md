# Software Supply-Chain Operations

This is the operator contract for dependency locks, software bills of
materials (SBOMs), vulnerability admission, immutable container identity,
release evidence, and provenance verification. The release workflows enforce
the same contract. Do not publish around a failed gate.

The release invariant is zero unexcepted Critical or High vulnerabilities in
the admitted source and image evidence. Scans include fixed and unfixed
findings (`ignore-unfixed=false`). A missing or stale database, malformed SBOM,
invalid exception, digest mismatch, failed tool, or incomplete artifact is a
release failure.

Implementation is not release certification: current source and image findings
still block admission. The committed image inventory records exact candidates,
not an approval to deploy them. Certification requires fresh passing evidence
or narrowly scoped, human-approved exceptions. As of September 10, 2026 UTC,
the canonical policy contains 287 native-package applicability exceptions, sixteen
Chroma exceptions for the supported embedded deployment, and four NLTK current-use
exceptions, owned by `rmusser01`
and valid through September 17 UTC. Exact scope and authorization are recorded
in the [initial OS activation](../Evidence/TASK-13013.7.24-os-exception-activation.md),
[Chroma disposition](../Evidence/TASK-13013.7.28-chroma-exclusions.md), and
[binary-package applicability review](../Evidence/TASK-13013.7.29-deployment-applicability.md), and
[native build and usage review](../Evidence/TASK-13013.7.30-native-applicability.md).
The TASK30 records distinguish absent affected code/features (153), current
librist dependency usage despite shipped affected code (10), current first-party
Perl usage despite shipped core code (10), and Expat's compiled parser entropy
path despite an unused helper remaining in the library (2).
The [native consumer review](../Evidence/TASK-13013.7.34-native-consumers.md)
adds 35 scoped GLib/Expat, mount/nsenter, gzip, infocmp and frontend ACL records.
Backend privileged mount and ACL prerequisites remain unresolved and gated.
The [NLTK usage disposition](../Evidence/TASK-13013.7.32-nltk-applicability.md)
covers the reviewed text/corpus callers and all direct locked consumers; it
does not claim a patched package or completed model-path qualification.
All other scanner matches remain subject to the admission gate. An unexcepted
scanner match is not proof of exploitable impact in the supported deployment.

## Pinned supply-chain tools

The workflow file `.github/workflows/sbom.yml` is the canonical source for
these immutable tool references. Update a version and digest together, review
the upstream release, and rerun the complete gate.

| Tool | Reviewed identity |
| --- | --- |
| uv 0.12.7 | `ghcr.io/astral-sh/uv:0.12.7@sha256:95f2aa1fe59274951cfe9b0cbc7972e879ff1004bc8945d130a32eb0dbd85945` |
| cdxgen 13.0.1 | `ghcr.io/cdxgen/cdxgen:v13@sha256:0be75639a833b59d1ba29b3c8ac00dfd2e41e7568d56b6c039007caadebebc0d` |
| CycloneDX CLI 0.33.1 | `docker.io/cyclonedx/cyclonedx-cli:0.33.1@sha256:252c2e26f468c25fea1e63ecde1bc3198ad6e9dbb57f5ed3236bddcb2281b3a7` |
| Trivy 0.74.0 | `ghcr.io/aquasecurity/trivy:0.74.0@sha256:62b1e65e8869bc4b4c6aa4fa2b21595256c7c2f6018a9d9ad61caf87187c1969` |

Bun 1.3.2 is the reviewed lockfile implementation for both Bun workspaces.
GitHub Actions and Docker base-image references are also pinned by commit or
digest and are covered by Dependabot.

## Verify dependency locks

Run from a clean repository checkout. Use the uv binary from the pinned image
above when the workstation's uv version is different.

```bash
uv --version
uv lock --check
uv sync --locked --no-dev --no-editable

(cd apps && bun --version && bun install --frozen-lockfile)
(cd admin-ui && bun --version && bun install --frozen-lockfile)

git diff --exit-code -- uv.lock apps/bun.lock admin-ui/bun.lock
```

Production Python profiles use the same universal `uv.lock` with explicit
extras. The app and audio worker use the default profile; the worker adds
`multiplayer`. A production Docker build
must use `uv sync --locked --no-dev --no-editable`; mutable `pip install`
resolution is not an accepted release input.

For an intentional dependency update, change the direct constraint or
Dependabot-owned input, regenerate the relevant lock with the reviewed tool,
review the lock diff, and rerun frozen installs, framework tests, builds, SBOMs,
and scans. Never hand-edit a resolved package entry.

## WebUI build-host requirements

The WebUI Docker build uses real Node 24 and the canonical `npm run build:prod`
Turbopack pipeline; Bun installs the frozen workspace dependencies. Provision
a 16 GiB-class build host or Docker VM. The 6 GiB Node heap setting does not
cap total build memory, including Turbopack's native allocations. The 8 GiB-class
Docker VM failed this build with memory exhaustion and is not a supported
build target. This is a build-time requirement, not a runtime memory requirement.

The canonical build command passed on a 15.61 GiB CI runner with the unchanged
600 KB shared and 900 KB route gzip budgets. The updated Docker image still
requires its own exact-artifact build and scan in CI; command-level validation
does not certify the image. Do not increase the budgets to accommodate a
different compiler pipeline.

## Generate and admit source SBOMs

Dispatch the canonical gate for the exact source commit:

```bash
SOURCE_REF=refs/heads/dev
gh workflow run sbom.yml --ref "$SOURCE_REF"
gh run list --workflow sbom.yml --branch "${SOURCE_REF#refs/heads/}" --limit 1
```

The workflow produces schema-validated CycloneDX JSON and checksums with these
stable names:

- `sbom-python-root.cdx.json`
- `sbom-apps-workspace.cdx.json`
- `sbom-admin-ui.cdx.json`
- `sbom-source-aggregate.cdx.json`

Python uses CycloneDX 1.5; the Bun workspace producers use CycloneDX 1.6.

It then produces complete Trivy JSON, policy-adjusted decisions,
`SHA256SUMS.source`, `SHA256SUMS.scan`, and scanner metadata. The applications
workspace SBOM explicitly contains the WebUI, browser extension, and shared UI
workspace roots. The Admin UI remains a separate lock and SBOM.

The dependency inventory is generated from the canonical parent Bun lock.
Child package identities are derived from their checked-in package manifests
because those directories do not own individual Bun locks. These identity
records do not replace or relax the required-only locked dependency inventory.
Verify this boundary from an activated project environment with Docker available:

```bash
TLDW_TEST_SBOM_DOCKER=1 python -m pytest -q \
  tldw_Server_API/tests/Supply_Chain/test_bun_package_metadata_integration.py
```

The Trivy vulnerability database must have a valid schema and an `UpdatedAt`
no more than 24 hours old (with five minutes of clock-skew tolerance). The
workflow downloads it once, records the scanner and database metadata, and
uses it without an implicit update for every decision. Never reuse a report
with a different database, policy revision, source commit, or target digest.

## Image set and identity

The target platform for certification is only `linux/amd64`; this release does not claim
multi-platform coverage. Each image is named by a readable tag and immutable
subject digest in `tag@sha256:` form. A tag alone, `latest@sha256`, shortened
digest, or uppercase digest is rejected by production preflight.

Project-built subjects:

| Name | Dockerfile | Publication |
| --- | --- | --- |
| `app` | `Dockerfiles/Dockerfile.prod` | version, major/minor, and `latest` after admission |
| `worker` | `Dockerfiles/Dockerfile.worker` | version, major/minor, and `latest` after admission |
| `audio-worker` | `Dockerfiles/Dockerfile.audio_gpu_worker` | version, major/minor, and `latest` after admission |
| `webui` | `Dockerfiles/Dockerfile.webui` | `build-and-scan-only` |
| `admin-ui` | `Dockerfiles/Dockerfile.admin-ui` | `build-and-scan-only` |

Third-party reference subjects are `caddy`, `postgres`, `redis`, `prometheus`,
`alertmanager`, and `grafana`. Their reviewed literals and the selected
platform manifests are in `.github/supply-chain/reference-images.json`.

Buildx may represent one `linux/amd64` image plus its attestations as an OCI
index. The subject/index digest identifies that whole immutable object. The
child manifest digest identifies the selected runnable `linux/amd64` manifest.
Release evidence records and cross-checks both. The scan is bound to the exact
subject and selected platform; promotion must reproduce the subject/index
digest exactly.

The three backend images receive maximum BuildKit provenance, an OCI SBOM, and
a signed GitHub build-provenance attestation before promotion. WebUI and Admin UI
receive the same build and signed evidence locally but are never pushed by
the release workflow. The reference-image records contain identity, SBOM, and
scan evidence only. Do not generate or claim third-party provenance for images
the project did not build.

### Embedded runtime evidence

The PR container check also loads each Python candidate's existing OCI layout
into Docker's containerd image store and runs it by the scanned subject digest.
It does not rebuild or substitute a mutable tag. The probe verifies the selected
`linux/amd64` manifest/config hashes, locked Chroma/PyJWT/cryptography
versions, real application-managed per-user collection/query isolation, absence
of TCP listeners while the embedded clients are active, bidirectional rejection
of foreign collection UUID reads and writes with original IDs and documents
preserved, and an ES256 round trip through PyJWT's cryptography backend. It runs
without network access or capabilities, with a read-only root and temporary
scratch storage.

The release backend job now runs the same probe for app, worker and audio-worker.
It pulls the immutable `linux/amd64` child selected from the scanned subject,
verifies the child manifest hash and the local image's repository digest and
platform, then runs that child with `--pull never` and the same isolation controls.
The release runtime JSON records the parent subject, platform manifest digest
and config digest taken from the authenticated manifest. Both the runtime JSON
and platform manifest join the image checksums and aggregate release file hashes.
Backend admission requires both the runtime step and vulnerability decision to
succeed. The source-admission prerequisite still applies; this workflow change
does not establish fresh release-image evidence until a release job actually runs.
See [TASK-13013.7.25](../Evidence/TASK-13013.7.25-release-runtime-admission.md).

The main-image publication workflow applies the same runtime check before alias
promotion. It selects the sole eligible `linux/amd64` child from the authenticated
build subject, records the probe's subject/platform/config identities and hashes
both runtime JSON and the child manifest. Promotion requires runtime and policy
success; source admission remains a prerequisite. This local wiring requires
fresh candidate evidence before it can support an artifact-specific assessment.
See [TASK-13013.7.27](../Evidence/TASK-13013.7.27-main-runtime-admission.md).

All six application-owned persistent Chroma client constructors explicitly select
the embedded Rust API, so `CHROMA_API_IMPL` cannot switch those paths to HTTP.
The declared Chroma minimum is 1.5.9, the existing locked and tested version;
resolved dependencies are unchanged. Explicit client/factory injection remains
outside this guarantee. This configuration hardening does not activate a Chroma
exception or qualify a new artifact. See
[TASK-13013.7.26](../Evidence/TASK-13013.7.26-embedded-chroma-settings.md).

`runtime-image-<name>.json` records these results, the lock hash, and the subject
and config digests; it is included in the evidence checksum manifest. Probe
failure rejects admission but does not suppress vulnerability scans or uploads.
These narrowly scoped checks are **not** a full application-startup audit or a
determination that a CVE is inapplicable. They never create exceptions or change
either vulnerability gate. Third-party server deployments and optional runtime
paths still need their own applicability review.

The runtime JSON also records exact-image observations for Perl integer/pointer
sizes and the presence of the two standard `systemd-homed` executable paths.
An absent executable is recorded as absent; a failed or malformed observation
rejects the probe. Host-side tests validate the collector but are not exact-image
evidence, and these facts do not establish broader module/function reachability
or make Perl, systemd, or frontend findings inapplicable.

JWT consumers use `PyJWT[crypto]`; the locked graph no longer includes
python-jose or its ecdsa fallback. A shared compatibility boundary preserves
existing issued tokens and claim/error handling, including numeric issued-at
validation and rejection of an unverifiable `at_hash`. OIDC still checks the
configured signing-algorithm allowlist, issuer, audience, and nonce. Deploy the
new locked environment or rebuilt images; updating an existing environment
without removing obsolete packages does not remove their vulnerability findings.

## Vulnerability exceptions

The canonical policy is
`.github/supply-chain/vulnerability-exceptions.json`; its schema is
`.github/supply-chain/vulnerability-exceptions.schema.json`. The default is an
empty exception list. Do not add an exception simply to make CI green.

The five OS records were explicitly approved in
[PR #2869](https://github.com/rmusser01/tldw_server/pull/2869#issuecomment-5613359176).
Their exact record identities and retained-report replay are recorded in
[TASK-13013.7.24](../Evidence/TASK-13013.7.24-os-exception-activation.md).
The requester subsequently directed activation of the sixteen exact Chroma
non-applicability records for the supported embedded profile; that authorization,
source/deployment scope and replay are recorded in
[TASK-13013.7.28](../Evidence/TASK-13013.7.28-chroma-exclusions.md).
The requester directed continued remediation and exclusion of confirmed
non-applicable findings. A further 72 exact image records, supported by package
contents, absent affected features or incompatible ABI prerequisites, are recorded
in [TASK-13013.7.29](../Evidence/TASK-13013.7.29-deployment-applicability.md).
Another 175 exact native-package records are documented in
[TASK-13013.7.30](../Evidence/TASK-13013.7.30-native-applicability.md), preserving
the prior 93 records. Current librist usage exclusions require reassessment when
native consumers or direct library calls change; Perl usage exclusions similarly
require reassessment for new Perl subprocesses or services. Neither claims absent code.
Four exact NLTK current-use records are documented in
[TASK-13013.7.32](../Evidence/TASK-13013.7.32-nltk-applicability.md). Their source
record also drives the existing dependency-review adapter; any uncovered added
occurrence prevents that advisory allowance. Reassess new model-management paths
or changed consumers. Prior candidate qualification remains incomplete.
The old OS-only comment does not authorize these later records; the linked
evidence records the conversation authorization and the PR review location.
Loading this policy
on September 18, 2026 UTC or later fails until expired records are removed or
replaced through the renewal process below.

Every exception is one exact vulnerability, component, package URL, installed
version, and severity. It requires a stable `id`, `vulnerability_id`,
`component`, `purl`, `installed_version`, `severity`, `owner`, `rationale`,
`mitigation`, human-review URL in `approval`, `created_on`, `expires_on`, and
nullable `supersedes`. Critical exceptions may last at most 7 days; High
exceptions may last at most 30 days. Shorter upstream deadlines win.

Approval must be visible in the repository and must come from an accountable
human reviewer. The rationale explains reachability and why remediation is not
currently possible; mitigation describes the active control and validation.
Stale, expired, overlong, cross-component, version-mismatched, or unused
exceptions fail admission.

To renew an exception, reassess the current package and vulnerability, obtain
new human approval, create a new record with new dates, and link the prior ID
through `supersedes`. Remove the prior record in the same reviewed change.
Never extend dates in place without a new assessment. Remediation followed by
a fresh clean scan is preferred to renewal.

## Refresh an image digest

Digest refresh is a security change, not a text-only version bump.

1. Choose a non-floating upstream version and resolve its OCI subject/index
   digest from the registry.
2. Select and record exactly one `linux/amd64` child manifest digest.
3. Pull or build the literal `tag@sha256:<subject>` and create its CycloneDX
   SBOM and complete Trivy JSON with the pinned, fresh database.
4. Evaluate the canonical exception policy with `ignore-unfixed=false` and
   require zero blockers and zero stale exception IDs.
5. For a reference image, update
   `.github/supply-chain/reference-images.json` and the corresponding
   `CADDY_IMAGE`, `POSTGRES_IMAGE`, `REDIS_IMAGE`, `PROMETHEUS_IMAGE`,
   `ALERTMANAGER_IMAGE`, or `GRAFANA_IMAGE` example together.
6. Run production preflight contracts and the source/image release contracts.
   Commit the reviewed identity and evidence result as one change.

Never copy a digest from an unauthenticated third-party report. Resolve it from
the registry, recompute the raw manifest hash, and compare the selected child
against the inventory.

## Cut and publish a container release

`make release`, `make release-patch`, and `make release-minor` create the
release commit, annotated stable tag, and an existing draft GitHub Release.
They do not publish the draft. Review its notes, then dispatch the workflow on
that exact tag:

```bash
RELEASE_TAG=v0.1.35 # Example only; substitute the intended release tag.
gh release view "$RELEASE_TAG" --json isDraft,tagName
gh workflow run publish-docker.yml \
  --ref "$RELEASE_TAG" \
  -f release_tag="$RELEASE_TAG" \
  -f confirmation="publish $RELEASE_TAG"
gh run list --workflow publish-docker.yml --limit 1
```

The confirmation must be the literal `publish <tag>`. The workflow accepts
stable semantic versions only and requires its `GITHUB_REF`, checked-out tag,
draft release tag, and source commit to agree.

The workflow then performs this ordered transaction:

1. Admit the exact source dependencies and a fresh scanner database.
2. Build unique app, worker, and audio-worker GHCR candidates; build WebUI and
   Admin UI as local OCI candidates.
3. SBOM and scan those five exact subjects plus all six reference images.
4. Validate checksums, policy decisions, subject/index versus child identity,
   provenance references and bundle hashes, and the complete `release-manifest.json`.
   Signature verification is a separate consumer step below.
5. Promote the three backend digests to full-version aliases and
   verify all three.
6. Promote the same digests to major, minor, and `latest` floating aliases and
   verify them again.
7. Upload the evidence assets, verify their presence, and publish the draft
   last.

If any producer, scanner, policy, evidence, attestation, registry, or upload
step fails, do not publish manually. Leave the GitHub Release as a draft,
retain the Actions evidence, fix the cause, and rerun the same tag only after
confirming any already-created version alias still resolves to the admitted
digest. A failed run may leave unique candidate tags; they are not release
aliases and may be cleaned up under the registry retention policy.

Full-version promotion refuses to overwrite a different existing digest and
treats registry lookup errors as failures, not missing tags. Resume failed jobs
using the original admitted candidates when possible. If rebuilding changes a
digest after a partial promotion, stop and investigate; do not delete or move
the existing version tag to force a retry. Tags remain mutable registry names,
so restrict independent writers and deploy by digest even with this guard.

## Verify release evidence and provenance

Download all four release assets, verify the outer checksum, then verify the
manifest against the unpacked evidence:

```bash
RELEASE_TAG=v0.1.35
gh release download "$RELEASE_TAG" \
  --pattern "release-evidence-*" \
  --pattern release-manifest.json \
  --pattern release-evidence.schema.json \
  --pattern SHA256SUMS.release
sha256sum -c SHA256SUMS.release
tar -xzf "release-evidence-$RELEASE_TAG.tar.gz"
source .venv/bin/activate
python Helper_Scripts/Supply_Chain/release_evidence.py verify \
  --manifest release-manifest.json \
  --evidence-dir release-evidence
```

For each published backend subject, authenticate to GHCR and verify the exact
digest against this repository and signer workflow:

```bash
IMAGE_REPOSITORY=ghcr.io/rmusser01/tldw_server
SUBJECT_DIGEST=sha256:replace-with-release-manifest-value
RELEASE_TAG=v0.1.35
gh attestation verify oci://"$IMAGE_REPOSITORY@$SUBJECT_DIGEST" \
  --repo rmusser01/tldw_server \
  --signer-workflow rmusser01/tldw_server/.github/workflows/publish-docker.yml \
  --source-ref "refs/tags/$RELEASE_TAG"
```

The evidence archive also contains `provenance-image-<name>.jsonl` bundles for
all five project-built subjects and the exact OCI subject bytes in
`subject-<name>.json`. Assembly and verification require GitHub CLI (`gh`) and
cryptographically verify every retained bundle against those bytes, the manifest's
repository and source commit, the release tag, and `publish-docker.yml` at the
same commit. The verified statement must contain exactly that image name and
digest. Missing bundles, mismatched subjects, unverifiable signatures, or missing
verification tooling fail closed before promotion.

The record's `provenance_ref` is a navigation link in the expected repository's
attestation namespace, not proof of authenticity. The verified bundle is the
authority; a GitHub URL alone can never satisfy admission. Confirm the manifest's
repository, tag, and source commit against your intended release before trusting it.

By default, `gh` may access the network to bootstrap or refresh its Sigstore trust
roots. For disconnected verification, obtain a trusted root independently using
`gh attestation trusted-root` on a trusted connected host, then pass
`--trusted-root /trusted/path/trusted_root.jsonl` to either evidence command.
Do not take trust roots from the untrusted release archive. No registry access
or frontend image publication is needed for this bundle-based verification.

PyPI trusted-publishing and PEP 740 consumer verification are documented in
[`PyPI_Publishing.md`](PyPI_Publishing.md).

## Evidence lifetime

A past clean scan is time-bound evidence, not a guarantee of future safety.
New advisories, database corrections, revoked upstream artifacts, or an expired
exception can invalidate an earlier decision without changing the image bytes.
Re-scan before deployment when the release evidence is older than the
organization's risk window, after a relevant advisory, and during every digest
refresh. Preserve the old report for audit; do not relabel it as current.

Primary references:

- [uv project locks](https://docs.astral.sh/uv/concepts/projects/layout/)
- [CycloneDX](https://cyclonedx.org/docs/1.5/json/)
- [Trivy image inputs](https://trivy.dev/docs/latest/guide/target/container_image/)
- [GitHub attestation verification](https://cli.github.com/manual/gh_attestation_verify)
- [PyPI attestations](https://docs.pypi.org/attestations/)
