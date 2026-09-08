# Native frontend runtime candidate renderer

This directory generates candidate-only Dockerfiles for WebUI and Admin UI. It
does not modify either canonical Dockerfile, build an application, publish an
image, or admit this design to production.

Run it from any directory and choose an output outside the canonical recipe:

```bash
python Dockerfiles/candidates/frontend/render.py \
  --application webui \
  --output /tmp/Dockerfile.webui.candidate
```

The renderer accepts only the exact Node 24.20.0 builder and runtime markers. It
preserves all bytes before the runtime marker and all bytes after it, replacing
only that marker with the candidate runtime block. Missing, duplicated, renamed,
or repinned canonical stages fail rather than producing an unreviewed recipe.

## Candidate runtime contract

The generated runtime is pinned to the reviewed Ubuntu 24.04 digest and installs
only `zlib1g=1:1.3.dfsg-3.1ubuntu2.2` through normal signed APT indexes. The build
retains the signed `InRelease` files and their SHA256 values, the exact acquired
zlib package SHA256, APT policy, and installed zlib/libc/libstdc++ versions under
`/usr/local/share/tldw-candidate-evidence`. It also requires libc6 to remain
`2.39-0ubuntu8.8`. These records bind the acquisition used by the experiment;
they are not a snapshot-reproducibility claim.

The unchanged builder supplies `/usr/local/bin/node`,
`/usr/local/bin/docker-entrypoint.sh`, and `/usr/local/LICENSE`; the candidate
restores `ENTRYPOINT ["docker-entrypoint.sh"]` explicitly. The official
[Node 24.20.0 Bookworm slim recipe](https://github.com/nodejs/docker-node/blob/c4eb0858f5c522521768d5b6dc1d9f1631d4854d/24/bookworm-slim/Dockerfile)
extracts the authenticated Node archive into `/usr/local` and copies the
entrypoint there. Direct inspection of the exact pinned amd64 artifact confirmed
all three paths and Node `v24.20.0`.

That exact artifact has no `/etc/ssl/certs`, `/usr/share/ca-certificates`, or
system CA bundle: the recipe auto-purges its temporary `ca-certificates` package.
The candidate therefore does not emit a broken CA copy. Copying the exact Node
executable preserves Node's embedded root store; native qualification must compare
its `tls.rootCertificates` hash and count and separately report any system bundle.

Unlike the baseline Node image, the candidate does not carry npm, npx, Yarn, or
the `nodejs` convenience symlink into the runtime. Both canonical applications
start with `node`, and their complete application `ENV`, `COPY`, `USER`,
`WORKDIR`, `EXPOSE`, `HEALTHCHECK`, and `CMD` remainder stays byte-for-byte
unchanged. This is not a bit-identical operating system and does not imply that
findings absent from another distribution's feed were fixed.

## Native compatibility qualification

The candidate workflow is `.github/workflows/frontend-runtime-candidate.yml`.
Its WebUI/Admin matrix runs on native `ubuntu-24.04`, with containerd OCI loading
and no QEMU setup. Both artifacts are built from the same checkout and arguments,
with maximum provenance and SBOM attestations. The workflow resolves their
hash-addressed configurations using `runtime_probe.py config` before loading.
It retains both OCI archives, generated recipe, source commit/status, build
metadata, archive hashes, subject/config identities and qualification evidence
for 14 days, including failed runs. It neither publishes images nor changes any
release admission policy. A scoped push on the design branch or manual dispatch
runs this candidate experiment; this implementation task does not trigger it.

For a local run, use a native Linux/x86_64 host and native Linux/amd64 Docker
server with the containerd image store. From the repository root, build/export
both recipes as local `linux/amd64` OCI archives. Keep the same arguments for
each pair: Admin uses `NEXT_PUBLIC_API_URL=http://backend:8000` in ordinary
standalone mode; WebUI uses quickstart and
`TLDW_INTERNAL_API_ORIGIN=http://backend:8000`. Other arguments match the
workflow. Resolve the build output subject digests (not tags or config IDs),
extract each archive to a separate layout, then bind and load each artifact:

```bash
# Set these to your actual retained layout/archive paths and build output digest.
python Helper_Scripts/Supply_Chain/runtime_probe.py config \
  --layout "$BASELINE_LAYOUT" --subject "$BASELINE_SUBJECT"
python Helper_Scripts/Supply_Chain/runtime_probe.py config \
  --layout "$CANDIDATE_LAYOUT" --subject "$CANDIDATE_SUBJECT"
docker load --input "$BASELINE_ARCHIVE"
docker load --input "$CANDIDATE_ARCHIVE"
python Dockerfiles/candidates/frontend/qualify.py \
  --application webui \
  --baseline "$BASELINE_SUBJECT" --candidate "$CANDIDATE_SUBJECT" \
  --evidence /tmp/frontend-webui-qualification
```

Use a new evidence directory for each invocation. Repeat with `--application
admin-ui` for the Admin pair. CI additionally supplies `evidence/inputs.json`
with the verified OCI/config/archive identities; local runs without that file
retain loaded subject identities and checkout input hashes but lack that
workflow binding. Retain the config command outputs and archives separately.

The qualifier rejects non-native host/daemon facts, mutable or ambiguous image
references, wrong loaded identities/platforms, and drift in canonical application
configuration. It checks actual Node version/ABI/executable hash, UID/GID/home,
workdir and ownership, embedded root count/hash, system CA bundle presence/hash,
and a real sharp PNG encode/resize/decode operation. The sharp probe uses Node's
normal resolver from the installed Next image optimizer context, matching Next's
own dependency lookup even when Bun nests sharp outside the application root.
It does not install dependencies or change the image's working directory.
Candidate dpkg versions are checked against the reviewed Ubuntu versions;
glibc versions are not compared
for equality between distributions.

Each invocation owns an internal network and uniquely named containers. The
controlled backend has the network alias `backend` and serves only
`/api/v1/health`; it uses the exact candidate Node image. No host ports or real
credentials are used. Every container has a read-only root filesystem, dropped
capabilities, no-new-privileges, 128 PID/1 GiB/2 CPU limits, the image's non-root
USER, and owned tmpfs mounts only at `/tmp` and the application's `.next/cache`.

Admin application containers receive one random, test-only `JWT_SECRET_KEY` per
qualification invocation, shared between its baseline and candidate. The key is
not supplied to diagnostics or the stub, saved in an image, or taken from real
credentials. Production `NODE_ENV` and normal runtime/auth validation remain
enabled. The qualifier verifies the effective nonempty signing fixture and
redacts its value from persisted command argv, output, inspections, logs,
errors, timeout partial output, and the final report. In-memory command results
remain available for validation.

WebUI must serve its root and pass its configured check. Admin must return
liveness/readiness 200 with the stub running, then readiness 503, liveness 200
and a failing configured check after the stub exits. The configured healthcheck
is validated byte-for-byte and its exact Node body is executed directly with a
five-second timeout; this avoids shell interpolation and does not claim to test
Docker's periodic health scheduler. Both applications and the stub must complete
SIGTERM within ten seconds with exit 0 or 143. Force removal is reserved for
cleanup of invocation-owned IDs, after failure evidence has been retained.

`qualification.json` has schema version 1 and explicit scope
`native-frontend-compatibility-not-release-admission`. All command argv, stdout,
stderr, timeouts, exit statuses, inspections (with the fixture key redacted),
diagnostic observations and cleanup outcomes remain under `commands/`, including failed controls. A failed
control returns a nonzero CLI status; cleanup failure also prevents success.

Local mocked tests are validation of the controls, not native application
qualification. ARM/emulated runs cannot pass. Even a native passing result
certifies only this exact artifact pair against the controlled health stub;
it is neither real-backend certification nor a vulnerability waiver.
`CVE-2026-85091` remains unresolved. Vulnerability comparison and production
adoption are separate stages.

## Frozen-database inventory and vulnerability comparison

After exact OCI binding succeeds, the workflow also records Syft 1.51.1
inventories and complete Trivy 0.74.0 and Grype 0.118.0 findings. All three
scanner containers are digest-pinned. Each application pair gets new Trivy and
Grype database directories; the databases are downloaded once, must be valid
and no older than 24 hours (with five minutes of clock skew), and are then used
offline for both baseline and candidate. Before/after database hashes prove that
the pair did not update between scans. Scanner versions, pinned-image
inspections, download logs, database metadata/status, hashes, scan logs and raw
JSON reports remain in the always-uploaded evidence directory.

All Grype commands run as the current runner UID/GID so the host can inspect
and hash Grype's owner-only database cache without relaxing its permissions.
They also receive a private `/tmp` tmpfs (`rw,nosuid,nodev,noexec,mode=1777`),
because the pinned scanner image's existing `/tmp` is not writable by that
identity. Database acquisition, freshness validation, offline scans and
before/after hash checks remain unchanged.

All three scanners read the validated OCI layouts; Trivy additionally addresses
the exact build subject within its read-only layout mount rather than a tag.
Every report is checked against its baseline or candidate config digest. Syft must
inventory Node 24.20.0 in both images and the reviewed zlib/libc versions in the
candidate. Scanning and comparison run after a failed native qualification when
OCI loading was successful, without changing or replacing the qualification
result; no scan runs when exact binding/loading failed.

The comparison CLI is intentionally fail-closed:

```bash
python Dockerfiles/candidates/frontend/compare.py \
  --baseline-trivy /path/baseline-trivy.json \
  --candidate-trivy /path/candidate-trivy.json \
  --baseline-grype /path/baseline-grype.json \
  --candidate-grype /path/candidate-grype.json \
  --baseline-config sha256:BASELINE_CONFIG \
  --candidate-config sha256:CANDIDATE_CONFIG \
  --output /path/vulnerability-comparison.json
```

Its schema retains Trivy and Grype separately, includes all severities and full
original rows (including duplicate package versions, paths and ecosystems), and
retains Grype ignored matches rather than hiding them. A baseline-only row is
classified `baseline-only-unproven`, never fixed merely because another feed or
distribution does not report it. For Debian-to-Ubuntu comparisons, Debian and
Ubuntu `deb` OS-package findings share a logical ecosystem key while their raw
distro types and namespaces remain intact; other package ecosystems and feeds
remain separate. The report always contains `admitted: false` and no fixed
claims. In particular, absence of the old zlib CVE-2026-27171 row
and Ubuntu vendor evidence for glibc require external corroboration; they are not
release admission. CVE-2026-85091 remains unresolved at its reported vendor
severity.
