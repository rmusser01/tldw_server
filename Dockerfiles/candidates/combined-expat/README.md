# Combined Expat candidate test tooling

TASK-13013.7.9 only. These helpers do not assemble, publish, promote or qualify an
image by themselves. Production recipes, dependency locks and scanner admission
are unchanged. `combined-expat-candidate.yml` now wires them into a native-only
assembly/application-control workflow; a successful run is still not admission.

`test-tools.py` emits eight exact universal wheel URLs and SHA-256 hashes from
the checked-in `uv.lock`. The reviewed versions and registry must match; missing,
duplicate or ambiguous records fail before any requirements are printed. It does
not resolve dependencies, download archives or install packages.

The intended **candidate-builder-only** integration is:

```sh
python Dockerfiles/candidates/combined-expat/test-tools.py uv.lock > /candidate/test-tools.requirements.txt
uv pip install --python /opt/tldw-venv/bin/python \
  --target /opt/expat-test-tools --no-deps --require-hashes --only-binary :all: \
  --requirement /candidate/test-tools.requirements.txt
```

Use the repository's digest-pinned uv 0.12.7 builder binary and retain its version
and install logs. Its target/hash options are documented in the
[official uv CLI reference](https://docs.astral.sh/uv/reference/cli/#uv-pip-install).
The candidate Dockerfile compares runtime-environment file hashes and symlink
targets before/after installation and retains both inventories. Its offline
runtime checks require the exact installed cohort. Native execution remains
required before treating that wiring as verified. The development extra is never
installed into the application environment.

`run-tests.py` is the application-test entry point after assembly:

```sh
/opt/tldw-venv/bin/python /candidate/run-tests.py \
  --root /app --tools /opt/expat-test-tools --report /evidence/application-tests.json
```

The launcher appends the tools directory after runtime import paths without
executing `.pth` files. It disables plugin autoload and inherited pytest options,
loads only the asyncio/timeout entry-point plugins explicitly, and retains normal
repository conftests. It requires the exact eight reviewed ingestion/chunking
test identities and successful setup, call and teardown for each. Skips, xfail,
non-strict xpass, missing tests and fixture errors fail the result even where
pytest itself would return zero. The JSON report is explicitly scoped to
`application-tests-only`.

Run this from the same immutable assembled image, as a non-root user with no
network. Include the tests/conftests and configuration at build time; do not mount
a host checkout over the application. The eventual controller must separately
bind interpreter/library/module paths and hashes to the qualified artifacts and
image identity. It must also enforce parser-version, rendering, FFmpeg capability,
SBOM and vulnerability gates. A passing local launcher run is not evidence that
the rebuilt Expat image passes those gates.

Regression coverage executes both helpers, including real small pytest suites
for each negative outcome, runtime-versus-test-tools import precedence, and a
`.pth` execution trap. The eight real application tests also pass through this
launcher in the task's local verification environment; that environment is not
the candidate image or its pinned test-tool cohort.

## Native assembly boundary

The workflow fetches metadata directly from GitHub and downloads three fixed
artifact IDs from the two approved successful runs. `assembly-inputs.py` checks
repository, workflow, branch, full commit, success and artifact/run bindings;
repeats every existing parser phase/inventory/hash gate; and verifies independently
reviewed payload SHA-256 pins before staging the replacements. Caller-provided
metadata alone is not authentication.

The FFmpeg input enters through a digest-pinned
[named OCI build context](https://docs.docker.com/reference/cli/docker/buildx/build/#build-context).
Its signed snapshot sources are inherited unchanged. Only candidate runtime
dependencies and the production build dependencies are acquired there. Python's
complete installation is restored before application dependency installation;
the final system Expat replacement happens after all runtime APT acquisition.
The builder uses production's locked/no-dev/non-editable sync with the qualified
Python explicitly selected. No alternative CPU-only or reduced dependency set
is substituted to fit a runner.

Root `.dockerignore` remains unchanged. A separate context produced with
`git archive HEAD` includes the tracked root conftest and complete test tree;
host databases, local environments and untracked test files are not imported.

The build retains an OCI archive with full provenance/SBOM attestations, then
exports the same cached result without attestations solely for local execution.
The runner's classic Docker store cannot import an explicitly attested image;
[Buildx rejects that exporter combination](https://github.com/docker/buildx/blob/v0.36.1/build/opt.go).
A tiny scratch-image preflight must prove both exports and their config equality
before the expensive application build. No registry publishing occurs, and the
execution representation is never substituted for canonical attested evidence.
`image-identity.py` binds the loaded config to the archive's hashed subject,
manifest and config metadata, requires native amd64/non-root execution, and
hashes the complete archive. It does not independently validate layer payloads;
the trusted build/export/load path owns that step. Consumers must recheck the
retained archive digest before loading or scanning it.

The image runs with no network, a read-only root filesystem, UID/GID 10001,
dropped capabilities, and bounded writable test/evidence directories. Controls
check four exact Python binaries, owning interpreter/library/parser identities
with and without `LD_LIBRARY_PATH`, both installed system parser versions, all
eight application tests, font discovery, drawtext/subtitles, and unresolved or
retired FFmpeg libraries. All failures remain fatal and available evidence is
uploaded even on failure.

Still outstanding after this assembly checkpoint: native results, full existing
FFmpeg baseline/capability/synthetic-media comparison, application import-path
and hash evidence, source-aware Syft/Trivy/Grype reports, final security review
and any separate production-adoption decision. No helper or workflow success
should be presented as vulnerability clearance while those gates remain open.
