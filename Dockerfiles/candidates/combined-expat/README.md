# Combined Expat candidate test tooling

TASK-13013.7.9 only. These helpers do not assemble, publish, promote or qualify an
image. Production recipes, dependency locks and scanner admission are unchanged.

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
The candidate workflow must additionally compare runtime-environment file hashes
before and after installation, retain tool identities, and verify the installed
cohort. That install wiring and native execution are still outstanding. Do not
install the development extra into the application environment.

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
