# PyPI Publishing Guide (`tldw-server`)

This guide sets up and uses the repository's PyPI release flow for the `tldw-server` package.

`tldw-server` on PyPI is the backend/API/CLI distribution only. It does not
bundle the Next.js WebUI from `apps/tldw-frontend`. During the frontend
licensing freeze, no WebUI Docker image or release artifact is published; run
an authorized frontend build from source when a UI is needed. The longer-term
artifact boundary is described in
[`Packaging_and_Distribution_Strategy.md`](Packaging_and_Distribution_Strategy.md).

## What This Repo Now Supports

- Local build/check helpers:
  - `make pypi-build`
  - `make pypi-check`
- CI packaging validation:
  - `.github/workflows/pypi-package.yml`
- PyPI publishing workflow (Trusted Publishing):
  - `.github/workflows/publish-pypi.yml`
- Backend/API-only artifact validation:
  - `make pypi-check` fails if frontend or Node build artifacts enter the
    wheel or source distribution.

## One-Time Setup (PyPI)

1. Create the project on PyPI (if it does not exist): `tldw-server`.
2. In PyPI project settings, add a trusted publisher:
   - Owner: `rmusser01`
   - Repository: `tldw_server`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
3. (Optional) Repeat for TestPyPI with environment `testpypi`.
4. In GitHub repo settings, create environments:
   - `pypi`
   - `testpypi` (optional)

GitHub can only manually dispatch a workflow when that workflow file exists on
the repository's default branch. Make sure `.github/workflows/publish-pypi.yml`
has landed on the default branch before relying on the Actions UI for a first
manual publish.

## Local Packaging Checks

The canonical release tools come from the exact `release` group in `uv.lock`.
Use uv 0.12.7, matching the immutable tool reference documented in
[`Software_Supply_Chain.md`](Software_Supply_Chain.md):

```bash
uv sync --locked --no-dev --no-editable --group release
```

```bash
# Build source + wheel distributions without a second dependency resolver
PYTHON=.venv/bin/python PYPI_BUILD_ARGS=--no-isolation make pypi-build

# Build + run twine metadata checks
PYTHON=.venv/bin/python PYPI_BUILD_ARGS=--no-isolation make pypi-check

# Require one wheel, one sdist, and retain their exact hashes
(cd dist && sha256sum ./*.whl ./*.tar.gz > SHA256SUMS)
(cd dist && sha256sum -c SHA256SUMS)
```

Optional wheel smoke test:

```bash
python -m venv /tmp/tldw-pypi-smoke
source /tmp/tldw-pypi-smoke/bin/activate
pip install --upgrade pip
pip install dist/*.whl
python -c "import tldw_Server_API; print('ok')"
```

The wheel smoke test intentionally imports the backend package only. It should
not attempt to start or validate the WebUI.

## Release Publish Flow

1. Bump the version in `pyproject.toml` and regenerate `uv.lock` with the
   reviewed uv 0.12.7 (`uv lock --offline`, then `uv lock --check --offline`).
   Commit both files together; `make release*` handles this for the canonical flow.
2. Create and push a Git tag (for example `v0.1.35`).
3. Create and review the draft GitHub Release from that tag.
4. Dispatch `publish-docker.yml` separately for verified container release
   publication.

For a manual PyPI publish, run `publish-pypi.yml` from Actions, select the
release tag/ref that matches the GitHub Release and Docker release publish, and
choose:

- `testpypi` for TestPyPI
- `pypi` for the real PyPI publish

Recommended manual operator sequence:

1. Run the locked `make pypi-check` command above or confirm the PR packaging
   check passed.
2. Run the isolated wheel smoke install shown above.
3. Publish to `testpypi`.
4. Install from TestPyPI in a fresh environment and import `tldw_Server_API`.
5. Publish to `pypi`.
6. Install from PyPI in a fresh environment and run `tldw-server --help` or a
   minimal startup smoke.

The workflow builds exactly one wheel and one source distribution in a job
without OIDC permission. It writes `SHA256SUMS`, verifies it, and uploads that
single checked artifact. Each publish job downloads the artifact, verifies
`SHA256SUMS` again, removes only the checksum file, and gives the unchanged two
distributions to the pinned publisher action within that run. TestPyPI and PyPI
are separate dispatch targets: separate runs build separate artifacts. Retain
and compare their hashes; do not assume byte-for-byte identity across runs.

## Trusted Publishing and PEP 740

Only the `publish-test` and `publish-pypi` jobs receive `id-token: write`. They
use PyPI Trusted Publishing rather than a long-lived API token, and both set
the publisher input explicitly:

```yaml
with:
  attestations: true
```

PyPI's PEP 740 publish attestations bind each distribution hash to the GitHub
OIDC publishing identity. They are publish attestations, not a substitute for
the earlier source, test, package-content, or checksum gates.

After publication, obtain the exact wheel filename from the PyPI release page
and verify it against this repository with the reviewed `pypi-attestations`
CLI. Its command-line interface is not yet a stable packaging standard, so pin
and record the verifier version used for an audit.

```bash
WHEEL=tldw_server-0.1.35-py3-none-any.whl
pypi-attestations verify pypi \
  --repository https://github.com/rmusser01/tldw_server \
  "pypi:$WHEEL"
```

Inspection alone is not verification. The verification must succeed for the
expected repository and exact filename; then compare the downloaded file's
SHA-256 with PyPI's file metadata and the workflow's `SHA256SUMS` evidence.

## Notes

- `publish-pypi.yml` has no GitHub Release trigger. A push to `main` that changes `pyproject.toml` may publish to PyPI only when the version is missing from PyPI and the workflow test gate passes.
- Manual dispatch remains available for TestPyPI and explicit PyPI publishes; TestPyPI is never published from a push event.
- Pushing to `main` continues to republish the rolling, scan-admitted backend
  GHCR snapshot through `publish-ghcr-main.yml`, independently of formal draft
  release publication.
- The Next.js WebUI is not a PyPI payload. Publish and validate it through the
  WebUI container or release-artifact flow.
- The default dependency set is intentionally broad and may be heavy for minimal installs.
- If you want a lighter default install later, move optional feature stacks (for example some STT/TTS stacks) into extras and keep base runtime lean.
