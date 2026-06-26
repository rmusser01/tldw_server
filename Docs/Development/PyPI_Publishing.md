# PyPI Publishing Guide (`tldw-server`)

This guide sets up and uses the repository's PyPI release flow for the `tldw-server` package.

`tldw-server` on PyPI is the backend/API/CLI distribution only. It does not
bundle the Next.js WebUI from `apps/tldw-frontend`. When a UI is needed, pair
the PyPI backend release with the separately published WebUI Docker image or
release artifact described in
[`Packaging_and_Distribution_Strategy.md`](Packaging_and_Distribution_Strategy.md).

## What This Repo Now Supports

- Local build/check helpers:
  - `make pypi-build`
  - `make pypi-check`
- CI packaging validation:
  - `.github/workflows/pypi-package.yml`
- Manual PyPI publishing workflow (Trusted Publishing):
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
publish.

## Local Packaging Checks

The packaging check requires the standard build tools plus Loguru for the
artifact-content guard:

```bash
python -m pip install build twine setuptools wheel loguru
```

```bash
# Build source + wheel distributions
make pypi-build

# Build + run twine metadata checks
make pypi-check
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

1. Bump version in `pyproject.toml`.
2. Create and push a Git tag (for example `v0.1.22`).
3. Publish a GitHub Release from that tag.
4. GitHub Actions runs `publish-docker.yml` for Docker release publication.

For PyPI publishing in this rollout, run `publish-pypi.yml` manually from
Actions, select the release tag/ref that matches the GitHub Release and Docker
release publish, and choose:

- `testpypi` for TestPyPI
- `pypi` for the real PyPI publish

Recommended operator sequence:

1. Run `make pypi-check` locally or confirm the PR packaging check passed.
2. Run the isolated wheel smoke install shown above.
3. Publish to `testpypi`.
4. Install from TestPyPI in a fresh environment and import `tldw_Server_API`.
5. Publish to `pypi`.
6. Install from PyPI in a fresh environment and run `tldw-server --help` or a
   minimal startup smoke.

## Notes

- `publish-pypi.yml` is manual-dispatch-only in this rollout; GitHub Release publication no longer triggers PyPI uploads.
- Pushing to `main` continues to republish rolling GHCR snapshots through `publish-ghcr-main.yml` before and independently of GitHub Release publication.
- The Next.js WebUI is not a PyPI payload. Publish and validate it through the
  WebUI container or release-artifact flow.
- The default dependency set is intentionally broad and may be heavy for minimal installs.
- If you want a lighter default install later, move optional feature stacks (for example some STT/TTS stacks) into extras and keep base runtime lean.
