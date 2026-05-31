# PyPI Publishing Guide (`tldw-server`)

This guide sets up and uses the repository's PyPI release flow for the `tldw-server` package.

## What This Repo Now Supports

- Local build/check helpers:
  - `make pypi-build`
  - `make pypi-check`
- CI packaging validation:
  - `.github/workflows/pypi-package.yml`
- Manual PyPI publishing workflow (Trusted Publishing):
  - `.github/workflows/publish-pypi.yml`

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

## Local Packaging Checks

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

## Release Publish Flow

1. Bump version in `pyproject.toml`.
2. Create and push a Git tag (for example `v0.1.22`).
3. Publish a GitHub Release from that tag.
4. GitHub Actions runs `publish-docker.yml` for Docker release publication.

For PyPI publishing in this rollout, run `publish-pypi.yml` manually from Actions, select the release tag/ref that matches the GitHub Release and Docker release publish, and choose:

- `testpypi` for TestPyPI
- `pypi` for the real PyPI publish

## Notes

- `publish-pypi.yml` is manual-dispatch-only in this rollout; GitHub Release publication no longer triggers PyPI uploads.
- Pushing to `main` continues to republish rolling GHCR snapshots through `publish-ghcr-main.yml` before and independently of GitHub Release publication.
- The default dependency set is intentionally broad and may be heavy for minimal installs.
- If you want a lighter default install later, move optional feature stacks (for example some STT/TTS stacks) into extras and keep base runtime lean.
