# Packaging and Distribution Strategy

This document covers two practical distribution concerns:

1. Slimmer container runtime for the API.
2. Distribution models for the Next.js WebUI.

The default release boundary is:

- **PyPI:** `tldw-server` backend/API/CLI only.
- **WebUI:** separate Docker image or release artifact.

Do not bundle `apps/tldw-frontend` into the root Python wheel unless the project
explicitly chooses a new single-artifact distribution model.

## 1) Slimmer Docker Runtime (API)

`Dockerfiles/Dockerfile.prod` now uses a multi-stage build:

- **Builder stage**: installs compiler/dev dependencies and builds Python deps.
- **Runtime stage**: contains only runtime libraries (`ffmpeg`, `libmagic1`, `libportaudio2`) and the app code.

Why this is the current best trade-off:

- Keeps heavy build packages out of production image layers.
- Preserves existing runtime behavior and entrypoint.
- Avoids coupling runtime to distro-specific `python3-pyaudio` packages.

If you want further size reductions later:

- Move heavyweight optional stacks from default dependencies into extras.
- Split API image variants (for example `core` vs `media` vs `full`).
- Consider separate worker images for audio/STT-heavy features.

## 2) WebUI Distribution Options

The WebUI (`apps/tldw-frontend`) now builds with `output: 'standalone'` in Next config, enabling artifact-based distribution.

### Option A: Ship WebUI as a Separate Artifact (Recommended)

Build WebUI independently and publish it as:

- a Docker image (`tldw-webui`), or
- a release tarball containing:
  - `.next/standalone`
  - `.next/static`
  - `public`

Example local build/bundle:

```bash
cd apps/tldw-frontend
npm install
npm run build
tar -czf ../../dist/tldw-webui-standalone.tar.gz .next/standalone .next/static public
```

Why recommended:

- Keeps PyPI package size and install path manageable.
- Clean separation of Python runtime vs Node runtime concerns.
- Lets WebUI release cadence differ from API if needed.

### Option B: Bundle WebUI with Python Package (`tldw-server`) (Not Recommended as Default)

Possible, but with notable drawbacks:

- Large wheel/sdist artifacts.
- PyPI users pull frontend assets even for API-only usage.
- Harder operational split between API and frontend rollout.

Use this only if your distribution model requires a single downloadable artifact.

## Suggested Default

- Publish the backend/API/CLI to PyPI (`tldw-server`) as the Python-first distribution.
- Publish WebUI as a separate deployable artifact (container or tarball).
- Document a "paired release" process so API/UI version compatibility is explicit.

The root PyPI package check validates that frontend and Node build artifacts do
not enter the `tldw-server` wheel or source distribution.
