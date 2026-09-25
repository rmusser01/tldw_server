# Paired Docker app bundle (candidate format)

This directory contains the source template for a signed, extractable
application bundle. The source checkout is not a downloadable release: its
start helpers contain control-image and key placeholders. Candidate packaging
fills those placeholders with one immutable control-image digest and trusted
key ID, then signs the release manifest. Do not run a helper whose
placeholders remain.

The current CI candidates use a job-local registry. Their image references
expire when the CI job ends, so an uploaded candidate bundle is evidence for
review, not an installer for another host. G2 browser setup, G4 two-instance
networking, and G12 runtime-support evidence remain open. No public paired
Docker release is available from these files yet.

An extracted candidate needs Docker Desktop or Docker Engine with Compose v2.
The host does not need Python, Node, Bun, a source checkout, PostgreSQL, or
Redis. Keep the extracted files together, including `manifest.json` and its
detached `manifest.sig`.

On macOS or Linux, run `./start.sh` from the extracted directory, then open
the printed loopback URL. On a desktop, the helper also tries to open the
browser after Compose reports healthy services. Set `TLDW_APP_NO_BROWSER=1`
to suppress that. Use `./status.sh` to inspect the services and
`./stop.sh` to stop them while retaining data. On Windows PowerShell, use
`./start.ps1`, `./status.ps1`, and `./stop.ps1`. Each script finds its bundle
relative to its own location, so the current working directory does not
matter.

The first start creates private, persistent instance identity and credentials
under `~/Library/Application Support/tldw/app` on macOS,
`${XDG_DATA_HOME:-~/.local/share}/tldw/app` on Linux, or
`%LOCALAPPDATA%\tldw\app` on Windows. Set `TLDW_APP_STATE_DIR` before the first
start to choose another location. Set `TLDW_APP_PUBLIC_PORT` before the first
start to use a free port other than 8080; later runs must keep that choice.
The host publishes only the gateway at `127.0.0.1:<port>`. Backend and WebUI
ports stay on the private Compose network.

The pinned control image verifies the manifest signature, platform, local
artifact hashes, required image references, and existing state before any
application container starts. The helper then pulls the image references by
their signed OCI digests. `stop` leaves both named data volumes intact. Keep
the private state directory and Docker volumes together when backing up or
moving an installation; the directory includes the instance API key.

Updates, backup/restore, and optional component replacement are not included
in this WP1 bundle. They require the separately planned host-helper work;
stopping and restarting this candidate retains the named volumes.
