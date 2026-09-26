# Paired Docker app bundle (candidate format)

This directory contains the source template for a signed, extractable
application bundle. The source checkout is not a downloadable release: its
start helpers contain control-image and key placeholders. Candidate packaging
fills those placeholders with one immutable control-image digest and trusted
key ID, then signs the release manifest. Do not run a helper whose
placeholders remain.

The current CI candidates use a job-local registry. Their image references
expire when the CI job ends, so an uploaded candidate bundle is evidence for
review, not an installer for another host. Each candidate records its actual
G2 Docker setup and G4 networking checks in bounded evidence. G12 release policy
remains open. No public paired Docker release is available from these files yet.

An extracted candidate needs Docker Desktop or Docker Engine with Compose v2.
The host does not need Python, Node, Bun, a source checkout, PostgreSQL, or
Redis. Keep the extracted files together, including `manifest.json` and its
detached `manifest.sig`.

On macOS or Linux, run `./start.sh` from the extracted directory, then open
the printed loopback URL. On a desktop, the helper also tries to open the
browser after the control probe verifies running identities, gateway paths,
a fresh cookie-only session, and its revocation. Set `TLDW_APP_NO_BROWSER=1`
to suppress that. Use `./status.sh` to inspect the services and
`./stop.sh` to stop them while retaining data. On Windows PowerShell, use
`./start.ps1`, `./status.ps1`, and `./stop.ps1`. Each script finds its bundle
relative to its own location, so the current working directory does not
matter. Windows helpers remain provisional: syntax checks do not qualify real
Windows Docker operation, private-state ACLs, or lifecycle/readiness behavior.
Do not treat the Linux container architecture matrix as Windows host validation.

The first start creates private, persistent instance identity and credentials
under `~/Library/Application Support/tldw/app` on macOS,
`${XDG_DATA_HOME:-~/.local/share}/tldw/app` on Linux, or
`%LOCALAPPDATA%\tldw\app` on Windows. Set `TLDW_APP_STATE_DIR` before the first
start to choose another location. Set `TLDW_APP_PUBLIC_PORT` before the first
start to use a free port other than 8080; later runs must keep that choice.
Before saving a first origin, a disposable Docker container checks the binding.
If 8080 is occupied, the helper offers an available alternative to set explicitly
and retry. An occupied explicit choice leaves no instance configuration, so a
retry with another port works. Port availability can change before startup;
established installations never silently change their saved origin.
The host publishes only the gateway at `127.0.0.1:<port>`. Backend and WebUI
ports stay on the private Compose network.

The pinned control image verifies the manifest signature, platform, local
artifact hashes (including Compose, README, and all six helpers), required image
references, and existing state before any
application container starts. The helper then pulls the image references by
their signed OCI digests. Inherited managed Compose values are cleared, so
images, credentials, cookie names, and the origin resolve from verified state.
Readiness uses only the pinned control runtime on the private network and sends
no master key in its HTTP requests. Failed readiness triggers stack cleanup;
failed cleanup reports the retained state/resource needed for recovery. `stop` leaves both named data volumes intact. Keep
the private state directory and Docker volumes together when backing up or
moving an installation; the directory includes the instance API key.

Updates, backup/restore, and optional component replacement are not included
in this WP1 bundle. They require the separately planned host-helper work;
stopping and restarting this candidate retains the named volumes.
