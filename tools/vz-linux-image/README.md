# vz_linux Reference Image

This directory holds the first reproducible local image/rootfs path for the
`vz_linux` helper and guest-agent roadmap.

## Current Scope

The current slice is intentionally narrow:

- build the guest-mode `tldw-agent`
- install it plus the guest service and workspace mount units into a rootfs-like directory
- emit a canonical bundle directory with `manifest.json`, `kernel`, optional
  `initrd`, and `rootfs.img`
- verify the staged layout with a smoke-check script

This is not yet a full distro image factory. It is the reproducible artifact path
that later VM/image work will consume.

## Inputs

- `TLDW_VZ_LINUX_IMAGE_ROOTFS`
  - rootfs directory to install into and verify
- `TLDW_VZ_LINUX_BUNDLE_KERNEL`
  - source kernel file to copy into the canonical bundle
- `TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE`
  - source rootfs disk image to copy into the canonical bundle
- `TLDW_VZ_LINUX_BUNDLE_INITRD`
  - optional initrd file to copy into the canonical bundle

## Builder Profiles

The Debian builder path uses repo-owned plain-text package profiles:

- `profiles/minimal.packages`
  - canonical reference image profile
- `profiles/debug.packages`
  - additive troubleshooting profile layered on top of `minimal`

Defaults and profile composition helpers live in
`scripts/builder-defaults.sh`.

Current pinned defaults:

- suite: `bookworm`
- architecture: `arm64`
- kernel package: `linux-image-arm64`

## Debian Rootfs Builder

`scripts/build-debian-rootfs.sh` is the first Linux-native entrypoint for
turning Debian inputs into a prepared rootfs directory.

Example dry run:

```bash
./scripts/build-debian-rootfs.sh --dry-run --profile minimal --output-rootfs /tmp/vz-rootfs
```

Real execution is Linux-only and currently expects root privileges for
`debootstrap`, chrooted package installation, and rootfs preparation.

The canonical staging path now also installs boot/debug affordances into the
rootfs:

- `/etc/modules-load.d/vsock.conf`
- `serial-getty@ttyS0.service` enablement
- `tldw-agent-guest.service`
- `workspace.mount`

## Rootfs Image Packing

`scripts/pack-rootfs-image.sh` turns a prepared rootfs directory into
`rootfs.img`.

The canonical packing path is directory-to-ext4 via `mke2fs -d`, which keeps
the source rootfs directory intact and avoids turning image packing into a
loop-mount-only workflow.

Example dry run:

```bash
./scripts/pack-rootfs-image.sh --dry-run --rootfs /tmp/vz-rootfs --output-image /tmp/rootfs.img
```

## Kernel And Initrd Extraction

`scripts/extract-kernel-artifacts.sh` copies the booted kernel and matching
initrd out of a prepared rootfs into the canonical bundle artifact names:

- `kernel`
- `initrd`

Example dry run:

```bash
./scripts/extract-kernel-artifacts.sh --dry-run --rootfs /tmp/vz-rootfs --output-dir /tmp/vz-boot
```

## Top-Level Debian Bundle Builder

`scripts/build-debian-bundle.sh` orchestrates the whole canonical builder flow:

1. build a Debian rootfs directory
2. pack it into `rootfs.img`
3. extract `kernel` and `initrd`
4. assemble the final canonical bundle
5. emit `build-info.json`

Example dry run:

```bash
./scripts/build-debian-bundle.sh --dry-run --output-dir /tmp/vz-linux-build
```

The dry run still writes `build-info.json` so provenance can be inspected
without a privileged Linux build host.

## Container Wrapper

`scripts/run-linux-builder-container.sh` is a thin wrapper around the native
Linux builder. It exists for execution convenience only and must not become a
second build implementation.

Example dry run:

```bash
./scripts/run-linux-builder-container.sh --dry-run --output-dir /tmp/vz-linux-build
```

## Quick Start

```bash
ROOTFS="$(mktemp -d "${TMPDIR:-/tmp}/vz-linux-rootfs.XXXXXX")"
KERNEL="$(mktemp "${TMPDIR:-/tmp}/vz-linux-kernel.XXXXXX")"
ROOTFS_IMG="$(mktemp "${TMPDIR:-/tmp}/vz-linux-rootfs-img.XXXXXX")"
BUNDLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vz-linux-bundle.XXXXXX")"

printf 'kernel' > "${KERNEL}"
printf 'rootfs' > "${ROOTFS_IMG}"

./scripts/install-agent.sh "${ROOTFS}"
TLDW_VZ_LINUX_IMAGE_ROOTFS="${ROOTFS}" bash ./scripts/smoke-check.sh
TLDW_VZ_LINUX_IMAGE_ROOTFS="${ROOTFS}" \
TLDW_VZ_LINUX_BUNDLE_KERNEL="${KERNEL}" \
TLDW_VZ_LINUX_BUNDLE_ROOTFS_IMAGE="${ROOTFS_IMG}" \
bash ./scripts/build-bundle.sh "${BUNDLE_DIR}"
```

## Expected Layout

```text
<rootfs>/
  workspace/
  usr/
    local/
      bin/
        tldw-agent-guest
  etc/
    systemd/
      system/
        tldw-agent-guest.service
        workspace.mount
        multi-user.target.wants/
          tldw-agent-guest.service
          workspace.mount

<bundle>/
  manifest.json
  kernel
  rootfs.img
  initrd  # optional
```

The install script now also stages `/workspace`, installs `workspace.mount`,
and enables both `workspace.mount` and `tldw-agent-guest.service` by creating
the expected `multi-user.target.wants/` symlinks inside the rootfs.

Guest-side kill-on-output-cap requires images rebuilt with the updated
`tldw-agent-guest` binary. Older images still boot and execute, but only the
host helper response cap is guaranteed.

## Helper Smoke

The canonical bundle can also drive the repeatable host-side E2E smoke:

```bash
./scripts/run-host-e2e-smoke.sh --dry-run --bundle "${BUNDLE_DIR}"
runtime_dir="$(mktemp -d "${TMPDIR:-/tmp}/tldw-vz-helper-e2e.XXXXXX")"
trap 'rm -rf "${runtime_dir}"' EXIT
chmod 700 "${runtime_dir}"

./scripts/run-host-e2e-smoke.sh \
  --bundle "${BUNDLE_DIR}" \
  --socket "${runtime_dir}/helper.sock" \
  --serial-log-dir "${runtime_dir}/serial"
```

On a prepared Apple silicon macOS host, that script builds the Swift helper
when needed, signs it with `--entitlements` unless the helper is already signed
with `com.apple.security.virtualization`, runs the helper-daemon bundle smoke,
starts a helper daemon for the Python sandbox runtime, runs real `vz_linux`
ephemeral execution, verifies same-session VM reuse, verifies recovery
diagnostics plus dry-run reconciliation repair planning, and stops the helper
on exit. The recovery step is non-destructive: it uses isolated test-store
metadata and does not terminate VMs, delete session controls, or run image-store
cleanup.

The helper refuses sockets whose parent directory is not owner-only. Do not put
the helper socket directly under `/tmp`; use the script defaults or create a
private `0700` runtime directory as shown above.

The lower-level helper smoke remains available for focused debugging:

```bash
TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1 \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH="${BUNDLE_DIR}" \
source ../../.venv/bin/activate && \
python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py -q
```

Use the direct module command for focused helper or bundle validation. Use
`run-host-e2e-smoke.sh` for the full operator workflow because it also covers
real sandbox execution, session VM reuse, and recovery dry-run planning.

The same script is the entrypoint for the host-gated GitHub Actions workflow at
`.github/workflows/vz-linux-host-gated.yml`. That workflow is intentionally
limited to prepared self-hosted Apple silicon runners labeled
`self-hosted`, `macOS`, `ARM64`, and `vz-linux`; normal hosted CI does not run
real VZ execution. The job is branch-gated to `main` and `dev` so manual
dispatch cannot run arbitrary feature-branch code on the self-hosted host. Set
repository variable
`TLDW_SANDBOX_VZ_LINUX_HOST_GATED_NIGHTLY=1` to enable the scheduled run, and
set `TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH` or pass the manual `bundle_path` input
to point at the canonical bundle on the runner.

See `Docs/Sandbox/vz-linux-host-gated-ci-acceptance-policy.md` for the
workflow acceptance policy, including manual/nightly gates, expected skips,
artifact upload expectations, branch allowlisting, and blocking regression
criteria.

## Image Store Registration

Canonical bundles can be registered in the sandbox image store for durable local
inventory, artifact hashes, build provenance, and run-clone planning:

```python
from pathlib import Path

from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore

store = SandboxImageStore(root_path="/var/lib/tldw/sandbox-images")
template_id = store.register_bundle(
    runtime="vz_linux",
    template_name="debian-bookworm-arm64",
    bundle_path=Path("/path/to/vz-linux-bundle"),
    labels={"suite": "bookworm", "profile": "minimal"},
)
```

The store writes:

```text
<image-store-root>/templates/vz_linux/debian-bookworm-arm64/manifest.json
```

The manifest records `kernel`, `rootfs.img`, optional `initrd`, artifact sizes,
SHA-256 hashes, labels, source path, registration time, and `build-info.json`
provenance when present. Canonical bundle registration writes
`artifact_format="tldw_bundle"`.

The manifest also has optional OCI/source provenance fields such as
`oci_image_ref`, `oci_platform`, manifest/config/layer digests, `registry`, and
`imported_at`. These fields are metadata scaffolding only; the helper still
boots the repo-owned bundle path and remains the source of truth for
bootability.
