# vz_linux Reference Image

This directory holds the first reproducible local image/rootfs path for the
`vz_linux` helper and guest-agent roadmap.

## Current Scope

The current slice is intentionally narrow:

- build the guest-mode `tldw-agent`
- install it plus the guest service unit into a rootfs-like directory
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
  usr/
    local/
      bin/
        tldw-agent-guest
  etc/
    systemd/
      system/
        tldw-agent-guest.service

<bundle>/
  manifest.json
  kernel
  rootfs.img
  initrd  # optional
```

The systemd unit is staged as the canonical guest-startup asset. The actual
long-lived in-guest transport wiring is completed in later VM boot and guest
execution slices.

## Helper Smoke

The canonical bundle can also drive the host-gated helper smoke:

```bash
TLDW_SANDBOX_MACOS_HELPER_DAEMON_SMOKE=1 \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_SMOKE=1 \
TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH="${BUNDLE_DIR}" \
source ../../.venv/bin/activate && \
python -m pytest tldw_Server_API/tests/sandbox/test_macos_virtualization_helper_daemon_host_gated.py -q
```

At the current slice, a prepared host should validate the canonical bundle and
move past the old `boot_not_implemented` path. The likely remaining ceiling is
guest readiness until the real vsock guest transport is wired.
