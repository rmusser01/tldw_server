# vz_linux Reference Image

This directory holds the first reproducible local image/rootfs path for the
`vz_linux` helper and guest-agent roadmap.

## Current Scope

The current slice is intentionally narrow:

- build the guest-mode `tldw-agent`
- install it into a rootfs-like directory at `/usr/local/bin/tldw-agent-guest`
- verify that layout with a smoke-check script

This is not yet a full distro image factory. It is the reproducible artifact path
that later VM/image work will consume.

## Inputs

- `TLDW_VZ_LINUX_IMAGE_ROOTFS`
  - rootfs directory to install into and verify

## Quick Start

```bash
ROOTFS="$(mktemp -d "${TMPDIR:-/tmp}/vz-linux-rootfs.XXXXXX")"
./scripts/install-agent.sh "${ROOTFS}"
TLDW_VZ_LINUX_IMAGE_ROOTFS="${ROOTFS}" bash ./scripts/smoke-check.sh
```

## Expected Layout

```text
<rootfs>/
  usr/
    local/
      bin/
        tldw-agent-guest
```
