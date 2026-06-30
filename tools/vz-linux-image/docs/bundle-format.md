# VZ Linux Canonical Bundle Format

The canonical `vz_linux` bundle is a directory with these required files:

- `manifest.json`
- `kernel`
- `rootfs.img`

Optional files:

- `initrd`

The manifest fields are:

- `bundle_version`
- `boot_mode`
- `kernel`
- `initrd`
- `rootfs`
- `guest_agent_path`
- `workspace_mount_tag`
- `vsock_port`

This bundle is the primary artifact the repo will build and validate. Raw
self-booting disk images remain a compatibility path handled separately by the
helper.
