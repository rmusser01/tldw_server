# VZ Linux Image Bundle And Boot Driver Design

**Date:** 2026-03-10

## Goal

Define the next pragmatic slice required to move `vz_linux` from helper/protocol
scaffolding to real VM boot on Apple silicon macOS hosts. The target is a
deterministic repo-owned canonical Linux image bundle plus a real helper-side boot
driver, while retaining weaker compatibility support for operator-supplied raw
bootable disk images.

## Current State

The repo now has:

- a real Swift helper daemon that serves the helper protocol over a Unix socket
- a real Python helper client that can talk to that daemon
- a migrated `tldw-agent` with guest-mode request/response serving
- a typed guest protocol bridge in the helper
- an opt-in cross-language smoke test proving the Python client can talk to the
  Swift daemon and receive honest `boot_not_implemented` failures

What is still missing is the actual Linux VM boot contract:

- no canonical boot artifact format
- no real helper-side Linux boot driver
- no real `virtiofs` / virtio socket configuration feeding a live VM
- no deterministic way for the helper to validate what `RunSpec.base_image` means

## Recommendation

Support both artifact styles, but make one of them canonical:

1. canonical path: an explicit repo-owned `vz_linux` image bundle
2. compatibility path: a raw self-booting disk image

The helper should strongly validate the canonical bundle and only heuristically
validate the compatibility path. The repo’s own reference-image tooling should
produce the canonical bundle, with a baked-in `systemd` service for
`tldw-agent-guest`.

## Artifact Model

The canonical `vz_linux` artifact should be a directory bundle under
`tools/vz-linux-image/` with at least:

- `manifest.json`
- `kernel`
- optional `initrd`
- `rootfs.img`

The manifest should declare:

- `bundle_version`
- `boot_mode`
- `kernel`
- optional `initrd`
- `rootfs`
- `guest_agent_path`
- `workspace_mount_tag`
- `vsock_port`

The helper should treat this bundle as the primary supported format.

Compatibility mode should still allow a raw self-booting disk image path, but the
helper should mark it as weaker validation and rely more heavily on runtime guest
readiness before declaring the VM healthy. The raw-disk path should stay a distinct
compatibility boot flow instead of pretending to be the canonical bundle format with
missing fields.

## Guest Startup Contract

The canonical bundle should assume a baked-in `systemd` service for
`tldw-agent-guest`.

That means the repo-owned path is:

- Linux boots from the manifest-declared kernel/initrd/rootfs
- `tldw-agent-guest` is already installed in the image
- a `systemd` unit starts it on boot
- the helper knows the expected workspace mount tag and vsock port from the
  manifest

This keeps guest readiness deterministic instead of depending on first-boot
provisioning or cloud-init timing.

## Helper Validation Model

The helper should discover boot mode itself.

### Canonical bundle validation

If the input path is a directory containing `manifest.json`, the helper should:

- parse the manifest
- verify `boot_mode` is supported
- verify `kernel` exists
- verify `rootfs.img` exists
- verify `initrd` exists if declared
- verify required guest metadata exists:
  - `vsock_port`
  - `workspace_mount_tag`
  - `guest_agent_path`

This path should be reported as `validationStrength=strong`.

### Raw disk compatibility validation

If the input path is a file, the helper should:

- verify the file exists
- treat it as `bootMode=raw_disk`
- report weaker guarantees in details/diagnostics

This path should be reported as `validationStrength=compatibility`.

## Helper Boot Flow

The helper should split resolution from boot:

- `BundleTemplateResolver`
  - reads `manifest.json`
  - validates the canonical bundle
  - returns a typed boot spec
- `RawDiskTemplateResolver`
  - validates only file existence and compatibility mode
  - returns a weaker boot spec

Both feed one normalized helper-side boot resolution interface, but not one flat
field bag. The helper should use two explicit boot-spec variants:

- bundle boot spec:
  - `bootMode`
  - `kernelPath`
  - optional `initrdPath`
  - `rootfsPath`
  - `workspaceMountTag`
  - `vsockPort`
  - `guestAgentPath`
  - `validationStrength`
- raw-disk compatibility boot spec:
  - `bootMode`
  - `diskImagePath`
  - `workspaceMountTag`
  - `vsockPort`
  - `guestAgentPath`
  - `bootLoaderKind`
  - `validationStrength`

### Canonical bundle boot path

1. resolve and strongly validate the bundle
2. construct `VZLinuxBootLoader`
3. attach `rootfs.img`
4. configure `virtiofs` using the manifest mount tag
5. configure virtio socket support
6. boot the VM
7. wait for guest-agent readiness over the declared vsock port
8. mark the VM healthy

### Raw disk compatibility path

1. validate the image file exists
2. construct the compatibility boot path for that image type, starting with an EFI
   loader for self-booting disk images
3. boot the VM
4. still require guest-agent readiness before the VM becomes healthy

Python should continue to see only helper protocol results. The complexity stays
inside the helper.

## Reference Image Tooling

`tools/vz-linux-image/` should become the source of the canonical bundle format.

The repo-owned reference-image path should:

- build `tldw-agent-guest`
- install it into a Linux rootfs
- install a `systemd` unit that starts it on boot
- produce `manifest.json`
- produce the canonical bundle layout

Raw disk-image support remains a compatibility path, not the repo’s primary
artifact.

## Testing Strategy

Testing should be staged around the canonical bundle.

### Unit tests

- manifest parsing and validation
- bundle-versus-raw resolver selection
- boot-spec normalization
- helper error mapping for missing kernel/initrd/rootfs/manifest fields

### Native helper tests

- canonical bundle creates a real boot config
- raw disk compatibility creates a compatibility boot config with its own boot-loader
  path
- helper returns honest validation data for both modes

### Reference image tests

- image tooling emits `manifest.json`
- bundle layout is correct
- `tldw-agent-guest` systemd unit is installed in the image assets

### Host-gated integration

- helper-daemon smoke still passes
- canonical-bundle smoke validates the bundle and reaches real VM boot on a
  prepared Apple silicon host
- existing `vz_linux` E2E prefers the canonical bundle path while keeping raw disk
  support available

## Success Criteria

This slice is complete when:

- the repo can produce one canonical `vz_linux` image bundle
- the helper can strongly validate it
- the helper can boot a real Linux VM from it
- guest-agent readiness is reached through the real boot path

After that, the next slice is real guest command execution over the already-built
guest protocol and helper bridge.
