# VZ Linux Debian Arm64 Builder Design

**Date:** 2026-03-11

## Goal

Define the most pragmatic long-term-stable path for producing a reproducible,
bootable Debian stable arm64 canonical bundle for `vz_linux`.

This design follows the subsystem-wide rules in
`Docs/Sandbox/sandbox-architecture-doctrine.md`, especially:

- canonical artifact path vs compatibility path
- layered readiness and runtime-owned truth
- required provenance metadata
- debug affordances in canonical VM images

The builder should stay repo-owned, Linux-built, and compatible with the
existing canonical bundle format already consumed by the Swift helper:

- `manifest.json`
- `kernel`
- optional `initrd`
- `rootfs.img`

## Current State

The repo already has:

- guest-agent staging in `tools/vz-linux-image/scripts/install-agent.sh`
- bundle assembly in `tools/vz-linux-image/scripts/build-bundle.sh`
- bundle metadata emission in `tools/vz-linux-image/scripts/write-manifest.sh`
- guest service and workspace mount units in `tools/vz-linux-image/systemd/`

What is still missing is the reproducible Linux image build path itself:

- no Debian rootfs bootstrap flow
- no package-profile model
- no `rootfs.img` packing flow
- no kernel/initrd extraction flow
- no top-level builder that turns Debian inputs into the canonical bundle

## Recommendation

Build the reference image on Linux only, with two supported execution modes:

- canonical: native Linux host or Linux VM
- wrapper: containerized Linux that invokes the same native scripts

The builder should be directory-first:

1. create a Debian arm64 rootfs directory
2. install packages and stage the guest agent/units into that directory
3. pack the rootfs directory into `rootfs.img`
4. extract the kernel and initrd artifacts
5. emit the final canonical bundle
6. emit a small `build-info.json` alongside the output artifacts capturing suite,
   profile, kernel package, and builder inputs

This is the most stable path because it keeps one real implementation and one
thin wrapper, avoids macOS-specific build tricks, and keeps every artifact
inspectable while the VM boot path is still being proven.

## Builder Architecture

The builder should live entirely under `tools/vz-linux-image/`.

Primary flow:

1. create a Debian stable arm64 rootfs directory with `debootstrap`
2. install the selected package profile into that rootfs
3. stage:
   - `tldw-agent-guest`
   - `tldw-agent-guest.service`
   - `workspace.mount`
4. enable the staged units inside the rootfs
5. pack the prepared rootfs directory into `rootfs.img`
6. extract the kernel and initrd from the built rootfs
7. emit the final canonical bundle with the existing manifest contract

The canonical output directory should keep intermediates by default:

- `rootfs/`
- `rootfs.img`
- `kernel`
- `initrd`
- `bundle/`
- `build-info.json`

Later cleanup flags can remove intermediates, but the early builder should favor
debuggability.

The canonical bundle is the strong-validation path. Weaker compatibility inputs
may still exist elsewhere in the subsystem, but this builder produces the
canonical repo-owned artifact family.

## Profiles And Packages

Package profiles should be repo-owned plain text files:

- `tools/vz-linux-image/profiles/minimal.packages`
- `tools/vz-linux-image/profiles/debug.packages`

Rules:

- `minimal` is the canonical reference image
- `debug` is additive on top of `minimal`
- the builder composes package sets by concatenating and de-duplicating package
  names

`minimal` should include only what the current VM path needs:

- `systemd`
- a Debian arm64 kernel package
- `initramfs-tools`
- minimal boot/filesystem utilities
- basic shell/core utilities for smoke commands and service startup

`debug` can add troubleshooting tools like:

- `procps`
- `iproute2`
- `strace`
- `less`
- `vim-tiny`

Kernel package selection should be pinned in one place, not inferred ad hoc.
That selection can live in a small builder config file or as a defaulted script
argument, but it should have exactly one source of truth.

## Script Layout

Add these scripts:

- `tools/vz-linux-image/scripts/build-debian-rootfs.sh`
  - runs `debootstrap`
  - installs selected profile packages
  - stages agent and units
  - enables units
- `tools/vz-linux-image/scripts/pack-rootfs-image.sh`
  - turns the prepared rootfs directory into `rootfs.img`
- `tools/vz-linux-image/scripts/extract-kernel-artifacts.sh`
  - copies out the bootable kernel and initrd
- `tools/vz-linux-image/scripts/build-debian-bundle.sh`
  - top-level orchestration script
  - produces the full output directory plus final canonical bundle
- `tools/vz-linux-image/scripts/run-linux-builder-container.sh`
  - thin wrapper that runs the same builder inside a Linux container

The wrapper should not become a second build implementation. It should only
mount inputs/outputs and call `build-debian-bundle.sh`.

## Operational Constraints

The builder should be explicit about platform and privilege requirements.

Native Linux path may require elevated privileges for:

- `debootstrap`
- chroot/nspawn package installation
- mount operations during package-install or inspection phases

Containerized Linux may require:

- privileged container execution
- mount device access

macOS is not a supported build host for this slice. macOS consumes the resulting
bundle but should not own the build.

If prerequisites are missing, scripts should fail fast with concrete messages
instead of attempting partial fallbacks.

## Testing Strategy

Testing should be layered:

### Unit-Style Script Tests

- package profile resolution
- argument parsing
- manifest emission
- output layout contracts

### Non-Privileged Smoke Tests

- guest-agent staging
- unit staging and enablement
- serial-console and vsock-startup staging
- package-list composition
- bundle-layout validation on fake artifacts

### Privileged Linux Integration

- build a real Debian stable arm64 rootfs
- pack `rootfs.img`
- extract `kernel` and `initrd`
- emit the canonical bundle
- verify that the final rootfs contains the guest agent and enabled units

### macOS Follow-On

- feed the produced bundle into the existing helper daemon smoke
- feed the produced bundle into the existing `vz_linux` E2E smoke

## Success Criteria

This slice is complete when:

- the repo can reproducibly build a Debian stable arm64 rootfs directory on
  Linux
- that rootfs can be packed into `rootfs.img`
- the builder can extract the matching kernel and initrd
- the builder emits the existing canonical bundle format without manual artifact
  assembly
- the builder emits `build-info.json` with enough provenance to explain exactly
  which suite, profile, kernel package, and inputs produced the bundle
- the staged rootfs contains:
  - `tldw-agent-guest`
  - `tldw-agent-guest.service`
  - `workspace.mount`
  - vsock module-loading configuration
  - serial-console enablement
  - enabled unit symlinks
- a prepared macOS Apple silicon host can consume the resulting bundle in the
  helper smoke and `vz_linux` E2E path

## Out Of Scope

This slice should not include:

- macOS-hosted image building
- Apple-container convenience builders on macOS
- APFS clone provisioning
- `vz_macos` image automation
- CI-hosted full image builds
- multiple distro families
- image caching or optimization work beyond basic builder ergonomics
