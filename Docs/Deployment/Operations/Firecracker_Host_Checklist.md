# Firecracker Host Checklist (Sandbox Runner)

Real Firecracker execution is currently blocked by
`firecracker_host_containment_not_implemented`. The runner does not establish
jailer-equivalent namespaces, privilege reduction, or host resource controls.
A configured binary or wrapper is not verified containment evidence. This is a
preparation checklist for future implementation and prepared-host validation,
not a way to enable the current real runner. It refers to:
`tldw_Server_API/app/core/Sandbox/runners/firecracker_runner.py`.

## Host Prerequisites

- [ ] Linux host (real mode hard-fails on non-Linux)
- [ ] `/dev/kvm` exists and is accessible to the service user
- [ ] Sufficient RAM and CPU for expected `SANDBOX_MAX_MEM_MB` and vCPU counts

## Binaries & Permissions

- [ ] `firecracker` installed and executable (`SANDBOX_FC_BIN` if non-default path)
- [ ] `virtiofsd` installed and executable (`SANDBOX_FC_VIRTIOFSD` if non-default path)
- [ ] Service user is in the `kvm` group (or has equivalent access to `/dev/kvm`)
- [ ] SELinux/AppArmor rules allow Firecracker + virtiofsd (if enabled)

## Kernel / RootFS Assets

- [ ] `SANDBOX_FC_KERNEL_PATH` points to a valid Linux kernel image
- [ ] `SANDBOX_FC_ROOTFS_PATH` points to a valid rootfs image (ext4 recommended)
- [ ] Rootfs is readable by the service user (mounted read-only by the runner)
- [ ] Kernel supports virtiofs (required by current runner)

## Guest Boot Flow (Critical)

The runner sets `init=/workspace/entry.sh` and relies on virtiofs. Ensure the
guest can mount `/workspace` before `entry.sh` runs.

- [ ] Provide an initramfs/init that mounts virtiofs, then execs `/workspace/entry.sh`
  OR
- [ ] Boot into a real init inside the rootfs, run a service that mounts virtiofs,
      then execs `/workspace/entry.sh`

## Environment Configuration

- [ ] `SANDBOX_FIRECRACKER_ENABLE_REAL=1` (enables real mode)
- [ ] `TLDW_SANDBOX_FIRECRACKER_FAKE_EXEC` is unset/false
- [ ] Optional: `SANDBOX_FC_BOOT_ARGS` (default adds `init=/workspace/entry.sh`)
- [ ] Optional: `SANDBOX_FC_USE_VIRTIOFS` (defaults to true; disabling currently fails)
- [ ] Optional: `TLDW_SANDBOX_FIRECRACKER_AVAILABLE=1` only for CI overrides (not prod)

## Networking & Isolation

- [ ] No network devices configured (runner is offline by design)
- [ ] Confirm no-network behavior matches the security model

## Smoke Test

- [ ] Start a simple run (e.g., `echo ok`) and verify `.sandbox_status.json`
- [ ] Verify stdout streaming and `run.log` tailing works
- [ ] Verify artifact capture and download
- [ ] Confirm run dir cleanup after completion

## Notes

- Real mode preflight checks: Linux, `/dev/kvm`, `firecracker`, and `virtiofsd`.
- The runner assumes virtiofs; disabling virtiofs currently raises an error.
