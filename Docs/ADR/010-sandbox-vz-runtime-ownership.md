# ADR-010: Sandbox VZ Runtime Ownership

**Status:** Accepted
**Date:** 2026-06-03
**Backfilled from:** `Docs/Design/2026-05-02-apple-containerization-evaluation.md`
**Decision owner:** Human requester approval of ADR backfill continuation
**Related task:** TASK-515
**Related spec/plan:** `Docs/superpowers/plans/2026-06-03-adr-follow-up-sprint-implementation-plan.md`

## Decision

Keep `vz_linux` as a repo-owned sandbox runtime path instead of requiring Apple `container`, while borrowing narrow OCI metadata and diagnostics ideas and treating networking as a separate policy milestone.

## Context

`vz_linux` already has a first-party execution path: the Python sandbox service owns admission and API behavior, the Swift `macos-vz-helper` owns prepared-host VM lifecycle and guest transport health, the Go `tldw-agent` owns guest command execution, and `SandboxImageStore` owns local bundle inventory and run clone manifests.

Apple `container` and `containerization` provide useful prior art for Apple silicon Linux VMs, OCI image handling, VM service topology, resource diagnostics, and vsock guest control. They are broader than this project needs, though. `tldw_server` is a sandbox control plane, not a Docker-compatible container engine, and the current `deny_all` meaning depends on not attaching a guest network device.

The source evaluation concluded that the project should preserve its working helper and guest-agent boundaries, use Apple projects as reference material, and move only narrow image-store metadata toward OCI compatibility before considering deeper package reuse.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Require Apple `container` for `vz_linux` | Makes the sandbox path depend on an external CLI and newer macOS virtualization/networking assumptions before the project intentionally changes host support. |
| Replace `macos-vz-helper` with Apple `container-apiserver` topology | Imports image, registry, network, and container UX assumptions that would blur sandbox policy ownership and add daemons before the existing helper responsibilities require them. |
| Replace `tldw-agent` with `vminitd` now | Adds guest-protocol churn before the current helper-backed execution path is stable enough; `vminitd` remains useful prior art for supervision and event semantics. |
| Keep image-store metadata entirely bundle-only | Blocks future OCI provenance and registry metadata even though metadata-only scaffolding can be added without changing helper bootability. |
| Enable vmnet networking as part of the same change | Attaching a network device changes the meaning of `deny_all`; networking needs its own policy design, diagnostics, and host-gated tests. |

## Consequences

`vz_linux` remains operator-repeatable without installing Apple `container`. The repo-owned bundle with `manifest.json`, `kernel`, `rootfs.img`, optional `initrd`, and guest-agent metadata remains the canonical near-term artifact for helper bootability.

Apple `container` and `containerization` remain reference material, not default runtime dependencies. Any direct Swift package reuse should be narrow, version-gated, and separately reviewed, starting with OCI manifest/digest parsing before rootfs construction or runtime primitives.

Image-store records may carry optional OCI/source metadata, but helper bootability still comes from template validation. Current bundles should continue to identify as `artifact_format=tldw_bundle`.

`deny_all` for `vz_linux` continues to mean no attached guest network device. Any vmnet allowlist or networking work must start with a separate policy milestone and likely a new ADR.

## Follow-up

- Use this ADR as the covering record for `INV-016`.
- Create a superseding ADR before requiring Apple `container`, replacing `macos-vz-helper` or `tldw-agent`, or changing `vz_linux` network policy semantics.
- Future implementation work can continue the OCI-aware image-store metadata seam without changing VM boot, helper lifecycle, networking, or guest execution.
