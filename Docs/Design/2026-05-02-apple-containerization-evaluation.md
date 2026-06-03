# Apple Containerization Evaluation For `vz_linux`

**Status:** Architecture evaluation for future sandbox runtime work.
**Covered by:** `Docs/ADR/010-sandbox-vz-runtime-ownership.md`
**Date:** 2026-05-02.

## Goal

Decide how Apple's `container` and `containerization` projects should influence
the `vz_linux` sandbox roadmap before changing helper, image-store, networking,
or guest-agent implementation.

This is not an implementation plan. It is a decision record for what to adopt,
defer, or reject.

## Sources Reviewed

- Apple [`container`](https://github.com/apple/container) README and technical overview.
- Apple [`containerization`](https://github.com/apple/containerization) README and package layout.
- Apple `container` feature docs for volumes, networking, logs, stats, resources,
  and Linux capabilities.
- Current `tldw_server` sandbox doctrine, operator notes, image-store design,
  helper protocol, canonical bundle format, and guest protocol.

## Current `tldw_server` Baseline

`vz_linux` now has a working first-party path:

- Python sandbox service owns admission, run/session identity, artifacts,
  queueing, audit, and public API behavior.
- Swift `macos-vz-helper` owns host readiness, template validation, VM lifecycle,
  VM status, and guest transport health.
- Go `tldw-agent` owns guest readiness and command execution over vsock.
- `SandboxImageStore` owns local inventory, hashes, provenance, and deterministic
  run clone manifests.
- `deny_all` currently means no guest network device for `vz_linux`.
- The canonical artifact is a repo-owned bundle with `manifest.json`, `kernel`,
  `rootfs.img`, optional `initrd`, and guest-agent metadata.

These seams should remain stable unless a replacement proves materially simpler
and safer.

## Upstream Findings

Apple `container` is a Swift CLI for running Linux containers as lightweight VMs
on Apple silicon. It consumes and produces OCI-compatible images, starts a
managed system service with `container system start`, and is currently tied to
newer macOS virtualization and networking behavior.

Apple `containerization` is the lower-level Swift package. It includes APIs for
OCI images and registries, ext4 filesystem creation, netlink, optimized Linux
kernel/rootfs work, lightweight VM runtime management, process execution, and
Rosetta-backed `linux/amd64` execution on Apple silicon. Its guest init system,
`vminitd`, exposes a gRPC API over vsock for runtime configuration and process
launch.

The technical overview describes a useful service topology:

- CLI client
- `container-apiserver` launch agent
- image/content XPC helper
- vmnet network XPC helper
- per-container Linux runtime helper

That topology is valuable prior art, but it should not be copied wholesale.
`tldw_server` is a sandbox control plane, not a general Docker-compatible
container engine.

## Evaluation Matrix

| Area | Apple approach | Current `tldw_server` approach | Recommendation |
| --- | --- | --- | --- |
| Runtime unit | One lightweight VM per container. | Ephemeral VM or same-session VM reuse per sandbox session. | Keep session-aware policy. Borrow per-workload VM assumptions, not the full container lifecycle model. |
| Service topology | CLI -> launch agent -> image/network helpers -> per-container runtime helper. | Python service -> one Swift helper -> guest agent. | Defer helper splitting until image/network responsibilities grow. Do not create more daemons prematurely. |
| Image format | OCI-compatible images and registry flows. | Repo-owned bundle plus filesystem image-store manifests. | Make image-store metadata OCI-aware next; keep current bundle as canonical near-term. |
| Filesystem generation | ext4 creation/population APIs. | Builder scripts produce rootfs image/bundle. | Evaluate `ContainerizationEXT4` separately as a future builder simplification candidate. |
| Kernel/rootfs | Optimized kernel and minimal rootfs for fast startup. | Debian stable arm64 bundle with debug affordances and systemd guest service. | Benchmark before replacing. Borrow kernel config ideas only after real host smoke remains reliable. |
| Guest control | `vminitd` gRPC over vsock. | `tldw-agent` JSON over vsock. | Keep `tldw-agent`; evaluate vminitd patterns for process supervision, signal, event, and log semantics. |
| Networking | vmnet networks, dedicated IPs, publish/localhost features. | No attached network for `deny_all`; allowlist unsupported. | Keep no-network default. Treat vmnet as future allowlist infrastructure only after policy design. |
| Logs/stats | Container logs, boot logs, resource stats. | Serial logs, helper details, diagnostics, startup warnings. | Borrow operator UX: boot logs and resource stats should become first-class diagnostics. |
| Dependency model | Swift package and CLI under active development. | In-repo helper with no Apple package dependency. | Do not depend on the CLI. Consider package reuse only behind a focused adapter and version gate. |

## Decisions

1. Do not require Apple `container` for `vz_linux`.

The current sandbox path must remain repo-owned and operator-repeatable without
installing Apple's CLI. The CLI is useful for comparison and local experiments,
not as a production prerequisite.

2. Do not replace `macos-vz-helper` with `container-apiserver`.

The helper protocol is already aligned to the sandbox contract. Apple
`container` is broader than needed and brings image, registry, network, and
container UX assumptions that would blur sandbox policy ownership.

3. Keep `tldw-agent` as the guest authority for now.

`vminitd` is strong prior art for a vsock guest-control service, but replacing
the existing guest agent would add churn before the current execution path is
fully stable. Borrow process-supervision and event semantics before considering
replacement.

4. Move image-store metadata toward OCI compatibility.

The next implementation work should add optional fields that can describe OCI
sources without changing current bundle boot:

- `oci_image_ref`
- `oci_platform`
- `oci_manifest_digest`
- `oci_config_digest`
- `oci_layer_digests`
- `registry`
- `imported_at`
- `artifact_format`

These should be metadata-only at first. Helper bootability still comes from
`validate_template`.

5. Treat networking as a separate policy milestone.

vmnet is the right Apple-native family for future `vz_linux` networking, but
attaching a guest network device changes the meaning of `deny_all`. Any vmnet
work must start with a policy design, helper diagnostics, and host-gated tests.

6. Evaluate Swift package reuse in narrow pieces.

If direct reuse is pursued, the safest order is:

1. `ContainerizationOCI` for manifest/digest/registry parsing.
2. `ContainerizationEXT4` for rootfs construction experiments.
3. `Containerization` runtime primitives only after helper API gaps are clear.

Do not add the full package graph as an incidental dependency. The package uses
modern Swift/macOS assumptions, and upstream source stability still needs review
before pinning it in this repo.

## Risks And Problems To Avoid

- Treating `apple/container` as an adversarial sandbox. It is a container engine;
  `tldw_server` still owns untrusted-code policy and audit.
- Requiring macOS 26 before the project intentionally changes host support.
- Silently changing `deny_all` by attaching a vmnet interface.
- Collapsing image-store truth and helper bootability truth into one layer.
- Replacing a working guest protocol before the current E2E path is boring.
- Adding Swift dependencies that make local helper builds slower or less
  predictable without removing meaningful custom code.
- Importing upstream code without preserving Apache-2.0 notices. The repo is
  GPL-3.0-only, so Apache-2.0 is not an obvious compatibility blocker, but
  dependency and notice handling still need review before direct reuse.

## Next Implementation Candidate

The most pragmatic next PR is **OCI-aware image-store metadata scaffolding**:

- extend `TemplateRecord` and persisted manifests with optional OCI/source fields
- keep `register_bundle()` behavior unchanged
- add explicit `artifact_format="tldw_bundle"` for current bundles
- add validation/tests that OCI metadata is bounded, deterministic, and
  diagnostics-safe
- update admin diagnostics and operator docs to show artifact format and OCI
  provenance when present

This creates the migration seam without changing VM boot, helper lifecycle,
networking, or guest execution.

## Deferred Implementation Candidates

- Containerization package spike in a disposable Swift prototype.
- vmnet allowlist/network policy design.
- vminitd vs `tldw-agent` protocol comparison.
- optimized kernel/rootfs benchmark against the current Debian stable arm64
  bundle.
- resource stats and boot-log diagnostics parity with Apple `container` UX.
