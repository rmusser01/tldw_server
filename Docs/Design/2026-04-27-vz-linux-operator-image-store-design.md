# vz_linux Operator Workflow And Image Store Design

## Goal

Make the merged `vz_linux` real-execution path repeatable for operators and less dependent on ad hoc `/tmp` bundle paths.

This PR intentionally focuses on two foundations:

- a single host-side smoke workflow that proves helper build/sign/start, bundle validation, ephemeral execution, session VM reuse, and helper shutdown
- a versioned image-store manifest path for canonical `vz_linux` bundles with hashes, provenance, cache listing, and safe garbage-collection planning

Full launchd installation, hosted Apple Silicon CI automation, and deeper runtime policy hardening are follow-up work. This keeps the next PR reviewable and minimizes risk after the large real-execution merge.

## Current State

`vz_linux` can now execute commands inside a real Linux VM on prepared Apple silicon macOS hosts. The helper, guest agent, vsock path, and real host E2E tests work, but the operational flow is still manual:

- helper build and codesign are separate commands
- helper daemon startup uses hand-picked sockets
- real E2E env variables must be assembled manually
- image bundles are usually referenced from temporary directories
- `SandboxImageStore` is an in-memory manifest stub, not a durable operator-facing bundle catalog

## Architecture

### Operator Smoke Workflow

Add `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` as the single local operator entrypoint. The script should:

- accept `--bundle PATH`, `--socket PATH`, `--serial-log-dir PATH`, `--helper PATH`, and `--entitlements PATH`
- support `--dry-run` to print the commands without starting VMs
- build the Swift helper when requested or when the helper binary is missing
- optionally codesign the helper with the supplied entitlements
- start the helper with an isolated socket and serial-log directory
- run the helper-daemon host smoke against the bundle
- run the real host E2E tests for ephemeral execution and session VM reuse
- stop the helper on exit via a trap

The script should not hide failures. If helper build, signing, validation, VM boot, exec, or reuse fails, the command exits non-zero with enough context to reproduce the failed stage.

### Image Store

Harden `tldw_Server_API/app/core/Sandbox/image_store.py` into a small filesystem-backed manifest store. The store should remain simple and local, not a full image registry.

The Apple [`container`](https://github.com/apple/container) and
[`containerization`](https://github.com/apple/containerization) projects are
relevant prior art for this layer because they use OCI-compatible images for
macOS-hosted Linux VM workloads. This design still keeps the near-term
repo-owned bundle format, but future image-store changes should avoid
assumptions that would block OCI manifests, layer digests, registry provenance,
or a later `vz_oci_linux` runtime from reusing the same run/session/audit model.

The store layout should be deterministic:

```text
<root>/
  templates/
    <runtime>/
      <name>/
        manifest.json
  runs/
    <run_id>/
```

Template manifests should capture:

- schema version
- template id
- runtime
- template name
- source path
- registered timestamp
- artifact paths
- artifact size and SHA-256 hash
- optional build provenance from `build-info.json`
- optional user labels

Registration should validate that artifact paths exist and should compute hashes at registration time. The helper remains the source of truth for bootability; the image store only owns inventory, provenance, and deterministic clone manifest planning.

### Documentation

Update operator docs to show:

- the single smoke command
- how to choose socket and serial-log paths
- how to pass local entitlements for ad hoc signing
- how to register/list/inspect image-store bundles
- what remains manual, especially launchd setup and CI runner provisioning

## Error Handling

- The smoke script should fail fast per stage and always stop the helper if it started one.
- Image-store registration should use explicit exceptions for missing paths, invalid manifests, duplicate template ids, and unsafe GC plans.
- GC should default to dry-run planning. Actual deletion must be explicit.

## Testing

The first PR should include:

- unit tests for image-store registration, durable reload, artifact hashes, provenance capture, clone manifest generation, and GC planning
- shell-script tests for dry-run behavior, required argument validation, and generated pytest/helper commands
- existing helper and real host E2E tests remain opt-in and are invoked by the smoke script on prepared hosts

## Deferred Work

- launchd install/uninstall commands
- managed helper auto-upgrade
- Apple Silicon host-gated GitHub Actions runner wiring
- APFS clone execution
- runtime policy hardening beyond manifest provenance and safer operator defaults
