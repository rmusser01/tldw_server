# VZ Host-Gated Evidence Artifact Design

**Date:** 2026-06-17
**Status:** Implementation slice
**Task:** `TASK-2332`

## Goal

Make the structured VZ Linux host-smoke evidence bundle first-class in the
host-gated workflow. Operators should be able to inspect one small evidence
artifact before falling back to raw helper logs, while the workflow keeps the
same prepared-host, branch-gated, manual/nightly execution model.

## Current State

`tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` now writes a structured
evidence bundle by default under:

```text
<runtime-dir>/evidence
```

The host-gated workflow currently uploads the entire runtime temp tree as
`vz-linux-host-gated-helper-logs`. That retains evidence indirectly, but it
does not make the evidence bundle obvious in the Actions UI and encourages
operators to inspect noisy raw logs first.

That broad runtime upload is also increasingly risky as the smoke wrapper now
uses a disposable image-store run bundle under the runtime directory by default.
Uploading the whole runtime tree can retain large rootfs clones and other
non-log runtime state that operators do not need for first-pass debugging.

## Design

The workflow should pass an explicit evidence directory under its private
runtime directory:

```bash
evidence_dir="${runtime_dir}/evidence"

bash tools/vz-linux-image/scripts/run-host-e2e-smoke.sh \
  --bundle "${TLDW_SANDBOX_VZ_LINUX_BUNDLE_PATH}" \
  --socket "${runtime_dir}/helper.sock" \
  --serial-log-dir "${runtime_dir}/serial" \
  --evidence-dir "${evidence_dir}" \
  ...
```

The workflow should upload that directory as a separate artifact:

```yaml
- name: Upload smoke evidence
  if: always()
  uses: actions/upload-artifact@<pinned-sha>
  with:
    name: vz-linux-host-gated-evidence
    path: ${{ runner.temp }}/tldw-vz-helper-ci/evidence/**
    if-no-files-found: ignore
```

The helper-log artifact should remain, but it should be narrowed to raw log
locations such as `${{ runner.temp }}/tldw-vz-helper-ci/serial/**`. Raw helper
stdout/stderr and serial logs are still useful when boot or helper startup
fails before evidence finalization. Disposable image-store clones, rootfs
images, sockets, and broad runtime state should not be included in the helper
log artifact.

## Operator Contract

Documentation should define the artifact priority:

1. Inspect `vz-linux-host-gated-evidence` first for structured run metadata,
   hashes, runtime paths, cleanup status, and phase outcomes.
2. Inspect `vz-linux-host-gated-helper-logs` when the evidence artifact is
   missing or when raw helper/serial logs are needed.

Missing evidence on a prepared-host run remains a blocking regression when the
smoke path reaches the wrapper. Early host-preparation failures may still
upload only helper/runtime logs, so `if-no-files-found: ignore` stays correct
for the evidence artifact.

## Risk Review

- Do not broaden workflow triggers or self-hosted runner trust. The job must
  stay manual/nightly and branch-gated to trusted refs.
- Do not upload broader filesystem paths. The evidence upload path must stay
  under `${{ runner.temp }}/tldw-vz-helper-ci/evidence/**`.
- Do not upload disposable image-store bundles or rootfs clones through the
  helper-log artifact.
- Do not remove raw helper logs yet. Evidence is structured, but not a complete
  replacement for failed boot debugging.
- Do not change VM lifecycle, helper signing, image-store cloning, failure
  drills, or pytest markers in this slice.
- Keep third-party actions pinned to immutable SHAs.

## Tests

Portable tests should cover:

- the workflow passes `--evidence-dir "${evidence_dir}"` to the smoke wrapper
- the workflow uploads a separate `vz-linux-host-gated-evidence` artifact with
  `if: always()` and `if-no-files-found: ignore`
- the helper-log artifact remains present but no longer uses the broad
  `${{ runner.temp }}/tldw-vz-helper-ci/**` runtime-tree glob
- docs/policy mention the evidence artifact as the primary operator artifact
  and define raw helper logs as a narrowed fallback artifact

Real VZ execution remains host-gated and is not required for normal CI.

## Non-Goals

- Running a real VZ VM from normal CI.
- Replacing helper-log upload with evidence-only upload.
- Adding artifact parsing, retention dashboards, or GitHub summary generation.
- Changing evidence file schema or wrapper evidence finalization behavior.
