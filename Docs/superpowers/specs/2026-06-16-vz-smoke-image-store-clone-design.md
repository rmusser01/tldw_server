# VZ Smoke Image-Store Clone Design

## Context

The 2026-06-16 prepared-host failure-drill evidence packet proved the real
`vz_linux` smoke path and failure drills, but it also exposed a stability gap:
the lower-level host smoke wrapper passed the canonical source bundle directly
to VM-executing stages. Real Virtualization.framework boots can update the
`rootfs.img` mtime and hash, so evidence collected after a run can no longer
prove that the canonical source bundle stayed immutable.

The existing image-store design already gives the right long-term direction:
register canonical templates, plan per-run clones, record provenance, and garbage
collect run artifacts separately. This slice should use that architecture
without broadening into full runner/session image-store migration.

## Goals

- Make `tools/vz-linux-image/scripts/run-host-e2e-smoke.sh` stop passing the
  canonical `--bundle` directory to any VM-executing stage by default.
- Materialize a private, disposable image-store run bundle for each smoke run.
- Keep the host smoke command as the single operator workflow.
- Preserve current dry-run and fake-helper testability.
- Make evidence packets distinguish source bundle hashes from disposable run
  bundle hashes.

## Non-Goals

- Do not migrate `VZLinuxRunner` session execution to mandatory image-store
  templates in this PR.
- Do not add image-store GC execution, background cache policy, or launchd
  integration.
- Do not add new hosted CI triggers or destructive scheduled drills.
- Do not require operators to pre-register templates before using smoke.

## Design

Add a small smoke-bundle materializer under `tools/vz-linux-image/scripts/`.
The materializer is a command-line bridge from the shell wrapper to
`SandboxImageStore`:

1. It accepts `--source-bundle`, `--store-root`, `--run-id`, and optional
   `--template-name`.
2. It creates/opens `SandboxImageStore(store_root)`.
3. It registers the source bundle as `vz_linux:<template-name>` with
   `allow_existing=True`.
4. It calls `prepare_run_clone(template_id=..., run_id=...)`.
5. It materializes each planned clone item into `<store-root>/runs/<run-id>/`.
6. It copies bundle metadata files such as `manifest.json` and `build-info.json`
   into the run directory so the run directory is a valid bundle.
7. It prints the run bundle path on stdout.

The shell wrapper keeps `--bundle` as the canonical source input, but internally
tracks two paths:

- `SOURCE_BUNDLE_PATH`: the operator-provided canonical bundle.
- `BUNDLE_PATH`: the disposable run bundle passed to helper bundle smoke, real
  host smoke, and optional failure drills.

Dry-run must not create directories or clone files. It should print the
materializer command that would run, set `BUNDLE_PATH` to the deterministic run
bundle path, and print later commands with that disposable path. This preserves
operator visibility and catches accidental direct-source regressions in tests.

## Clone Semantics

The materializer should prefer APFS copy-on-write clone behavior on macOS using
the platform `clonefile(2)` syscall when available. If COW clone fails or is
unavailable, it should fall back to `shutil.copy2()` so Linux test hosts and
non-APFS volumes still work. A fallback copy is acceptable for this slice
because the safety property is source immutability; COW performance is an
optimization.

The run directory should be private when created through the wrapper because it
lives under the existing owner-only runtime directory. The materializer should
still create its store/run directories with owner-only permissions where
possible to preserve the trust boundary if called directly.

## Error Handling

- Missing source bundle artifacts remain fatal before helper startup.
- A materialization failure must abort before helper startup or VM execution.
- The cleanup trap should continue to stop the helper and remove the socket; it
  does not need to delete the image-store run directory because it is under the
  operator-selected runtime/artifact root and is useful evidence.
- If `cp -c` fails, the fallback copy should not hide permission or missing-file
  errors from the fallback path.

## Test Strategy

- Add tests for the materializer itself: it registers a bundle, writes a run
  clone manifest, creates a runnable bundle directory, preserves source file
  contents, and does not mutate source artifacts.
- Add shell-wrapper tests proving dry-run emits the materializer command and
  later VM stages use the disposable run bundle instead of the source bundle.
- Update fake-helper shell tests so the new pre-pytest materializer invocation
  is represented without booting a VM.
- Keep existing host-gated and workflow contract tests portable.

## Documentation Updates

- Update `tools/vz-linux-image/README.md` and operator notes to explain that
  `--bundle` is now a source bundle and that VM stages use a disposable
  image-store run bundle.
- Update the prepared-host evidence tracker guidance to record both source
  bundle hashes and run bundle hashes when available.
- Leave the direct-bundle mutability residual gap until fresh prepared-host
  evidence proves the new smoke path preserves the source bundle.
