# VZ Host-Gated Evidence Summary Advisory Design

**Date:** 2026-06-17
**Status:** Design slice
**Task:** `TASK-2381`

## Goal

Add an advisory, host-independent summary step for VZ Linux host-gated smoke
evidence. Operators should be able to open the GitHub Actions run and see a
small Markdown summary of what evidence was produced, which files are missing,
what the smoke wrapper reported as its final exit code, and where to look next.

This is a reporting layer only. It must not change helper startup, VM
execution, image-store cloning, evidence generation, artifact upload semantics,
or the host-gated trust model.

## Current State

The smoke wrapper writes structured evidence under:

```text
<runtime-dir>/evidence
```

The host-gated workflow uploads that directory as
`vz-linux-host-gated-evidence`, and uploads narrowed helper logs separately as
`vz-linux-host-gated-helper-logs`.

That gives operators the right artifacts, but the first diagnostic step is
still manual: download the evidence artifact, inspect JSON/hash/status files,
then decide whether raw helper logs are needed. A GitHub step summary can make
the common case faster without changing any VZ execution path.

## Design

Add a portable evidence summarizer script that accepts an evidence directory and
writes Markdown to the GitHub step summary file when available:

```bash
python tools/vz-linux-image/scripts/summarize-host-e2e-evidence.py \
  --evidence-dir "${RUNNER_TEMP}/tldw-vz-helper-ci/evidence"
```

The implementation should default to `$GITHUB_STEP_SUMMARY` when it is set and
append to that file for GitHub Actions runs. It should fall back to stdout for
local/operator runs. The script should be usable on any host with Python and
must not require Virtualization.framework, a prepared VZ bundle, or a running
helper.

The summarizer is a read-only diagnostic tool. It must not create, modify, or
delete the evidence directory or any evidence file.

The workflow should run the summarizer with `if: always()` after the smoke
wrapper has had a chance to finalize evidence and before or near the artifact
upload steps. The summary step must exit `0` in this first advisory slice, even
when the evidence directory is missing or malformed. GitHub Actions will still
preserve the primary smoke failure because the failing smoke step remains the
job's authoritative result.

## Evidence Contract

The summary should treat these files as expected evidence:

- `host-smoke-evidence.json`
- `source-bundle-hashes-before.txt`
- `source-bundle-hashes-after.txt`
- `run-bundle-hashes.txt`
- `runtime-paths.txt`
- `cleanup-status.txt`

The implementation should inspect only these direct child paths under the
configured evidence directory. It should use `lstat`-style checks, skip
symlinks and non-regular files with warnings, and cap JSON reads so a corrupt
or malicious file cannot exhaust memory in an operator diagnostic path.

When `host-smoke-evidence.json` is present and valid, the summary should report:

- evidence directory path
- evidence file presence checklist
- final smoke exit code when recorded
- phase outcomes when recorded
- cleanup status when recorded
- run/source hash file presence
- runtime path pointers when recorded
- artifact/log pointers when recorded

The summary should not parse or embed raw serial logs, helper stdout/stderr, or
guest command output. It may mention paths, file names, sizes, and status values
already present in the evidence bundle.

## Error Handling

Missing evidence directory:

- write a warning summary that evidence is unavailable
- list the expected evidence directory
- explain that this may indicate an early setup/preflight failure
- exit `0`

Missing `host-smoke-evidence.json` with some files present:

- write a warning summary that structured metadata is missing
- still show the file presence checklist
- point operators to the evidence artifact and helper-log artifact
- exit `0`

Malformed JSON:

- write a warning summary that JSON could not be parsed
- include only the parse error class/message, not raw file contents
- still show the file presence checklist
- exit `0`

Unexpected filesystem errors while probing evidence:

- degrade to a warning row for the affected file/path
- avoid recursive directory traversal outside the configured evidence directory
- treat symlinks, directories, devices, and oversized JSON as unavailable
  evidence rather than reading through them
- exit `0` in advisory mode

## Workflow Contract

The host-gated workflow should preserve this order:

1. Run managed host smoke.
2. Run advisory evidence summary with `if: always()`.
3. Upload `vz-linux-host-gated-evidence` with `if: always()`.
4. Upload `vz-linux-host-gated-helper-logs` with `if: always()`.

The summary step must not use broad runtime globs and must not change the
artifact paths. It should read only the configured evidence directory.

## Risk Review

- Advisory mode can hide defects if operators treat warnings as success.
  Mitigation: label the summary clearly as advisory and keep the smoke step as
  the only authoritative pass/fail result in this slice.
- Summary output can leak sensitive data if it embeds logs. Mitigation: never
  include raw log contents, helper stdout/stderr, guest output, or environment
  variables.
- A diagnostic reader can become a filesystem probe if it follows arbitrary
  links. Mitigation: inspect only known direct child evidence files, skip
  symlinks/non-regular files, and avoid recursive traversal.
- Missing evidence after a smoke failure is still useful signal, but failing the
  summary step would obscure the original failure. Mitigation: always exit `0`
  for malformed/missing evidence until a later strict-mode design is approved.
- Parsing too much evidence schema now would make future schema changes harder.
  Mitigation: use optional reads with graceful fallbacks and treat unknown JSON
  fields as ignored.
- Running the summary before evidence finalization would produce false missing
  warnings. Mitigation: place it after the smoke wrapper command, which owns
  evidence finalization through its exit trap.

## Tests

Portable tests should cover:

- complete evidence directory produces Markdown with final exit code, phase
  outcomes, cleanup status, expected file checklist, and artifact/log pointers
- missing evidence directory produces a warning summary and exits `0`
- malformed `host-smoke-evidence.json` produces a warning summary, keeps the
  file checklist, and exits `0`
- partial evidence without JSON still summarizes present/missing files
- workflow contract includes an always-run summary step pointed at
  `${{ runner.temp }}/tldw-vz-helper-ci/evidence`

Real VZ execution remains host-gated/manual and is not required for normal CI.

## Non-Goals

- Failing CI based on evidence summary quality.
- Downloading or parsing GitHub artifacts through the GitHub API.
- Parsing raw serial/helper logs.
- Changing smoke evidence schema or wrapper evidence finalization.
- Changing helper lifecycle, VM provisioning, image-store clone behavior, or
  cleanup semantics.
- Adding dashboards, PR comments, or automatic issue updates.

## Future Work

A later strict-mode slice can make missing or malformed evidence fail
host-gated runs once prepared-host evidence quality is stable. Another later
slice can aggregate evidence summaries into the prepared-host evidence tracker,
but this advisory slice should stay limited to local files and GitHub step
summary output.
