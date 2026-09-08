# Candidate-only FFmpeg wheel remediation

Task: TASK-13013.7.12, child of TASK-13013.7. Related evidence: TASK-13013.7.9.
Status: approach approved; written design awaiting review. No rebuild implemented.

## Decision and boundaries

Rebuild `av==18.1.0` and `opencv-python==5.0.0.93` from verified owning-package
sources against a completely repaired FFmpeg 8.x source baseline. Install those
two complete wheels into a new candidate derived from the retained combined
candidate. Preserve package versions, dependency requirements and capabilities;
identify custom builds by wheel build tags, artifact hashes and provenance, never
by pretending that the upstream binary wheels have changed.

This is an architectural change to candidate construction, not production
adoption. Production Dockerfiles, `pyproject.toml`, `uv.lock`, release admission,
scanner thresholds, ignore lists and VEX remain unchanged. No registry publishing,
dependency removal, OpenCV headless substitution, unrelated workspace edits or
ABI-incompatible library replacement is authorized. The existing standalone
FFmpeg 9 recipe and signed Debian snapshot inputs remain unchanged.

The alternatives are a larger binding port to FFmpeg 9, or individual
non-applicability decisions. Neither is selected: PyAV's installed release
documents FFmpeg 8.x support, and current static evidence does not justify
clearing every reported wheel finding. Changing PATH cannot repair wheel-local
shared libraries.

## Baseline evidence

The retained canonical OCI archive has SHA-256
`864b2e19887641bbe2f6505d0b28f44ccec3cb1cc3d1de4f65063f5211c1c478`;
its subject is
`08f4a090041b1d87d779e1436073910c0b6c4afc2ffcb9a6d957a94c307b45bb`.
Producer: run `34177651966`, commit
`20fbadd2dc23ff3c186153f581a9b1dbd2731b24`, artifact `10038017399`.
The retained image and its successful native Expat/application/standalone-FFmpeg
evidence are immutable inputs, not results transferable to a modified image.

Diagnostic Grype evidence identified 43 wheel/component matches across 22 CVE
IDs: 22 matches against OpenCV's FFmpeg 8.1.1, and 21 against PyAV's FFmpeg 8.1.2.
All remain `needs_review`; these are not 43 independently confirmed exploits.
Retained reports are in
`/private/tmp/task-13013-7-combined-scans.vpkHOr/`, including `wheel-triage.json`,
`wheel-triage.md`, `maintenance-reverse-checks.json` and unchanged scanner JSON.
These temporary paths aid local recovery, but are not durable release evidence.
Implementation must retain hash-bound copies of the input reports and relevant
upstream evidence in its qualification artifact before depending on them in CI.

The coverage ledger must preserve each of the 43 input identities, including both
owning-wheel copies where reported. Its complete CVE set is CVE-2026-8461,
64830–64835, 65703–65706, 66036–66041 and 70628–70632.

## Source-preparation gate

Use an authenticated FFmpeg 8.1 release baseline and an explicitly ordered set of
upstream repairs. Pin the full source identity, archive hash, signature and signer
verification evidence where upstream supplies signatures. Pin owning-package
sources, submodules, build dependencies, compiler image and acquisition tools.
An unsigned source must be identified as such and anchored to reviewed official
repository/registry metadata and content hashes, not described as signed.

The inspected maintenance snapshot `7ba069f4f11d126f52a740156dbab6476a8a865a`
is investigation evidence, not an approved complete patch set. Its archive hash
is `6eb57fb4de774031a491919a90dc74bbdf617d38bfeacbef492586ca2184ddc9`.
Several unsafe source paths remain there. Reverse-apply failure alone does not
establish a missing fix, and a clean reverse check is not behavioral validation.

For every ledger entry record the original component/CVE, source file and unsafe
condition, upstream repair and prerequisites, baseline presence/absence,
resulting source hash, and regression evidence. Resolve the complete floodfill
allocation repair and hqdn3d prerequisite history before preparing the patch
series. Verify MagicYUV's repair in the chosen baseline. Use actual RSCC and
CineForm repairs, not their cited feature-introduction commits. Reject missing
prerequisites, ambiguous coverage, patch drift, partial application or unreviewed
local changes. No compilation or qualification claim proceeds with an incomplete
source ledger. Source resolution is implementation work, not a fabricated pin in
this design.

Preserve all affected code that is built today; do not disable vulnerable codecs,
filters or hardware support as a substitute for repair. An already-fixed or
absent source condition requires explicit evidence, never an automatic scanner
exception. New findings discovered during rebuilding remain blocking for review.

## Build and installation contract

Add candidate-local wheel preparation/build/qualification files under
`Dockerfiles/candidates/ffmpeg-wheels/`, an opt-in candidate workflow under
`.github/workflows/`, and focused tests under `tldw_Server_API/tests/Supply_Chain/`.
Reuse the existing immutable-input and image-identity contracts where suitable;
do not introduce a second general release framework or modify the old producer.

Build on native Linux amd64 with the retained candidate's Python ABI and compatible
system ABI. Use each owning project's supported source build and wheel-repair
process. Produce self-contained repaired wheels with private bundled libraries,
as the existing wheel packaging does. Do not make both bindings depend on a
mutable system FFmpeg or a global loader override. Each library copy must map
back to the repaired source, even when both builds share source preparation.

Pin and hash the full build-tool/dependency closure; prohibit incidental resolution
or fetching during offline compilation. Preserve the build configuration and
transitive native dependencies, including license notices and corresponding
source obligations. Compare each binding's own original build/capability baseline
against its rebuilt wheel, not against the standalone FFmpeg 9 baseline. Unexpected
feature loss or a required package-version change requires a revised design.

Derive a new image from the exact retained candidate. First verify installed wheel
ownership and file inventories. Replace only the two owning distributions using
complete hash-verified local wheels with dependency resolution disabled. Remove
their superseded owned files through the package installation mechanism; reject
ambiguous ownership and leftover old libraries. No broad directory deletion or
bare `.so` swap. Verify all runtime files outside the two reviewed ownership sets
are unchanged, including Python/Expat, application files, other dependencies,
standalone FFmpeg and system packages. Build/test tooling remains outside the
runtime environment. Export a new attested OCI archive and bind its archive,
subject, manifest, config, producer commit and wheel/source hashes together.

## Native qualification

All runtime controls execute against that same new immutable image, non-root,
offline, with a read-only root, dropped capabilities and bounded writable scratch.
No host application or dependency tree may shadow image code. Retain failed runs
and gate-specific exit status. Missing evidence, skipped required tests, crashes,
timeouts and mismatched hashes fail qualification.

Required evidence comprises:

1. Source-level security regressions for every repaired condition, using upstream
   tests or bounded local fixtures. Confirm the original unsafe behavior before
   accepting the repaired result where safely reproducible. Hardware-dependent
   claims require suitable native hardware; software-only CI cannot clear them.
2. PyAV audio decode/resample and PCM, WAV, MP3, FLAC, Opus and AAC round trips;
   Faster-Whisper's actual decode boundary without model downloads; the actual TTS
   streaming writer's supported formats. Require sample/channel/rate/content
   checks, not merely successful imports or subprocess exit status.
3. OpenCV image/color/resize controls through its actual RapidOCR consumer, plus
   retained video read/write capability controls. Use pinned offline fixtures and
   OCR assets where needed; unavailable assets cannot turn into passing skips.
4. Both bindings imported and exercised separately and in both orders within one
   process. Capture actual mapped libraries, paths, hashes and dependency closure
   after media operations. Reject old wheel copies, unexpected system/standalone
   FFmpeg resolution, unresolved dependencies and cross-binding symbol conflicts.
5. Re-run the existing eight application tests with 24 passing phases, import
   provenance, Python/Expat identity, rendering and standalone FFmpeg capability
   controls on the derivative. Preserve the original baselines and accepted
   retirements; previous-image success is not derivative success.
6. Fresh pinned Syft, Trivy and Grype evidence against the exact exported subject,
   with DB identity/timestamps, raw output hashes, and source-derived coverage for
   bundled/static native dependencies that binary cataloging misses. Link both
   wheel SBOMs and source/patch evidence to actual installed binary hashes.

Security review distinguishes a source repair from a scanner's version-only
match. Preserve raw matches and document reconciliation separately; do not edit
results, spoof versions, add ignore rules or change release admission. This task
does not clear unrelated Debian, Expat, Chroma, NLTK or source-inventory findings.
Any required evidence gap remains visible and blocks the corresponding clearance.

## Completion and recovery

Qualification requires the complete source ledger, both wheel artifacts, native
behavior and mapping evidence, unchanged unrelated-runtime inventory, full raw
scanner evidence and an independent security/compatibility review. Production
adoption and PR merge are separate decisions and still require all existing gates.

On failure keep the prior candidate, publish no runtime artifact to a registry,
retain available logs and do not overwrite an earlier evidence bundle or version
tag. Retries use new run/artifact identities. Reassess after three failed attempts
at the same issue rather than weakening a gate.

After written-design approval, prepare
`IMPLEMENTATION_PLAN_task_13013_7_12_ffmpeg_wheels.md` with four stages: verified
source ledger; deterministic owning-wheel builds; isolated derivative and native
controls; source-aware scans and review. Each stage records tests, success criteria
and status. This design approval does not assert that unresolved repairs are
already identified or that the candidate is safe to release.
