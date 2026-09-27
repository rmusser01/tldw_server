# TASK-13376.11: Retain the pending Bun child inside BuildKit

Scope: one private dependency diagnostic for the already approved paired-bundle
qualification. This is not a production fix, candidate retry or release gate.

## Stage 1: Compare the failing and successful boundaries
**Goal**: Choose a different evidence-gathering approach after ordinary retries.
**Success Criteria**: Record the observed difference without guessing a cause.
**Tests**: Inspect exact-source native amd64 log and prior container evidence.
**Status**: Complete

The unchanged local candidate stalled in a BuildKit dependency RUN for 1090
seconds without progress. Native amd64 installed dependencies in 12.74 seconds;
the isolated arm64 container installed 3665 packages in 85.08 seconds. The latter
identified active Puppeteer and canvas scripts, but did not reproduce the hang.
Earlier timed-out probes did not preserve the pending child's script identity.
These observations do not establish a Bun or Puppeteer root cause.

Hypothesis to investigate: the pending process must be captured in the actual
BuildKit RUN boundary; a successful container install cannot identify it.
Preserve argv, cwd, executable, package metadata, CPU/I/O and thread wait states
for owned descendants while keeping the install inputs unchanged.

## Stage 2: Prepare and run one bounded BuildKit diagnostic
**Goal**: Preserve evidence even when BuildKit discards a failed RUN layer.
**Success Criteria**: Unchanged Bun 1.3.2 digest, frozen lockfile and production
COPY inputs; one install with 300-second timeout and 15-second kill grace.
**Tests**: Shell syntax; strict evidence decoder fixtures; private log retention;
ownership-bound outer process guard; baseline Docker inventory preservation.
**Status**: In Progress

Private recipe and evidence live under
`/private/tmp/task13376-buildkit-bun-stall-3ef013bd94`. The recipe emits encoded
records into a private host log. Decode only anchored BuildKit frames with
allowlisted relative paths, validated base64 and conflicting-duplicate rejection.
Do not read process environments, publish raw logs or accept the diagnostic as a
product artifact. The build exports cache only and never tags/loads a candidate.
An outer 420-second guard may terminate only its recorded process group after
verifying that the child is still running and owns that group. Preserve baseline
containers, volumes and images; do not prune shared Docker state.

## Stage 3: Classify evidence and update qualification records
**Goal**: Report actual outcome and remaining uncertainty.
**Success Criteria**: Verified exit, retained snapshots and resource preservation;
no root-cause claim on non-reproduction; any production change separately reviewed.
**Tests**: Evidence completeness/identity checks; `git diff --check`; current native
CI status. Bandit is not applicable to tracked Markdown/task-only changes.
**Status**: Not Started

Update TASK-13376.11, parent TASK-13376 and the existing acceptance review. Commit
the plan and task before removing only this completed plan, preserving its final
version privately and in history. The broader qualification plan stays open.
Do not trigger duplicate native builds by pushing documentation while current
arm64 jobs are still running. Do not retry or cancel those jobs.
