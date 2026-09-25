# VZ Missing Agent Startup Drill

Task: TASK-13243.9. This extends the existing opt-in real Apple Silicon failure workflow. The canonical Debian bundle remains a read-only input.

## Stage 1: Establish the fault contract
**Goal**: Distinguish a guest that reached its agent service from a VM that never booted.
**Success Criteria**: A test-only launcher in a disposable rootfs reads a fresh nonce from the guest workspace, atomically publishes a bounded proof matching the requested VM ID, and does not connect to VSock unless the negative-control challenge explicitly permits the preserved original agent.
**Tests**: Portable proof parser rejects missing, stale, malformed, oversized, non-regular, and symlinked files. The normal opt-in cannot enable this drill by itself.
**Status**: Complete

## Stage 2: Integrate the disposable workflow
**Goal**: Install and verify the launcher and preserved agent through the existing offline preparer VM; add one positive and one negative live case to the existing image-store workflow.
**Success Criteria**: An installer verifies both bytes after ext4 modification, and every case uses fresh healthy and fault clones. The negative control changes only the challenge given to the same launcher and proves actual fault-VM execution.
**Tests**: Portable workflow contract tests cover profile registration, installer selection, and exact negative-control acceptance.
**Status**: Complete

## Stage 3: Prove behavior on Apple Silicon
**Goal**: Run the complete opt-in real VM workflow with a signed isolated helper.
**Success Criteria**: The positive case has fresh guest-written startup proof, bounded `guest_transport_timeout`, no guest exec, and no reusable VM/control state. The negative case boots the preserved original agent and executes. Recovery runs two commands in one healthy VM. All VM/helper/disk cleanup and source hashes pass.
**Tests**: Targeted portable pytest, formatter/lint, Bandit, and real host workflow with retained receipt and logs.
**Status**: In Progress

Preflight on 2026-09-25 found that the previously accepted canonical bundle's
`rootfs.img` is absent. Portable tests passed, but live acceptance is pending a
new or restored bootable source image and an isolated real helper run.

## Stage 4: Review and record
**Goal**: Update operator instructions and evidence, review the diff and test output, and commit the substantive slice.
**Success Criteria**: Evidence names the exact input hashes, helper, faults, negative control, cleanup, and any remaining coverage limits. The Backlog task links the artifacts and verification.
**Tests**: Fresh diff check and receipt integrity checks.
**Status**: Not Started

## Design checks

- A transport timeout alone cannot pass: the nonce and VM ID must come from a guest-written workspace proof observed before failed-run cleanup.
- The control cannot pass by bypassing host validation: it boots the same launcher and execs the preserved original agent.
- An installer failure cannot modify the canonical input or silently count as a guest timeout.
- Unexpected helper, source-hash, disk-handle, or cleanup failures fail the entire workflow and retain evidence.
