# Sandbox Architecture Doctrine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a repo-level sandbox architecture doctrine, align the active `vz_linux` builder plans with it, and expose the doctrine as the canonical reference for future sandbox work.

**Architecture:** Treat the doctrine as a stable subsystem document under `Docs/Sandbox/`, not a milestone plan. Update the active `vz_linux` Debian-builder design and implementation plan to reference the doctrine and absorb the most important cross-project lessons, then surface the doctrine in the sandbox README so future work can find it.

**Tech Stack:** Markdown docs, existing sandbox docs under `Docs/Sandbox/`, current `vz_linux` design/plan docs, repo README-style sandbox documentation

---

### Task 1: Add The Repo-Level Doctrine Document

**Files:**
- Create: `Docs/Sandbox/sandbox-architecture-doctrine.md`
- Test/Verify: manual review

**Step 1: Write the doctrine document**

Include sections for:

- trusted control plane vs untrusted compute
- layered readiness model
- source-of-truth ownership
- fail-closed runtime contract
- host helper vs guest agent protocol boundaries
- canonical artifact path vs compatibility path
- provenance and audit expectations
- lifecycle and reconciliation rules
- runtime-specific guidance for `seatbelt`, `vz_linux`, and `vz_macos`

**Step 2: Review the document for drift risks**

Check that the doctrine:

- does not hard-code one milestone's implementation details
- captures subsystem rules rather than task-specific TODOs
- clearly distinguishes durable architecture from convenience paths

**Step 3: Commit**

```bash
git add Docs/Sandbox/sandbox-architecture-doctrine.md
git commit -m "docs(sandbox): add architecture doctrine"
```

### Task 2: Align The Active Debian-Builder Design With The Doctrine

**Files:**
- Modify: `Docs/Plans/2026-03-11-vz-linux-debian-builder-design.md`

**Step 1: Update the design doc**

Add explicit alignment with the doctrine:

- mention that the builder follows the canonical-artifact rule from the doctrine
- require `build-info.json`
- distinguish canonical bundle output from weaker compatibility artifacts
- require debug affordances such as serial console and vsock module loading
- note that macOS convenience builder flows stay out of scope

**Step 2: Review for consistency**

Verify the updated design still matches the current helper and image-bundle
direction and does not promise APFS/image-store work in the same slice.

**Step 3: Commit**

```bash
git add Docs/Plans/2026-03-11-vz-linux-debian-builder-design.md
git commit -m "docs(vz_linux): align debian builder design with sandbox doctrine"
```

### Task 3: Align The Active Debian-Builder Implementation Plan

**Files:**
- Modify: `Docs/Plans/2026-03-11-vz-linux-debian-builder-implementation-plan.md`

**Step 1: Update the implementation plan**

Adjust the tasks so they reflect doctrine rules:

- `pack-rootfs-image.sh` should prefer directory-to-ext4 packing without
  loop-mount dependence when possible
- builder tasks should include serial-console and vsock module-loading staging
- `build-info.json` should be a required output artifact
- the plan should reference the doctrine as the durable architecture source

**Step 2: Review for TDD/task granularity**

Ensure each task still follows the existing plan style:

- failing test
- run it and confirm failure
- minimal implementation
- rerun and confirm pass
- commit

**Step 3: Commit**

```bash
git add Docs/Plans/2026-03-11-vz-linux-debian-builder-implementation-plan.md
git commit -m "docs(vz_linux): align debian builder plan with sandbox doctrine"
```

### Task 4: Surface The Doctrine In Sandbox Documentation

**Files:**
- Modify: `tldw_Server_API/app/core/Sandbox/README.md`

**Step 1: Add a doctrine reference**

Add a short note in the sandbox README that future runtime work should reference
`Docs/Sandbox/sandbox-architecture-doctrine.md` for subsystem-wide rules on:

- readiness layering
- runtime/source-of-truth ownership
- canonical vs compatibility artifact paths
- audit and provenance expectations

**Step 2: Review for discoverability**

Check that the README points to the doctrine without turning the README into a
duplicate of the doctrine doc.

**Step 3: Commit**

```bash
git add tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(sandbox): surface architecture doctrine"
```

### Task 5: Final Verification And Combined Documentation Commit

**Files:**
- Verify: `Docs/Sandbox/sandbox-architecture-doctrine.md`
- Verify: `Docs/Plans/2026-03-11-sandbox-architecture-doctrine-design.md`
- Verify: `Docs/Plans/2026-03-11-sandbox-architecture-doctrine-implementation-plan.md`
- Verify: `Docs/Plans/2026-03-11-vz-linux-debian-builder-design.md`
- Verify: `Docs/Plans/2026-03-11-vz-linux-debian-builder-implementation-plan.md`
- Verify: `tldw_Server_API/app/core/Sandbox/README.md`

**Step 1: Review the complete doc set**

Check that:

- the doctrine doc is the stable reference point
- the builder docs reference and follow it
- the sandbox README exposes it
- no document contradicts the current helper/builder direction

**Step 2: Run a narrow verification sweep**

Run:

```bash
git diff -- Docs/Sandbox/sandbox-architecture-doctrine.md Docs/Plans/2026-03-11-sandbox-architecture-doctrine-design.md Docs/Plans/2026-03-11-sandbox-architecture-doctrine-implementation-plan.md Docs/Plans/2026-03-11-vz-linux-debian-builder-design.md Docs/Plans/2026-03-11-vz-linux-debian-builder-implementation-plan.md tldw_Server_API/app/core/Sandbox/README.md
```

Expected: only the intended doctrine and alignment changes are present.

**Step 3: Commit**

```bash
git add Docs/Sandbox/sandbox-architecture-doctrine.md Docs/Plans/2026-03-11-sandbox-architecture-doctrine-design.md Docs/Plans/2026-03-11-sandbox-architecture-doctrine-implementation-plan.md Docs/Plans/2026-03-11-vz-linux-debian-builder-design.md Docs/Plans/2026-03-11-vz-linux-debian-builder-implementation-plan.md tldw_Server_API/app/core/Sandbox/README.md
git commit -m "docs(sandbox): codify architecture doctrine"
```
