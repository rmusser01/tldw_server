# Sandbox Architecture Doctrine Design

**Date:** 2026-03-11

## Goal

Capture the durable architectural lessons from recent sandbox reference reviews
and the current `vz_linux` work as a single repo-level doctrine that future
sandbox plans must reference.

The doctrine should guide:

- `seatbelt`
- `vz_linux`
- `vz_macos`
- sandbox diagnostics
- image/template lifecycle
- future runtime helpers and guest agents

## Problem

The sandbox subsystem now has multiple runtime families and multiple active plan
streams. Without a stable doctrine, the same architectural questions get
re-litigated in every plan:

- where readiness truth lives
- whether diagnostics match runtime truth
- how helpers and guest agents should be split
- how canonical artifacts differ from compatibility paths
- how recovery, audit, and provenance should work

That drift is already becoming visible as the `vz_linux` work grows from runtime
identity to helper protocol, guest transport, and image-building.

## Recommendation

Create one repo-level doctrine doc under `Docs/Sandbox/` and treat it as the
stable architecture layer for the subsystem.

Then update the current `vz_linux` Debian-builder design and implementation plan
to align explicitly with that doctrine.

The doctrine should codify the strongest reusable lessons from the reviewed
projects:

- Bouvet for host-helper and guest-agent separation, explicit image and vsock
  mechanics, and builder debuggability
- Gobii for strict trusted control plane versus untrusted compute, layered
  lifecycle truth, and operational auditability
- Nono for explicit policy ownership, provenance, and single-source-of-truth
  discipline
- CUA/Lume for keeping native virtualization separate from higher-level agent
  behavior
- CodeRunner as a reminder that convenience paths should not become the
  canonical runtime model

## Doctrine Structure

The doctrine should define subsystem-wide rules for:

- trusted control plane vs untrusted compute
- layered readiness model
- source-of-truth ownership
- fail-closed runtime contract
- protocol boundaries and versioning
- image/template and builder provenance
- debug affordances for canonical VM images
- lifecycle, reconciliation, and recovery
- observability and audit expectations

It should not be a milestone plan. It should be a stable reference document that
other plans cite.

## Immediate Impact On Current `vz_linux` Work

The doctrine should immediately tighten the current Debian-builder stream:

- `pack-rootfs-image.sh` should prefer a directory-to-ext4 path that avoids
  loop-mount dependence when possible
- canonical images should always include serial-console and vsock-startup
  affordances
- builder metadata should be required output
- canonical bundle output should be distinct from weaker compatibility artifacts
- the builder plan should explicitly reference the doctrine doc

## Success Criteria

This design is complete when:

- a repo-level doctrine doc exists in `Docs/Sandbox/`
- the current Debian-builder design and implementation plan reference it
- the sandbox README points future runtime work to the doctrine
- future runtime plans have a clear durable reference instead of repeating the
  same architectural arguments
