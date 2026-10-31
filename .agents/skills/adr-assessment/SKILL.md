---
name: adr-assessment
description: Assess durable architecture decisions while drafting repo specs, plans, and PRs; link an existing ADR or record why no ADR is needed.
---

# ADR Assessment

Use this repo-local workflow when a spec, implementation plan, or PR may establish or change a durable architecture rule.

1. Search `Docs/ADR/README.md` and the relevant module docs for an existing decision.
2. Decide whether the work changes a lasting rule for module boundaries, public APIs, persistence, security, worker ownership, provider integration, WebUI/extension conventions, major dependencies, or repository workflow. Follow the criteria in `Docs/ADR/README.md`.
3. Record a short `ADR check` in the spec or plan and in the Backlog task: `ADR required: yes/no`, `ADR path: ...` when applicable, and one sentence explaining the choice. Carry the result into the PR description.
4. If an accepted ADR already governs the work, link it. If the decision changes, draft a new ADR with `Docs/ADR/000-template.md` and supersede the old record; do not rewrite accepted rationale.
5. If this is a historical backfill, verify current implementation and owner decision first. Use `Status: Accepted` only for a still-governing decision and record `Backfilled from:`. Keep uncertain candidates in the inventory.

Use the same assessment for a substantial Superpowers spec before its implementation plan is written. Small fixes with no durable rule need only a brief `ADR required: no` reason; they do not need a new record.
