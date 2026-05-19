# Phase 4.2 Deployment Docs Inventory

**Date:** 2026-04-25

**Status:** Inventory complete; docs owner review and edits pending.

## Purpose

Inventory the deployment and getting-started documentation before any Phase 4.2 refresh work. This is a planning artifact only. It identifies the source docs, published mirrors, deployment-mode coverage, and review decisions needed before edits.

## Method

Static file inventory from:

```bash
rg --files Docs/Getting_Started Docs/Deployment Docs/Published/Getting_Started Docs/Published/Deployment
```

No prose validation, link check, or CI docs gate was run in this pass.

## Source Doc Sets

### Getting Started

Source docs under `Docs/Getting_Started/` currently cover:

- `README.md`
- `QUICKSTART.md`
- `ARCHITECTURE.md`
- `TROUBLESHOOTING.md`
- `Profile_Local_Single_User.md`
- `Profile_Docker_Single_User.md`
- `Profile_Docker_Multi_User_Postgres.md`
- `First_Time_Audio_Setup_CPU.md`
- `First_Time_Audio_Setup_GPU_Accelerated.md`
- `GPU_STT_Addon.md`
- `Getting-Started-with-HA-Guide.md`
- `onboarding_manifest.yaml`

Published mirrors under `Docs/Published/Getting_Started/` exist for the profile and audio setup docs, `README.md`, and `onboarding_manifest.yaml`.

Source-only signals in this checkout:

- `QUICKSTART.md`
- `ARCHITECTURE.md`
- `TROUBLESHOOTING.md`
- `Getting-Started-with-HA-Guide.md`

These should not be copied into `Docs/Published` until the publishing flow is confirmed.

### Deployment

Source docs under `Docs/Deployment/` currently cover:

- first-time production setup
- long-term admin operations
- minimal deployment
- horizontal scaling
- offline and air-gapped operation
- Postgres migration
- resource requirements
- setup wizard
- embeddings deployment
- OpenAI-compatible strict mode
- reverse proxy examples
- CDN/static assets
- sidecar workers and sidecar templates
- database and FTS SQL references
- operations runbooks
- monitoring, alerts, scrape samples, exemplars, and nightly eval fixtures
- systemd and launchd worker service files

Published mirrors under `Docs/Published/Deployment/` exist for most core deployment docs, operations docs, database/FTS references, systemd files, and launchd files.

Source-only signals in this checkout:

- `Docs/Deployment/Monitoring/`
- some monitoring alert/eval sample artifacts

Follow-up inspection found that `Helper_Scripts/refresh_docs_published.sh` promotes `Docs/Deployment/Monitoring` to top-level `Docs/Published/Monitoring`, not to `Docs/Published/Deployment/Monitoring`. The monitoring tree should still be reviewed as operational reference material before deciding whether that publishing shape is correct.

## Deployment Mode Coverage

| Deployment mode | Source docs | Published mirror signal | Review need |
| --- | --- | --- | --- |
| Local single-user | `Profile_Local_Single_User.md`, `QUICKSTART.md`, `TROUBLESHOOTING.md` | Profile mirrored; quickstart/troubleshooting source-only | Confirm whether quickstart remains source-only or becomes published onboarding. |
| Docker single-user + WebUI | `Profile_Docker_Single_User.md`, `minimal-deploy.md`, `Reverse_Proxy_Examples.md`, `cdn-static-assets.md` | Core docs mirrored | Check Docker/WebUI commands against current compose files before prose edits. |
| Docker multi-user + Postgres | `Profile_Docker_Multi_User_Postgres.md`, `Postgres_Migration_Guide.md`, `Database/postgres-rls-policies.sql` | Core docs and SQL mirrored | Confirm AuthNZ/Postgres fixture guidance matches current test and deploy setup. |
| First-time production | `First_Time_Production_Setup.md`, `Long_Term_Admin_Guide.md`, `resource-requirements.md`, `setup-wizard-guide.md` | Mirrored | Assign docs owner review before changing production hardening language. |
| Horizontal scaling / HA | `horizontal-scaling.md`, `Sidecar_Workers.md`, `Getting-Started-with-HA-Guide.md` | Horizontal scaling and sidecar docs mirrored; HA guide source-only | Decide whether the HA guide is canonical, draft, or onboarding-only before publishing. |
| Offline / air-gapped | `offline-air-gapped.md` | Mirrored | Validate provider/model download assumptions before edits. |
| Sidecar workers | `Sidecar_Workers.md`, `Sidecar_Workers.template.md`, `systemd/`, `launchd/` | Mirrored | Treat service files as deploy artifacts; validate with owner before formatting churn. |
| Audio and GPU setup | CPU/GPU first-time audio docs, `GPU_STT_Addon.md` | Mirrored | Validate against current optional extras and model setup docs. |
| Monitoring and alerts | `Monitoring/` | Refreshed to top-level `Docs/Published/Monitoring` by the publishing script | Decide whether top-level published monitoring docs are the intended operator-doc shape. |
| OpenAI compatibility | `OpenAI_Compat_Strict_Mode.md` | Mirrored | Ensure strict-mode docs stay aligned with API compatibility tests. |

## Risk Flags

- The docs publishing flow is now identified, but owner confirmation is still needed before Phase 4.2 edits. Editing both source and published mirrors manually could create drift.
- Onboarding/docs gates are active in related PR CI, so broad docs rewrites could make failures harder to triage.
- The HA guide appears in source but not in published docs. Confirm whether it is ready for published docs before mirroring it.
- Monitoring docs are promoted to top-level published docs by the refresh script. Confirm whether that is intentional before changing the publishing shape.
- Service files under `systemd/` and `launchd/` are deployment artifacts, not prose. Avoid reformatting them unless a validation command exists.

## Recommended First Phase 4.2 Slice

When Phase 4.2 is unblocked:

1. Confirm whether source docs are canonical and whether `Docs/Published` is generated or manually maintained.
2. Pick source-only edits first, preferably a deployment mode matrix in `Docs/Getting_Started/README.md` or `Docs/Deployment/First_Time_Production_Setup.md`.
3. Validate links and commands in the edited source docs.
4. Update published mirrors only through the accepted publishing flow.
5. Run the onboarding/docs gate that owns these files before PR handoff.

Draft refresh plan:

- `Docs/superpowers/plans/2026-04-25-phase4-2-deployment-docs-refresh-plan.md`

## Do Not Do Yet

- Do not rewrite the deployment docs while Phase 2/3 closeout and PR `#1125` remain unstable.
- Do not manually mirror source docs to `Docs/Published` until the publishing flow is confirmed.
- Do not change service files without a validation path.
- Do not mix deployment docs cleanup with runtime deployment behavior changes.

## Handoff Checklist

- [ ] Docs owner confirms canonical source and published mirror flow.
- [ ] Owner decides whether the HA guide should be published.
- [ ] Owner decides whether monitoring docs should be published.
- [ ] Deployment mode matrix is reviewed against current scripts and compose files.
- [x] Onboarding/docs gate command is identified before edits.
- [ ] Onboarding/docs gate command is run after edits.
